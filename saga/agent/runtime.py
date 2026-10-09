from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from saga.agent.augmentation_planner import apply_selected_plan_to_intent, build_augmentation_plan, should_auto_apply_plan
from saga.agent.benefit_evidence import build_benefit_evidence_report
from saga.agent.benefit_estimator import build_benefit_context
from saga.agent.candidate_pilot import build_candidate_recipe_pilot
from saga.agent.intent_recognizer import recognize_request
from saga.agent.llm_client import LLMConfig
from saga.agent.llm_intent import build_intent_failure_result, run_llm_intent_recognizer
from saga.agent.llm_planner import build_planner_failure_result, run_llm_planner
from saga.agent.multi_dataset_profile import build_multi_dataset_profile
from saga.agent.plan_critic import critique_agent_plan
from saga.agent.plan_verifier import verify_plan
from saga.agent.planning_evidence import build_planning_evidence
from saga.agent.recipe_generator import generate_recipe_from_agent_state
from saga.agent.request_constraints import infer_request_constraints
from saga.agent.skill_registry import save_skill_registry
from saga.core.config import load_llm_config, save_json, save_text
from saga.core.provenance import build_run_provenance
from saga.core.saga_config import load_saga_config
from saga.data.format_bridge import compile_and_validate_format
from saga.data.profile import profile_dataset
from saga.executor.recipe_executor import execute_recipe
from saga.memory import record_agent_run_memory
from saga.memory.policy_bank import retrieve_memory_context


AGENT_RUN_VERSION = "saga_agent_run_v1"


def run_agent(
    request: str,
    output_dir: str | Path,
    dataset_root: str | Path | None = None,
    style_transfer_config: str | Path | None = None,
    diffusion_lora_config: str | Path | None = None,
    traditional_aug_config: str | Path | None = None,
    saga_config_path: str | Path | None = None,
    llm_config: LLMConfig | None = None,
    use_llm_intent: bool = False,
    use_llm_planner: bool = False,
    memory_dir: str | Path = "runs/memory",
    use_memory: bool = True,
    dry_run: bool = True,
    sample_limit: int = 120,
    image_probe_limit: int = 200,
    bridge_sample_limit: int = 500,
    execute: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    saga_config = load_saga_config(saga_config_path)
    save_json(output_path / "saga_config_effective.json", saga_config)
    llm_runtime_config = saga_config.get("llm") or {}
    use_llm_intent = bool(use_llm_intent or llm_runtime_config.get("use_intent_by_default", False))
    use_llm_planner = bool(use_llm_planner or llm_runtime_config.get("use_planner_by_default", False))
    if llm_config is None and (use_llm_intent or use_llm_planner) and llm_runtime_config.get("config"):
        llm_config = LLMConfig.from_mapping(load_llm_config(llm_runtime_config["config"]))
    run_provenance = build_run_provenance(output_path) if saga_config.get("execution", {}).get("record_provenance", True) else {}
    memory_config = saga_config.get("memory") or {}
    if memory_dir == "runs/memory":
        memory_dir = memory_config.get("dir", memory_dir)
    memory_enabled = bool(memory_config.get("enabled", True))
    retrieve_memory = bool(use_memory and memory_enabled)
    memory_limit = int(memory_config.get("retrieval_limit", 5))
    skill_configs = saga_config.get("skill_configs") or {}
    style_transfer_config = style_transfer_config or skill_configs.get("style_transfer")
    diffusion_lora_config = diffusion_lora_config or skill_configs.get("diffusion_lora")
    traditional_aug_config = traditional_aug_config or skill_configs.get("traditional_augmentation")
    model_to_pov_config = skill_configs.get("model_to_pov_scene")
    geodiff_sar_config = skill_configs.get("geodiff_sar")
    gaussian_splatting_config = skill_configs.get("gaussian_splatting")
    classification_eval_config = skill_configs.get("classification_evaluation")
    background_generation_config = skill_configs.get("background_generation")
    save_skill_registry(output_path)

    request_result = recognize_request(text=request, output_dir=output_path, dataset_root=dataset_root)
    intent_spec_for_run = request_result["intent_spec"]
    format_hints_for_run = request_result["format_hints"]
    intent_effective_spec_path = output_path / "intent_spec.json"
    llm_intent_result = None
    if use_llm_intent:
        if llm_config is None:
            raise ValueError("agent-run --use-llm-intent requires --llm-config.")
        try:
            llm_intent_result = run_llm_intent_recognizer(
                text=request,
                rule_intent_spec=request_result["intent_spec"],
                rule_format_hints=request_result["format_hints"],
                llm_config=llm_config,
                output_dir=output_path,
                dataset_root=dataset_root,
            )
        except Exception as exc:
            llm_intent_result = build_intent_failure_result(
                rule_intent_spec=request_result["intent_spec"],
                rule_format_hints=request_result["format_hints"],
                error=exc,
                output_dir=output_path,
            )
        intent_spec_for_run = llm_intent_result["effective_intent_spec"]
        format_hints_for_run = llm_intent_result["merged_format_hints"]
        intent_effective_spec_path = output_path / "intent_effective_spec.json"
        save_json(output_path / "format_hints.json", format_hints_for_run)

    intent = intent_spec_for_run.get("intent", {})
    request_constraints = infer_request_constraints(intent_spec_for_run, output_dir=output_path)
    root = Path(dataset_root or infer_dataset_root_from_intent(intent)).expanduser().resolve()
    if intent.get("task") == "raysar_synthesis" and root.exists() and root.is_file():
        root = root.parent.resolve()

    profile_report = profile_dataset(
        root=root,
        output_dir=output_path,
        request=request,
        intent_spec=intent_spec_for_run,
        format_hints=format_hints_for_run,
        sample_limit=sample_limit,
        image_probe_limit=image_probe_limit,
    )
    bridge_report = compile_and_validate_format(
        root=root,
        output_dir=output_path,
        raw_profile=load_json_like(output_path / "raw_profile.json"),
        intent_spec=intent_spec_for_run,
        format_hints=format_hints_for_run,
        sample_limit=bridge_sample_limit,
    )
    dataset_profile = load_json_like(output_path / "dataset_profile.json")
    raw_profile = load_json_like(output_path / "raw_profile.json")
    validated_profile = load_json_like(output_path / "validated_dataset_profile.json")
    multi_dataset_profile = build_multi_dataset_profile(
        intent_spec=intent_spec_for_run,
        primary_raw_profile=raw_profile,
        primary_dataset_profile=dataset_profile,
        output_dir=output_path,
        request=request,
        format_hints=format_hints_for_run,
        sample_limit=max(20, min(sample_limit, 80)),
        image_probe_limit=max(20, min(image_probe_limit, 120)),
    )
    planning_evidence = build_planning_evidence(
        dataset_root=root,
        output_dir=output_path,
        validated_profile=validated_profile,
        intent_spec=intent_spec_for_run,
        dry_run=True,
    )
    memory_context = (
        retrieve_memory_context(
            raw_profile=raw_profile,
            task=intent_spec_for_run.get("intent", {}).get("task"),
            dataset_root=root.as_posix(),
            memory_dir=memory_dir,
            limit=memory_limit,
        )
        if retrieve_memory
        else {"schema_version": "saga_memory_retrieval_v1", "match_count": 0, "matches": []}
    )
    save_json(output_path / "memory_retrieval.json", memory_context)
    benefit_context = build_benefit_context(
        raw_profile=raw_profile,
        validated_profile=validated_profile,
        bridge_report=bridge_report,
        intent_spec=intent_spec_for_run,
        memory_context=memory_context,
        output_dir=output_path,
    )
    benefit_context["planning_evidence"] = planning_evidence
    augmentation_plan = build_augmentation_plan(
        intent_spec=intent_spec_for_run,
        benefit_context=benefit_context,
        request_constraints=request_constraints,
        multi_dataset_profile=multi_dataset_profile,
        memory_context=memory_context,
        output_dir=output_path,
        planner_config=saga_config.get("planner") or {},
    )
    benefit_context["augmentation_plan"] = augmentation_plan
    planner_config = saga_config.get("planner") or {}
    if (
        not use_llm_planner
        and planner_config.get("auto_apply_selected_augmentation_plan", True)
        and should_auto_apply_plan(intent_spec_for_run, augmentation_plan)
    ):
        intent_spec_for_run = apply_selected_plan_to_intent(intent_spec_for_run, augmentation_plan)
        intent_effective_spec_path = output_path / "intent_effective_spec.json"
        save_json(intent_effective_spec_path, intent_spec_for_run)

    planner_result = None
    plan_verification = None
    effective_intent_spec_path = intent_effective_spec_path
    if use_llm_planner:
        if llm_config is None:
            raise ValueError("agent-run --use-llm-planner requires --llm-config.")
        try:
            planner_result = run_llm_planner(
                request=request,
                intent_spec=intent_spec_for_run,
                raw_profile=raw_profile,
                dataset_profile=dataset_profile,
                bridge_report=bridge_report,
                validated_profile=validated_profile,
                llm_config=llm_config,
                output_dir=output_path,
                memory_context=memory_context,
                benefit_context=benefit_context,
            )
        except Exception as exc:
            planner_result = build_planner_failure_result(
                intent_spec=intent_spec_for_run,
                error=exc,
                output_dir=output_path,
            )
        effective_intent_spec_path = output_path / "effective_intent_spec.json"
        plan_verification = verify_plan(
            proposal=planner_result["proposal"],
            guardrail=planner_result["guardrail"],
            bridge_report=bridge_report,
            output_dir=output_path,
        )
    planner_guardrail = planner_result.get("guardrail", {}) if planner_result else {}
    plan_ranking = planner_result.get("plan_ranking", {}) if planner_result else {}

    state = {
        "schema_version": AGENT_RUN_VERSION,
        "request": request,
        "output_dir": output_path.as_posix(),
        "dataset_root": root.as_posix(),
        "task": intent_spec_for_run.get("intent", {}).get("task"),
        "dry_run": dry_run,
        "saga_config_path": str(saga_config_path) if saga_config_path else "configs/saga.yaml",
        "saga_config_effective_path": (output_path / "saga_config_effective.json").as_posix(),
        "run_provenance_path": (output_path / "run_provenance.json").as_posix() if run_provenance else None,
        "intent_spec_path": (output_path / "intent_spec.json").as_posix(),
        "intent_effective_spec_path": intent_effective_spec_path.as_posix(),
        "intent_mode": "llm_intent_proposal_rule_guardrail" if use_llm_intent else "rule_based",
        "intent_llm_proposal_path": (output_path / "intent_llm_proposal.json").as_posix() if llm_intent_result else None,
        "intent_guardrail_path": (output_path / "intent_guardrail.json").as_posix() if llm_intent_result else None,
        "effective_intent_spec_path": effective_intent_spec_path.as_posix(),
        "planner_mode": "llm_proposal_rule_guardrail" if use_llm_planner else "rule_based",
        "planner_proposal_path": (output_path / "planner_proposal.json").as_posix() if planner_result else None,
        "planner_guardrail_path": (output_path / "planner_guardrail.json").as_posix() if planner_result else None,
        "planner_clarification_path": (output_path / "planner_clarification.json").as_posix() if planner_result else None,
        "plan_verification_path": (output_path / "plan_verification.json").as_posix() if plan_verification else None,
        "plan_verification_status": plan_verification.get("status") if plan_verification else None,
        "skill_registry_path": (output_path / "skill_registry.json").as_posix(),
        "memory_retrieval_path": (output_path / "memory_retrieval.json").as_posix(),
        "memory_match_count": memory_context.get("match_count", 0),
        "planning_evidence_path": (output_path / "planning_evidence" / "planning_evidence.json").as_posix(),
        "request_constraints_path": (output_path / "request_constraints.json").as_posix(),
        "multi_dataset_profile_path": (output_path / "multi_dataset_profile.json").as_posix(),
        "augmentation_plan_path": (output_path / "augmentation_plan.json").as_posix(),
        "augmentation_selected_plan_id": augmentation_plan.get("selected_plan_id"),
        "augmentation_selected_skill": augmentation_plan.get("selected_skill"),
        "augmentation_selected_recipe_task": augmentation_plan.get("selected_recipe_task"),
        "dataset_need_profile_path": (output_path / "dataset_need_profile.json").as_posix(),
        "skill_effect_cards_path": (output_path / "skill_effect_cards.json").as_posix(),
        "skill_utility_path": (output_path / "skill_utility.json").as_posix(),
        "plan_utility_report_path": (output_path / "plan_utility_report.json").as_posix() if planner_result else None,
        "selected_plan_utility_score": plan_ranking.get("selected_utility_score"),
        "planner_decision": planner_guardrail.get("decision"),
        "planner_execution_allowed": planner_guardrail.get("execution_allowed"),
        "planner_requires_clarification": planner_guardrail.get("requires_clarification"),
        "planner_question_count": len(planner_guardrail.get("clarification_questions") or []),
        "format_hints_path": (output_path / "format_hints.json").as_posix(),
        "raw_profile_path": (output_path / "raw_profile.json").as_posix(),
        "dataset_profile_path": (output_path / "dataset_profile.json").as_posix(),
        "format_bridge_path": (output_path / "format_bridge.json").as_posix(),
        "validated_profile_path": (output_path / "validated_dataset_profile.json").as_posix(),
        "profile_status": profile_report.get("dataset_profile", {}).get("status"),
        "bridge_valid": bridge_report.get("valid"),
        "planner_ready": bridge_report.get("profile_level_after_validation") == 2,
    }
    recipe_path = output_path / "recipe.yaml"
    recipe = generate_recipe_from_agent_state(
        state=state,
        output_path=recipe_path,
        style_transfer_config=style_transfer_config,
        diffusion_lora_config=diffusion_lora_config,
        traditional_aug_config=traditional_aug_config,
        model_to_pov_config=model_to_pov_config,
        geodiff_sar_config=geodiff_sar_config,
        gaussian_splatting_config=gaussian_splatting_config,
        background_generation_config=background_generation_config,
        classification_eval_config=classification_eval_config,
        max_repair_trials=int((saga_config.get("execution") or {}).get("max_repair_trials", 3)),
    )
    state["recipe_path"] = recipe_path.as_posix()
    state["recipe_id"] = recipe.recipe_id
    state["candidate_pilot_path"] = None
    state["plan_critic_path"] = None
    state["benefit_evidence_path"] = None

    candidate_pilot = build_candidate_recipe_pilot(
        run_dir=output_path,
        output_dir=output_path / "candidate_pilot",
        top_k=int((saga_config.get("planner") or {}).get("pilot_top_k", 3)),
        pilot_sample_count=int((saga_config.get("planner") or {}).get("pilot_sample_count", 8)),
        execute=False,
        dry_run=True,
    )
    state["candidate_pilot_path"] = (output_path / "candidate_pilot" / "candidate_pilot.json").as_posix()
    state["candidate_pilot_count"] = candidate_pilot.get("candidate_count")

    plan_critic = critique_agent_plan(
        run_dir=output_path,
        output_dir=output_path,
        state=state,
        bridge_report=bridge_report,
        augmentation_plan=augmentation_plan,
        recipe=recipe.to_dict(),
        request_constraints=request_constraints,
    )
    state["plan_critic_path"] = (output_path / "plan_critic.json").as_posix()
    state["plan_critic_status"] = plan_critic.get("status")
    state["plan_critic_high_findings"] = (plan_critic.get("severity_counts") or {}).get("high", 0)

    execution_report = None
    should_execute = execute
    if execute and not dry_run and (
        planner_blocks_real_execution(planner_guardrail, plan_verification)
        or plan_critic.get("status") == "blocked"
    ):
        should_execute = False
        state["execution_status"] = "blocked_by_planner"
        state["execution_blocked_reason"] = (
            "PlanCritic found high-severity issues."
            if plan_critic.get("status") == "blocked"
            else planner_block_reason(planner_guardrail, plan_verification)
        )
        state["execution_report_path"] = None

    if should_execute:
        execution_report = execute_recipe(
            recipe_path=recipe_path,
            output_dir=output_path / "execution",
            dry_run=dry_run,
        )
        state["execution_report_path"] = (output_path / "execution" / "recipe_execution.json").as_posix()
        state["execution_status"] = execution_report.get("status")

    benefit_evidence = build_benefit_evidence_report(
        run_dir=output_path,
        output_dir=output_path,
        execution_report=execution_report,
        state=state,
    )
    state["benefit_evidence_path"] = (output_path / "benefit_evidence.json").as_posix()
    state["benefit_evidence_level"] = benefit_evidence.get("evidence_level")
    state["benefit_evidence_label"] = benefit_evidence.get("evidence_label")
    state["downstream_claim_allowed"] = benefit_evidence.get("downstream_claim_allowed")
    plan_critic = critique_agent_plan(
        run_dir=output_path,
        output_dir=output_path,
        state=state,
        bridge_report=bridge_report,
        augmentation_plan=augmentation_plan,
        recipe=recipe.to_dict(),
        request_constraints=request_constraints,
        benefit_evidence=benefit_evidence,
    )
    state["plan_critic_status"] = plan_critic.get("status")
    state["plan_critic_high_findings"] = (plan_critic.get("severity_counts") or {}).get("high", 0)

    state["elapsed_seconds"] = round(time.time() - started, 3)
    save_json(output_path / "agent_state.json", state)
    save_text(output_path / "agent_run.md", render_agent_run_markdown(state, bridge_report, execution_report))
    memory_report = record_agent_run_memory(output_path, memory_dir=memory_dir) if memory_enabled else {}
    state["memory_report_path"] = (
        (Path(memory_report["memory_dir"]) / "policy_memory_latest.json").as_posix()
        if memory_report
        else None
    )
    save_json(output_path / "agent_state.json", state)
    save_text(output_path / "agent_run.md", render_agent_run_markdown(state, bridge_report, execution_report))
    return state


def planner_blocks_real_execution(guardrail: dict[str, Any], plan_verification: dict[str, Any] | None = None) -> bool:
    if not guardrail:
        return False
    if guardrail.get("requires_clarification"):
        return True
    if guardrail.get("valid") is False:
        return True
    if plan_verification and plan_verification.get("status") != "passed":
        return True
    return False


def planner_block_reason(guardrail: dict[str, Any], plan_verification: dict[str, Any] | None = None) -> str:
    if guardrail.get("requires_clarification"):
        return "Planner requires clarification before real execution."
    if guardrail.get("valid") is False:
        return "Planner guardrail did not validate the LLM proposal."
    if plan_verification and plan_verification.get("status") != "passed":
        return f"Plan verification status is {plan_verification.get('status')!r}; real execution requires 'passed'."
    return "Planner blocked real execution."


def infer_dataset_root_from_intent(intent: dict[str, Any]) -> str:
    if intent.get("task") == "raysar_synthesis":
        scene_or_contributions = (
            intent.get("pov_scene")
            or intent.get("model_file")
            or intent.get("contributions_txt")
            or intent.get("dataset_source")
        )
        if scene_or_contributions:
            path = Path(str(scene_or_contributions)).expanduser()
            return str(path.parent if path.suffix else path)
    if intent.get("task") == "background_generation":
        return "myproject/scene_gen_segment_large/data/controls"
    if intent.get("task") == "geodiff_sar_generation":
        dataset_source = intent.get("dataset_source")
        if dataset_source:
            return str(dataset_source)
        model_file = intent.get("model_file")
        if model_file:
            return str(Path(str(model_file)).expanduser().parent)
    baseline_dataset = intent.get("baseline_dataset")
    if baseline_dataset:
        return str(baseline_dataset)
    dataset_source = intent.get("dataset_source")
    if dataset_source:
        return str(dataset_source)
    content_source = intent.get("content_source")
    if content_source:
        return str(content_source)
    raise ValueError("agent-run needs --dataset-root when the request does not include a content dataset path.")


def load_json_like(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_agent_run_markdown(
    state: dict[str, Any],
    bridge_report: dict[str, Any],
    execution_report: dict[str, Any] | None,
) -> str:
    lines = [
        "# SAGA Agent Run",
        "",
        f"- Request: {state['request']}",
        f"- Dataset root: `{state['dataset_root']}`",
        f"- Dry run: {state['dry_run']}",
        f"- Intent mode: `{state.get('intent_mode')}`",
        f"- Planner mode: `{state.get('planner_mode')}`",
        f"- Profile status: {state.get('profile_status')}",
        f"- Bridge valid: {state.get('bridge_valid')}",
        f"- Planner ready: {state.get('planner_ready')}",
        f"- Augmentation selected skill: `{state.get('augmentation_selected_skill')}`",
        f"- Augmentation selected recipe task: `{state.get('augmentation_selected_recipe_task')}`",
        f"- Recipe: `{state.get('recipe_path')}`",
        "",
        "## Artifacts",
        "",
        f"- Intent: `{state['intent_spec_path']}`",
        f"- Intent effective: `{state.get('intent_effective_spec_path')}`",
        f"- Raw profile: `{state['raw_profile_path']}`",
        f"- Effective intent: `{state.get('effective_intent_spec_path')}`",
        f"- Format bridge: `{state['format_bridge_path']}`",
        f"- Validated profile: `{state['validated_profile_path']}`",
    ]
    if state.get("saga_config_effective_path"):
        lines.append(f"- SAGA config: `{state.get('saga_config_effective_path')}`")
    if state.get("run_provenance_path"):
        lines.append(f"- Run provenance: `{state.get('run_provenance_path')}`")
    if state.get("request_constraints_path"):
        lines.append(f"- Request constraints: `{state.get('request_constraints_path')}`")
    if state.get("multi_dataset_profile_path"):
        lines.append(f"- Multi-dataset profile: `{state.get('multi_dataset_profile_path')}`")
    if state.get("augmentation_plan_path"):
        lines.append(f"- Augmentation plan: `{state.get('augmentation_plan_path')}`")
    if state.get("candidate_pilot_path"):
        lines.append(f"- Candidate pilot: `{state.get('candidate_pilot_path')}` ({state.get('candidate_pilot_count')} candidates)")
    if state.get("plan_critic_path"):
        lines.append(f"- Plan critic: `{state.get('plan_critic_path')}` ({state.get('plan_critic_status')})")
    if state.get("benefit_evidence_path"):
        lines.append(
            f"- Benefit evidence: `{state.get('benefit_evidence_path')}` "
            f"(level {state.get('benefit_evidence_level')}: {state.get('benefit_evidence_label')})"
        )
    if state.get("intent_llm_proposal_path"):
        lines.append(f"- LLM intent proposal: `{state.get('intent_llm_proposal_path')}`")
    if state.get("intent_guardrail_path"):
        lines.append(f"- Intent guardrail: `{state.get('intent_guardrail_path')}`")
    if state.get("planner_proposal_path"):
        lines.append(f"- Planner proposal: `{state.get('planner_proposal_path')}`")
    if state.get("planner_guardrail_path"):
        lines.append(f"- Planner guardrail: `{state.get('planner_guardrail_path')}`")
    if state.get("planner_clarification_path"):
        lines.append(f"- Planner clarification: `{state.get('planner_clarification_path')}`")
    if state.get("plan_verification_path"):
        lines.append(f"- Plan verification: `{state.get('plan_verification_path')}`")
    if state.get("skill_registry_path"):
        lines.append(f"- Skill registry: `{state.get('skill_registry_path')}`")
    if state.get("memory_retrieval_path"):
        lines.append(f"- Memory retrieval: `{state.get('memory_retrieval_path')}` ({state.get('memory_match_count')} matches)")
    if state.get("planning_evidence_path"):
        lines.append(f"- Planning evidence: `{state.get('planning_evidence_path')}`")
    if state.get("dataset_need_profile_path"):
        lines.append(f"- Dataset need profile: `{state.get('dataset_need_profile_path')}`")
    if state.get("skill_effect_cards_path"):
        lines.append(f"- Skill effect cards: `{state.get('skill_effect_cards_path')}`")
    if state.get("skill_utility_path"):
        lines.append(f"- Skill utility: `{state.get('skill_utility_path')}`")
    if state.get("plan_utility_report_path"):
        lines.append(f"- Plan utility: `{state.get('plan_utility_report_path')}`")
    if state.get("planner_decision"):
        lines.extend(
            [
                "",
                "## Planner",
                "",
                f"- Decision: `{state.get('planner_decision')}`",
                f"- Plan verification: `{state.get('plan_verification_status')}`",
                f"- Selected plan utility: {state.get('selected_plan_utility_score')}",
                f"- Execution allowed: {state.get('planner_execution_allowed')}",
                f"- Requires clarification: {state.get('planner_requires_clarification')}",
                f"- Question count: {state.get('planner_question_count')}",
                f"- Plan critic: `{state.get('plan_critic_status')}` "
                f"({state.get('plan_critic_high_findings')} high findings)",
                f"- Benefit evidence: level {state.get('benefit_evidence_level')} "
                f"`{state.get('benefit_evidence_label')}`",
                f"- Downstream claim allowed: {state.get('downstream_claim_allowed')}",
            ]
        )
    lines.extend(["", "## Validation", ""])
    validation = bridge_report.get("validation", {})
    lines.append(f"- Profile level after validation: {bridge_report.get('profile_level_after_validation')}")
    lines.append(f"- Required fields: `{bridge_report.get('required_fields')}`")
    lines.append(f"- Field coverage: `{validation.get('field_coverage')}`")
    lines.append(f"- Missing required count: {validation.get('missing_required_count')}")
    if execution_report:
        lines.extend(["", "## Execution", ""])
        lines.append(f"- Status: {execution_report.get('status')}")
        lines.append(f"- Report: `{state.get('execution_report_path')}`")
        for step in execution_report.get("steps", []):
            lines.append(f"- `{step.get('step_id')}` {step.get('skill')} -> {step.get('status')}")
    elif state.get("execution_status") == "blocked_by_planner":
        lines.extend(["", "## Execution", ""])
        lines.append("- Status: `blocked_by_planner`")
        lines.append(f"- Reason: {state.get('execution_blocked_reason')}")
    lines.append("")
    return "\n".join(lines)

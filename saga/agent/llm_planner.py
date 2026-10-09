from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from saga.agent.json_utils import loads_json_object
from saga.agent.llm_client import LLMConfig, OpenAICompatibleClient
from saga.agent.plan_ranker import normalize_candidate_plans, rank_candidate_plans, selected_candidate_from_ranking, write_plan_ranking
from saga.agent.planning_evidence import compact_planning_evidence
from saga.agent.benefit_estimator import write_plan_utility_report
from saga.agent.skill_registry import executable_skill_names, planned_skill_names, skill_registry_for_prompt
from saga.core.config import save_json, save_text


LLM_PLANNER_VERSION = "saga_llm_planner_proposal_v1"
LLM_PLANNER_GUARDRAIL_VERSION = "saga_llm_planner_guardrail_v1"

EXECUTABLE_SKILLS = executable_skill_names()
PLANNED_SKILLS = planned_skill_names()

SUPPORTED_RECIPE_TASKS = {
    "style_transfer",
    "diffusion_lora_generation",
    "geodiff_sar_generation",
    "gaussian_splatting_completion",
    "gan_generation",
    "traditional_augmentation",
    "pseudocolor_transform",
    "background_generation",
    "target_background_composition",
    "raysar_synthesis",
    "classification_evaluation",
    "classification",
    "object_detection",
    "segmentation",
    "augmentation_or_generation",
    "unknown",
}

EXECUTABLE_RECIPE_TASKS = {
    "style_transfer",
    "diffusion_lora_generation",
    "geodiff_sar_generation",
    "gaussian_splatting_completion",
    "gan_generation",
    "traditional_augmentation",
    "pseudocolor_transform",
    "background_generation",
    "target_background_composition",
    "raysar_synthesis",
    "classification_evaluation",
}

MODEL_FAMILIES = {"flux", "sd3", "sdxl"}
SAFE_PATH_RE = re.compile(r"^[\w\u4e00-\u9fff./@:+()（）\[\]\- ]+$")


def run_llm_planner(
    request: str,
    intent_spec: dict[str, Any],
    raw_profile: dict[str, Any],
    dataset_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    validated_profile: dict[str, Any],
    llm_config: LLMConfig,
    output_dir: str | Path | None = None,
    memory_context: dict[str, Any] | None = None,
    benefit_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask an LLM for a high-level plan, then guardrail it into planner artifacts.

    The LLM output is a proposal, not an executable recipe. Deterministic code
    decides which fields can influence the effective intent used by the recipe
    compiler.
    """

    client = OpenAICompatibleClient(llm_config)
    prompt = build_planner_prompt(
        request=request,
        intent_spec=intent_spec,
        raw_profile=raw_profile,
        dataset_profile=dataset_profile,
        bridge_report=bridge_report,
        validated_profile=validated_profile,
        memory_context=memory_context,
        benefit_context=benefit_context,
    )
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are SAGA's high-level SAR data augmentation planner. "
                    "Return only JSON. You propose intent and recipe strategy, "
                    "but you never execute commands and never claim unvalidated "
                    "dataset semantics as facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw_response = loads_json_object(content)
    candidate_plans = normalize_candidate_plans(raw_response)
    plan_ranking = rank_candidate_plans(
        candidate_plans,
        intent_spec=intent_spec,
        bridge_report=bridge_report,
        benefit_context=benefit_context,
    )
    proposal = normalize_planner_proposal(selected_candidate_from_ranking(candidate_plans, plan_ranking))
    guardrail = validate_planner_proposal(proposal, intent_spec=intent_spec, bridge_report=bridge_report)
    effective_intent_spec = apply_planner_proposal_to_intent(
        intent_spec=intent_spec,
        proposal=proposal,
        guardrail=guardrail,
    )
    result = {
        "schema_version": LLM_PLANNER_VERSION,
        "mode": "llm_proposal_rule_guardrail",
        "candidate_plans": candidate_plans,
        "plan_ranking": plan_ranking,
        "benefit_context_summary": compact_benefit_context(benefit_context or {}),
        "proposal": proposal,
        "guardrail": guardrail,
        "effective_intent_spec": effective_intent_spec,
    }
    if output_dir:
        write_planner_artifacts(output_dir, result)
    return result


def build_planner_failure_result(
    intent_spec: dict[str, Any],
    error: Exception,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    intent = intent_spec.get("intent", {})
    proposal = {
        "schema_version": LLM_PLANNER_VERSION,
        "plan_id": "llm_failed",
        "planner": "llm_failed",
        "task": intent.get("task") or "unknown",
        "confidence": 0.0,
        "goals": list(intent.get("goals") or []),
        "strategy": "LLM planner failed; SAGA will fall back to the deterministic rule planner.",
        "recipe_intent_updates": {},
        "pipeline": [],
        "required_validations": [],
        "risks": ["llm_planner_failed"],
        "clarification_questions": [],
        "rationale": f"{type(error).__name__}: {error}",
    }
    guardrail = {
        "schema_version": LLM_PLANNER_GUARDRAIL_VERSION,
        "valid": False,
        "selected_task": proposal["task"],
        "rule_task": intent.get("task"),
        "executable_recipe_task": proposal["task"] in EXECUTABLE_RECIPE_TASKS,
        "executable_steps": [],
        "planned_skills": [],
        "unsupported_skills": [],
        "issues": [
            {
                "severity": "high",
                "name": "llm_planner_failed",
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
            }
        ],
        "blocking_issue_count": 1,
        "warning_issue_count": 0,
        "clarification_questions": [],
        "requires_clarification": False,
        "execution_allowed": False,
        "decision": "fallback_to_rule_planner",
    }
    effective_intent_spec = apply_planner_proposal_to_intent(
        intent_spec=intent_spec,
        proposal=proposal,
        guardrail=guardrail,
    )
    result = {
        "schema_version": LLM_PLANNER_VERSION,
        "mode": "llm_proposal_rule_guardrail",
        "candidate_plans": [proposal],
        "plan_ranking": {
            "schema_version": "saga_plan_ranking_v1",
            "candidate_count": 1,
            "selected_plan_id": proposal.get("plan_id", "llm_failed"),
            "selected_task": proposal.get("task"),
            "ranked_plans": [
                {
                    "plan_id": proposal.get("plan_id", "llm_failed"),
                    "task": proposal.get("task"),
                    "score": -100,
                    "rank_reasons": ["llm_planner_failed"],
                    "candidate": proposal,
                }
            ],
        },
        "proposal": proposal,
        "guardrail": guardrail,
        "effective_intent_spec": effective_intent_spec,
    }
    if output_dir:
        write_planner_artifacts(output_dir, result)
    return result


def build_planner_prompt(
    request: str,
    intent_spec: dict[str, Any],
    raw_profile: dict[str, Any],
    dataset_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    validated_profile: dict[str, Any],
    memory_context: dict[str, Any] | None = None,
    benefit_context: dict[str, Any] | None = None,
) -> str:
    context = {
        "user_request": request,
        "current_rule_intent_spec": intent_spec,
        "raw_dataset_profile_summary": compact_raw_profile(raw_profile),
        "dataset_profile_summary": compact_dataset_profile(dataset_profile),
        "format_bridge_summary": compact_bridge_report(bridge_report),
        "validated_profile_summary": compact_validated_profile(validated_profile),
        "memory_retrieval_summary": compact_memory_context(memory_context or {}),
        "benefit_context_summary": compact_benefit_context(benefit_context or {}),
        "available_skill_registry": build_skill_registry(),
            "planner_contract": {
                "llm_role": [
                    "infer high-level task and user goal",
                    "propose two to four candidate recipe strategies unless the user explicitly names exactly one skill",
                    "use dataset needs and skill utility evidence to prefer plans likely to help downstream tasks",
                    "compare alternatives by expected downstream value, controllability, data requirements, runtime, GPU cost, and evaluation risk",
                    "choose lightweight observer metrics by default and avoid classifier training unless the user explicitly asks for downstream evidence",
                    "identify missing validations or clarification questions",
                    "select only from the provided skill registry",
                ],
            "llm_must_not": [
                "execute tools",
                "invent paths not present in the request/profile",
                "treat unvalidated filename semantics as facts",
                "infer caption text contents unless text previews are explicitly provided",
                "misread numeric image/txt pairs as numeric class labels; it only means same-stem files such as 1.png and 1.txt",
                "use unavailable skills as executable",
                "treat RaySAR as a default classification augmentation method; RaySAR is a physics simulation skill unless the user explicitly asks to use its outputs for classification experiments",
            ],
            "guardrail_capabilities": [
                "If DatasetFormatSpec is valid, DiffusionLoRAGenerationSkill can stage metadata-derived txt captions when original txt captions are missing.",
                "Use exclude_filters for requests such as exclude(pauli), not normal equality filters.",
                "GeoDiffSARSkill may run with a partial DatasetFormatSpec when the request provides real SAR training images, a 3D model, and explicit inference geometry.",
                "BackgroundGenerationSkill may run without a user dataset because it uses the packaged trained background generator and control bank.",
            ],
            "parameter_policy": {
                "target_count": "Respect explicit user counts. Otherwise choose small counts for speed, moderate counts for balanced runs, and larger counts only for relaxed/high-quality requests.",
                "training_epochs": "Respect explicit epochs. Otherwise use short training for speed, 10-ish epochs for balanced LoRA/GAN pilots, and 25-ish epochs for high-quality relaxed requests.",
                "heavy_skills": "Prefer dry-run/pilot recipes unless the user explicitly allows real execution. Expensive skills include DiffusionLoRAGenerationSkill, GeoDiffSARSkill, BackgroundGenerationSkill, and ClassificationEvaluationSkill.",
                "downstream_evidence": "Use QualityEvaluationSkill, DistributionEvaluationSkill, SARArtifactEvaluationSkill, LeakageCheckSkill, and DuplicateNearDuplicateSkill as default lightweight probes. Add ClassificationEvaluationSkill only if requested.",
            },
            "required_output_schema": {
                "candidate_plans": [
                    {
                        "plan_id": "short stable id",
                        "task": "one of style_transfer, diffusion_lora_generation, geodiff_sar_generation, gaussian_splatting_completion, gan_generation, traditional_augmentation, pseudocolor_transform, background_generation, target_background_composition, raysar_synthesis, classification_evaluation, classification, object_detection, segmentation, augmentation_or_generation, unknown",
                        "confidence": "number from 0 to 1",
                        "goals": ["string"],
                        "strategy": "short string",
                        "recipe_intent_updates": {
                            "task": "optional task override",
                            "style_source": "optional path if missing from rule intent",
                            "content_source": "optional path if missing from rule intent",
                            "dataset_source": "optional path if missing from rule intent",
                            "pov_scene": "optional RaySAR-compatible .pov path for raysar_synthesis",
                            "model_file": "optional OBJ/STL/PLY/GLB/point-cloud path for RaySAR model-to-POV compilation",
                            "parameters_file": "optional RaySAR parameters file",
                            "contributions_txt": "optional existing Contributions.txt path for postprocess-only RaySAR",
                            "baseline_dataset": "optional baseline dataset path for classification_evaluation",
                            "augmented_dataset": "optional augmented dataset path for classification_evaluation",
                            "model_family": "optional flux/sd3/sdxl",
                            "target_count": "optional positive integer",
                            "training_epochs": "optional positive integer for LoRA training epochs",
                            "width": "optional RaySAR/render image width in pixels",
                            "height": "optional RaySAR/render image height in pixels",
                            "scene_prompt": "optional natural-language scene/background prompt for background_generation",
                            "target_source": "optional target image directory for target_background_composition",
                            "background_source": "optional background image directory for target_background_composition",
                            "mask_source": "optional mask directory for target_background_composition",
                            "blend_mode": "optional feather/laplacian/poisson",
                            "raysar_geometry": {
                                "incidence_angle_deg": "optional number",
                                "depression_angle_deg": "optional number",
                                "azimuth_deg": "optional number",
                                "target_extent_m": "optional number",
                                "sensor_plane_m": "optional number",
                                "range_distance_m": "optional number",
                                "input_up_axis": "optional x/y/z",
                                "azimuth_sweep": {"start": "number", "stop": "number", "step": "number"},
                                "azimuth_values": ["optional explicit azimuth numbers"],
                            },
                            "filters": {"field": "scalar value"},
                            "exclude_filters": {"field": ["scalar values to exclude"]},
                        },
                        "pipeline": [
                            {
                                "id": "short stable step id",
                                "skill": "skill name from registry",
                                "depends_on": ["step ids"],
                                "execution_status": "executable, planned, or blocked",
                                "params": {},
                                "reason": "why this step is useful",
                            }
                        ],
                        "expected_benefits": ["string"],
                        "required_validations": ["string"],
                        "risks": ["string"],
                        "clarification_questions": ["string"],
                        "rationale": "short explanation",
                    }
                ],
                "note": "If you can only provide one plan, return candidate_plans with one item.",
            },
        },
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def build_skill_registry() -> list[dict[str, Any]]:
    return skill_registry_for_prompt()


def normalize_planner_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    body = raw.get("planner_proposal", raw)
    updates = body.get("recipe_intent_updates") or {}
    if not isinstance(updates, dict):
        updates = {}
    pipeline = body.get("pipeline") or []
    if not isinstance(pipeline, list):
        pipeline = []
    return {
        "schema_version": body.get("schema_version", LLM_PLANNER_VERSION),
        "plan_id": str(body.get("plan_id") or body.get("id") or "selected_plan"),
        "planner": body.get("planner", "llm"),
        "task": str(body.get("task") or updates.get("task") or "unknown"),
        "confidence": clamp_float(body.get("confidence"), default=0.5),
        "goals": normalize_string_list(body.get("goals")),
        "strategy": str(body.get("strategy") or ""),
        "recipe_intent_updates": updates,
        "pipeline": normalize_pipeline(pipeline),
        "required_validations": normalize_string_list(body.get("required_validations")),
        "risks": normalize_string_list(body.get("risks")),
        "clarification_questions": normalize_string_list(body.get("clarification_questions")),
        "rationale": str(body.get("rationale") or ""),
    }


def normalize_pipeline(pipeline: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for idx, raw_step in enumerate(pipeline):
        if not isinstance(raw_step, dict):
            continue
        skill = str(raw_step.get("skill") or "")
        step_id = safe_step_id(str(raw_step.get("id") or raw_step.get("step_id") or f"step_{idx + 1}"))
        normalized.append(
            {
                "id": step_id,
                "skill": skill,
                "depends_on": normalize_string_list(raw_step.get("depends_on")),
                "execution_status": str(raw_step.get("execution_status") or inferred_skill_status(skill)),
                "params": raw_step.get("params") if isinstance(raw_step.get("params"), dict) else {},
                "reason": str(raw_step.get("reason") or raw_step.get("description") or ""),
            }
        )
    return normalized


def validate_proposal_contract(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not proposal.get("task"):
        issues.append({"severity": "high", "name": "missing_task"})
    if not proposal.get("strategy"):
        issues.append({"severity": "medium", "name": "missing_strategy"})
    if not proposal.get("rationale"):
        issues.append({"severity": "low", "name": "missing_rationale"})
    if not proposal.get("goals"):
        issues.append({"severity": "medium", "name": "missing_goals"})

    pipeline = proposal.get("pipeline") or []
    if not pipeline:
        issues.append({"severity": "medium", "name": "empty_pipeline_proposal"})
        return issues

    step_ids = [str(step.get("id") or "") for step in pipeline if isinstance(step, dict)]
    duplicate_ids = sorted({step_id for step_id in step_ids if step_id and step_ids.count(step_id) > 1})
    for step_id in duplicate_ids:
        issues.append({"severity": "high", "name": "duplicate_step_id", "step_id": step_id})

    known_ids = {step_id for step_id in step_ids if step_id}
    allowed_status = {"executable", "planned", "blocked"}
    for step in pipeline:
        step_id = str(step.get("id") or "")
        if not step_id:
            issues.append({"severity": "high", "name": "missing_step_id"})
        if not step.get("skill"):
            issues.append({"severity": "high", "name": "missing_step_skill", "step_id": step_id})
        status = str(step.get("execution_status") or inferred_skill_status(str(step.get("skill") or "")))
        if status not in allowed_status:
            issues.append(
                {
                    "severity": "medium",
                    "name": "invalid_execution_status",
                    "step_id": step_id,
                    "value": status,
                }
            )
        for dep in step.get("depends_on") or []:
            if str(dep) not in known_ids:
                issues.append(
                    {
                        "severity": "high",
                        "name": "missing_step_dependency",
                        "step_id": step_id,
                        "dependency": str(dep),
                    }
                )
    return issues


def validate_planner_proposal(
    proposal: dict[str, Any],
    intent_spec: dict[str, Any],
    bridge_report: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    issues.extend(validate_proposal_contract(proposal))
    task = normalize_task(proposal.get("task"))
    if task not in SUPPORTED_RECIPE_TASKS:
        issues.append({"severity": "high", "name": "unsupported_task", "value": proposal.get("task")})
        task = "unknown"

    unsupported_skills = []
    planned_skills = []
    executable_steps = []
    for step in proposal.get("pipeline", []):
        skill = str(step.get("skill") or "")
        execution_status = str(step.get("execution_status") or inferred_skill_status(skill))
        if skill in EXECUTABLE_SKILLS:
            executable_steps.append(step.get("id"))
            if execution_status == "blocked":
                issues.append(
                    {
                        "severity": "medium",
                        "name": "executable_skill_marked_blocked",
                        "skill": skill,
                        "step_id": step.get("id"),
                    }
                )
        elif skill in PLANNED_SKILLS:
            planned_skills.append(skill)
            if execution_status == "executable":
                issues.append(
                    {
                        "severity": "high",
                        "name": "planned_skill_marked_executable",
                        "skill": skill,
                        "step_id": step.get("id"),
                    }
                )
        else:
            unsupported_skills.append(skill)
            issues.append({"severity": "high", "name": "unsupported_skill", "skill": skill, "step_id": step.get("id")})

    current_task = intent_spec.get("intent", {}).get("task")
    if current_task in EXECUTABLE_RECIPE_TASKS and task in EXECUTABLE_RECIPE_TASKS and current_task != task:
        issues.append(
            {
                "severity": "medium",
                "name": "task_conflict_with_rule_intent",
                "rule_task": current_task,
                "llm_task": task,
                "resolution": "keep_rule_task",
            }
        )

    if task == "style_transfer" and bridge_report.get("profile_level_after_validation") != 2:
        issues.append(
            {
                "severity": "medium",
                "name": "style_transfer_without_validated_metadata",
                "resolution": "recipe may still run, but metadata filters are not trusted",
            }
        )

    updates = proposal.get("recipe_intent_updates") or {}
    for key in (
        "style_source",
        "content_source",
        "dataset_source",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
        "baseline_dataset",
        "augmented_dataset",
        "target_source",
        "background_source",
        "mask_source",
    ):
        value = updates.get(key)
        if value and not is_safe_path_like(str(value)):
            issues.append({"severity": "high", "name": "unsafe_path_update", "field": key, "value": str(value)[:200]})
    blend_mode = str(updates.get("blend_mode") or "").strip().lower()
    if blend_mode and blend_mode not in {"feather", "laplacian", "poisson", "hard"}:
        issues.append({"severity": "medium", "name": "unsupported_blend_mode", "value": blend_mode})
    raysar_geometry = updates.get("raysar_geometry") or {}
    if raysar_geometry and not isinstance(raysar_geometry, dict):
        issues.append({"severity": "high", "name": "invalid_raysar_geometry"})
    elif isinstance(raysar_geometry, dict):
        issues.extend(validate_raysar_geometry_update(raysar_geometry))
    for key in ("width", "height"):
        if updates.get(key) is not None and safe_positive_int(updates.get(key), upper=16384) is None:
            issues.append({"severity": "medium", "name": f"invalid_{key}", "value": updates.get(key)})

    filters = updates.get("filters") or {}
    if filters and not isinstance(filters, dict):
        issues.append({"severity": "high", "name": "invalid_filter_update", "value": filters})
    elif isinstance(filters, dict):
        issues.extend(validate_filter_updates(filters=filters, bridge_report=bridge_report, task=task))
    exclude_filters = updates.get("exclude_filters") or {}
    if exclude_filters and not isinstance(exclude_filters, dict):
        issues.append({"severity": "high", "name": "invalid_exclude_filter_update", "value": exclude_filters})
    elif isinstance(exclude_filters, dict):
        issues.extend(validate_exclude_filter_updates(filters=exclude_filters, bridge_report=bridge_report, task=task))

    blocking_issues = [issue for issue in issues if issue.get("severity") == "high"]
    clarification_questions = build_guardrail_clarification_questions(
        proposal=proposal,
        issues=issues,
        bridge_report=bridge_report,
    )
    decision = planner_decision(
        blocking_issues=blocking_issues,
        clarification_questions=clarification_questions,
        task=task,
        proposal=proposal,
    )
    return {
        "schema_version": LLM_PLANNER_GUARDRAIL_VERSION,
        "valid": not blocking_issues,
        "selected_task": task,
        "rule_task": current_task,
        "executable_recipe_task": task in EXECUTABLE_RECIPE_TASKS,
        "executable_steps": executable_steps,
        "planned_skills": sorted(set(planned_skills)),
        "unsupported_skills": sorted(set(skill for skill in unsupported_skills if skill)),
        "issues": issues,
        "blocking_issue_count": len(blocking_issues),
        "warning_issue_count": len([issue for issue in issues if issue.get("severity") == "medium"]),
        "clarification_questions": clarification_questions,
        "requires_clarification": bool(clarification_questions),
        "execution_allowed": not blocking_issues and task in EXECUTABLE_RECIPE_TASKS and not clarification_questions,
        "decision": decision,
    }


def validate_filter_updates(filters: dict[str, Any], bridge_report: dict[str, Any], task: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    validation = bridge_report.get("validation") or {}
    coverage = validation.get("field_coverage") or {}
    for field, value in filters.items():
        if not is_safe_filter_name(str(field)) or not is_scalar(value):
            issues.append({"severity": "high", "name": "unsafe_filter_update", "field": str(field)})
            continue
        field_coverage = coverage.get(str(field))
        if field_coverage is None:
            severity = "medium" if task in EXECUTABLE_RECIPE_TASKS else "low"
            issues.append(
                {
                    "severity": severity,
                    "name": "filter_field_not_in_validated_profile",
                    "field": str(field),
                    "value": value,
                    "coverage": None,
                }
            )
        elif float(field_coverage) < 1.0:
            severity = "medium" if float(field_coverage) > 0 else "high"
            issues.append(
                {
                    "severity": severity,
                    "name": "filter_field_incomplete_coverage",
                    "field": str(field),
                    "value": value,
                    "coverage": field_coverage,
                }
            )
    return issues


def validate_exclude_filter_updates(filters: dict[str, Any], bridge_report: dict[str, Any], task: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    flattened: dict[str, Any] = {}
    for field, value in filters.items():
        if not is_safe_filter_name(str(field)) or not is_safe_exclude_filter_value(value):
            issues.append({"severity": "high", "name": "unsafe_exclude_filter_update", "field": str(field)})
            continue
        values = normalize_exclude_filter_values(value)
        if values:
            flattened[str(field)] = values[0]
    issues.extend(validate_filter_updates(filters=flattened, bridge_report=bridge_report, task=task))
    for issue in issues:
        if issue.get("name") == "filter_field_not_in_validated_profile":
            issue["name"] = "exclude_filter_field_not_in_validated_profile"
        elif issue.get("name") == "filter_field_incomplete_coverage":
            issue["name"] = "exclude_filter_field_incomplete_coverage"
    return issues


def build_guardrail_clarification_questions(
    proposal: dict[str, Any],
    issues: list[dict[str, Any]],
    bridge_report: dict[str, Any],
) -> list[str]:
    questions: list[str] = []
    for question in normalize_string_list(proposal.get("clarification_questions")):
        if not is_optional_preference_question(question) and not is_resolved_by_guardrail_policy(question, bridge_report):
            add_unique_question(questions, question)
    if not bridge_report.get("valid") and not proposal_can_run_without_dataset_format(proposal):
        add_unique_question(
            questions,
            "当前 DatasetFormatSpec 验证未通过。请补充数据路径/文件名/标签字段中关键字段的含义。",
        )
    for issue in issues:
        name = issue.get("name")
        if name == "filter_field_incomplete_coverage":
            add_unique_question(
                questions,
                f"筛选字段 `{issue.get('field')}` 在数据集中覆盖不完整，是否仍要使用该字段筛选？",
            )
        elif name == "filter_field_not_in_validated_profile":
            add_unique_question(
                questions,
                f"筛选字段 `{issue.get('field')}` 未通过格式验证。请说明它应该从文件名、目录名还是标签文件中解析。",
            )
        elif name == "exclude_filter_field_incomplete_coverage":
            add_unique_question(
                questions,
                f"排除筛选字段 `{issue.get('field')}` 在数据集中覆盖不完整，是否仍要使用该字段排除样本？",
            )
        elif name == "exclude_filter_field_not_in_validated_profile":
            add_unique_question(
                questions,
                f"排除筛选字段 `{issue.get('field')}` 未通过格式验证。请说明它应该从文件名、目录名还是标签文件中解析。",
            )
        elif name == "missing_step_dependency":
            add_unique_question(
                questions,
                "LLM 规划的步骤依赖不完整。是否允许 SAGA 回退到规则 recipe？",
            )
    return questions[:10]


def proposal_can_run_without_dataset_format(proposal: dict[str, Any]) -> bool:
    task = normalize_task(proposal.get("task") or (proposal.get("recipe_intent_updates") or {}).get("task"))
    updates = proposal.get("recipe_intent_updates") or {}
    if task == "background_generation":
        return True
    if task == "geodiff_sar_generation" and (
        (updates.get("dataset_source") or updates.get("dataset_root")) and updates.get("model_file")
    ):
        return True
    if task == "raysar_synthesis" and (
        updates.get("pov_scene") or updates.get("model_file") or updates.get("contributions_txt")
    ):
        return True
    if task == "raysar_synthesis":
        for step in proposal.get("pipeline") or []:
            params = step.get("params") if isinstance(step, dict) else {}
            if not isinstance(params, dict):
                continue
            if params.get("pov_scene") or params.get("model_file") or params.get("contributions_txt"):
                return True
            if step.get("skill") in {"ModelToPOVSceneCompilerSkill", "RaySARSynthesisSkill"}:
                return True
    return False


def is_optional_preference_question(question: str) -> bool:
    text = question.lower()
    optional_markers = [
        "export include",
        "include the lora weights",
        "training metadata",
        "desired output format",
        "output format",
        "image size",
        "generated image size",
        "preference for the generated",
        "preferred base model",
        "base model for lora",
        "model for lora",
        "specific polarization",
        "polarization combinations",
        "class-conditioned",
        "directory, archive",
        "archive",
        "保存格式",
        "导出格式",
        "底模",
        "基座模型",
        "极化组合",
        "类别条件",
        "是否包含",
        "是否导出",
        "both azimuth angles",
        "sweep with start",
        "start=",
        "stop=",
        "step=",
        "端点",
        "包含起止",
        "包含首尾",
    ]
    return any(marker in text for marker in optional_markers)


def is_resolved_by_guardrail_policy(question: str, bridge_report: dict[str, Any]) -> bool:
    text = question.lower()
    if "no text caption" in text or "caption files" in text or "generate captions" in text:
        return bool(bridge_report.get("valid"))
    if "polarization" in text and "caption" in text:
        return True
    if ("base model" in text or "model for lora" in text or "底模" in text or "基座模型" in text) and (
        "lora" in text or "lo-ra" in text
    ):
        return True
    if "polarization" in text and ("specific" in text or "combination" in text or "all available" in text):
        return bridge_field_coverage(bridge_report, "polarization") >= 0.95
    if ("极化" in question and "组合" in question) or ("极化" in question and "限定" in question):
        return bridge_field_coverage(bridge_report, "polarization") >= 0.95
    if ("class-conditioned" in text or "class conditioned" in text or "类别条件" in question) and "polarization" in text:
        return bridge_field_coverage(bridge_report, "class") >= 0.95 and bridge_field_coverage(bridge_report, "polarization") >= 0.95
    return False


def bridge_field_coverage(bridge_report: dict[str, Any], field: str) -> float:
    coverage = (bridge_report.get("validation") or {}).get("field_coverage") or {}
    try:
        return float(coverage.get(field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def add_unique_question(questions: list[str], question: str) -> None:
    cleaned = str(question).strip()
    if cleaned and cleaned not in questions:
        questions.append(cleaned)


def planner_decision(
    blocking_issues: list[dict[str, Any]],
    clarification_questions: list[str],
    task: str,
    proposal: dict[str, Any],
) -> str:
    if blocking_issues:
        return "fallback_to_rule_planner"
    if clarification_questions and task not in EXECUTABLE_RECIPE_TASKS:
        return "needs_clarification"
    if clarification_questions:
        return "accept_with_clarification_warnings"
    if proposal.get("pipeline"):
        return "accept_with_guardrails"
    return "accept_rule_recipe_only"


def apply_planner_proposal_to_intent(
    intent_spec: dict[str, Any],
    proposal: dict[str, Any],
    guardrail: dict[str, Any],
) -> dict[str, Any]:
    effective = deepcopy(intent_spec)
    intent = effective.setdefault("intent", {})
    updates = proposal.get("recipe_intent_updates") or {}
    applied: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    if not guardrail.get("valid"):
        effective["planner_notes"] = {
            "source": "llm_planner_guardrail",
            "decision": guardrail.get("decision"),
            "execution_allowed": guardrail.get("execution_allowed"),
            "requires_clarification": guardrail.get("requires_clarification"),
            "applied_updates": applied,
            "ignored_updates": [{"field": "*", "reason": "guardrail_invalid"}],
            "proposal_task": proposal.get("task"),
            "clarification_questions": guardrail.get("clarification_questions", []),
        }
        return effective

    proposed_task = normalize_task(updates.get("task") or proposal.get("task"))
    current_task = intent.get("task") or "unknown"
    if should_apply_task(current_task, proposed_task):
        intent["task"] = proposed_task
        applied.append({"field": "task", "value": proposed_task})
    elif proposed_task and proposed_task != current_task:
        ignored.append({"field": "task", "value": proposed_task, "reason": f"kept_rule_task:{current_task}"})

    for key in (
        "style_source",
        "content_source",
        "dataset_source",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
        "baseline_dataset",
        "augmented_dataset",
        "target_source",
        "background_source",
        "mask_source",
    ):
        value = updates.get(key)
        if not value:
            continue
        if intent.get(key):
            ignored.append({"field": key, "value": value, "reason": "rule_intent_already_has_value"})
        elif is_safe_path_like(str(value)):
            intent[key] = str(value)
            applied.append({"field": key, "value": str(value)})
        else:
            ignored.append({"field": key, "value": str(value)[:200], "reason": "unsafe_path_like_value"})

    blend_mode = str(updates.get("blend_mode") or "").strip().lower()
    if blend_mode:
        if blend_mode in {"feather", "laplacian", "poisson", "hard"}:
            intent["blend_mode"] = blend_mode
            applied.append({"field": "blend_mode", "value": blend_mode})
        else:
            ignored.append({"field": "blend_mode", "value": blend_mode, "reason": "unsupported_blend_mode"})

    model_family = str(updates.get("model_family") or "").lower()
    if model_family:
        if model_family in MODEL_FAMILIES:
            intent["model_family"] = model_family
            applied.append({"field": "model_family", "value": model_family})
        else:
            ignored.append({"field": "model_family", "value": model_family, "reason": "unsupported_model_family"})

    if updates.get("target_count") is not None:
        target_count = safe_positive_int(updates.get("target_count"), upper=10000)
        if target_count is not None:
            intent["target_count"] = target_count
            applied.append({"field": "target_count", "value": target_count})
        else:
            ignored.append({"field": "target_count", "value": updates.get("target_count"), "reason": "not_positive_int"})

    if updates.get("scene_prompt") not in (None, ""):
        prompt = str(updates.get("scene_prompt")).strip()
        if len(prompt) <= 500:
            intent["scene_prompt"] = prompt
            applied.append({"field": "scene_prompt", "value": prompt})
        else:
            ignored.append({"field": "scene_prompt", "value": prompt[:120], "reason": "too_long"})

    if updates.get("training_epochs") is not None:
        training_epochs = safe_positive_int(updates.get("training_epochs"), upper=10000)
        if training_epochs is not None:
            intent["training_epochs"] = training_epochs
            applied.append({"field": "training_epochs", "value": training_epochs})
        else:
            ignored.append({"field": "training_epochs", "value": updates.get("training_epochs"), "reason": "not_positive_int"})

    for key in ("width", "height"):
        if updates.get(key) is None:
            continue
        size = safe_positive_int(updates.get(key), upper=16384)
        if size is not None:
            intent[key] = size
            applied.append({"field": key, "value": size})
        else:
            ignored.append({"field": key, "value": updates.get(key), "reason": "not_positive_render_size"})

    raysar_geometry = updates.get("raysar_geometry") if isinstance(updates.get("raysar_geometry"), dict) else {}
    if raysar_geometry:
        merged_geometry = dict(intent.get("raysar_geometry") or {})
        for key, value in sanitize_raysar_geometry_update(raysar_geometry).items():
            merged_geometry[key] = value
            applied.append({"field": f"raysar_geometry.{key}", "value": value})
        intent["raysar_geometry"] = merged_geometry

    filter_updates = updates.get("filters") or {}
    if isinstance(filter_updates, dict):
        filters = dict(intent.get("filters") or {})
        exclude_filters = normalize_exclude_filters(intent.get("exclude_filters") or {})
        for key, value in filter_updates.items():
            exclude_value = parse_exclude_filter_value(value)
            if exclude_value is not None and is_safe_filter_name(str(key)):
                add_exclude_filter_value(exclude_filters, str(key), exclude_value)
                applied.append({"field": f"exclude_filters.{key}", "value": exclude_value})
            elif is_safe_filter_name(str(key)) and is_scalar(value):
                filters[str(key)] = value
                applied.append({"field": f"filters.{key}", "value": value})
            else:
                ignored.append({"field": f"filters.{key}", "reason": "unsafe_or_non_scalar_filter"})
        intent["filters"] = filters

    exclude_updates = updates.get("exclude_filters") if isinstance(updates.get("exclude_filters"), dict) else {}
    exclude_filters = normalize_exclude_filters(intent.get("exclude_filters") or {})
    for key, value in exclude_updates.items():
        if not is_safe_filter_name(str(key)):
            ignored.append({"field": f"exclude_filters.{key}", "reason": "unsafe_filter_name"})
            continue
        values = normalize_exclude_filter_values(value)
        if not values:
            ignored.append({"field": f"exclude_filters.{key}", "reason": "empty_or_unsafe_exclude_filter"})
            continue
        for item in values:
            add_exclude_filter_value(exclude_filters, str(key), item)
        applied.append({"field": f"exclude_filters.{key}", "value": values})
    intent["exclude_filters"] = exclude_filters

    merged_goals = []
    for goal in list(intent.get("goals") or []) + list(proposal.get("goals") or []):
        if goal and goal not in merged_goals:
            merged_goals.append(goal)
    if merged_goals:
        intent["goals"] = merged_goals

    effective["planner_notes"] = {
        "source": "llm_planner_guardrail",
        "decision": guardrail.get("decision"),
        "execution_allowed": guardrail.get("execution_allowed"),
        "requires_clarification": guardrail.get("requires_clarification"),
        "proposal_task": proposal.get("task"),
        "strategy": proposal.get("strategy"),
        "confidence": proposal.get("confidence"),
        "applied_updates": applied,
        "ignored_updates": ignored,
        "planned_skills": guardrail.get("planned_skills", []),
        "unsupported_skills": guardrail.get("unsupported_skills", []),
        "risks": proposal.get("risks", []),
        "clarification_questions": guardrail.get("clarification_questions") or proposal.get("clarification_questions", []),
    }
    return effective


def write_planner_artifacts(output_dir: str | Path, result: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    if result.get("plan_ranking"):
        write_plan_ranking(path, result["plan_ranking"])
        if result["plan_ranking"].get("plan_utility_report"):
            write_plan_utility_report(path, result["plan_ranking"]["plan_utility_report"])
    elif result.get("candidate_plans"):
        save_json(path / "candidate_plans.json", {"schema_version": "saga_candidate_plans_v1", "plans": result["candidate_plans"]})
    save_json(path / "planner_proposal.json", result["proposal"])
    save_json(path / "planner_guardrail.json", result["guardrail"])
    save_json(path / "effective_intent_spec.json", result["effective_intent_spec"])
    save_json(
        path / "planner_clarification.json",
        {
            "schema_version": "saga_planner_clarification_v1",
            "requires_clarification": result["guardrail"].get("requires_clarification"),
            "questions": result["guardrail"].get("clarification_questions", []),
            "decision": result["guardrail"].get("decision"),
        },
    )
    save_text(path / "planner_report.md", render_planner_markdown(result))


def render_planner_markdown(result: dict[str, Any]) -> str:
    proposal = result.get("proposal", {})
    guardrail = result.get("guardrail", {})
    ranking = result.get("plan_ranking", {})
    notes = result.get("effective_intent_spec", {}).get("planner_notes", {})
    lines = [
        "# SAGA LLM Planner",
        "",
        f"- Mode: `{result.get('mode')}`",
        f"- Candidate plans: {ranking.get('candidate_count')}",
        f"- Selected plan: `{ranking.get('selected_plan_id')}`",
        f"- Selected utility: {ranking.get('selected_utility_score')}",
        f"- Proposal task: `{proposal.get('task')}`",
        f"- Strategy: {proposal.get('strategy')}",
        f"- Confidence: {proposal.get('confidence')}",
        f"- Guardrail decision: `{guardrail.get('decision')}`",
        f"- Guardrail valid: {guardrail.get('valid')}",
        f"- Execution allowed: {guardrail.get('execution_allowed')}",
        f"- Requires clarification: {guardrail.get('requires_clarification')}",
        f"- Executable steps proposed: `{guardrail.get('executable_steps')}`",
        f"- Planned skills proposed: `{guardrail.get('planned_skills')}`",
        f"- Unsupported skills: `{guardrail.get('unsupported_skills')}`",
        "",
        "## Applied Updates",
        "",
    ]
    applied = notes.get("applied_updates") or []
    if not applied:
        lines.append("- None")
    else:
        for item in applied:
            lines.append(f"- `{item.get('field')}` = `{item.get('value')}`")
    lines.extend(["", "## Ignored Updates", ""])
    ignored = notes.get("ignored_updates") or []
    if not ignored:
        lines.append("- None")
    else:
        for item in ignored:
            lines.append(f"- `{item.get('field')}`: {item.get('reason')}")
    lines.extend(["", "## Guardrail Issues", ""])
    issues = guardrail.get("issues") or []
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('name')}` ({issue.get('severity')}): `{issue}`")
    lines.extend(["", "## Clarification Questions", ""])
    questions = guardrail.get("clarification_questions") or proposal.get("clarification_questions") or []
    if not questions:
        lines.append("- None")
    else:
        for question in questions:
            lines.append(f"- {question}")
    lines.append("")
    return "\n".join(lines)


def compact_raw_profile(raw_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": raw_profile.get("root"),
        "total_files": raw_profile.get("total_files"),
        "image_count": raw_profile.get("image_count"),
        "suffix_counts": raw_profile.get("suffix_counts") or raw_profile.get("image_suffix_counts"),
        "sidecar_counts": raw_profile.get("sidecar_counts"),
        "sample_paths": (raw_profile.get("sample_paths") or raw_profile.get("sample_image_paths") or [])[:30],
        "path_patterns": (raw_profile.get("path_patterns") or raw_profile.get("pattern_groups") or [])[:20],
        "numeric_token_summary": raw_profile.get("numeric_token_summary"),
    }


def parse_exclude_filter_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = re.fullmatch(r"(?:exclude|not|except)\((.+)\)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if text.lower().startswith("not "):
        return text[4:].strip()
    return None


def is_safe_exclude_filter_value(value: Any) -> bool:
    if is_scalar(value):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(is_scalar(item) for item in value)
    return False


def normalize_exclude_filters(value: Any) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    if not isinstance(value, dict):
        return normalized
    for key, raw in value.items():
        if not is_safe_filter_name(str(key)):
            continue
        values = normalize_exclude_filter_values(raw)
        if values:
            normalized[str(key)] = values
    return normalized


def normalize_exclude_filter_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    values = []
    for item in raw_values:
        if not is_scalar(item):
            continue
        parsed = parse_exclude_filter_value(item)
        value_to_add = parsed if parsed is not None else item
        if value_to_add is not None and str(value_to_add) != "":
            values.append(value_to_add)
    return sorted({str(item): item for item in values}.values(), key=lambda item: str(item))


def add_exclude_filter_value(filters: dict[str, list[Any]], key: str, value: Any) -> None:
    existing = filters.setdefault(key, [])
    existing.append(value)
    filters[key] = sorted({str(item): item for item in existing}.values(), key=lambda item: str(item))


def validate_raysar_geometry_update(value: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sanitized = sanitize_raysar_geometry_update(value)
    for key in value.keys():
        if key not in sanitized:
            issues.append({"severity": "medium", "name": "invalid_raysar_geometry_field", "field": str(key)})
    return issues


def sanitize_raysar_geometry_update(value: dict[str, Any]) -> dict[str, Any]:
    allowed_numeric = {
        "incidence_angle_deg": (1.0, 89.0),
        "depression_angle_deg": (1.0, 89.0),
        "azimuth_deg": (-360.0, 360.0),
        "target_extent_m": (0.001, 100000.0),
        "sensor_plane_m": (0.001, 1000000.0),
        "range_distance_m": (0.001, 10000000.0),
        "pitch_deg": (-360.0, 360.0),
        "roll_deg": (-360.0, 360.0),
        "scale_factor": (0.000001, 1000000.0),
    }
    sanitized: dict[str, Any] = {}
    for key, bounds in allowed_numeric.items():
        if value.get(key) is None:
            continue
        try:
            number = float(value[key])
        except (TypeError, ValueError):
            continue
        if bounds[0] <= number <= bounds[1]:
            sanitized[key] = number
    for key in ("input_up_axis", "pov_height_axis"):
        axis = str(value.get(key) or "").lower()
        if axis in {"x", "y", "z"}:
            sanitized[key] = axis
    sweep = value.get("azimuth_sweep")
    if isinstance(sweep, dict):
        try:
            start = float(sweep["start"])
            stop = float(sweep["stop"])
            step = float(sweep["step"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if -3600 <= start <= 3600 and -3600 <= stop <= 3600 and 0 < abs(step) <= 360:
                sanitized["azimuth_sweep"] = {"start": start, "stop": stop, "step": step}
    values = value.get("azimuth_values")
    if isinstance(values, list):
        cleaned = []
        for item in values[:360]:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if -3600 <= number <= 3600:
                cleaned.append(number)
        if cleaned:
            sanitized["azimuth_values"] = cleaned
    return sanitized


def compact_dataset_profile(dataset_profile: dict[str, Any]) -> dict[str, Any]:
    body = dataset_profile.get("dataset_profile", dataset_profile)
    return {
        "status": body.get("status"),
        "profile_level": body.get("profile_level"),
        "task_candidates": body.get("task_candidates"),
        "known": body.get("known"),
        "unknown": body.get("unknown"),
        "issues": (body.get("issues") or [])[:20],
        "clarification_questions": (body.get("clarification_questions") or [])[:10],
    }


def compact_bridge_report(bridge_report: dict[str, Any]) -> dict[str, Any]:
    validation = bridge_report.get("validation") or {}
    return {
        "valid": bridge_report.get("valid"),
        "profile_level_after_validation": bridge_report.get("profile_level_after_validation"),
        "required_fields": bridge_report.get("required_fields"),
        "compiled_rules": (bridge_report.get("compiled_rules") or [])[:12],
        "field_coverage": validation.get("field_coverage"),
        "missing_required_count": validation.get("missing_required_count"),
        "missing_required_groups": (validation.get("missing_required_groups") or [])[:8],
        "next_step": bridge_report.get("next_step"),
    }


def compact_validated_profile(validated_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": validated_profile.get("status"),
        "profile_level": validated_profile.get("profile_level"),
        "planner_ready": validated_profile.get("planner_ready"),
        "dataset_config": validated_profile.get("dataset_config"),
        "format_spec": validated_profile.get("format_spec"),
        "required_fields": validated_profile.get("required_fields"),
        "field_coverage": validated_profile.get("field_coverage"),
        "missing_required_count": validated_profile.get("missing_required_count"),
    }


def compact_benefit_context(benefit_context: dict[str, Any]) -> dict[str, Any]:
    need_profile = benefit_context.get("dataset_need_profile") or {}
    skill_utility = benefit_context.get("skill_utility") or {}
    augmentation_plan = benefit_context.get("augmentation_plan") or {}
    planning_evidence = benefit_context.get("planning_evidence") or {}
    return {
        "top_dataset_needs": need_profile.get("top_needs", [])[:6],
        "planning_evidence": compact_planning_evidence(planning_evidence) if planning_evidence else {},
        "top_skill_recommendations": [
            {
                "skill": item.get("skill"),
                "status": item.get("status"),
                "utility_score": item.get("utility_score"),
                "verdict": item.get("verdict"),
                "matched_needs": item.get("matched_needs", [])[:4],
            }
            for item in (skill_utility.get("top_recommendations") or [])[:8]
        ],
        "deterministic_augmentation_plan": {
            "selected_plan_id": augmentation_plan.get("selected_plan_id"),
            "selected_skill": augmentation_plan.get("selected_skill"),
            "selected_recipe_task": augmentation_plan.get("selected_recipe_task"),
            "top_ranked_plans": [
                {
                    "plan_id": item.get("plan_id"),
                    "skill": item.get("skill"),
                    "status": item.get("status"),
                    "recipe_task": item.get("recipe_task"),
                    "planner_score": item.get("planner_score"),
                    "params": item.get("params"),
                    "rationale": item.get("rationale"),
                }
                for item in (augmentation_plan.get("ranked_plans") or [])[:5]
            ],
        },
        "note": "These are prior-based estimates, not downstream experimental results.",
    }


def compact_memory_context(memory_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_count": memory_context.get("match_count", 0),
        "matches": [
            {
                "score": item.get("score"),
                "reasons": item.get("reasons"),
                "task": item.get("task"),
                "recipe_id": item.get("recipe_id"),
                "selected_skills": item.get("selected_skills", []),
                "selected_skill_params": item.get("selected_skill_params", {}),
                "execution_status": (item.get("execution") or {}).get("status"),
                "quality_status": (item.get("quality") or {}).get("status"),
                "quality_trigger_count": (item.get("quality") or {}).get("trigger_count"),
                "sar_artifact_trigger_count": (item.get("sar_artifacts") or {}).get("trigger_count"),
                "classification_outcome": (item.get("outcomes") or {}).get("classification"),
                "planner": item.get("planner"),
                "reuse_notes": item.get("reuse_notes", []),
            }
            for item in (memory_context.get("matches") or [])[:5]
        ],
        "policy_summary": memory_context.get("policy_summary", {}),
    }


def normalize_task(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def inferred_skill_status(skill: str) -> str:
    if skill in EXECUTABLE_SKILLS:
        return "executable"
    if skill in PLANNED_SKILLS:
        return "planned"
    return "blocked"


def should_apply_task(current_task: str, proposed_task: str) -> bool:
    if not proposed_task or proposed_task == "unknown":
        return False
    if current_task in {"unknown", "augmentation_or_generation", None}:
        return proposed_task in SUPPORTED_RECIPE_TASKS
    return current_task == proposed_task


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def safe_step_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "step"


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def safe_positive_int(value: Any, upper: int) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or number > upper:
        return None
    return number


def is_safe_path_like(value: str) -> bool:
    if not value or len(value) > 500:
        return False
    if any(token in value for token in [";", "|", "&&", "`", "$(", "\n", "\r"]):
        return False
    return bool(SAFE_PATH_RE.match(value))


def is_safe_filter_name(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value))


def is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None

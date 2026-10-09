from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text


PLAN_CRITIC_VERSION = "saga_plan_critic_v1"


def critique_agent_plan(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    state: dict[str, Any] | None = None,
    bridge_report: dict[str, Any] | None = None,
    augmentation_plan: dict[str, Any] | None = None,
    recipe: dict[str, Any] | None = None,
    request_constraints: dict[str, Any] | None = None,
    benefit_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else run_path
    state = state or load_optional_mapping(run_path / "agent_state.json")
    bridge_report = bridge_report or load_optional_mapping(run_path / "format_bridge.json")
    augmentation_plan = augmentation_plan or load_optional_mapping(run_path / "augmentation_plan.json")
    recipe = recipe or load_optional_mapping(run_path / "recipe.yaml")
    request_constraints = request_constraints or load_optional_mapping(run_path / "request_constraints.json")
    benefit_evidence = benefit_evidence or load_optional_mapping(run_path / "benefit_evidence.json")

    findings = []
    findings.extend(check_format_bridge(bridge_report, state=state, augmentation_plan=augmentation_plan, recipe=recipe))
    findings.extend(check_selected_plan(augmentation_plan, recipe))
    findings.extend(check_recipe_observers(recipe, request_constraints))
    findings.extend(check_task_skill_boundaries(state, augmentation_plan, recipe, request_constraints))
    findings.extend(check_benefit_claims(state, recipe, request_constraints, benefit_evidence))
    findings.extend(check_resource_and_execution_policy(augmentation_plan, recipe, state))

    severity_counts = count_severities(findings)
    status = "blocked" if severity_counts.get("high", 0) else "warning" if findings else "passed"
    report = {
        "schema_version": PLAN_CRITIC_VERSION,
        "run_dir": run_path.as_posix(),
        "status": status,
        "severity_counts": severity_counts,
        "finding_count": len(findings),
        "findings": findings,
        "selected_plan": {
            "plan_id": augmentation_plan.get("selected_plan_id"),
            "skill": augmentation_plan.get("selected_skill"),
            "recipe_task": augmentation_plan.get("selected_recipe_task"),
            "recipe_id": recipe.get("recipe_id"),
            "recipe_task_actual": recipe.get("task"),
        },
        "decision": {
            "real_execution_safe": status != "blocked",
            "needs_user_confirmation": status == "blocked" or bool(severity_counts.get("medium")),
            "downstream_claim_allowed": bool((benefit_evidence or {}).get("downstream_claim_allowed")),
        },
        "notes": [
            "PlanCritic is a bounded verifier. It does not execute skills and does not replace deterministic recipe validation.",
            "High-severity findings should block real execution unless the user explicitly overrides the risk.",
            "Medium findings should be surfaced in reports and reviewed before scaling up.",
        ],
    }
    save_json(out / "plan_critic.json", report)
    save_text(out / "plan_critic.md", render_plan_critic_markdown(report))
    return report


def check_format_bridge(
    bridge: dict[str, Any],
    state: dict[str, Any] | None = None,
    augmentation_plan: dict[str, Any] | None = None,
    recipe: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    findings = []
    task = (recipe or {}).get("task") or (state or {}).get("task")
    selected_skill = (augmentation_plan or {}).get("selected_skill") or (state or {}).get("augmentation_selected_skill")
    bridge_optional = task == "geodiff_sar_generation" or selected_skill == "GeoDiffSARSkill"
    if bridge.get("valid") is False:
        if bridge_optional:
            findings.append(
                finding(
                    "medium",
                    "format_bridge_partial_for_geodiff",
                    (
                        "DatasetFormatSpec validation did not reach a semantic profile, but GeoDiff-SAR can still run "
                        "when real SAR images, captions, a 3D model, and explicit inference geometry are available."
                    ),
                    evidence={"profile_level_after_validation": bridge.get("profile_level_after_validation")},
                )
            )
        else:
            findings.append(
                finding(
                    "high",
                    "format_bridge_invalid",
                    "DatasetFormatSpec validation failed; recipe execution may read labels or metadata incorrectly.",
                    evidence={"profile_level_after_validation": bridge.get("profile_level_after_validation")},
                )
            )
    elif bridge.get("profile_level_after_validation") not in (2, 3, None):
        findings.append(
            finding(
                "medium",
                "semantic_profile_not_task_ready",
                "Dataset profile did not reach validated semantic/task-ready level.",
                evidence={"profile_level_after_validation": bridge.get("profile_level_after_validation")},
            )
        )
    validation = bridge.get("validation") or {}
    missing = validation.get("missing_required_count")
    if isinstance(missing, int) and missing > 0:
        severity = "low" if bridge_optional else "medium"
        message = (
            "Some requested dataset metadata fields were missing. For GeoDiff-SAR this is a provenance/evaluation warning "
            "unless those fields are needed for sample selection."
            if bridge_optional
            else "Some required metadata fields were missing during format validation."
        )
        findings.append(
            finding(
                severity,
                "required_metadata_missing",
                message,
                evidence={"missing_required_count": missing, "field_coverage": validation.get("field_coverage")},
            )
        )
    return findings


def check_selected_plan(augmentation_plan: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    selected = selected_ranked_plan(augmentation_plan)
    if not selected:
        findings.append(finding("high", "no_selected_plan", "Planner did not select a ranked plan."))
        return findings
    if selected.get("status") != "executable":
        findings.append(
            finding(
                "high",
                "selected_skill_not_executable",
                "Selected skill is not executable in the current SAGA registry.",
                evidence={"skill": selected.get("skill"), "status": selected.get("status")},
            )
        )
    if selected.get("recipe_task") and recipe.get("task") != selected.get("recipe_task"):
        findings.append(
            finding(
                "high",
                "recipe_task_mismatch",
                "Compiled recipe task does not match the selected planner task.",
                evidence={"selected_recipe_task": selected.get("recipe_task"), "recipe_task": recipe.get("task")},
            )
        )
    if not recipe.get("pipeline"):
        findings.append(finding("high", "recipe_pipeline_empty", "Compiled recipe has no executable pipeline steps."))
    return findings


def check_recipe_observers(recipe: dict[str, Any], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    steps = recipe.get("pipeline") or []
    skills = {step.get("skill") for step in steps if isinstance(step, dict)}
    task = recipe.get("task")
    generation_tasks = {
        "traditional_augmentation",
        "diffusion_lora_generation",
        "gan_generation",
        "style_transfer",
        "background_generation",
        "target_background_composition",
        "raysar_synthesis",
        "geodiff_sar_generation",
        "pseudocolor_transform",
    }
    if task in generation_tasks and "QualityEvaluationSkill" not in skills:
        findings.append(finding("medium", "quality_observer_missing", "Generation recipe lacks QualityEvaluationSkill."))
    if task in generation_tasks - {"pseudocolor_transform"} and "SARArtifactEvaluationSkill" not in skills:
        findings.append(finding("medium", "sar_artifact_observer_missing", "SAR generation recipe lacks SARArtifactEvaluationSkill."))
    if task in {"diffusion_lora_generation", "gan_generation"} and "DistributionEvaluationSkill" not in skills:
        findings.append(finding("medium", "distribution_probe_missing", "Learned generation recipe lacks DistributionEvaluationSkill."))
    if task == "target_background_composition" and "DistributionEvaluationSkill" not in skills:
        findings.append(finding("medium", "composition_distribution_probe_missing", "Composition recipe should compare fused scenes against a background/reference bank."))
    if constraints.get("needs_downstream_evidence") and "ClassificationEvaluationSkill" not in skills:
        findings.append(
            finding(
                "medium",
                "requested_downstream_evaluator_missing",
                "User/request constraints ask for downstream evidence, but no classification evaluator is in the recipe.",
            )
        )
    if "ClassificationEvaluationSkill" in skills:
        if "LeakageCheckSkill" not in skills:
            findings.append(finding("high", "classification_without_leakage_gate", "Classification evaluator requires leakage gate."))
        if "DuplicateNearDuplicateSkill" not in skills:
            findings.append(
                finding("high", "classification_without_duplicate_gate", "Classification evaluator requires duplicate/near-duplicate gate.")
            )
    return findings


def check_task_skill_boundaries(
    state: dict[str, Any],
    augmentation_plan: dict[str, Any],
    recipe: dict[str, Any],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    findings = []
    selected_skill = augmentation_plan.get("selected_skill") or state.get("augmentation_selected_skill")
    task = recipe.get("task") or state.get("task")
    request = str(state.get("request") or "").lower()
    if selected_skill == "BackgroundGenerationSkill" and task == "classification":
        findings.append(
            finding(
                "medium",
                "background_generation_for_target_classification",
                "Background generation is indirect for target-chip classification unless composition is requested.",
            )
        )
    if selected_skill == "TargetBackgroundCompositionSkill":
        inputs = recipe.get("inputs") or {}
        if not inputs.get("target_source"):
            findings.append(finding("high", "composition_missing_target_source", "Target/background composition requires a target image source."))
        if not inputs.get("background_source"):
            findings.append(finding("high", "composition_missing_background_source", "Target/background composition requires a background image source."))
    if selected_skill in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"} and "分类" in request and "物理" not in request and "raysar" not in request:
        findings.append(
            finding(
                "medium",
                "raysar_as_default_classification_aug",
                "RaySAR is a physics simulation skill and should not be treated as default classification augmentation.",
            )
        )
    if selected_skill == "PseudocolorSkill" and not any(word in request for word in ["伪彩", "pseudocolor", "可视化", "rgb"]):
        findings.append(
            finding(
                "medium",
                "pseudocolor_not_explicitly_requested",
                "Pseudocolor should be selected only for visualization or explicit RGB feature-ablation requests.",
            )
        )
    if constraints.get("gpu_budget") == "limited" and selected_skill in {"DiffusionLoRAGenerationSkill", "GeoDiffSARSkill", "BackgroundGenerationSkill"}:
        findings.append(
            finding(
                "medium",
                "heavy_skill_under_limited_gpu",
                "Selected skill is GPU-heavy while request constraints indicate limited GPU budget.",
                evidence={"selected_skill": selected_skill},
            )
        )
    if selected_skill == "GeoDiffSARSkill":
        request_has_prior = any(word in request for word in ["3d", "三维", "模型", "物理先验", "geodiff", "controlnet"])
        if not request_has_prior:
            findings.append(
                finding(
                    "medium",
                    "geodiff_without_explicit_physical_prior",
                    "GeoDiff-SAR should be used for sparse-angle completion when a 3D/physical prior or ControlNet condition workflow is available.",
                )
            )
    return findings


def check_benefit_claims(
    state: dict[str, Any],
    recipe: dict[str, Any],
    constraints: dict[str, Any],
    benefit_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    findings = []
    if not benefit_evidence:
        return findings
    if benefit_evidence.get("evidence_level", 0) < 4 and constraints.get("needs_downstream_evidence"):
        findings.append(
            finding(
                "medium",
                "downstream_claim_not_supported_yet",
                "Request asks for downstream evidence, but current evidence level does not allow downstream benefit claims.",
                evidence={
                    "evidence_level": benefit_evidence.get("evidence_level"),
                    "evidence_label": benefit_evidence.get("evidence_label"),
                },
            )
        )
    if benefit_evidence.get("downstream_claim_allowed") is False and any(
        word in str(state.get("request") or "").lower()
        for word in ["提升", "收益", "准确率", "improve", "accuracy", "benefit"]
    ):
        findings.append(
            finding(
                "low",
                "final_report_should_avoid_downstream_claim",
                "Final reports should phrase results as candidate augmentation evidence, not proven task improvement.",
            )
        )
    return findings


def check_resource_and_execution_policy(augmentation_plan: dict[str, Any], recipe: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []
    selected = selected_ranked_plan(augmentation_plan)
    policy = selected.get("execution_policy") if selected else {}
    if policy.get("real_run_requires_user_run_flag") and state.get("dry_run") is False and state.get("execution_status") == "blocked_by_planner":
        findings.append(
            finding(
                "medium",
                "real_execution_blocked_by_planner",
                "Planner blocked real execution; user confirmation or clarification is required.",
            )
        )
    if len([item for item in augmentation_plan.get("ranked_plans") or [] if item.get("execution_policy", {}).get("executable")]) < 2:
        findings.append(
            finding(
                "low",
                "limited_candidate_diversity",
                "Planner has fewer than two executable candidate recipes to compare in pilot mode.",
            )
        )
    return findings


def selected_ranked_plan(augmentation_plan: dict[str, Any]) -> dict[str, Any]:
    selected_id = augmentation_plan.get("selected_plan_id")
    for item in augmentation_plan.get("ranked_plans") or []:
        if item.get("plan_id") == selected_id:
            return item
    return {}


def finding(severity: str, code: str, message: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "evidence": evidence or {},
    }


def count_severities(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in findings:
        severity = str(item.get("severity") or "low")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def render_plan_critic_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Plan Critic",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Selected skill: `{(report.get('selected_plan') or {}).get('skill')}`",
        f"- Recipe task: `{(report.get('selected_plan') or {}).get('recipe_task_actual')}`",
        f"- Findings: {report.get('finding_count')}",
        f"- Real execution safe: `{(report.get('decision') or {}).get('real_execution_safe')}`",
        "",
        "## Findings",
        "",
    ]
    if not report.get("findings"):
        lines.append("- None")
    else:
        for item in report.get("findings") or []:
            lines.append(f"- `{item.get('severity')}` `{item.get('code')}`: {item.get('message')}")
    lines.append("")
    return "\n".join(lines)


def load_optional_mapping(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {}
    return load_mapping(value)

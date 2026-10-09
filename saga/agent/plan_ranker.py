from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from saga.agent.benefit_estimator import evaluate_plan_utility, write_plan_utility_report
from saga.agent.skill_registry import executable_skill_names, planned_skill_names
from saga.core.config import save_json, save_text


PLAN_RANKING_VERSION = "saga_plan_ranking_v1"

SUPPORTED_TASKS = {
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
EXECUTABLE_TASKS = {
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


def normalize_candidate_plans(raw: dict[str, Any]) -> list[dict[str, Any]]:
    body = raw.get("planner_proposal", raw)
    candidates = body.get("candidate_plans")
    if isinstance(candidates, list) and candidates:
        return [normalize_candidate_plan(item, index=idx) for idx, item in enumerate(candidates) if isinstance(item, dict)]
    return [normalize_candidate_plan(body, index=0)]


def normalize_candidate_plan(raw: dict[str, Any], index: int) -> dict[str, Any]:
    updates = raw.get("recipe_intent_updates") if isinstance(raw.get("recipe_intent_updates"), dict) else {}
    task = str(raw.get("task") or updates.get("task") or "unknown")
    plan_id = safe_plan_id(str(raw.get("plan_id") or raw.get("id") or f"{task}_plan_{index + 1}"))
    pipeline = raw.get("pipeline") if isinstance(raw.get("pipeline"), list) else []
    return {
        "plan_id": plan_id,
        "planner": raw.get("planner", "llm"),
        "task": task,
        "confidence": clamp_float(raw.get("confidence"), default=0.5),
        "goals": normalize_string_list(raw.get("goals")),
        "strategy": str(raw.get("strategy") or ""),
        "recipe_intent_updates": updates,
        "pipeline": [normalize_step(step, idx) for idx, step in enumerate(pipeline) if isinstance(step, dict)],
        "expected_benefits": normalize_string_list(raw.get("expected_benefits")),
        "required_validations": normalize_string_list(raw.get("required_validations")),
        "risks": normalize_string_list(raw.get("risks")),
        "clarification_questions": normalize_string_list(raw.get("clarification_questions")),
        "cost": raw.get("cost") if isinstance(raw.get("cost"), dict) else {},
        "rationale": str(raw.get("rationale") or ""),
    }


def normalize_step(raw: dict[str, Any], index: int) -> dict[str, Any]:
    skill = str(raw.get("skill") or "")
    return {
        "id": safe_step_id(str(raw.get("id") or raw.get("step_id") or f"step_{index + 1}")),
        "skill": skill,
        "depends_on": normalize_string_list(raw.get("depends_on")),
        "execution_status": str(raw.get("execution_status") or inferred_skill_status(skill)),
        "params": raw.get("params") if isinstance(raw.get("params"), dict) else {},
        "reason": str(raw.get("reason") or raw.get("description") or ""),
    }


def rank_candidate_plans(
    candidates: list[dict[str, Any]],
    intent_spec: dict[str, Any],
    bridge_report: dict[str, Any],
    benefit_context: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    ranked = [
        score_candidate(
            candidate,
            intent_spec=intent_spec,
            bridge_report=bridge_report,
            benefit_context=benefit_context,
        )
        for candidate in candidates
    ]
    ranked = sorted(ranked, key=lambda item: item["score"], reverse=True)
    selected = ranked[0] if ranked else None
    plan_utility_report = build_plan_utility_report(ranked, selected)
    report = {
        "schema_version": PLAN_RANKING_VERSION,
        "candidate_count": len(ranked),
        "selected_plan_id": selected.get("plan_id") if selected else None,
        "selected_task": selected.get("task") if selected else None,
        "selected_utility_score": selected.get("plan_utility", {}).get("utility_score") if selected else None,
        "ranked_plans": ranked,
        "plan_utility_report": plan_utility_report,
    }
    if output_dir:
        write_plan_ranking(output_dir, report)
        if plan_utility_report:
            write_plan_utility_report(output_dir, plan_utility_report)
    return report


def score_candidate(
    candidate: dict[str, Any],
    intent_spec: dict[str, Any],
    bridge_report: dict[str, Any],
    benefit_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executable_names = executable_skill_names()
    planned_names = planned_skill_names()
    task = str(candidate.get("task") or "unknown")
    rule_task = intent_spec.get("intent", {}).get("task")
    pipeline = candidate.get("pipeline") or []
    executable_count = 0
    planned_count = 0
    unsupported_count = 0
    gpu_required = False
    reasons = []

    score = int(float(candidate.get("confidence") or 0.0) * 20)
    plan_utility = evaluate_plan_utility(candidate, benefit_context)
    utility_score = float(plan_utility.get("utility_score") or 0.0)
    score += int((utility_score - 0.5) * 24)
    if utility_score >= 0.65:
        reasons.append("high_plan_utility")
    elif utility_score >= 0.4:
        reasons.append("moderate_plan_utility")
    else:
        reasons.append("low_plan_utility")
    if task in EXECUTABLE_TASKS:
        score += 30
        reasons.append("executable_task")
    elif task in SUPPORTED_TASKS:
        score += 8
        reasons.append("supported_but_not_executable_task")
    else:
        score -= 40
        reasons.append("unsupported_task")

    if rule_task and rule_task == task:
        score += 10
        reasons.append("matches_rule_intent")
    elif rule_task in EXECUTABLE_TASKS and task in EXECUTABLE_TASKS:
        score -= 12
        reasons.append("conflicts_with_rule_intent")

    if bridge_report.get("valid"):
        score += 8
        reasons.append("format_bridge_valid")
    else:
        bridge_optional = task in {"geodiff_sar_generation", "raysar_synthesis", "background_generation"}
        score -= 4 if bridge_optional else 12
        reasons.append("format_bridge_partial_optional" if bridge_optional else "format_bridge_invalid")

    memory_policy = memory_policy_adjustment(candidate, benefit_context)
    if memory_policy["adjustment"]:
        score += int(memory_policy["adjustment"] * 20)
        reasons.append(memory_policy["reason"])
    for step in pipeline:
        skill = str(step.get("skill") or "")
        if skill in executable_names:
            executable_count += 1
            score += 5
        elif skill in planned_names:
            planned_count += 1
            score -= 4
        else:
            unsupported_count += 1
            score -= 20
        if any(keyword in skill.lower() for keyword in ["lora", "diffusion", "geodiff", "gaussian"]):
            gpu_required = True

    if not pipeline:
        score -= 6
        reasons.append("empty_pipeline")
    if candidate.get("clarification_questions"):
        score -= 8
        reasons.append("needs_clarification")
    if unsupported_count:
        reasons.append("has_unsupported_skills")
    if planned_count:
        reasons.append("has_planned_skills")

    return {
        "plan_id": candidate.get("plan_id"),
        "task": task,
        "score": score,
        "rank_reasons": reasons,
        "confidence": candidate.get("confidence"),
        "strategy": candidate.get("strategy"),
        "executable_step_count": executable_count,
        "planned_step_count": planned_count,
        "unsupported_step_count": unsupported_count,
        "gpu_required": gpu_required,
        "requires_clarification": bool(candidate.get("clarification_questions")),
        "plan_utility": plan_utility,
        "memory_policy": memory_policy,
        "candidate": candidate,
    }


def memory_policy_adjustment(candidate: dict[str, Any], benefit_context: dict[str, Any] | None) -> dict[str, Any]:
    context = (benefit_context or {}).get("memory_context") or {}
    global_learning = context.get("global_policy_learning") or {}
    priors = global_learning.get("skill_priors") or {}
    names = candidate_skill_names(candidate)
    if not names:
        return {"adjustment": 0.0, "reason": "no_memory_policy_skill"}
    adjustments = []
    used = []
    for name in names:
        prior = priors.get(name)
        if not isinstance(prior, dict):
            continue
        try:
            adjustment = float(prior.get("planner_adjustment") or 0.0)
        except (TypeError, ValueError):
            adjustment = 0.0
        if adjustment:
            adjustments.append(adjustment)
            used.append({"skill": name, "planner_adjustment": adjustment, "recommendation": prior.get("recommendation")})
    if not adjustments:
        return {"adjustment": 0.0, "reason": "no_matching_memory_prior"}
    average = sum(adjustments) / len(adjustments)
    return {
        "adjustment": round(max(-0.18, min(0.18, average)), 3),
        "reason": "positive_memory_policy_prior" if average > 0 else "negative_memory_policy_prior",
        "skills": used,
    }


def candidate_skill_names(candidate: dict[str, Any]) -> list[str]:
    names = []
    for step in candidate.get("pipeline") or []:
        skill = str(step.get("skill") or "")
        if skill.endswith("Skill") and skill not in names:
            names.append(skill)
    if names:
        return names
    task_to_skill = {
        "traditional_augmentation": "TraditionalAugmentationSkill",
        "diffusion_lora_generation": "DiffusionLoRAGenerationSkill",
        "geodiff_sar_generation": "GeoDiffSARSkill",
        "gaussian_splatting_completion": "GaussianSplattingCompletionSkill",
        "gan_generation": "GANImageToImageSkill",
        "style_transfer": "StyleTransferSkill",
        "pseudocolor_transform": "PseudocolorSkill",
        "background_generation": "BackgroundGenerationSkill",
        "target_background_composition": "TargetBackgroundCompositionSkill",
        "raysar_synthesis": "RaySARSweepSynthesisSkill",
        "classification_evaluation": "ClassificationEvaluationSkill",
    }
    skill = task_to_skill.get(str(candidate.get("task") or ""))
    return [skill] if skill else []


def build_plan_utility_report(ranked: list[dict[str, Any]], selected: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": "saga_plan_utility_v1",
        "selected_plan_id": selected.get("plan_id") if selected else None,
        "selected_utility_score": selected.get("plan_utility", {}).get("utility_score") if selected else None,
        "ranked_plans": [
            {
                "plan_id": item.get("plan_id"),
                "task": item.get("task"),
                **(item.get("plan_utility") or {}),
            }
            for item in ranked
        ],
    }


def selected_candidate_from_ranking(candidates: list[dict[str, Any]], ranking: dict[str, Any]) -> dict[str, Any]:
    selected_id = ranking.get("selected_plan_id")
    for candidate in candidates:
        if candidate.get("plan_id") == selected_id:
            return candidate
    return candidates[0] if candidates else {}


def write_plan_ranking(output_dir: str | Path, report: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    save_json(path / "candidate_plans.json", {"schema_version": "saga_candidate_plans_v1", "plans": [item["candidate"] for item in report.get("ranked_plans", [])]})
    save_json(path / "plan_ranking.json", report)
    save_text(path / "plan_ranking.md", render_plan_ranking_markdown(report))


def render_plan_ranking_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Plan Ranking",
        "",
        f"- Candidate count: {report.get('candidate_count')}",
        f"- Selected plan: `{report.get('selected_plan_id')}`",
        f"- Selected task: `{report.get('selected_task')}`",
        "",
    ]
    for idx, item in enumerate(report.get("ranked_plans", []), start=1):
        lines.extend(
            [
                f"## {idx}. {item.get('plan_id')}",
                "",
                f"- Task: `{item.get('task')}`",
                f"- Score: {item.get('score')}",
                f"- Confidence: {item.get('confidence')}",
                f"- Strategy: {item.get('strategy')}",
                f"- Plan utility: {item.get('plan_utility', {}).get('utility_score')}",
                f"- Utility verdict: {item.get('plan_utility', {}).get('verdict')}",
                f"- Executable steps: {item.get('executable_step_count')}",
                f"- Planned steps: {item.get('planned_step_count')}",
                f"- Unsupported steps: {item.get('unsupported_step_count')}",
                f"- Reasons: `{item.get('rank_reasons')}`",
                "",
            ]
        )
    return "\n".join(lines)


def inferred_skill_status(skill: str) -> str:
    if skill in executable_skill_names():
        return "executable"
    if skill in planned_skill_names():
        return "planned"
    return "blocked"


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)


def safe_plan_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "candidate_plan"


def safe_step_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return cleaned or "step"

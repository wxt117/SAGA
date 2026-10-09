from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text


RUN_EXPLANATION_VERSION = "saga_run_explanation_v1"
RUN_COMPARISON_VERSION = "saga_run_comparison_v1"


def explain_agent_run(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    state = load_optional_mapping(run_path / "agent_state.json")
    augmentation_plan = load_optional_mapping(run_path / "augmentation_plan.json")
    request_constraints = load_optional_mapping(run_path / "request_constraints.json")
    dataset_need_profile = load_optional_mapping(run_path / "dataset_need_profile.json")
    skill_utility = load_optional_mapping(run_path / "skill_utility.json")
    memory_retrieval = load_optional_mapping(run_path / "memory_retrieval.json")
    recipe = load_optional_mapping(run_path / "recipe.yaml")
    execution = load_optional_mapping(run_path / "execution" / "recipe_execution.json")

    selected_plan = find_selected_augmentation_plan(augmentation_plan)
    explanation = {
        "schema_version": RUN_EXPLANATION_VERSION,
        "run_dir": run_path.as_posix(),
        "request": state.get("request"),
        "task": state.get("task"),
        "dataset_root": state.get("dataset_root"),
        "selected": {
            "plan_id": augmentation_plan.get("selected_plan_id"),
            "skill": augmentation_plan.get("selected_skill"),
            "recipe_task": augmentation_plan.get("selected_recipe_task"),
            "recipe_id": state.get("recipe_id") or recipe.get("recipe_id"),
            "execution_status": state.get("execution_status") or execution.get("status"),
        },
        "why": build_selection_explanation(
            selected_plan=selected_plan,
            request_constraints=request_constraints,
            dataset_need_profile=dataset_need_profile,
            skill_utility=skill_utility,
            memory_retrieval=memory_retrieval,
        ),
        "top_needs": dataset_need_profile.get("top_needs") or [],
        "top_ranked_plans": compact_ranked_plans(augmentation_plan),
        "recipe_pipeline": compact_recipe_pipeline(recipe),
        "execution_summary": compact_execution(execution),
        "artifacts": {
            "agent_state": (run_path / "agent_state.json").as_posix(),
            "augmentation_plan": (run_path / "augmentation_plan.json").as_posix(),
            "recipe": (run_path / "recipe.yaml").as_posix(),
            "execution": (run_path / "execution" / "recipe_execution.json").as_posix(),
        },
    }
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else run_path
    save_json(target_dir / "plan_explanation.json", explanation)
    save_text(target_dir / "plan_explanation.md", render_plan_explanation_markdown(explanation))
    return explanation


def compare_agent_runs(
    run_dirs: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    rows = [summarize_agent_run(Path(run_dir).expanduser().resolve()) for run_dir in run_dirs]
    comparison = {
        "schema_version": RUN_COMPARISON_VERSION,
        "run_count": len(rows),
        "runs": rows,
        "ranking": rank_run_summaries(rows),
        "notes": [
            "Comparison ranks completed runs by downstream evaluator evidence first, then observer quality, then execution status.",
            "Dry-run evaluator results are treated as unavailable downstream evidence.",
        ],
    }
    out = Path(output_dir).expanduser().resolve()
    save_json(out / "run_comparison.json", comparison)
    save_text(out / "run_comparison.md", render_run_comparison_markdown(comparison))
    return comparison


def summarize_agent_run(run_path: Path) -> dict[str, Any]:
    state = load_optional_mapping(run_path / "agent_state.json")
    augmentation_plan = load_optional_mapping(run_path / "augmentation_plan.json")
    execution = load_optional_mapping(run_path / "execution" / "recipe_execution.json")
    export_report = load_optional_mapping(run_path / "augmented_dataset" / "export_report.json")
    evaluator_report = find_evaluator_report(run_path, execution)
    quality, sar_artifacts, distribution = summarize_observer_steps(execution)
    classification_delta = safe_float(evaluator_report.get("delta")) if evaluator_report else None
    classification_verdict = evaluator_report.get("verdict") if evaluator_report else None
    return {
        "run_dir": run_path.as_posix(),
        "request": state.get("request"),
        "task": state.get("task"),
        "dataset_root": state.get("dataset_root"),
        "selected_skill": augmentation_plan.get("selected_skill") or state.get("augmentation_selected_skill"),
        "selected_recipe_task": augmentation_plan.get("selected_recipe_task") or state.get("augmentation_selected_recipe_task"),
        "recipe_id": state.get("recipe_id") or execution.get("recipe_id"),
        "execution_status": state.get("execution_status") or execution.get("status"),
        "dry_run": execution.get("dry_run", state.get("dry_run")),
        "bridge_valid": state.get("bridge_valid"),
        "quality": quality,
        "sar_artifacts": sar_artifacts,
        "distribution": distribution,
        "classification": {
            "available": bool(evaluator_report) and evaluator_report.get("baseline") is not None and evaluator_report.get("augmented") is not None,
            "status": evaluator_report.get("status") if evaluator_report else None,
            "primary_metric": evaluator_report.get("primary_metric") if evaluator_report else None,
            "baseline": safe_float(evaluator_report.get("baseline")) if evaluator_report else None,
            "augmented": safe_float(evaluator_report.get("augmented")) if evaluator_report else None,
            "delta": classification_delta,
            "verdict": classification_verdict,
            "confidence": evaluator_report.get("confidence") if evaluator_report else None,
            "leakage_risk": evaluator_report.get("leakage_risk") if evaluator_report else None,
        },
        "export": {
            "status": export_report.get("status"),
            "generated_image_count": export_report.get("generated_image_count"),
            "manifest_count": export_report.get("manifest_count"),
            "output_dir": export_report.get("output_dir"),
        },
    }


def build_selection_explanation(
    *,
    selected_plan: dict[str, Any],
    request_constraints: dict[str, Any],
    dataset_need_profile: dict[str, Any],
    skill_utility: dict[str, Any],
    memory_retrieval: dict[str, Any],
) -> list[dict[str, Any]]:
    facts = []
    if request_constraints:
        facts.append(
            {
                "source": "request_constraints",
                "summary": {
                    "speed_priority": request_constraints.get("speed_priority"),
                    "quality_priority": request_constraints.get("quality_priority"),
                    "gpu_budget": request_constraints.get("gpu_budget"),
                    "target_count": request_constraints.get("target_count"),
                    "training_epochs": request_constraints.get("training_epochs"),
                    "needs_downstream_evidence": request_constraints.get("needs_downstream_evidence"),
                },
            }
        )
    if dataset_need_profile:
        facts.append({"source": "dataset_need_profile", "summary": dataset_need_profile.get("top_needs") or []})
    if selected_plan:
        facts.append(
            {
                "source": "selected_augmentation_plan",
                "summary": {
                    "planner_score": selected_plan.get("planner_score"),
                    "base_utility": selected_plan.get("base_utility"),
                    "adjustments": selected_plan.get("adjustments") or [],
                    "params": selected_plan.get("params") or {},
                    "expected_benefits": selected_plan.get("expected_benefits") or [],
                    "risks": selected_plan.get("risks") or [],
                    "rationale": selected_plan.get("rationale"),
                },
            }
        )
    selected_skill = selected_plan.get("skill") if selected_plan else None
    if selected_skill:
        for item in skill_utility.get("ranked_skills") or []:
            if item.get("skill") == selected_skill:
                facts.append(
                    {
                        "source": "skill_utility",
                        "summary": {
                            "utility_score": item.get("utility_score"),
                            "need_match": item.get("need_match"),
                            "metadata_readiness": item.get("metadata_readiness"),
                            "request_alignment": item.get("request_alignment"),
                            "memory_outcome": item.get("memory_outcome"),
                            "matched_needs": item.get("matched_needs") or [],
                        },
                    }
                )
                break
    policy_summary = memory_retrieval.get("policy_summary") or {}
    if policy_summary:
        facts.append({"source": "policy_memory", "summary": policy_summary})
    return facts


def find_selected_augmentation_plan(augmentation_plan: dict[str, Any]) -> dict[str, Any]:
    selected_id = augmentation_plan.get("selected_plan_id")
    for item in augmentation_plan.get("ranked_plans") or []:
        if item.get("plan_id") == selected_id:
            return item
    return {}


def compact_ranked_plans(augmentation_plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in (augmentation_plan.get("ranked_plans") or [])[:8]:
        rows.append(
            {
                "plan_id": item.get("plan_id"),
                "skill": item.get("skill"),
                "status": item.get("status"),
                "recipe_task": item.get("recipe_task"),
                "planner_score": item.get("planner_score"),
                "params": item.get("params"),
                "rationale": item.get("rationale"),
            }
        )
    return rows


def compact_recipe_pipeline(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.get("id"),
            "skill": step.get("skill"),
            "depends_on": step.get("depends_on") or [],
            "params": compact_step_params(step.get("params") or {}),
        }
        for step in recipe.get("pipeline", [])
    ]


def compact_step_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "config",
        "dataset_root",
        "content",
        "style",
        "target_count",
        "training_epochs",
        "model_family",
        "filters",
        "exclude_filters",
        "max_trials",
        "baseline_dataset",
        "augmented_from_step",
    )
    return {key: params.get(key) for key in keys if params.get(key) not in (None, "", {}, [])}


def compact_execution(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": execution.get("status"),
        "dry_run": execution.get("dry_run"),
        "step_count": len(execution.get("steps") or []),
        "steps": [
            {
                "step_id": step.get("step_id"),
                "skill": step.get("skill"),
                "status": step.get("status"),
                "trigger_count": step.get("trigger_count") or len(step.get("triggers") or []),
            }
            for step in execution.get("steps", [])
        ],
    }


def summarize_observer_steps(execution: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    quality: dict[str, Any] = {}
    sar_artifacts: dict[str, Any] = {}
    distribution: dict[str, Any] = {}
    for step in execution.get("steps") or []:
        if step.get("quality_report"):
            report = step.get("quality_report") or {}
            quality = {"status": report.get("status"), "trigger_count": len(report.get("triggers") or [])}
        if step.get("sar_artifact_report"):
            report = step.get("sar_artifact_report") or {}
            sar_artifacts = {"status": report.get("status"), "trigger_count": len(report.get("triggers") or [])}
        if step.get("distribution_report"):
            report = step.get("distribution_report") or {}
            distribution = {"status": report.get("status"), "trigger_count": len(report.get("triggers") or []), "metrics": report.get("metrics") or {}}
    return quality, sar_artifacts, distribution


def find_evaluator_report(run_path: Path, execution: dict[str, Any]) -> dict[str, Any]:
    direct = load_optional_mapping(run_path / "recipe_artifacts" / "classification_evaluation" / "evaluator_report.json")
    if direct:
        return direct
    for step in execution.get("steps") or []:
        report = step.get("evaluator_report")
        if isinstance(report, dict) and report:
            return report
    return {}


def rank_run_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=run_rank_score, reverse=True)
    return [
        {
            "rank": idx,
            "run_dir": item.get("run_dir"),
            "selected_skill": item.get("selected_skill"),
            "score": run_rank_score(item),
            "classification_verdict": (item.get("classification") or {}).get("verdict"),
            "classification_delta": (item.get("classification") or {}).get("delta"),
            "execution_status": item.get("execution_status"),
            "quality_trigger_count": (item.get("quality") or {}).get("trigger_count"),
            "sar_artifact_trigger_count": (item.get("sar_artifacts") or {}).get("trigger_count"),
        }
        for idx, item in enumerate(ranked, start=1)
    ]


def run_rank_score(row: dict[str, Any]) -> float:
    score = 0.0
    if row.get("execution_status") in {"succeeded", "dry_run", "warning"}:
        score += 1.0
    classification = row.get("classification") or {}
    delta = safe_float(classification.get("delta"))
    if classification.get("verdict") == "improved":
        score += 3.0
    elif classification.get("verdict") == "regressed":
        score -= 3.0
    if delta is not None:
        score += max(-1.0, min(1.0, delta))
    score -= 0.25 * int((row.get("quality") or {}).get("trigger_count") or 0)
    score -= 0.25 * int((row.get("sar_artifacts") or {}).get("trigger_count") or 0)
    return round(score, 4)


def render_plan_explanation_markdown(explanation: dict[str, Any]) -> str:
    selected = explanation.get("selected") or {}
    lines = [
        "# SAGA Plan Explanation",
        "",
        f"- Run: `{explanation.get('run_dir')}`",
        f"- Task: `{explanation.get('task')}`",
        f"- Selected skill: `{selected.get('skill')}`",
        f"- Selected recipe task: `{selected.get('recipe_task')}`",
        f"- Execution status: `{selected.get('execution_status')}`",
        "",
        "## Why This Plan",
        "",
    ]
    for item in explanation.get("why") or []:
        lines.append(f"### {item.get('source')}")
        lines.append("")
        lines.append(f"`{item.get('summary')}`")
        lines.append("")
    lines.extend(["## Ranked Plans", ""])
    for item in explanation.get("top_ranked_plans") or []:
        lines.append(
            f"- `{item.get('plan_id')}`: {item.get('skill')} -> {item.get('recipe_task')} "
            f"(score={item.get('planner_score')})"
        )
    lines.extend(["", "## Pipeline", ""])
    for step in explanation.get("recipe_pipeline") or []:
        lines.append(f"- `{step.get('id')}` {step.get('skill')} params=`{step.get('params')}`")
    lines.append("")
    return "\n".join(lines)


def render_run_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# SAGA Run Comparison",
        "",
        f"- Run count: {comparison.get('run_count')}",
        "",
        "## Ranking",
        "",
    ]
    for item in comparison.get("ranking") or []:
        lines.append(
            f"- {item.get('rank')}. `{item.get('run_dir')}` skill={item.get('selected_skill')} "
            f"score={item.get('score')} verdict={item.get('classification_verdict')} "
            f"delta={item.get('classification_delta')}"
        )
    lines.extend(["", "## Runs", ""])
    for row in comparison.get("runs") or []:
        lines.extend(
            [
                f"### {row.get('run_dir')}",
                "",
                f"- Skill: `{row.get('selected_skill')}`",
                f"- Task: `{row.get('selected_recipe_task')}`",
                f"- Execution: `{row.get('execution_status')}` dry_run={row.get('dry_run')}",
                f"- Classification: `{row.get('classification')}`",
                f"- Quality: `{row.get('quality')}`",
                f"- SAR artifacts: `{row.get('sar_artifacts')}`",
                f"- Export: `{row.get('export')}`",
                "",
            ]
        )
    return "\n".join(lines)


def load_optional_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_mapping(path)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

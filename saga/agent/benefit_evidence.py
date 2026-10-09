from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text


BENEFIT_EVIDENCE_VERSION = "saga_benefit_evidence_v1"


def build_benefit_evidence_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    execution_report: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else run_path
    state = state or load_optional_mapping(run_path / "agent_state.json")
    execution = execution_report or load_optional_mapping(run_path / "execution" / "recipe_execution.json")
    recipe = load_optional_mapping(run_path / "recipe.yaml")
    augmentation_plan = load_optional_mapping(run_path / "augmentation_plan.json")

    summaries = summarize_execution_evidence(execution)
    level, label, reasons = infer_evidence_level(
        state=state,
        execution=execution,
        recipe=recipe,
        summaries=summaries,
    )
    downstream_claim_allowed = level >= 5 and summaries["classification"].get("verdict") == "improved"
    report = {
        "schema_version": BENEFIT_EVIDENCE_VERSION,
        "run_dir": run_path.as_posix(),
        "request": state.get("request"),
        "task": state.get("task") or recipe.get("task"),
        "selected_skill": augmentation_plan.get("selected_skill") or state.get("augmentation_selected_skill"),
        "selected_recipe_task": augmentation_plan.get("selected_recipe_task") or state.get("augmentation_selected_recipe_task"),
        "evidence_level": level,
        "evidence_label": label,
        "level_reasons": reasons,
        "downstream_claim_allowed": downstream_claim_allowed,
        "lightweight_claim_allowed": level >= 2,
        "summaries": summaries,
        "risks": infer_evidence_risks(summaries, state, recipe),
        "recommended_next_action": recommend_next_action(level, summaries, state, recipe),
        "notes": [
            "BenefitEvidenceReport separates generated-data usability evidence from downstream task-gain evidence.",
            "Quality, distribution, SAR artifact, duplicate, and leakage metrics are lightweight probes, not final downstream proof.",
            "Downstream benefit claims require a valid task evaluator result that is not invalidated by leakage or near-duplicate gates.",
        ],
    }
    save_json(out / "benefit_evidence.json", report)
    save_text(out / "benefit_evidence.md", render_benefit_evidence_markdown(report))
    return report


def summarize_execution_evidence(execution: dict[str, Any]) -> dict[str, Any]:
    steps = execution.get("steps") if isinstance(execution, dict) else []
    steps = steps if isinstance(steps, list) else []
    quality_step = find_step(steps, "evaluate_outputs")
    distribution_step = find_step(steps, "evaluate_distribution")
    sar_step = find_step(steps, "evaluate_sar_artifacts")
    leakage_step = find_step_by_skill(steps, "LeakageCheckSkill")
    duplicate_step = find_step_by_skill(steps, "DuplicateNearDuplicateSkill")
    classification_step = find_step_by_skill(steps, "ClassificationEvaluationSkill")
    export_step = find_step(steps, "export_dataset")
    generation_steps = [step for step in steps if is_generation_step(step)]

    quality_report = nested_report(quality_step, "quality_report")
    distribution_report = nested_report(distribution_step, "distribution_report")
    sar_report = nested_report(sar_step, "sar_artifact_report")
    leakage_report = nested_report(leakage_step, "leakage_report")
    duplicate_report = nested_report(duplicate_step, "duplicate_report")
    classifier_report = nested_report(classification_step, "evaluator_report")
    if not classifier_report:
        skill_report = nested_report(classification_step, "skill_report")
        classifier_report = skill_report.get("evaluator_report") if isinstance(skill_report, dict) else {}

    return {
        "execution": {
            "status": execution.get("status"),
            "dry_run": execution.get("dry_run"),
            "step_count": len(steps),
            "generation_step_count": len(generation_steps),
            "generation_steps": [
                {
                    "step_id": step.get("step_id"),
                    "skill": step.get("skill"),
                    "status": step.get("status"),
                    "image_count": step.get("image_count"),
                    "expected_count": step.get("expected_count"),
                }
                for step in generation_steps
            ],
        },
        "quality": summarize_observer(quality_step, quality_report, "quality"),
        "distribution": summarize_observer(distribution_step, distribution_report, "distribution"),
        "sar_artifacts": summarize_observer(sar_step, sar_report, "sar_artifacts"),
        "leakage": {
            "available": bool(leakage_step or leakage_report),
            "status": leakage_report.get("status") or leakage_step.get("status"),
            "trigger_count": safe_int(leakage_report.get("trigger_count") or leakage_step.get("trigger_count")) or 0,
            "overlap_count": leakage_overlap_count(leakage_report),
        },
        "duplicates": {
            "available": bool(duplicate_step or duplicate_report),
            "status": duplicate_report.get("status") or duplicate_step.get("status"),
            "pair_count": safe_int(duplicate_report.get("pair_count") or duplicate_step.get("pair_count")) or 0,
            "hamming_threshold": duplicate_report.get("hamming_threshold"),
        },
        "classification": summarize_classification(classifier_report, classification_step, leakage_report, duplicate_report),
        "export": {
            "available": bool(export_step),
            "status": export_step.get("status"),
            "generated_image_count": export_step.get("generated_image_count"),
            "manifest_count": export_step.get("manifest_count"),
            "output_dir": export_step.get("output_dir"),
        },
    }


def infer_evidence_level(
    *,
    state: dict[str, Any],
    execution: dict[str, Any],
    recipe: dict[str, Any],
    summaries: dict[str, Any],
) -> tuple[int, str, list[str]]:
    reasons: list[str] = []
    execution_status = summaries["execution"].get("status") or state.get("execution_status")
    dry_run = bool(summaries["execution"].get("dry_run", state.get("dry_run")))
    if not execution:
        return 1, "planned_only", ["No recipe execution report is available."]
    if execution_status not in {"succeeded", "warning", "dry_run"}:
        return 1, "execution_failed_or_blocked", [f"Execution status is {execution_status!r}."]
    if dry_run:
        reasons.append("Recipe executed in dry-run mode; generated samples were not fully materialized.")
        return 1, "dry_run_plan_evidence", reasons

    level = 1
    label = "generated_without_evaluation"
    generation_count = generated_count(summaries)
    if generation_count > 0:
        level = 2
        label = "generated_quality_screen_pending"
        reasons.append(f"Generated/exported image count is {generation_count}.")
    quality = summaries["quality"]
    sar = summaries["sar_artifacts"]
    if observer_passed(quality) and observer_passed_or_unavailable(sar):
        level = max(level, 2)
        label = "quality_gate_passed"
        reasons.append("Quality/SAR artifact observers did not raise blocking triggers.")
    distribution = summaries["distribution"]
    duplicates = summaries["duplicates"]
    leakage = summaries["leakage"]
    if (
        observer_passed_or_unavailable(quality)
        and observer_passed_or_unavailable(sar)
        and observer_passed_or_unavailable(distribution)
        and int(duplicates.get("pair_count") or 0) == 0
        and int(leakage.get("trigger_count") or 0) == 0
        and (distribution.get("available") or duplicates.get("available") or leakage.get("available"))
    ):
        level = max(level, 3)
        label = "lightweight_probe_passed"
        reasons.append("Lightweight probes did not show distribution, duplicate, or leakage blockers.")
    if metadata_coverage_improved(recipe, summaries):
        level = max(level, 4)
        label = "metadata_or_slice_coverage_improved"
        reasons.append("Recipe/export evidence indicates metadata or slice coverage support.")
    classification = summaries["classification"]
    if classification.get("available"):
        if classification.get("verdict") == "improved" and not classification.get("invalidated_by_data_gate"):
            level = 5
            label = "downstream_task_improved"
            reasons.append("Downstream classification evaluator reports improvement and data gates did not invalidate it.")
        elif classification.get("invalidated_by_data_gate"):
            level = min(level, 3)
            label = "downstream_result_invalidated_by_data_gate"
            reasons.append("Classification result exists but leakage/duplicate gates invalidated the downstream claim.")
        else:
            level = max(level, 4)
            label = "downstream_task_evaluated_without_improvement"
            reasons.append("Downstream evaluator ran, but no improvement verdict is available.")
    return level, label, reasons or ["Execution evidence was recorded but no positive gate was identified."]


def infer_evidence_risks(summaries: dict[str, Any], state: dict[str, Any], recipe: dict[str, Any]) -> list[dict[str, Any]]:
    risks = []
    for key, label in (
        ("quality", "quality_observer"),
        ("distribution", "distribution_proxy"),
        ("sar_artifacts", "sar_artifact_observer"),
    ):
        summary = summaries.get(key) or {}
        if int(summary.get("trigger_count") or 0) > 0:
            risks.append({"source": label, "severity": "medium", "detail": summary.get("triggers") or []})
    if int((summaries.get("leakage") or {}).get("trigger_count") or 0) > 0:
        risks.append({"source": "leakage", "severity": "high", "detail": "Leakage gate triggered."})
    if int((summaries.get("duplicates") or {}).get("pair_count") or 0) > 0:
        risks.append({"source": "duplicates", "severity": "high", "detail": "Near-duplicate pairs were detected."})
    task = state.get("task") or recipe.get("task")
    selected_skill = state.get("augmentation_selected_skill")
    if selected_skill == "BackgroundGenerationSkill" and task == "background_generation":
        risks.append(
            {
                "source": "task_scope",
                "severity": "low",
                "detail": "Background images are scene assets; classification benefit is indirect until composition/evaluation.",
            }
        )
    if selected_skill == "TargetBackgroundCompositionSkill":
        risks.append(
            {
                "source": "composition_quality",
                "severity": "medium",
                "detail": "Target/background fusion can introduce mask, scale, and domain mismatch errors; use quality, distribution, and annotation checks before downstream claims.",
            }
        )
    if selected_skill == "GeoDiffSARSkill":
        risks.append(
            {
                "source": "physical_prior_diffusion",
                "severity": "medium",
                "detail": "GeoDiff-SAR outputs depend on GEFM quality, 3D-prior fit, and diffusion ControlNet behavior; sparse-angle benefit needs observer or downstream validation.",
            }
        )
    if selected_skill in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
        risks.append(
            {
                "source": "domain_gap",
                "severity": "medium",
                "detail": "RaySAR outputs are physically interpretable but can have sim-to-real domain gap.",
            }
        )
    return risks


def recommend_next_action(level: int, summaries: dict[str, Any], state: dict[str, Any], recipe: dict[str, Any]) -> str:
    if level == 1:
        if not summaries["execution"].get("status"):
            return "Execute the recipe in dry-run or pilot mode to collect observer evidence before scaling up."
        if summaries["execution"].get("dry_run"):
            return "Run the selected recipe on a small pilot batch before making quality or benefit claims."
        return "Inspect execution failures and rerun only after the recipe executes successfully."
    if any(int((summaries.get(key) or {}).get("trigger_count") or 0) > 0 for key in ("quality", "distribution", "sar_artifacts")):
        return "Run bounded repair on the recipe, then compare benefit evidence before accepting outputs."
    if int((summaries.get("duplicates") or {}).get("pair_count") or 0) > 0:
        return "Remove duplicate/near-duplicate generated samples before writing this run into positive policy memory."
    if int((summaries.get("leakage") or {}).get("trigger_count") or 0) > 0:
        return "Fix split leakage before using downstream evaluator results."
    if level < 5 and downstream_evidence_requested(state, recipe):
        return "Run the downstream evaluator with leakage and duplicate gates enabled."
    if level >= 3:
        return "Outputs are acceptable under lightweight probes; run downstream evaluator only if claiming task gain."
    return "Inspect generated samples and lightweight observer reports before scaling up."


def summarize_observer(step: dict[str, Any], report: dict[str, Any], name: str) -> dict[str, Any]:
    triggers = report.get("triggers") or step.get("triggers") or []
    return {
        "available": bool(step or report),
        "status": report.get("status") or step.get("status"),
        "trigger_count": len(triggers),
        "triggers": triggers[:20],
        "metrics": report.get("metrics") or step.get("metrics") or {},
        "image_count": report.get("image_count") or step.get("image_count"),
        "report_name": name,
    }


def summarize_classification(report: dict[str, Any], step: dict[str, Any], leakage: dict[str, Any], duplicates: dict[str, Any]) -> dict[str, Any]:
    report = report or {}
    step = step or {}
    leakage = leakage or {}
    duplicates = duplicates or {}
    metrics = report.get("metrics") or step.get("metrics") or {}
    primary = metrics.get("primary") if isinstance(metrics, dict) else {}
    baseline = safe_float(report.get("baseline") or (primary or {}).get("baseline"))
    augmented = safe_float(report.get("augmented") or (primary or {}).get("augmented"))
    delta = safe_float(report.get("delta") or (primary or {}).get("delta"))
    if delta is None and baseline is not None and augmented is not None:
        delta = augmented - baseline
    verdict = report.get("verdict") or (primary or {}).get("verdict")
    if not verdict and delta is not None:
        verdict = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    leakage_trigger_count = safe_int(leakage.get("trigger_count")) or 0
    duplicate_pair_count = safe_int(duplicates.get("pair_count")) or 0
    invalidated = leakage_trigger_count > 0 or duplicate_pair_count > 0
    return {
        "available": bool(report or step) and baseline is not None and augmented is not None,
        "status": report.get("status") or step.get("status"),
        "primary_metric": report.get("primary_metric") or (primary or {}).get("metric"),
        "baseline": baseline,
        "augmented": augmented,
        "delta": delta,
        "verdict": "invalidated_by_data_gate" if invalidated and verdict else verdict,
        "raw_verdict": verdict,
        "invalidated_by_data_gate": invalidated,
        "data_gates": {
            "leakage_trigger_count": leakage_trigger_count,
            "duplicate_pair_count": duplicate_pair_count,
        },
    }


def render_benefit_evidence_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Benefit Evidence Report",
        "",
        f"- Run: `{report.get('run_dir')}`",
        f"- Task: `{report.get('task')}`",
        f"- Selected skill: `{report.get('selected_skill')}`",
        f"- Evidence level: **{report.get('evidence_level')}** (`{report.get('evidence_label')}`)",
        f"- Downstream claim allowed: `{report.get('downstream_claim_allowed')}`",
        f"- Recommended next action: {report.get('recommended_next_action')}",
        "",
        "## Reasons",
        "",
    ]
    for reason in report.get("level_reasons") or []:
        lines.append(f"- {reason}")
    lines.extend(["", "## Probe Summary", ""])
    summaries = report.get("summaries") or {}
    for key in ("quality", "distribution", "sar_artifacts", "leakage", "duplicates", "classification", "export"):
        value = summaries.get(key) or {}
        lines.append(f"- `{key}`: `{compact_summary(value)}`")
    risks = report.get("risks") or []
    if risks:
        lines.extend(["", "## Risks", ""])
        for risk in risks:
            lines.append(f"- `{risk.get('source')}` ({risk.get('severity')}): {risk.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def compact_summary(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "available",
        "status",
        "trigger_count",
        "pair_count",
        "overlap_count",
        "image_count",
        "generated_image_count",
        "manifest_count",
        "verdict",
        "delta",
    )
    return {key: value.get(key) for key in keys if value.get(key) not in (None, "", {}, [])}


def is_generation_step(step: dict[str, Any]) -> bool:
    skill = str(step.get("skill") or "")
    return skill in {
        "StyleTransferSkill",
        "DiffusionLoRAGenerationSkill",
        "SARLoRAGenerationSkill",
        "TraditionalAugmentationSkill",
        "TraditionalAugSkill",
        "GANImageToImageSkill",
        "GaussianSplattingCompletionSkill",
        "PseudocolorSkill",
        "BackgroundGenerationSkill",
        "GeoDiffSARSkill",
        "RaySARSynthesisSkill",
        "RaySARSweepSynthesisSkill",
    }


def observer_passed(summary: dict[str, Any]) -> bool:
    if not summary.get("available"):
        return False
    return int(summary.get("trigger_count") or 0) == 0 and str(summary.get("status")) not in {"failed", "blocked"}


def observer_passed_or_unavailable(summary: dict[str, Any]) -> bool:
    if not summary.get("available"):
        return True
    return observer_passed(summary)


def generated_count(summaries: dict[str, Any]) -> int:
    export_count = safe_int((summaries.get("export") or {}).get("generated_image_count"))
    if export_count:
        return export_count
    total = 0
    for step in (summaries.get("execution") or {}).get("generation_steps") or []:
        total += safe_int(step.get("image_count")) or 0
    return total


def metadata_coverage_improved(recipe: dict[str, Any], summaries: dict[str, Any]) -> bool:
    task = recipe.get("task")
    if task in {"diffusion_lora_generation", "traditional_augmentation", "gan_generation"}:
        return generated_count(summaries) > 0 and (summaries.get("export") or {}).get("manifest_count") is not None
    return False


def downstream_evidence_requested(state: dict[str, Any], recipe: dict[str, Any]) -> bool:
    request = str(state.get("request") or "").lower()
    return any(keyword in request for keyword in ["下游", "准确率", "识别率", "分类评估", "benchmark", "accuracy", "classifier"])


def find_step(steps: list[dict[str, Any]], step_id: str) -> dict[str, Any]:
    for step in steps:
        if step.get("step_id") == step_id or step.get("id") == step_id:
            return step
    return {}


def find_step_by_skill(steps: list[dict[str, Any]], skill: str) -> dict[str, Any]:
    for step in steps:
        if step.get("skill") == skill:
            return step
    return {}


def nested_report(step: dict[str, Any], key: str) -> dict[str, Any]:
    value = step.get(key) if isinstance(step, dict) else None
    return value if isinstance(value, dict) else {}


def leakage_overlap_count(report: dict[str, Any]) -> int:
    total = 0
    for item in report.get("overlaps") or []:
        total += safe_int(item.get("hash_overlap_count")) or 0
        total += safe_int(item.get("stem_overlap_count")) or 0
    return total


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_optional_mapping(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {}
    return load_mapping(value)

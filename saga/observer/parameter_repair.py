from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from saga.core.config import save_json, save_text
from saga.core.recipe import SagaRecipe, RecipeStep


PARAMETER_REPAIR_VERSION = "saga_parameter_repair_plan_v1"


DIFFUSION_LORA_BOUNDS: dict[str, dict[str, Any]] = {
    "training_epochs": {"min": 5, "max": 100, "type": "int"},
    "max_train_samples": {"min": 32, "max": 512, "type": "int"},
    "prompt_count": {"min": 1, "max": 128, "type": "int"},
    "inference_steps": {"min": 8, "max": 40, "type": "int"},
    "inference_guidance": {"min": 2.0, "max": 6.0, "type": "float"},
    "inference_cfg_scale": {"min": 0.5, "max": 4.0, "type": "float"},
    "lora_multiplier": {"min": 0.6, "max": 1.2, "type": "float"},
}

TRADITIONAL_AUG_BOUNDS: dict[str, dict[str, Any]] = {
    "blur_probability": {"min": 0.0, "max": 0.2, "type": "float"},
}

STYLE_TRANSFER_BOUNDS: dict[str, dict[str, Any]] = {
    "content_weight": {"min": 0.1, "max": 0.8, "type": "float"},
    "lr": {"min": 0.005, "max": 0.08, "type": "float"},
    "steps": {"min": 50, "max": 300, "type": "int"},
}


def build_parameter_repair_plan(
    recipe: SagaRecipe,
    evaluation: dict[str, Any] | None,
    output_dir: str | Path | None = None,
    max_trials: int = 3,
    trial_index: int = 0,
    dry_run: bool = False,
    step_results: dict[str, dict[str, Any]] | None = None,
    execution_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a bounded, auditable parameter patch plan.

    This module does not run expensive skills. It turns observer evidence into a
    revised recipe artifact that can be reviewed or executed explicitly.
    """

    merged_evaluation = evaluation or evaluation_from_execution_report(execution_report or {})
    step_results = step_results or step_results_from_execution_report(execution_report or {})
    triggers = list(merged_evaluation.get("triggers") or [])
    metrics = collect_metrics(merged_evaluation)
    warnings = build_diagnostic_warnings(merged_evaluation, metrics)

    if dry_run:
        status = "dry_run"
    elif trial_index >= max_trials:
        status = "blocked"
    elif recipe.task not in {"diffusion_lora_generation", "traditional_augmentation", "style_transfer"}:
        status = "unsupported"
    else:
        status = "not_needed"

    patches: list[dict[str, Any]] = []
    blockers: list[str] = []
    mutable_step = find_step(
        recipe.pipeline,
        {"DiffusionLoRAGenerationSkill", "SARLoRAGenerationSkill", "TraditionalAugmentationSkill", "TraditionalAugSkill", "StyleTransferSkill"},
    )
    skill_report = find_skill_report(step_results, mutable_step.id if mutable_step else None)

    if status not in {"dry_run", "blocked", "unsupported"}:
        if mutable_step is None:
            status = "blocked"
            blockers.append("No mutable generation/augmentation step was found in the recipe.")
        else:
            patches = propose_patches_for_step(step=mutable_step, skill_report=skill_report, triggers=triggers, metrics=metrics)
            status = "planned" if patches else ("manual_review" if triggers or warnings else "not_needed")

    if status == "dry_run" and mutable_step is not None:
        patches = propose_patches_for_step(step=mutable_step, skill_report=skill_report, triggers=triggers, metrics=metrics)

    revised_recipe = None
    revised_recipe_path = None
    if patches and status in {"planned", "dry_run"}:
        revised_recipe = apply_parameter_patches(recipe, patches, trial_index=trial_index)

    report = {
        "schema_version": PARAMETER_REPAIR_VERSION,
        "status": status,
        "bounded": True,
        "dry_run": dry_run,
        "source_recipe_id": recipe.recipe_id,
        "task": recipe.task,
        "max_trials": int(max_trials),
        "trial_index": int(trial_index),
        "remaining_trials": max(0, int(max_trials) - int(trial_index)),
        "trigger_count": len(triggers),
        "triggers": triggers,
        "metrics": metrics,
        "diagnostic_warnings": warnings,
        "blockers": blockers,
        "patches": patches,
        "patch_count": len(patches),
        "auto_rerun": False,
        "notes": [
            "Parameter repair is evidence-triggered and bounded by a whitelist.",
            "This plan writes a revised recipe but does not execute it automatically.",
            "Expensive reruns must be started explicitly through saga execute-recipe or agent-run.",
        ],
    }

    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        save_json(path / "parameter_repair_plan.json", report)
        save_text(path / "parameter_repair_plan.md", render_parameter_repair_markdown(report))
        if revised_recipe:
            revised_recipe_path = path / "revised_recipe.yaml"
            save_text(revised_recipe_path, yaml.safe_dump(revised_recipe, allow_unicode=True, sort_keys=False))
            report["artifacts"] = {
                "parameter_repair_plan_json": (path / "parameter_repair_plan.json").as_posix(),
                "parameter_repair_plan_md": (path / "parameter_repair_plan.md").as_posix(),
                "revised_recipe_yaml": revised_recipe_path.as_posix(),
            }
            save_json(path / "parameter_repair_plan.json", report)
            save_text(path / "parameter_repair_plan.md", render_parameter_repair_markdown(report))
        else:
            report["artifacts"] = {
                "parameter_repair_plan_json": (path / "parameter_repair_plan.json").as_posix(),
                "parameter_repair_plan_md": (path / "parameter_repair_plan.md").as_posix(),
            }
            save_json(path / "parameter_repair_plan.json", report)
            save_text(path / "parameter_repair_plan.md", render_parameter_repair_markdown(report))

    return report


def propose_patches_for_step(
    step: RecipeStep,
    skill_report: dict[str, Any] | None,
    triggers: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if step.skill in {"DiffusionLoRAGenerationSkill", "SARLoRAGenerationSkill"}:
        return propose_diffusion_lora_patches(step=step, skill_report=skill_report, triggers=triggers, metrics=metrics)
    if step.skill in {"TraditionalAugmentationSkill", "TraditionalAugSkill"}:
        return propose_traditional_aug_patches(step=step, skill_report=skill_report, triggers=triggers, metrics=metrics)
    if step.skill == "StyleTransferSkill":
        return propose_style_transfer_patches(step=step, skill_report=skill_report, triggers=triggers, metrics=metrics)
    return []


def propose_diffusion_lora_patches(
    step: RecipeStep,
    skill_report: dict[str, Any] | None,
    triggers: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    trigger_names = {str(item.get("name")) for item in triggers}
    sar_fid_lite = safe_float(metrics.get("sar_fid_lite"))
    histogram_jsd = safe_float(metrics.get("histogram_jsd"))
    diversity = safe_float(metrics.get("generated_diversity_mean_l2"))

    if "high_sar_fid_lite" in trigger_names or (sar_fid_lite is not None and sar_fid_lite > 120.0):
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="training_epochs",
            new_value=raise_training_epochs(current_value(step, skill_report, "training_epochs")),
            reason="Generated SAR-lite feature distribution is far from the reference; increase LoRA adaptation budget.",
            trigger="high_sar_fid_lite",
        )
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="max_train_samples",
            new_value=double_or_default(current_value(step, skill_report, "max_train_samples"), default=128),
            reason="Use more balanced metadata-caption training samples before retrying.",
            trigger="high_sar_fid_lite",
        )
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="inference_steps",
            new_value=increment(current_value(step, skill_report, "inference_steps"), default=20, amount=4),
            reason="Allow a slightly longer denoising trajectory for the repaired generation trial.",
            trigger="high_sar_fid_lite",
        )

    if "high_histogram_jsd" in trigger_names or (histogram_jsd is not None and histogram_jsd > 0.25):
        epochs = safe_int(current_value(step, skill_report, "training_epochs"))
        if epochs is None or epochs < 25:
            add_patch(
                patches,
                seen,
                step,
                skill_report,
                parameter="training_epochs",
                new_value=max(25, raise_training_epochs(epochs)),
                reason="Histogram mismatch can indicate under-trained domain statistics; raise epochs to a conservative floor.",
                trigger="high_histogram_jsd",
            )
        else:
            add_patch(
                patches,
                seen,
                step,
                skill_report,
                parameter="lora_multiplier",
                new_value=decrement_float(current_value(step, skill_report, "lora_multiplier"), default=1.0, amount=0.1),
                reason="Histogram mismatch after sufficient training may indicate overly strong LoRA conditioning.",
                trigger="high_histogram_jsd",
            )
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="inference_guidance",
            new_value=decrement_float(current_value(step, skill_report, "inference_guidance"), default=3.5, amount=0.25),
            reason="Reduce prompt guidance slightly to avoid intensity-distribution drift.",
            trigger="high_histogram_jsd",
        )

    if "low_generated_diversity" in trigger_names or (diversity is not None and diversity < 0.02):
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="prompt_count",
            new_value=double_or_default(current_value(step, skill_report, "prompt_count"), default=32),
            reason="Increase the number of caption prompts sampled for generation.",
            trigger="low_generated_diversity",
        )
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="lora_multiplier",
            new_value=decrement_float(current_value(step, skill_report, "lora_multiplier"), default=1.0, amount=0.1),
            reason="Lower LoRA strength to reduce mode collapse risk.",
            trigger="low_generated_diversity",
        )

    quality_names = {
        "flat_outputs",
        "low_dynamic_range_outputs",
        "black_heavy_outputs",
        "white_heavy_outputs",
    }
    if trigger_names & quality_names:
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="inference_steps",
            new_value=increment(current_value(step, skill_report, "inference_steps"), default=20, amount=4),
            reason="Low-level quality triggers suggest a safer, longer inference pass before changing training.",
            trigger="image_quality_trigger",
        )
        add_patch(
            patches,
            seen,
            step,
            skill_report,
            parameter="inference_guidance",
            new_value=increment_float(current_value(step, skill_report, "inference_guidance"), default=3.5, amount=0.25),
            reason="Increase conditioning slightly for low-information generated images.",
            trigger="image_quality_trigger",
        )

    return patches


def propose_traditional_aug_patches(
    step: RecipeStep,
    skill_report: dict[str, Any] | None,
    triggers: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    trigger_names = {str(item.get("name")) for item in triggers}
    if trigger_names & {"low_target_compactness", "target_fragmentation", "stripe_artifacts", "background_gradient_artifacts"}:
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="operations",
            path=["params", "skill_overrides", "operations"],
            old_value=current_traditional_value(step, skill_report, "operations"),
            new_value=["hflip", "vflip", "rotate", "intensity_scale"],
            bounds={"type": "list"},
            reason="SAR structure/artifact triggers call for disabling crop/noise/blur in the repaired traditional augmentation recipe.",
            trigger="sar_artifact_trigger",
        )
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="rotate_degrees",
            path=["params", "skill_overrides", "rotate_degrees"],
            old_value=current_traditional_value(step, skill_report, "rotate_degrees"),
            new_value=[-10.0, -5.0, 5.0, 10.0, 180.0],
            bounds={"type": "list"},
            reason="Use gentler rotations to reduce target-structure damage.",
            trigger="sar_artifact_trigger",
        )
    if trigger_names & {"flat_outputs", "low_dynamic_range_outputs", "black_heavy_outputs", "white_heavy_outputs"}:
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="intensity_scale_range",
            path=["params", "skill_overrides", "intensity_scale_range"],
            old_value=current_traditional_value(step, skill_report, "intensity_scale_range"),
            new_value=[0.9, 1.1],
            bounds={"type": "range"},
            reason="Reduce intensity perturbation range after dynamic-range observer triggers.",
            trigger="image_quality_trigger",
        )
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="blur_probability",
            path=["params", "skill_overrides", "blur_probability"],
            old_value=current_traditional_value(step, skill_report, "blur_probability"),
            new_value=0.0,
            bounds=TRADITIONAL_AUG_BOUNDS["blur_probability"],
            reason="Disable blur when generated samples have low information or weak target compactness.",
            trigger="image_quality_trigger",
        )
    return patches


def propose_style_transfer_patches(
    step: RecipeStep,
    skill_report: dict[str, Any] | None,
    triggers: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    trigger_names = {str(item.get("name")) for item in triggers}
    if trigger_names & {"low_target_compactness", "target_fragmentation", "background_gradient_artifacts", "black_heavy_outputs"}:
        content_weight = current_style_value(step, skill_report, "content_weight")
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="content_weight",
            path=["params", "skill_overrides", "content_weight"],
            old_value=content_weight,
            new_value=increment_float(content_weight, default=0.25, amount=0.1),
            bounds=STYLE_TRANSFER_BOUNDS["content_weight"],
            reason="Increase content preservation after over-transfer or target-structure triggers.",
            trigger="style_overtransfer_trigger",
        )
        lr = current_style_value(step, skill_report, "lr")
        add_patch_with_bounds(
            patches,
            seen,
            step,
            parameter="lr",
            path=["params", "skill_overrides", "lr"],
            old_value=lr,
            new_value=decrement_float(lr, default=0.05, amount=0.01),
            bounds=STYLE_TRANSFER_BOUNDS["lr"],
            reason="Reduce style-transfer optimization aggressiveness.",
            trigger="style_overtransfer_trigger",
        )
    return patches


def add_patch(
    patches: list[dict[str, Any]],
    seen: set[tuple[str, tuple[str, ...]]],
    step: RecipeStep,
    skill_report: dict[str, Any] | None,
    parameter: str,
    new_value: Any,
    reason: str,
    trigger: str,
) -> None:
    spec = patch_spec(parameter)
    if spec is None:
        return
    bounded = clamp_value(new_value, DIFFUSION_LORA_BOUNDS[parameter])
    old_value = current_value(step, skill_report, parameter)
    if old_value is not None and values_equal(old_value, bounded):
        return
    key = (step.id, tuple(spec["path"]))
    if key in seen:
        return
    seen.add(key)
    patches.append(
        {
            "step_id": step.id,
            "skill": step.skill,
            "parameter": parameter,
            "path": spec["path"],
            "old_value": old_value,
            "new_value": bounded,
            "bounds": DIFFUSION_LORA_BOUNDS[parameter],
            "trigger": trigger,
            "reason": reason,
            "auto_apply": True,
        }
    )


def add_patch_with_bounds(
    patches: list[dict[str, Any]],
    seen: set[tuple[str, tuple[str, ...]]],
    step: RecipeStep,
    parameter: str,
    path: list[str],
    old_value: Any,
    new_value: Any,
    bounds: dict[str, Any],
    reason: str,
    trigger: str,
) -> None:
    bounded = clamp_value(new_value, bounds) if bounds.get("type") in {"int", "float"} else new_value
    if old_value is not None and values_equal(old_value, bounded):
        return
    key = (step.id, tuple(path))
    if key in seen:
        return
    seen.add(key)
    patches.append(
        {
            "step_id": step.id,
            "skill": step.skill,
            "parameter": parameter,
            "path": path,
            "old_value": old_value,
            "new_value": bounded,
            "bounds": bounds,
            "trigger": trigger,
            "reason": reason,
            "auto_apply": True,
        }
    )


def patch_spec(parameter: str) -> dict[str, Any] | None:
    paths = {
        "training_epochs": ["params", "training_epochs"],
        "max_train_samples": ["params", "skill_overrides", "max_train_samples"],
        "prompt_count": ["params", "skill_overrides", "prompt_count"],
        "inference_steps": ["params", "skill_overrides", "adapter", "inference_steps"],
        "inference_guidance": ["params", "skill_overrides", "adapter", "inference_guidance"],
        "inference_cfg_scale": ["params", "skill_overrides", "adapter", "inference_cfg_scale"],
        "lora_multiplier": ["params", "skill_overrides", "adapter", "lora_multiplier"],
    }
    path = paths.get(parameter)
    return {"path": path} if path else None


def current_value(step: RecipeStep, skill_report: dict[str, Any] | None, parameter: str) -> Any:
    skill_report = skill_report or {}
    adapter_config = skill_report.get("adapter_config") or {}
    preparation = skill_report.get("preparation") or {}
    overrides = step.params.get("skill_overrides") or {}
    override_adapter = overrides.get("adapter") or {}
    if parameter == "training_epochs":
        return step.params.get("training_epochs") or skill_report.get("training_epochs") or adapter_config.get("max_train_epochs")
    if parameter == "max_train_samples":
        return overrides.get("max_train_samples") or preparation.get("max_train_samples")
    if parameter == "prompt_count":
        return overrides.get("prompt_count")
    if parameter in {"inference_steps", "inference_guidance", "inference_cfg_scale", "lora_multiplier"}:
        return override_adapter.get(parameter) or adapter_config.get(parameter)
    return None


def current_traditional_value(step: RecipeStep, skill_report: dict[str, Any] | None, parameter: str) -> Any:
    overrides = step.params.get("skill_overrides") or {}
    if parameter in overrides:
        return overrides.get(parameter)
    config = (skill_report or {}).get("config") or {}
    return config.get(parameter)


def current_style_value(step: RecipeStep, skill_report: dict[str, Any] | None, parameter: str) -> Any:
    overrides = step.params.get("skill_overrides") or {}
    if parameter in overrides:
        return overrides.get(parameter)
    adapter_config = (skill_report or {}).get("adapter_config") or {}
    return adapter_config.get(parameter)


def apply_parameter_patches(recipe: SagaRecipe, patches: list[dict[str, Any]], trial_index: int = 0) -> dict[str, Any]:
    revised = copy.deepcopy(recipe.to_dict())
    revised["recipe_id"] = f"{recipe.recipe_id}_repair{int(trial_index) + 1}"
    revised.setdefault("metadata", {})
    revised["metadata"]["parent_recipe_id"] = recipe.recipe_id
    revised["metadata"]["repair_trial_index"] = int(trial_index) + 1
    revised["metadata"].setdefault("repair_history", [])
    revised["metadata"]["repair_history"].append(
        {
            "source": PARAMETER_REPAIR_VERSION,
            "patch_count": len(patches),
            "patched_parameters": [patch.get("parameter") for patch in patches],
        }
    )
    steps = {str(step.get("id")): step for step in revised.get("pipeline", [])}
    for patch in patches:
        step = steps.get(str(patch.get("step_id")))
        if not step:
            continue
        set_nested(step, list(patch.get("path") or []), patch.get("new_value"))
    return revised


def set_nested(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if path:
        current[path[-1]] = value


def find_step(steps: list[RecipeStep], skills: set[str]) -> RecipeStep | None:
    for step in steps:
        if step.skill in skills:
            return step
    return None


def find_skill_report(step_results: dict[str, dict[str, Any]], step_id: str | None) -> dict[str, Any] | None:
    if step_id and isinstance(step_results.get(step_id), dict):
        report = step_results[step_id].get("skill_report")
        if isinstance(report, dict):
            return report
    for result in step_results.values():
        report = result.get("skill_report") if isinstance(result, dict) else None
        if isinstance(report, dict) and report.get("skill") in {
            "DiffusionLoRAGenerationSkill",
            "SARLoRAGenerationSkill",
            "TraditionalAugmentationSkill",
            "TraditionalAugSkill",
            "StyleTransferSkill",
        }:
            return report
    return None


def collect_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in ("metrics",):
        if isinstance(evaluation.get(key), dict):
            metrics.update(evaluation[key])
    secondary = evaluation.get("secondary_evaluation")
    if isinstance(secondary, dict) and isinstance(secondary.get("metrics"), dict):
        metrics.update(secondary["metrics"])
    return metrics


def build_diagnostic_warnings(evaluation: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    fid_pytorch = safe_float(metrics.get("fid_pytorch"))
    generated_count = safe_int(evaluation.get("generated_image_count") or evaluation.get("image_count"))
    secondary = evaluation.get("secondary_evaluation") or {}
    if generated_count is None and isinstance(secondary, dict):
        generated_count = safe_int(secondary.get("generated_image_count"))
    if fid_pytorch is not None and fid_pytorch > 150.0:
        warnings.append(
            {
                "name": "high_standard_fid_experimental",
                "severity": "medium",
                "metric": "fid_pytorch",
                "value": fid_pytorch,
                "message": "Standard FID is high, but SAGA treats it as diagnostic for small SAR batches rather than a direct auto-repair trigger.",
            }
        )
    if generated_count is not None and generated_count < 20:
        warnings.append(
            {
                "name": "small_generated_evaluation_batch",
                "severity": "low",
                "generated_count": generated_count,
                "message": "Distribution metrics from fewer than 20 generated samples should be used cautiously.",
            }
        )
    return warnings


def evaluation_from_execution_report(report: dict[str, Any]) -> dict[str, Any]:
    steps = report.get("steps") if isinstance(report, dict) else []
    if not isinstance(steps, list):
        return {}
    primary: dict[str, Any] = {}
    secondary: dict[str, Any] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        if not primary and isinstance(step.get("quality_report"), dict):
            primary = step["quality_report"]
        if not secondary and isinstance(step.get("distribution_report"), dict):
            secondary = step["distribution_report"]
    if primary and secondary:
        merged = dict(primary)
        merged["triggers"] = list(primary.get("triggers") or []) + list(secondary.get("triggers") or [])
        merged["secondary_evaluation"] = secondary
        return merged
    return primary or secondary or {}


def step_results_from_execution_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    steps = report.get("steps") if isinstance(report, dict) else []
    if not isinstance(steps, list):
        return {}
    return {str(step.get("step_id")): step for step in steps if isinstance(step, dict) and step.get("step_id")}


def render_parameter_repair_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Parameter Repair Plan",
        "",
        f"- Status: {report.get('status')}",
        f"- Source recipe: `{report.get('source_recipe_id')}`",
        f"- Task: `{report.get('task')}`",
        f"- Trial: {report.get('trial_index')} / {report.get('max_trials')}",
        f"- Patch count: {report.get('patch_count')}",
        "",
        "## Patches",
        "",
    ]
    patches = report.get("patches") or []
    if not patches:
        lines.append("- None")
    else:
        for patch in patches:
            lines.append(
                f"- `{patch.get('step_id')}.{patch.get('parameter')}`: "
                f"`{patch.get('old_value')}` -> `{patch.get('new_value')}`; {patch.get('reason')}"
            )
    warnings = report.get("diagnostic_warnings") or []
    if warnings:
        lines.extend(["", "## Diagnostic Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning.get('name')}`: {warning.get('message')}")
    lines.append("")
    return "\n".join(lines)


def clamp_value(value: Any, bounds: dict[str, Any]) -> Any:
    numeric = safe_float(value)
    if numeric is None:
        numeric = float(bounds["min"])
    numeric = min(max(numeric, float(bounds["min"])), float(bounds["max"]))
    if bounds.get("type") == "int":
        return int(round(numeric))
    return round(float(numeric), 4)


def raise_training_epochs(value: Any) -> int:
    current = safe_int(value) or 10
    return int(math.ceil(current * 1.5))


def double_or_default(value: Any, default: int) -> int:
    current = safe_int(value)
    return int(default if current is None else current * 2)


def increment(value: Any, default: int, amount: int) -> int:
    current = safe_int(value)
    return int((default if current is None else current) + amount)


def increment_float(value: Any, default: float, amount: float) -> float:
    current = safe_float(value)
    return float((default if current is None else current) + amount)


def decrement_float(value: Any, default: float, amount: float) -> float:
    current = safe_float(value)
    return float((default if current is None else current) - amount)


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
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    except (TypeError, ValueError):
        return None
    return None


def values_equal(left: Any, right: Any) -> bool:
    left_num = safe_float(left)
    right_num = safe_float(right)
    if left_num is not None and right_num is not None:
        return abs(left_num - right_num) < 1e-9
    return left == right

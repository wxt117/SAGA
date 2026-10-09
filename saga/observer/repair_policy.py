from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.core.recipe import SagaRecipe
from saga.observer.parameter_repair import build_parameter_repair_plan


REPAIR_POLICY_VERSION = "saga_repair_policy_v1"

DEFAULT_ALLOWED_ACTIONS = [
    "inspect_skill_log",
    "rerun_failed_subset",
    "adjust_normalization",
    "switch_preprocess_mode",
    "reduce_style_strength",
    "reduce_artifact_strength",
    "adjust_traditional_aug_policy",
    "review_sar_structure",
    "reject_bad_samples",
    "keep_original",
]


def build_repair_policy_report(
    evaluation: dict[str, Any],
    output_dir: str | Path | None = None,
    max_trials: int = 3,
    auto_rerun: bool = False,
    dry_run: bool = False,
    recipe: SagaRecipe | None = None,
    step_results: dict[str, dict[str, Any]] | None = None,
    trial_index: int = 0,
) -> dict[str, Any]:
    triggers = list(evaluation.get("triggers") or [])
    if dry_run:
        status = "dry_run"
        reason = "Repair policy is recorded but not applied during dry-run."
    elif triggers:
        status = "triggered"
        reason = "One or more observer triggers require bounded repair planning."
    else:
        status = "not_triggered"
        reason = "Observer did not raise quality triggers."

    parameter_repair = None
    parameter_repair_dir = Path(output_dir).expanduser().resolve() / "parameter_repair" if output_dir else None
    if recipe is not None:
        parameter_repair = build_parameter_repair_plan(
            recipe=recipe,
            evaluation=evaluation,
            output_dir=parameter_repair_dir,
            max_trials=max_trials,
            trial_index=trial_index,
            dry_run=dry_run,
            step_results=step_results,
        )

    report = {
        "schema_version": REPAIR_POLICY_VERSION,
        "status": status,
        "reason": reason,
        "max_trials": max_trials,
        "auto_rerun": bool(auto_rerun),
        "bounded": True,
        "triggers": triggers,
        "allowed_actions": DEFAULT_ALLOWED_ACTIONS,
        "suggested_actions": suggest_actions(triggers),
        "parameter_repair": parameter_repair,
        "fallback": [
            "reject_samples_with_failed_quality_checks",
            "keep_original_samples_for_failed_outputs",
            "stop_and_request_user_confirmation_if_repair_exceeds_max_trials",
        ],
        "notes": [
            "This policy does not rerun expensive skills automatically.",
            "Automatic repair may only mutate whitelisted parameters and must record every trial.",
        ],
    }
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        save_json(path / "repair_policy.json", report)
        save_text(path / "repair_policy.md", render_repair_markdown(report))
    return report


def suggest_actions(triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trigger in triggers:
        name = str(trigger.get("name"))
        for action in actions_for_trigger(name):
            if action["action"] in seen:
                continue
            seen.add(action["action"])
            actions.append(action)
    return actions


def actions_for_trigger(name: str) -> list[dict[str, Any]]:
    mapping: dict[str, list[dict[str, Any]]] = {
        "output_dir_missing": [
            {
                "action": "inspect_skill_log",
                "parameters": ["stdout", "stderr", "command", "returncode"],
                "reason": "The skill may not have created its output directory.",
            },
            {
                "action": "rerun_failed_subset",
                "parameters": ["same_recipe_step", "same_seed"],
                "reason": "Retry only after confirming the command and paths are valid.",
            },
        ],
        "output_empty": [
            {
                "action": "inspect_skill_log",
                "parameters": ["stdout", "stderr", "command", "returncode"],
                "reason": "Empty outputs usually indicate skill failure or wrong input paths.",
            },
            {
                "action": "rerun_failed_subset",
                "parameters": ["same_recipe_step", "same_seed"],
                "reason": "The selected subset can be retried without changing dataset semantics.",
            },
        ],
        "output_count_mismatch": [
            {
                "action": "inspect_skill_log",
                "parameters": ["adapter.limit", "overwrite", "stdout", "stderr"],
                "reason": "Count mismatch can be caused by adapter limits, skipped files, or failed samples.",
            },
            {
                "action": "rerun_failed_subset",
                "parameters": ["missing_outputs_only"],
                "reason": "Only missing outputs should be rerun.",
            },
        ],
        "unreadable_images": [
            {
                "action": "reject_bad_samples",
                "parameters": ["unreadable_outputs"],
                "reason": "Unreadable files should not enter the augmented training set.",
            },
            {
                "action": "rerun_failed_subset",
                "parameters": ["unreadable_outputs_only"],
                "reason": "Regenerate only samples that failed image decoding.",
            },
        ],
        "black_heavy_outputs": [
            {
                "action": "switch_preprocess_mode",
                "parameters": ["percentile", "log_percentile"],
                "reason": "SAR outputs with excessive black pixels often need stronger dynamic-range normalization.",
            },
            {
                "action": "reduce_style_strength",
                "parameters": ["increase_content_weight", "reduce_steps_or_lr"],
                "reason": "Style transfer may be overpowering target structure.",
            },
        ],
        "white_heavy_outputs": [
            {
                "action": "adjust_normalization",
                "parameters": ["reduce_clipping", "percentile_high"],
                "reason": "White-heavy outputs suggest saturation or clipping.",
            }
        ],
        "flat_outputs": [
            {
                "action": "adjust_normalization",
                "parameters": ["percentile_low", "percentile_high", "bit_depth"],
                "reason": "Flat images usually indicate normalization or decoding issues.",
            }
        ],
        "low_dynamic_range_outputs": [
            {
                "action": "adjust_normalization",
                "parameters": ["percentile", "log_percentile"],
                "reason": "Low dynamic range weakens downstream target features.",
            },
            {
                "action": "keep_original",
                "parameters": ["failed_outputs"],
                "reason": "Fallback to original content is safer than injecting low-information generated samples.",
            },
        ],
        "high_histogram_jsd": [
            {
                "action": "adjust_prompt_or_caption_policy",
                "parameters": ["include_intensity_terms", "include_sar_domain_tokens", "review_metadata_captions"],
                "reason": "Generated grayscale distribution differs from the reference; captions or prompts may not constrain SAR intensity statistics enough.",
            },
            {
                "action": "increase_lora_training_budget",
                "parameters": ["more_train_samples", "more_epochs", "higher_lora_rank_if_needed"],
                "reason": "A one-epoch quick LoRA may not learn the reference SAR distribution.",
            },
        ],
        "high_sar_fid_lite": [
            {
                "action": "increase_lora_training_budget",
                "parameters": ["more_train_samples", "more_epochs"],
                "reason": "Generated features are far from the reference distribution under SAR-lite statistics.",
            },
            {
                "action": "adjust_normalization",
                "parameters": ["input_percentile_normalization", "grayscale_export_policy"],
                "reason": "Feature mismatch may come from representation or dynamic-range mismatch.",
            },
        ],
        "low_generated_diversity": [
            {
                "action": "increase_prompt_diversity",
                "parameters": ["balanced_metadata_prompts", "seed_schedule", "prompt_templates"],
                "reason": "Generated outputs are too similar under lightweight features.",
            }
        ],
        "stripe_artifacts": [
            {
                "action": "adjust_normalization",
                "parameters": ["destripe_or_percentile_normalization", "review_frequency_artifacts"],
                "reason": "Strong stripe energy can indicate preprocessing artifacts or over-aggressive generation.",
            },
            {
                "action": "reject_bad_samples",
                "parameters": ["samples_triggering_stripe_artifacts"],
                "reason": "Striped SAR samples can bias downstream models toward non-target artifacts.",
            },
        ],
        "background_gradient_artifacts": [
            {
                "action": "adjust_normalization",
                "parameters": ["background_flattening", "percentile_window"],
                "reason": "Smooth generated gradients are often non-SAR background artifacts or normalization drift.",
            },
            {
                "action": "reduce_artifact_strength",
                "parameters": ["lower_style_strength", "lower_guidance", "review_blending"],
                "reason": "Reducing transfer/generation strength may preserve SAR background statistics.",
            },
        ],
        "low_target_compactness": [
            {
                "action": "review_sar_structure",
                "parameters": ["target_mask_proxy", "worst_samples"],
                "reason": "Diffuse bright responses can mean the augmentation damaged target scattering structure.",
            },
            {
                "action": "adjust_traditional_aug_policy",
                "parameters": ["disable_large_rotation", "reduce_crop_strength", "lower_noise_std"],
                "reason": "For traditional augmentation, target compactness failures usually call for gentler transforms.",
            },
        ],
        "target_fragmentation": [
            {
                "action": "adjust_traditional_aug_policy",
                "parameters": ["lower_speckle_std", "lower_gaussian_std", "disable_blur"],
                "reason": "Fragmented bright responses can be caused by excessive noise or blur-like transforms.",
            },
            {
                "action": "reject_bad_samples",
                "parameters": ["fragmented_outputs"],
                "reason": "Highly fragmented generated targets are risky for target recognition training.",
            },
        ],
        "off_center_target": [
            {
                "action": "adjust_traditional_aug_policy",
                "parameters": ["center_crop_only", "avoid_translation_like_preprocess"],
                "reason": "Classification-style SAR target chips usually assume the target remains centered.",
            },
            {
                "action": "reject_bad_samples",
                "parameters": ["off_center_outputs"],
                "reason": "Off-center target chips may change the intended training distribution.",
            },
        ],
        "sar_artifact_read_errors": [
            {
                "action": "reject_bad_samples",
                "parameters": ["unreadable_artifact_samples"],
                "reason": "Unreadable samples should not enter the augmented dataset.",
            }
        ],
    }
    return mapping.get(
        name,
        [
            {
                "action": "inspect_skill_log",
                "parameters": ["observer_trigger", name],
                "reason": "Unhandled trigger type should be inspected before automatic repair.",
            }
        ],
    )


def render_repair_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Repair Policy",
        "",
        f"- Status: {report.get('status')}",
        f"- Reason: {report.get('reason')}",
        f"- Max trials: {report.get('max_trials')}",
        f"- Auto rerun: {report.get('auto_rerun')}",
        "",
        "## Triggers",
        "",
    ]
    triggers = report.get("triggers") or []
    if not triggers:
        lines.append("- None")
    else:
        for trigger in triggers:
            lines.append(f"- `{trigger.get('name')}` ({trigger.get('severity')}): {trigger.get('message')}")
    lines.extend(["", "## Suggested Actions", ""])
    actions = report.get("suggested_actions") or []
    if not actions:
        lines.append("- None")
    else:
        for action in actions:
            lines.append(f"- `{action.get('action')}`: {action.get('reason')}")
    parameter_repair = report.get("parameter_repair") or {}
    if parameter_repair:
        lines.extend(["", "## Parameter Repair", ""])
        lines.append(f"- Status: {parameter_repair.get('status')}")
        lines.append(f"- Patch count: {parameter_repair.get('patch_count')}")
        artifacts = parameter_repair.get("artifacts") or {}
        if artifacts.get("revised_recipe_yaml"):
            lines.append(f"- Revised recipe: `{artifacts.get('revised_recipe_yaml')}`")
    lines.append("")
    return "\n".join(lines)

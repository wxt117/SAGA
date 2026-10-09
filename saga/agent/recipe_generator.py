from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from saga.core.config import load_mapping, save_json, save_text
from saga.core.recipe import SagaRecipe, RecipeStep


def generate_recipe_from_agent_state(
    state: dict[str, Any],
    output_path: str | Path,
    style_transfer_config: str | Path | None = None,
    diffusion_lora_config: str | Path | None = None,
    traditional_aug_config: str | Path | None = None,
    model_to_pov_config: str | Path | None = None,
    geodiff_sar_config: str | Path | None = None,
    gaussian_splatting_config: str | Path | None = None,
    background_generation_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
) -> SagaRecipe:
    intent_spec = (
        load_optional_mapping(state.get("effective_intent_spec_path"))
        or load_optional_mapping(state.get("intent_spec_path"))
        or state.get("intent_spec")
        or {}
    )
    validated_profile = load_optional_mapping(state.get("validated_profile_path")) or {}
    if state.get("validated_profile_path"):
        validated_profile["_validated_profile_path"] = state.get("validated_profile_path")
    bridge_report = load_optional_mapping(state.get("format_bridge_path")) or {}
    planner_proposal = load_optional_mapping(state.get("planner_proposal_path")) or {}
    planner_guardrail = load_optional_mapping(state.get("planner_guardrail_path")) or {}
    output_dir = Path(state["output_dir"]).expanduser().resolve()
    recipe = generate_recipe(
        request_text=str(state.get("request", "")),
        intent_spec=intent_spec,
        dataset_profile_path=state.get("dataset_profile_path"),
        validated_profile=validated_profile,
        bridge_report=bridge_report,
        output_dir=output_dir,
        style_transfer_config=style_transfer_config,
        diffusion_lora_config=diffusion_lora_config,
        traditional_aug_config=traditional_aug_config,
        model_to_pov_config=model_to_pov_config,
        geodiff_sar_config=geodiff_sar_config,
        gaussian_splatting_config=gaussian_splatting_config,
        background_generation_config=background_generation_config,
        classification_eval_config=classification_eval_config,
        max_repair_trials=max_repair_trials,
        planner_mode=str(state.get("planner_mode") or "rule_based"),
        planner_proposal=planner_proposal,
        planner_guardrail=planner_guardrail,
    )
    write_recipe(recipe, output_path)
    return recipe


def generate_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    style_transfer_config: str | Path | None = None,
    diffusion_lora_config: str | Path | None = None,
    traditional_aug_config: str | Path | None = None,
    model_to_pov_config: str | Path | None = None,
    geodiff_sar_config: str | Path | None = None,
    gaussian_splatting_config: str | Path | None = None,
    background_generation_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    task = intent.get("task") or "unknown"
    if task == "style_transfer":
        return generate_style_transfer_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            style_transfer_config=style_transfer_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "diffusion_lora_generation":
        return generate_diffusion_lora_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            diffusion_lora_config=diffusion_lora_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "geodiff_sar_generation":
        return generate_geodiff_sar_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            geodiff_sar_config=geodiff_sar_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "gaussian_splatting_completion":
        return generate_gaussian_splatting_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            gaussian_splatting_config=gaussian_splatting_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "gan_generation":
        return generate_gan_generation_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "traditional_augmentation":
        return generate_traditional_augmentation_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            traditional_aug_config=traditional_aug_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "pseudocolor_transform":
        return generate_pseudocolor_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            classification_eval_config=classification_eval_config,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "background_generation":
        return generate_background_generation_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            background_generation_config=background_generation_config,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "target_background_composition":
        return generate_target_background_composition_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "raysar_synthesis":
        return generate_raysar_synthesis_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            classification_eval_config=classification_eval_config,
            max_repair_trials=max_repair_trials,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    if task == "classification_evaluation":
        return generate_classification_evaluation_recipe(
            request_text=request_text,
            intent_spec=intent_spec,
            dataset_profile_path=dataset_profile_path,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            output_dir=output_dir,
            classification_eval_config=classification_eval_config,
            planner_mode=planner_mode,
            planner_proposal=planner_proposal,
            planner_guardrail=planner_guardrail,
        )
    return generate_profile_only_recipe(
        request_text=request_text,
        intent_spec=intent_spec,
        dataset_profile_path=dataset_profile_path,
        validated_profile=validated_profile,
        bridge_report=bridge_report,
        output_dir=output_dir,
        planner_mode=planner_mode,
        planner_proposal=planner_proposal,
        planner_guardrail=planner_guardrail,
    )


def generate_style_transfer_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    style_transfer_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("style_transfer", intent)
    content_source = resolve_recipe_path(intent.get("content_source"))
    style_source = resolve_recipe_path(intent.get("style_source"))
    filters, exclude_filters = normalize_filters_for_recipe(
        filters=intent.get("filters") or {},
        exclude_filters=intent.get("exclude_filters") or {},
    )
    run_dir = Path(output_dir).expanduser().resolve()
    selected_dir = run_dir / "recipe_artifacts" / "selected_content"
    skill_output_dir = run_dir / "recipe_artifacts" / "style_transfer_output"
    export_dir = run_dir / "augmented_dataset"
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record profile and validation artifacts for this run.",
        ),
        RecipeStep(
            id="select_content",
            skill="FileSelectionSkill",
            depends_on=["inspect_inputs"],
            params={
                "content_root": content_source,
                "output_dir": selected_dir.as_posix(),
                "dataset_config": validated_profile.get("dataset_config"),
                "filters": filters,
                "exclude_filters": exclude_filters,
                "preserve_tree": False,
                "fail_if_empty": True,
            },
            description="Select content images according to validated metadata filters.",
        ),
        RecipeStep(
            id="run_style_transfer",
            skill="StyleTransferSkill",
            depends_on=["select_content"],
            params={
                "content": selected_dir.as_posix(),
                "style": style_source,
                "output_dir": skill_output_dir.as_posix(),
                "config": str(style_transfer_config) if style_transfer_config else "configs/skills/style_transfer.yaml",
            },
            description="Run the wrapped SAR style-transfer skill on selected content images.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_style_transfer"],
            params={
                "input_dir": skill_output_dir.as_posix(),
                "expected_from_step": "select_content",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.85,
                    "max_white_pixel_ratio": 0.85,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate generated image count, readability, and simple SAR image-quality statistics.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_style_transfer"],
            params={
                "input_from_step": "run_style_transfer",
                "sample_limit": 100,
                "image_size": 128,
            },
            description="Evaluate SAR-specific artifacts such as stripes, smooth background gradients, target compactness, and centeredness.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create a bounded repair plan if observer triggers are raised.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_style_transfer",
                "selection_from_step": "select_content",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package generated outputs, manifest, quality report, and provenance into an augmented dataset artifact.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=content_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="style_transfer",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "content_source": content_source,
            "style_source": style_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "exclude_filters": exclude_filters,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": validated_profile.get("planner_ready", False),
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "This recipe is deterministic and executable by saga execute-recipe.",
                "LLM intent parsing is allowed, but dataset filtering depends on validated metadata.",
            ],
        },
    )


def generate_diffusion_lora_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    diffusion_lora_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("diffusion_lora_generation", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "diffusion_lora"
    generated_dir = skill_run_dir / "generated_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count") or 20
    training_epochs = intent.get("training_epochs")
    filters, exclude_filters = normalize_filters_for_recipe(
        filters=intent.get("filters") or {},
        exclude_filters=intent.get("exclude_filters") or {},
    )
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record raw profile and optional format validation artifacts for this LoRA run.",
        ),
        RecipeStep(
            id="run_diffusion_lora",
            skill="DiffusionLoRAGenerationSkill",
            depends_on=["inspect_inputs"],
            params={
                "dataset_root": dataset_source,
                "output_dir": skill_run_dir.as_posix(),
                "config": str(diffusion_lora_config) if diffusion_lora_config else "configs/skills/diffusion_lora.yaml",
                "model_family": intent.get("model_family") or "flux",
                "target_count": target_count,
                "training_epochs": training_epochs,
                "dataset_config": validated_profile.get("dataset_config"),
                "filters": filters,
                "exclude_filters": exclude_filters,
                "auto_caption_from_metadata": True,
                "train": True,
                "infer": True,
            },
            description="Inspect caption data, generate qinglong LoRA train config, then train/infer or dry-run commands.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_diffusion_lora"],
            params={
                "input_dir": generated_dir.as_posix(),
                "expected_from_step": "run_diffusion_lora",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.9,
                    "max_white_pixel_ratio": 0.9,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate generated image count, readability, and simple SAR image-quality statistics.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_diffusion_lora"],
            params={
                "generated_from_step": "run_diffusion_lora",
                "reference_from_step": "run_diffusion_lora",
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count,
                "image_size": 64,
                "standard_fid": True,
                "standard_fid_dims": 2048,
                "standard_fid_batch_size": 16,
                "thresholds": {
                    "max_sar_fid_lite": 120.0,
                    "max_histogram_jsd": 0.25,
                    "min_generated_diversity": 0.02,
                },
            },
            description="Compare generated samples with the reference training distribution using fast FID-style metrics.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_diffusion_lora"],
            params={
                "input_from_step": "run_diffusion_lora",
                "sample_limit": 100,
                "image_size": 128,
            },
            description="Evaluate SAR-specific artifacts such as stripes, smooth background gradients, target compactness, and centeredness.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create a bounded repair plan if LoRA generation observer triggers are raised.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_diffusion_lora",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
                "write_caption_sidecars": True,
            },
            description="Package LoRA generated outputs, manifest, quality report, and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="diffusion_lora_generation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "model_family": intent.get("model_family") or "flux",
            "target_count": target_count,
            "training_epochs": training_epochs,
            "exclude_filters": exclude_filters,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "This skill wraps text-caption LoRA training and inference only.",
                "ControlNet and GeoDiff-SAR physical-prior generation are intentionally excluded from this recipe.",
                "agent-run and execute-recipe default to dry-run unless expensive execution is explicitly enabled.",
            ],
        },
    )


def generate_geodiff_sar_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    geodiff_sar_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("geodiff_sar_generation", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    model_file = resolve_recipe_path(intent.get("model_file"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "geodiff_sar"
    generated_dir = skill_run_dir / "generated_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count") or 20
    training_epochs = intent.get("training_epochs")
    raysar_geometry = dict(intent.get("raysar_geometry") or {})
    filters = intent.get("filters") or {}
    if filters.get("azimuth_deg") is not None and raysar_geometry.get("azimuth_values") is None:
        raysar_geometry["azimuth_values"] = [filters["azimuth_deg"]]
    if filters.get("depression_angle_deg") is not None and raysar_geometry.get("depressions") is None:
        raysar_geometry["depressions"] = [filters["depression_angle_deg"]]
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
                "dataset_source": dataset_source,
                "model_file": model_file,
            },
            description="Record dataset/profile artifacts and GeoDiff-SAR component inputs for provenance.",
        ),
        RecipeStep(
            id="run_geodiff_sar",
            skill="GeoDiffSARSkill",
            depends_on=["inspect_inputs"],
            params={
                "dataset_root": dataset_source,
                "model_file": model_file,
                "output_dir": skill_run_dir.as_posix(),
                "config": str(geodiff_sar_config) if geodiff_sar_config else "configs/skills/geodiff_sar.yaml",
                "target_azimuths": raysar_geometry.get("azimuth_values"),
                "azimuth_sweep": raysar_geometry.get("azimuth_sweep"),
                "depressions": raysar_geometry.get("depressions"),
                "target_count": target_count,
                "training_epochs": training_epochs,
                "controlnet_weights": intent.get("controlnet_weights"),
                "controlnet_init_weights": intent.get("controlnet_init_weights"),
                "prompt": intent.get("prompt"),
                "train": True,
                "infer": True,
                "extract_real_gefm": True,
                "render_gefm": True,
            },
            description="Run GeoDiff-SAR composite workflow: real GEFM extraction, ControlNet training, 3D GEFM rendering, and inference.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_geodiff_sar"],
            params={
                "input_dir": generated_dir.as_posix(),
                "expected_from_step": "run_geodiff_sar",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.95,
                    "max_white_pixel_ratio": 0.95,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate generated GeoDiff-SAR image count, readability, and basic image statistics.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_geodiff_sar"],
            params={
                "generated_from_step": "run_geodiff_sar",
                "reference_dir": dataset_source,
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count,
                "image_size": 64,
                "standard_fid": True,
                "standard_fid_dims": 2048,
                "standard_fid_batch_size": 16,
                "thresholds": {
                    "max_sar_fid_lite": 140.0,
                    "max_histogram_jsd": 0.28,
                    "min_generated_diversity": 0.02,
                },
            },
            description="Compare GeoDiff-SAR outputs with the real training distribution using lightweight FID-style proxies.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_geodiff_sar"],
            params={
                "input_from_step": "run_geodiff_sar",
                "sample_limit": 100,
                "image_size": 128,
                "thresholds": {
                    "max_trigger_fraction": 0.35,
                    "max_center_offset": 0.45,
                },
            },
            description="Check SAR-specific artifacts and target structure consistency after physical-prior generation.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create bounded repair suggestions for ControlNet strength, condition quality, or prompt/normalization issues.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_geodiff_sar",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package GeoDiff-SAR generated outputs with provenance and physical-prior warnings.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="geodiff_sar_generation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "model_file": model_file,
            "target_count": target_count,
            "training_epochs": training_epochs,
            "raysar_geometry": raysar_geometry,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "GeoDiff-SAR is selected for high-quality sparse-azimuth completion or physical-prior target generation.",
                "The skill composes real-image GEFM extraction, 3D-model GEFM rendering, ControlNet training, and ControlNet inference.",
                "Downstream task benefit must still be validated by SAGA observers or an explicit downstream evaluator.",
            ],
        },
    )


def generate_gaussian_splatting_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    gaussian_splatting_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("gaussian_splatting_completion", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "gaussian_splatting"
    rendered_dir = skill_run_dir / "rendered_views"
    export_dir = run_dir / "augmented_dataset"
    raysar_geometry = dict(intent.get("raysar_geometry") or {})
    target_azimuths = intent.get("target_azimuths") or raysar_geometry.get("azimuth_values")
    target_count = intent.get("target_count") or (len(target_azimuths) if isinstance(target_azimuths, list) else 20)
    render_resolution = intent.get("render_resolution") or 128
    filters = intent.get("filters") or {}
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
                "dataset_source": dataset_source,
                "project_root": resolve_recipe_path(intent.get("project_root") or "myproject/SAR GS V1"),
            },
            description="Record sparse-view input/profile artifacts and Gaussian splatting project binding for provenance.",
        ),
        RecipeStep(
            id="run_gaussian_splatting_completion",
            skill="GaussianSplattingCompletionSkill",
            depends_on=["inspect_inputs"],
            params={
                "dataset_root": dataset_source,
                "output_dir": skill_run_dir.as_posix(),
                "config": str(gaussian_splatting_config) if gaussian_splatting_config else "configs/skills/gaussian_splatting.yaml",
                "project_root": resolve_recipe_path(intent.get("project_root") or "myproject/SAR GS V1"),
                "target_azimuths": target_azimuths,
                "azimuth_sweep": intent.get("azimuth_sweep") or raysar_geometry.get("azimuth_sweep"),
                "target_count": target_count,
                "render_resolution": render_resolution,
                "iterations": intent.get("iterations"),
                "train": True,
                "render": True,
            },
            description="Run the lightweight SAR Gaussian splatting completion workflow or materialize its dry-run commands.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_gaussian_splatting_completion"],
            params={
                "input_dir": rendered_dir.as_posix(),
                "expected_from_step": "run_gaussian_splatting_completion",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.95,
                    "max_white_pixel_ratio": 0.95,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate rendered Gaussian splatting views for count, readability, and basic image statistics.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_gaussian_splatting_completion"],
            params={
                "generated_from_step": "run_gaussian_splatting_completion",
                "reference_dir": dataset_source,
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count,
                "image_size": 64,
                "standard_fid": True,
                "standard_fid_dims": 2048,
                "standard_fid_batch_size": 16,
                "thresholds": {
                    "max_sar_fid_lite": 140.0,
                    "max_histogram_jsd": 0.28,
                    "min_generated_diversity": 0.02,
                },
            },
            description="Compare Gaussian splatting rendered views with the reference sparse-view SAR distribution.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_gaussian_splatting_completion"],
            params={
                "input_from_step": "run_gaussian_splatting_completion",
                "sample_limit": 100,
                "image_size": 128,
                "thresholds": {
                    "max_trigger_fraction": 0.35,
                    "max_center_offset": 0.45,
                },
            },
            description="Check reconstruction artifacts and target structure consistency in Gaussian splatting outputs.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create bounded repair suggestions for render resolution, normalization, or sample rejection thresholds.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_gaussian_splatting_completion",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package Gaussian splatting completion outputs with provenance and observer reports.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="gaussian_splatting_completion",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "project_root": resolve_recipe_path(intent.get("project_root") or "myproject/SAR GS V1"),
            "target_count": target_count,
            "target_azimuths": target_azimuths,
            "render_resolution": render_resolution,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "Gaussian splatting is selected as the lightweight sparse-view completion option.",
                "The recipe records project binding to myproject/SAR GS V1 and renders target views through dry-run commands unless explicitly enabled.",
                "Reconstruction quality and downstream benefit remain subject to observer and downstream evaluator evidence.",
            ],
        },
    )


def generate_traditional_augmentation_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    traditional_aug_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("traditional_augmentation", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "traditional_augmentation"
    generated_dir = skill_run_dir / "generated_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count")
    multiplier = intent.get("multiplier") or 2
    filters, exclude_filters = normalize_filters_for_recipe(
        filters=intent.get("filters") or {},
        exclude_filters=intent.get("exclude_filters") or {},
    )
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record dataset profile and optional format validation artifacts for this traditional augmentation run.",
        ),
        RecipeStep(
            id="run_traditional_augmentation",
            skill="TraditionalAugmentationSkill",
            depends_on=["inspect_inputs"],
            params={
                "dataset_root": dataset_source,
                "output_dir": skill_run_dir.as_posix(),
                "config": str(traditional_aug_config) if traditional_aug_config else "configs/skills/traditional_augmentation.yaml",
                "dataset_config": validated_profile.get("dataset_config"),
                "filters": filters,
                "exclude_filters": exclude_filters,
                "target_count": target_count,
                "multiplier": multiplier,
            },
            description="Apply fast SAR-aware label-preserving transforms such as flips, rotations, crops, intensity and speckle perturbations.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_traditional_augmentation"],
            params={
                "input_dir": generated_dir.as_posix(),
                "expected_from_step": "run_traditional_augmentation",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.9,
                    "max_white_pixel_ratio": 0.9,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate augmented image count, readability, and simple SAR image-quality statistics.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_traditional_augmentation"],
            params={
                "input_from_step": "run_traditional_augmentation",
                "sample_limit": 100,
                "image_size": 128,
            },
            description="Evaluate SAR-specific artifacts such as stripes, smooth background gradients, target compactness, and centeredness.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_traditional_augmentation"],
            params={
                "generated_from_step": "run_traditional_augmentation",
                "reference_dir": dataset_source,
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count or 50,
                "image_size": 64,
                "standard_fid": True,
                "standard_fid_dims": 2048,
                "standard_fid_batch_size": 16,
                "thresholds": {
                    "max_sar_fid_lite": 120.0,
                    "max_histogram_jsd": 0.25,
                    "min_generated_diversity": 0.02,
                },
            },
            description="Compare traditional augmented samples with the reference data using fast FID-style distribution metrics.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create a bounded repair report if traditional augmentation output checks fail.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_traditional_augmentation",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package traditional augmented outputs, manifest, quality report, and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="traditional_augmentation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "target_count": target_count,
            "multiplier": multiplier,
            "exclude_filters": exclude_filters,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "Traditional augmentation is fast and deterministic.",
                "Transforms are intended as conservative label-preserving SAR baselines.",
                "Downstream task benefit still requires ClassificationEvaluationSkill or a task-specific evaluator.",
            ],
        },
    )


def generate_pseudocolor_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    classification_eval_config: str | Path | None = None,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("pseudocolor_transform", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    run_dir = Path(output_dir).expanduser().resolve()
    selected_dir = run_dir / "recipe_artifacts" / "pseudocolor_input"
    skill_run_dir = run_dir / "recipe_artifacts" / "pseudocolor"
    pseudo_dir = skill_run_dir / "pseudocolor_images"
    export_dir = run_dir / "augmented_dataset"
    filters, exclude_filters = normalize_filters_for_recipe(
        filters=intent.get("filters") or {},
        exclude_filters=intent.get("exclude_filters") or {},
    )
    colormap = intent.get("colormap") or "sar"
    preserve_tree = bool(intent.get("preserve_tree", True))
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record dataset profile and optional format validation artifacts for this pseudocolor run.",
        ),
        RecipeStep(
            id="select_input",
            skill="FileSelectionSkill",
            depends_on=["inspect_inputs"],
            params={
                "content_root": dataset_source,
                "output_dir": selected_dir.as_posix(),
                "dataset_config": validated_profile.get("dataset_config"),
                "filters": filters,
                "exclude_filters": exclude_filters,
                "preserve_tree": preserve_tree,
                "fail_if_empty": True,
            },
            description="Select input images according to validated metadata filters before pseudocolor conversion.",
        ),
        RecipeStep(
            id="run_pseudocolor",
            skill="PseudocolorSkill",
            depends_on=["select_input"],
            params={
                "input_dir": selected_dir.as_posix(),
                "output_dir": skill_run_dir.as_posix(),
                "colormap": colormap,
                "preserve_tree": preserve_tree,
            },
            description="Apply the lightweight SAR pseudocolor visualization/feature transform.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_pseudocolor"],
            params={
                "input_dir": pseudo_dir.as_posix(),
                "expected_from_step": "select_input",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.95,
                    "max_white_pixel_ratio": 0.95,
                    "min_luma_dynamic_range": 4,
                    "low_luma_dynamic_range": 8,
                },
            },
            description="Evaluate pseudocolor output count and basic image readability.",
        ),
        RecipeStep(
            id="duplicate_check",
            skill="DuplicateNearDuplicateSkill",
            depends_on=["run_pseudocolor"],
            params={
                "input_from_step": "run_pseudocolor",
                "output_dir": (run_dir / "recipe_artifacts" / "data_gates" / "duplicate_check").as_posix(),
                "sample_limit": 5000,
                "hash_size": 16,
                "hamming_threshold": 6,
            },
            description="Check repeated pseudocolor outputs before export or optional classifier use.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["evaluate_outputs", "duplicate_check"],
            params={
                "source_from_step": "run_pseudocolor",
                "quality_from_step": "evaluate_outputs",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package pseudocolor outputs with manifest, reports, and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="pseudocolor_transform",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "colormap": colormap,
            "preserve_tree": preserve_tree,
            "exclude_filters": exclude_filters,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "Pseudocolor is treated as visualization or an explicit RGB feature ablation, not physically new SAR data.",
                "It is not selected as the default for standard grayscale SAR training.",
                "Downstream task benefit still requires ClassificationEvaluationSkill if the pseudocolor outputs are used for training claims.",
            ],
        },
    )


def generate_background_generation_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    background_generation_config: str | Path | None = None,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("background_generation", intent)
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "background_generation"
    generated_dir = skill_run_dir / "background_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count") or 4
    scene_prompt = intent.get("scene_prompt") or request_text
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record profile artifacts before background generation.",
        ),
        RecipeStep(
            id="run_background_generation",
            skill="BackgroundGenerationSkill",
            depends_on=["inspect_inputs"],
            params={
                "scene_prompt": scene_prompt,
                "output_dir": skill_run_dir.as_posix(),
                "config": str(background_generation_config) if background_generation_config else "configs/skills/background_generation.yaml",
                "target_count": target_count,
                "num_images": target_count,
                "width": intent.get("width") or 512,
                "height": intent.get("height") or 512,
                "cuda": intent.get("cuda"),
            },
            description="Generate SAR background scenes from the packaged segmentation-controlled model without user-side training.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_background_generation"],
            params={
                "input_dir": generated_dir.as_posix(),
                "expected_from_step": "run_background_generation",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.92,
                    "max_white_pixel_ratio": 0.92,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate generated background count, readability, and dynamic range.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_background_generation"],
            params={
                "input_from_step": "run_background_generation",
                "sample_limit": 100,
                "image_size": 128,
                "thresholds": {
                    "min_target_compactness": 0.0,
                    "max_trigger_fraction": 0.8,
                },
            },
            description="Evaluate SAR-specific visual artifacts while treating outputs as backgrounds rather than centered targets.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create bounded repair suggestions if generated background observers trigger.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_background_generation",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package generated background scenes with manifests and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=resolve_recipe_path(intent.get("baseline_dataset")) if intent.get("baseline_dataset") else None,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="background_generation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "scene_prompt": scene_prompt,
            "target_count": target_count,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
        },
        filters=intent.get("filters") or {},
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "BackgroundGenerationSkill uses packaged trained weights and does not ask users to train background models.",
                "Generated backgrounds are scene assets; they are not default target-chip classification augmentation.",
                "Task benefit should be evaluated after composition or task-specific use.",
            ],
        },
    )


def generate_gan_generation_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("gan_generation", intent)
    dataset_source = resolve_recipe_path(intent.get("dataset_source") or intent_spec.get("dataset_root"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "gan_generation"
    generated_dir = skill_run_dir / "generated_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count") or 20
    training_epochs = intent.get("training_epochs")
    variant = intent.get("gan_variant") or intent.get("variant") or "gpu"
    filters, exclude_filters = normalize_filters_for_recipe(
        filters=intent.get("filters") or {},
        exclude_filters=intent.get("exclude_filters") or {},
    )
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record dataset profile and validation artifacts for this GAN run.",
        ),
        RecipeStep(
            id="run_gan_generation",
            skill="GANImageToImageSkill",
            depends_on=["inspect_inputs"],
            params={
                "dataset_root": dataset_source,
                "output_dir": skill_run_dir.as_posix(),
                "variant": variant,
                "target_count": target_count,
                "epochs": training_epochs,
                "train": True,
                "infer": True,
                "dataset_config": validated_profile.get("dataset_config"),
                "filters": filters,
                "exclude_filters": exclude_filters,
            },
            description="Run or dry-run the DCGAN wrapper for fast simple-target SAR generation.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_gan_generation"],
            params={
                "input_dir": generated_dir.as_posix(),
                "expected_from_step": "run_gan_generation",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.9,
                    "max_white_pixel_ratio": 0.9,
                    "min_luma_dynamic_range": 8,
                    "low_luma_dynamic_range": 16,
                },
            },
            description="Evaluate generated GAN image count and basic SAR readability.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_gan_generation"],
            params={
                "generated_from_step": "run_gan_generation",
                "reference_from_step": "run_gan_generation",
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count,
                "image_size": 64,
                "standard_fid": True,
                "standard_fid_dims": 2048,
                "standard_fid_batch_size": 16,
                "thresholds": {
                    "max_sar_fid_lite": 120.0,
                    "max_histogram_jsd": 0.25,
                    "min_generated_diversity": 0.02,
                },
            },
            description="Compare GAN outputs with the training distribution using lightweight proxy metrics.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_gan_generation"],
            params={
                "input_from_step": "run_gan_generation",
                "sample_limit": 100,
                "image_size": 128,
            },
            description="Evaluate SAR-specific GAN artifacts and target compactness.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create bounded repair suggestions if GAN observer triggers are raised.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_gan_generation",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package GAN generated outputs with manifest, reports, and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=dataset_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="gan_generation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_source": dataset_source,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "target_count": target_count,
            "training_epochs": training_epochs,
            "variant": variant,
            "exclude_filters": exclude_filters,
        },
        filters=filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "GAN generation is treated as a fast learned baseline for simple targets.",
                "Use downstream classification and duplicate checks before claiming benefit.",
            ],
        },
    )


def generate_target_background_composition_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    classification_eval_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("target_background_composition", intent)
    target_source = resolve_recipe_path(intent.get("target_source") or intent.get("dataset_source") or intent_spec.get("dataset_root"))
    background_source = resolve_recipe_path(intent.get("background_source") or intent.get("background_dir"))
    mask_source = resolve_optional_recipe_path(intent.get("mask_source") or intent.get("mask_dir"))
    run_dir = Path(output_dir).expanduser().resolve()
    skill_run_dir = run_dir / "recipe_artifacts" / "target_background_composition"
    composed_dir = skill_run_dir / "composed_images"
    export_dir = run_dir / "augmented_dataset"
    target_count = intent.get("target_count") or 20
    blend_mode = intent.get("blend_mode") or "feather"
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record dataset profile and optional format validation artifacts before target/background composition.",
        ),
        RecipeStep(
            id="run_target_background_composition",
            skill="TargetBackgroundCompositionSkill",
            depends_on=["inspect_inputs"],
            params={
                "target_dir": target_source,
                "background_dir": background_source,
                "mask_dir": mask_source,
                "output_dir": skill_run_dir.as_posix(),
                "config": "configs/skills/target_background_composition.yaml",
                "target_count": target_count,
                "blend_mode": blend_mode,
                "mask_mode": intent.get("mask_mode") or "auto",
                "placement_policy": intent.get("placement_policy") or "random",
            },
            description="Fuse target chips into background scenes with SAR-aware mask, intensity matching, and feather/laplacian/Poisson blending.",
        ),
        RecipeStep(
            id="evaluate_outputs",
            skill="QualityEvaluationSkill",
            depends_on=["run_target_background_composition"],
            params={
                "input_dir": composed_dir.as_posix(),
                "expected_from_step": "run_target_background_composition",
                "sample_limit": 100,
                "thresholds": {
                    "allow_count_mismatch": False,
                    "max_black_pixel_ratio": 0.95,
                    "max_white_pixel_ratio": 0.95,
                    "min_luma_dynamic_range": 6,
                    "low_luma_dynamic_range": 12,
                },
            },
            description="Evaluate composited image count, readability, and dynamic range.",
        ),
        RecipeStep(
            id="evaluate_distribution",
            skill="DistributionEvaluationSkill",
            depends_on=["run_target_background_composition"],
            params={
                "generated_from_step": "run_target_background_composition",
                "reference_dir": background_source,
                "reference_sample_limit": 200,
                "generated_sample_limit": target_count,
                "image_size": 64,
                "standard_fid": False,
                "thresholds": {
                    "max_sar_fid_lite": 160.0,
                    "max_histogram_jsd": 0.35,
                    "min_generated_diversity": 0.01,
                },
            },
            description="Compare composed scenes to background bank with lightweight distribution metrics.",
        ),
        RecipeStep(
            id="evaluate_sar_artifacts",
            skill="SARArtifactEvaluationSkill",
            depends_on=["run_target_background_composition"],
            params={
                "input_from_step": "run_target_background_composition",
                "sample_limit": 100,
                "image_size": 128,
            },
            description="Evaluate SAR-specific artifacts after target/background fusion.",
        ),
        RecipeStep(
            id="repair_policy",
            skill="RepairPolicySkill",
            depends_on=["evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts"],
            params={
                "evaluator_step": "evaluate_outputs",
                "distribution_step": "evaluate_distribution",
                "sar_artifact_step": "evaluate_sar_artifacts",
                "max_trials": int(max_repair_trials),
                "auto_rerun": False,
            },
            description="Create bounded repair suggestions if fusion observers trigger.",
        ),
        RecipeStep(
            id="export_dataset",
            skill="ExportDatasetSkill",
            depends_on=["repair_policy"],
            params={
                "source_from_step": "run_target_background_composition",
                "quality_from_step": "evaluate_outputs",
                "repair_from_step": "repair_policy",
                "output_dir": export_dir.as_posix(),
                "mode": "copy",
                "include_originals": False,
                "require_quality_pass": False,
            },
            description="Package composited outputs, masks, manifests, quality reports, and provenance.",
        ),
    ]
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=target_source,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="target_background_composition",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "target_source": target_source,
            "background_source": background_source,
            "mask_source": mask_source,
            "target_count": target_count,
            "blend_mode": blend_mode,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
        },
        filters=intent.get("filters") or {},
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": True,
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "TargetBackgroundCompositionSkill performs deterministic image fusion, not background model training.",
                "Auto masks are heuristics; detection labels and downstream benefit claims require validation.",
                "Feather/laplacian modes are default SAR-safe baselines; Poisson is available when OpenCV is installed.",
            ],
        },
    )


def generate_raysar_synthesis_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    classification_eval_config: str | Path | None = None,
    model_to_pov_config: str | Path | None = None,
    max_repair_trials: int = 3,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("raysar_synthesis", intent)
    pov_scene = resolve_recipe_path(intent.get("pov_scene") or intent.get("dataset_source"))
    model_file = resolve_recipe_path(intent.get("model_file"))
    if pov_scene and Path(str(pov_scene)).suffix.lower() in {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae", ".3ds", ".xyz"}:
        model_file = model_file or pov_scene
        pov_scene = None
    if pov_scene and model_file and Path(str(pov_scene)).suffix.lower() != ".pov":
        pov_scene = None
    parameters_file = resolve_recipe_path(intent.get("parameters_file"))
    contributions_txt = resolve_recipe_path(intent.get("contributions_txt"))
    baseline_dataset = resolve_recipe_path(intent.get("baseline_dataset")) if intent.get("baseline_dataset") else None
    run_dir = Path(output_dir).expanduser().resolve()
    compile_dir = run_dir / "recipe_artifacts" / "model_to_pov_scene"
    skill_run_dir = run_dir / "recipe_artifacts" / "raysar_synthesis"
    maps_dir = skill_run_dir / "postprocess" / "Maps"
    export_dir = run_dir / "augmented_dataset"
    width = intent.get("width") or 512
    height = intent.get("height") or 512
    raysar_geometry = dict(intent.get("raysar_geometry") or {})
    filters = intent.get("filters") or {}
    geometry_filter_keys = {"incidence_angle_deg", "depression_angle_deg", "azimuth_deg", "elevation_angle_deg"}
    recipe_filters = {key: value for key, value in filters.items() if key not in geometry_filter_keys}
    for source_key, target_key in (
        ("incidence_angle_deg", "incidence_angle_deg"),
        ("depression_angle_deg", "depression_angle_deg"),
        ("azimuth_deg", "azimuth_deg"),
    ):
        if filters.get(source_key) is not None and raysar_geometry.get(target_key) is None:
            raysar_geometry[target_key] = filters[source_key]
    is_sweep = bool(
        model_file
        and not pov_scene
        and not contributions_txt
        and (raysar_geometry.get("azimuth_sweep") or raysar_geometry.get("azimuth_values"))
    )
    if is_sweep:
        skill_run_dir = run_dir / "recipe_artifacts" / "raysar_sweep"
        maps_dir = skill_run_dir / "sweep_maps"
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
                "pov_scene": pov_scene,
                "model_file": model_file,
            },
            description="Record profile artifacts and RaySAR scene path for provenance.",
        )
    ]
    raysar_depends_on = ["inspect_inputs"]
    raysar_params: dict[str, Any] = {
        "pov_scene": pov_scene,
        "parameters_file": parameters_file,
        "contributions_txt": contributions_txt,
        "output_dir": skill_run_dir.as_posix(),
        "config": "configs/skills/raysar.yaml",
        "width": width,
        "height": height,
        "render": not bool(contributions_txt),
        "postprocess": True,
        "sar_intersection": True,
        "auto_fix_scene": True,
        "build_if_missing": True,
    }
    if is_sweep:
        sweep_params = {
            "model_file": model_file,
            "output_dir": skill_run_dir.as_posix(),
            "config": "configs/skills/raysar_sweep.yaml",
            "parameters_file": parameters_file,
            "width": width,
            "height": height,
            "render": True,
            "postprocess": True,
        }
        for key in (
            "azimuth_sweep",
            "azimuth_values",
            "incidence_angle_deg",
            "depression_angle_deg",
            "target_extent_m",
            "scale_factor",
            "input_up_axis",
            "pov_height_axis",
            "range_distance_m",
            "sensor_plane_m",
        ):
            if raysar_geometry.get(key) is not None:
                sweep_params[key] = raysar_geometry[key]
        steps.append(
            RecipeStep(
                id="run_raysar_sweep",
                skill="RaySARSweepSynthesisSkill",
                depends_on=["inspect_inputs"],
                params=sweep_params,
                description="Compile and simulate multiple RaySAR views by sweeping target azimuth/aspect angles.",
            )
        )
        generation_step_id = "run_raysar_sweep"
    elif model_file and not pov_scene and not contributions_txt:
        compile_params = {
            "model_file": model_file,
            "output_dir": compile_dir.as_posix(),
            "config": str(model_to_pov_config) if model_to_pov_config else "configs/skills/model_to_pov_scene.yaml",
        }
        for key in (
            "incidence_angle_deg",
            "depression_angle_deg",
            "azimuth_deg",
            "pitch_deg",
            "roll_deg",
            "target_extent_m",
            "scale_factor",
            "input_up_axis",
            "pov_height_axis",
            "range_distance_m",
            "sensor_plane_m",
        ):
            if raysar_geometry.get(key) is not None:
                compile_params[key] = raysar_geometry[key]
        steps.append(
            RecipeStep(
                id="compile_pov_scene",
                skill="ModelToPOVSceneCompilerSkill",
                depends_on=["inspect_inputs"],
                params=compile_params,
                description="Compile the user-provided 3D target model into a RaySAR-compatible POV scene.",
            )
        )
        raysar_depends_on = ["compile_pov_scene"]
        raysar_params["pov_scene"] = None
        raysar_params["pov_scene_from_step"] = "compile_pov_scene"
        generation_step_id = "run_raysar_synthesis"
    else:
        generation_step_id = "run_raysar_synthesis"
    if not is_sweep:
        steps.append(
            RecipeStep(
                id="run_raysar_synthesis",
                skill="RaySARSynthesisSkill",
                depends_on=raysar_depends_on,
                params=raysar_params,
                description="Render RaySAR scattering contributions through adapted POV-Ray, then post-process reflection maps.",
            )
        )
    steps.extend(
        [
            RecipeStep(
                id="evaluate_outputs",
                skill="QualityEvaluationSkill",
                depends_on=[generation_step_id],
                params={
                    "input_dir": maps_dir.as_posix(),
                    "expected_from_step": generation_step_id,
                    "expected_count": None if is_sweep else 3,
                    "sample_limit": 20,
                    "thresholds": {
                        "allow_count_mismatch": True,
                        "max_black_pixel_ratio": 0.98,
                        "max_white_pixel_ratio": 0.98,
                        "min_luma_dynamic_range": 1,
                        "low_luma_dynamic_range": 4,
                        "max_flat_image_fraction": 0.8,
                        "max_low_dynamic_range_fraction": 0.8,
                    },
                },
                description="Evaluate generated RaySAR map readability without treating physics maps as ordinary real SAR samples.",
            ),
            RecipeStep(
                id="evaluate_sar_artifacts",
                skill="SARArtifactEvaluationSkill",
                depends_on=[generation_step_id],
                params={
                    "input_from_step": generation_step_id,
                    "sample_limit": 20,
                    "image_size": 128,
                    "thresholds": {
                        "min_target_compactness": 0.02,
                        "max_center_offset": 0.6,
                        "max_trigger_fraction": 0.8,
                    },
                },
                description="Check SAR map artifacts and target compactness before export.",
            ),
            RecipeStep(
                id="repair_policy",
                skill="RepairPolicySkill",
                depends_on=["evaluate_outputs", "evaluate_sar_artifacts"],
                params={
                    "evaluator_step": "evaluate_outputs",
                    "sar_artifact_step": "evaluate_sar_artifacts",
                    "max_trials": int(max_repair_trials),
                    "auto_rerun": False,
                },
                description="Create bounded repair suggestions for scene, camera, or postprocess parameter issues.",
            ),
            RecipeStep(
                id="export_dataset",
                skill="ExportDatasetSkill",
                depends_on=["repair_policy"],
                params={
                    "source_from_step": generation_step_id,
                    "quality_from_step": "evaluate_outputs",
                    "repair_from_step": "repair_policy",
                    "output_dir": export_dir.as_posix(),
                    "mode": "copy",
                    "include_originals": False,
                    "require_quality_pass": False,
                },
                description="Package RaySAR reflection maps with provenance and warnings about real-synthetic domain shift.",
            ),
        ]
    )
    maybe_append_classification_evaluation_step(
        steps=steps,
        intent=intent,
        baseline_dataset=baseline_dataset,
        augmented_from_step="export_dataset",
        output_dir=run_dir / "recipe_artifacts" / "classification_evaluation",
        baseline_dataset_config=validated_profile.get("dataset_config"),
        classification_eval_config=classification_eval_config,
    )
    return SagaRecipe(
        recipe_id=recipe_id,
        task="raysar_synthesis",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "pov_scene": pov_scene,
            "model_file": model_file,
            "parameters_file": parameters_file,
            "contributions_txt": contributions_txt,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
            "width": width,
            "height": height,
            "raysar_geometry": raysar_geometry,
        },
        filters=recipe_filters,
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": bool(pov_scene or model_file or contributions_txt),
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "RaySAR is treated as a physical simulation skill, not a generic learned generator.",
                "Executable input can be a RaySAR-compatible .pov scene, an existing Contributions.txt, or a common 3D model compiled by ModelToPOVSceneCompilerSkill.",
                "3D mesh scenes use coarse POV-Ray material reflectivity and still require real-synthetic domain-gap evaluation.",
                "Real-synthetic domain gap must be evaluated before downstream benefit is written into memory.",
            ],
        },
    )


def generate_classification_evaluation_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    classification_eval_config: str | Path | None = None,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    intent = intent_spec.get("intent", {})
    recipe_id = stable_recipe_id("classification_evaluation", intent)
    baseline_dataset = resolve_recipe_path(
        intent.get("baseline_dataset") or intent.get("dataset_source") or intent_spec.get("dataset_root")
    )
    augmented_dataset = resolve_recipe_path(intent.get("augmented_dataset"))
    val_dataset = resolve_recipe_path(intent.get("val_dataset"))
    run_dir = Path(output_dir).expanduser().resolve()
    eval_dir = run_dir / "recipe_artifacts" / "classification_evaluation"
    baseline_config = validated_profile.get("dataset_config") if baseline_dataset else None

    leakage_dir = run_dir / "recipe_artifacts" / "data_gates" / "leakage_check"
    duplicate_dir = run_dir / "recipe_artifacts" / "data_gates" / "duplicate_check"
    steps = [
        RecipeStep(
            id="inspect_inputs",
            skill="DatasetProfileReportSkill",
            params={
                "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                "validated_profile_path": validated_profile_artifact_path(validated_profile),
                "format_bridge_valid": bridge_report.get("valid"),
            },
            description="Record dataset profile and format validation artifacts before downstream evaluation.",
        ),
        RecipeStep(
            id="leakage_check",
            skill="LeakageCheckSkill",
            depends_on=["inspect_inputs"],
            params={
                "baseline_dataset": baseline_dataset,
                "augmented_dataset": augmented_dataset,
                "val_dataset": val_dataset,
                "output_dir": leakage_dir.as_posix(),
                "sample_limit": 20000,
            },
            description="Check exact hash and same-stem overlaps before classification evaluation.",
        ),
        RecipeStep(
            id="duplicate_check",
            skill="DuplicateNearDuplicateSkill",
            depends_on=["inspect_inputs"],
            params={
                "input_dir": augmented_dataset or baseline_dataset,
                "output_dir": duplicate_dir.as_posix(),
                "sample_limit": 5000,
                "hash_size": 16,
                "hamming_threshold": 6,
            },
            description="Check near-duplicate samples before classification evaluation.",
        ),
        RecipeStep(
            id="evaluate_classification",
            skill="ClassificationEvaluationSkill",
            depends_on=["leakage_check", "duplicate_check"],
            params={
                "baseline_dataset": baseline_dataset,
                "augmented_dataset": augmented_dataset,
                "val_dataset": val_dataset,
                "baseline_dataset_config": baseline_config,
                "augmented_dataset_config": intent.get("augmented_dataset_config"),
                "val_dataset_config": intent.get("val_dataset_config"),
                "output_dir": eval_dir.as_posix(),
                "config": str(classification_eval_config) if classification_eval_config else "configs/skills/classification_evaluation.yaml",
            },
            description="Compile and optionally run a baseline-vs-augmented downstream classification benchmark.",
        ),
    ]
    return SagaRecipe(
        recipe_id=recipe_id,
        task="classification_evaluation",
        planner=planner_mode,
        inputs={
            "request": request_text,
            "baseline_dataset": baseline_dataset,
            "augmented_dataset": augmented_dataset,
            "val_dataset": val_dataset,
            "baseline_dataset_config": baseline_config,
            "format_spec": validated_profile.get("format_spec"),
        },
        filters=intent.get("filters") or {},
        pipeline=steps,
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": bool(baseline_dataset),
            "format_bridge_valid": bridge_report.get("valid"),
            "notes": [
                "This recipe is an evaluator DAG, not an augmentation DAG.",
                "Dry-run compiles benchmark data and reproducible training/validation commands.",
                "Real downstream benefit is only claimed after the evaluator records baseline and augmented metrics.",
            ],
        },
    )


def maybe_append_classification_evaluation_step(
    *,
    steps: list[RecipeStep],
    intent: dict[str, Any],
    baseline_dataset: str | None,
    augmented_from_step: str,
    output_dir: Path,
    baseline_dataset_config: str | None = None,
    classification_eval_config: str | Path | None = None,
) -> None:
    goals = set(intent.get("goals") or [])
    if "downstream_benefit_validation" not in goals:
        return
    if not baseline_dataset:
        return
    if any(step.id == "evaluate_classification" for step in steps):
        return
    gate_dir = output_dir.parent / "data_gates"
    steps.append(
        RecipeStep(
            id="leakage_check",
            skill="LeakageCheckSkill",
            depends_on=[augmented_from_step],
            params={
                "baseline_dataset": baseline_dataset,
                "augmented_from_step": augmented_from_step,
                "output_dir": (gate_dir / "leakage_check").as_posix(),
                "sample_limit": 20000,
            },
            description="Check leakage between original and exported augmented data before downstream classification evaluation.",
        )
    )
    steps.append(
        RecipeStep(
            id="duplicate_check",
            skill="DuplicateNearDuplicateSkill",
            depends_on=[augmented_from_step],
            params={
                "input_from_step": augmented_from_step,
                "output_dir": (gate_dir / "duplicate_check").as_posix(),
                "sample_limit": 5000,
                "hash_size": 16,
                "hamming_threshold": 6,
            },
            description="Check duplicates and near-duplicates in the exported augmented data before recording classifier benefit.",
        )
    )
    steps.append(
        RecipeStep(
            id="evaluate_classification",
            skill="ClassificationEvaluationSkill",
            depends_on=["leakage_check", "duplicate_check"],
            params={
                "baseline_dataset": baseline_dataset,
                "augmented_from_step": augmented_from_step,
                "baseline_dataset_config": baseline_dataset_config,
                "augmented_dataset_config": None,
                "val_dataset_config": None,
                "output_dir": output_dir.as_posix(),
                "config": str(classification_eval_config) if classification_eval_config else "configs/skills/classification_evaluation.yaml",
            },
            description="Compile and optionally run downstream classification evaluation for the exported augmented dataset.",
        )
    )


def generate_profile_only_recipe(
    request_text: str,
    intent_spec: dict[str, Any],
    dataset_profile_path: str | Path | None,
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path,
    planner_mode: str = "rule_based",
    planner_proposal: dict[str, Any] | None = None,
    planner_guardrail: dict[str, Any] | None = None,
) -> SagaRecipe:
    task = intent_spec.get("intent", {}).get("task") or "unknown"
    return SagaRecipe(
        recipe_id=stable_recipe_id(task, intent_spec.get("intent", {})),
        task=task,
        planner=planner_mode,
        inputs={
            "request": request_text,
            "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
            "dataset_config": validated_profile.get("dataset_config"),
            "format_spec": validated_profile.get("format_spec"),
        },
        filters=intent_spec.get("intent", {}).get("filters") or {},
        pipeline=[
            RecipeStep(
                id="inspect_inputs",
                skill="DatasetProfileReportSkill",
                params={
                    "dataset_profile_path": str(dataset_profile_path) if dataset_profile_path else None,
                    "validated_profile_path": validated_profile_artifact_path(validated_profile),
                    "format_bridge_valid": bridge_report.get("valid"),
                },
                description="Record profile and validation artifacts for this run.",
            )
        ],
        metadata={
            "generator": "saga_rule_recipe_generator_v1",
            "planner_mode": planner_mode,
            "planner_guardrail_decision": (planner_guardrail or {}).get("decision"),
            "planner_selected_plan_id": (planner_proposal or {}).get("plan_id"),
            "planner_execution_allowed": (planner_guardrail or {}).get("execution_allowed"),
            "planner_requires_clarification": (planner_guardrail or {}).get("requires_clarification"),
            "planner_clarification_questions": (planner_guardrail or {}).get("clarification_questions", []),
            "planner_strategy": (planner_proposal or {}).get("strategy"),
            "planner_rationale": (planner_proposal or {}).get("rationale"),
            "planner_ready": validated_profile.get("planner_ready", False),
            "format_bridge_valid": bridge_report.get("valid"),
            "output_dir": Path(output_dir).expanduser().resolve().as_posix(),
        },
    )


def write_recipe(recipe: SagaRecipe, output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    rendered = yaml.safe_dump(recipe.to_dict(), allow_unicode=True, sort_keys=False)
    save_text(path, rendered)
    save_json(path.with_suffix(".json"), recipe.to_dict())


def load_optional_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    value = Path(path)
    if not value.exists():
        return None
    return load_mapping(value)


def resolve_recipe_path(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return Path(str(value)).expanduser().resolve().as_posix()


def resolve_optional_recipe_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (dict, list, tuple, set)) and not value:
        return None
    return Path(str(value)).expanduser().resolve().as_posix()


def validated_profile_artifact_path(validated_profile: dict[str, Any]) -> str | None:
    value = validated_profile.get("_validated_profile_path") or validated_profile.get("profile_path")
    if not value:
        return None
    return Path(str(value)).expanduser().resolve().as_posix()


def stable_recipe_id(task: str, intent: dict[str, Any]) -> str:
    pieces = [task or "unknown"]
    seen_piece_values = {str(pieces[0])}
    for key in (
        "content_source",
        "style_source",
        "dataset_source",
        "baseline_dataset",
        "augmented_dataset",
        "model_family",
        "target_count",
        "training_epochs",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
    ):
        value = intent.get(key)
        if value:
            piece = Path(str(value)).name
            if piece not in seen_piece_values:
                pieces.append(piece)
                seen_piece_values.add(piece)
    filters = intent.get("filters") or {}
    for key, value in sorted(filters.items()):
        pieces.append(f"{key}_{value}")
    exclude_filters = intent.get("exclude_filters") or {}
    for key, value in sorted(exclude_filters.items()):
        pieces.append(f"exclude_{key}_{value}")
    raw = "_".join(pieces)
    value = re.sub(r"\W+", "_", raw, flags=re.UNICODE).strip("_")
    return (value or "saga_recipe")[:120]


def normalize_filters_for_recipe(
    filters: dict[str, Any],
    exclude_filters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    normalized_filters: dict[str, Any] = {}
    normalized_excludes = normalize_exclude_filters(exclude_filters)
    for key, value in (filters or {}).items():
        if not is_safe_filter_name(str(key)):
            continue
        exclude_value = parse_exclude_filter_value(value)
        if exclude_value is not None:
            add_exclude_filter_value(normalized_excludes, str(key), exclude_value)
        else:
            normalized_filters[str(key)] = value
    return normalized_filters, normalized_excludes


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
        if item is None or isinstance(item, (dict, list, tuple, set)):
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


def is_safe_filter_name(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value))

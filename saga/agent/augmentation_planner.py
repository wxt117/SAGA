from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.agent.planning_evidence import compact_planning_evidence
from saga.core.config import save_json, save_text


AUGMENTATION_PLAN_VERSION = "saga_augmentation_plan_v1"

EXECUTABLE_RECIPE_TASKS = {
    "TraditionalAugmentationSkill": "traditional_augmentation",
    "DiffusionLoRAGenerationSkill": "diffusion_lora_generation",
    "GeoDiffSARSkill": "geodiff_sar_generation",
    "GaussianSplattingCompletionSkill": "gaussian_splatting_completion",
    "GANImageToImageSkill": "gan_generation",
    "StyleTransferSkill": "style_transfer",
    "PseudocolorSkill": "pseudocolor_transform",
    "BackgroundGenerationSkill": "background_generation",
    "TargetBackgroundCompositionSkill": "target_background_composition",
    "RaySARSynthesisSkill": "raysar_synthesis",
    "RaySARSweepSynthesisSkill": "raysar_synthesis",
}


def build_augmentation_plan(
    *,
    intent_spec: dict[str, Any],
    benefit_context: dict[str, Any],
    request_constraints: dict[str, Any],
    multi_dataset_profile: dict[str, Any],
    memory_context: dict[str, Any],
    output_dir: str | Path | None = None,
    planner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planner_config = planner_config or {}
    intent = dict(intent_spec.get("intent") or {})
    intent["_raw_text"] = intent_spec.get("raw_text") or ""
    skill_rank = benefit_context.get("skill_utility", {}).get("ranked_skills", [])
    planning_evidence = benefit_context.get("planning_evidence") or {}
    candidates = [
        build_candidate(
            skill=item,
            intent=intent,
            constraints=request_constraints,
            multi_dataset_profile=multi_dataset_profile,
            memory_context=memory_context,
            planning_evidence=planning_evidence,
        )
        for item in skill_rank
        if is_augmentation_or_evaluator_skill(str(item.get("skill") or ""))
    ]
    candidates = [candidate for candidate in candidates if candidate]
    ranked = sorted(candidates, key=lambda item: (item["planner_score"], plan_tie_breaker(item, intent)), reverse=True)
    selected = select_plan(ranked, prefer_executable=bool(planner_config.get("prefer_executable_skills", True)))
    candidate_limit = safe_int(planner_config.get("candidate_plan_limit"))
    if candidate_limit and candidate_limit > 0:
        ranked = ranked[:candidate_limit]
        if selected and not any(item.get("plan_id") == selected.get("plan_id") for item in ranked):
            ranked = [selected] + ranked[: max(0, candidate_limit - 1)]
    report = {
        "schema_version": AUGMENTATION_PLAN_VERSION,
        "task": intent.get("task"),
        "request_goals": intent.get("goals") or [],
        "constraints": request_constraints,
        "selected_plan_id": selected.get("plan_id") if selected else None,
        "selected_recipe_task": selected.get("recipe_task") if selected else None,
        "selected_skill": selected.get("skill") if selected else None,
        "ranked_plans": ranked,
        "planner_config": {
            "candidate_plan_limit": candidate_limit,
            "prefer_executable_skills": bool(planner_config.get("prefer_executable_skills", True)),
        },
        "planning_evidence_summary": compact_planning_evidence(planning_evidence) if planning_evidence else {},
        "memory_policy_summary": memory_context.get("policy_summary", {}),
        "alignment": multi_dataset_profile.get("alignment", {}),
        "notes": [
            "This is SAGA's deterministic task-specific augmentation planner.",
            "It compares candidate augmentation strategies using dataset needs, request constraints, skill utility, and policy memory.",
            "LLM planner may use this as evidence, but execution still compiles through deterministic recipes.",
        ],
    }
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        save_json(path / "augmentation_plan.json", report)
        save_text(path / "augmentation_plan.md", render_augmentation_plan_markdown(report))
    return report


def build_candidate(
    *,
    skill: dict[str, Any],
    intent: dict[str, Any],
    constraints: dict[str, Any],
    multi_dataset_profile: dict[str, Any],
    memory_context: dict[str, Any],
    planning_evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    name = str(skill.get("skill") or "")
    recipe_task = EXECUTABLE_RECIPE_TASKS.get(name)
    status = str(skill.get("status") or "planned")
    base = float(skill.get("utility_score") or 0.0)
    adjustments, rationale = planner_adjustments(
        name=name,
        intent=intent,
        constraints=constraints,
        multi_dataset_profile=multi_dataset_profile,
        memory_context=memory_context,
        planning_evidence=planning_evidence,
    )
    score = clamp(base + sum(item["delta"] for item in adjustments))
    params = default_params_for_skill(
        name=name,
        intent=intent,
        constraints=constraints,
        multi_dataset_profile=multi_dataset_profile,
        memory_context=memory_context,
        planning_evidence=planning_evidence,
    )
    plan_id = safe_plan_id(f"{name}_{recipe_task or 'planned'}")
    pipeline = pipeline_for_skill(name=name, recipe_task=recipe_task, include_evaluator=constraints.get("needs_downstream_evidence"))
    return {
        "plan_id": plan_id,
        "skill": name,
        "status": status,
        "recipe_task": recipe_task,
        "planner_score": round(score, 3),
        "base_utility": base,
        "adjustments": adjustments,
        "params": params,
        "recipe_intent_updates": recipe_intent_updates(name=name, intent=intent, params=params),
        "pipeline": pipeline,
        "expected_benefits": expected_benefits(name),
        "risks": risks(name),
        "rationale": rationale,
        "execution_policy": {
            "executable": status == "executable" and recipe_task is not None,
            "real_run_requires_user_run_flag": name
            in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill", "StyleTransferSkill", "ClassificationEvaluationSkill"},
            "downstream_evaluator_recommended": bool(constraints.get("needs_downstream_evidence")),
        },
    }


def planner_adjustments(
    *,
    name: str,
    intent: dict[str, Any],
    constraints: dict[str, Any],
    multi_dataset_profile: dict[str, Any],
    memory_context: dict[str, Any],
    planning_evidence: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    adjustments = []
    goals = set(intent.get("goals") or [])
    task = intent.get("task")
    text = str(intent.get("_raw_text") or "").lower()
    speed = constraints.get("speed_priority")
    quality = constraints.get("quality_priority")
    gpu = constraints.get("gpu_budget")
    evidence_summary = (planning_evidence or {}).get("summary") or {}
    raysar_geometry = intent.get("raysar_geometry") or {}
    caption_evidence = evidence_summary.get("metadata_caption") or {}
    balance_evidence = evidence_summary.get("dataset_balance") or {}
    preprocess_evidence = evidence_summary.get("preprocess") or {}
    balance_deficit = safe_int(balance_evidence.get("total_recommended_additions")) or 0
    balance_fields = set(balance_evidence.get("fields") or [])
    physical_prior_available = bool(intent.get("model_file") or intent.get("pov_scene")) or any(
        keyword in text for keyword in ["3d", "三维", "3d模型", "模型", "几何先验", "物理先验"]
    )
    radiometric_preservation = any(
        keyword in text
        for keyword in ["辐射定标", "绝对幅度", "绝对幅值", "保持灰度", "物理灰度", "不改变灰度", "radiometric", "absolute amplitude"]
    )
    physics_simulation_excluded = any(
        keyword in text
        for keyword in [
            "不需要物理仿真",
            "不要物理仿真",
            "不做物理仿真",
            "不用物理仿真",
            "不需要raysar",
            "不要raysar",
            "不做raysar",
            "不用raysar",
            "no raysar",
            "without raysar",
            "no physics simulation",
            "without physics simulation",
        ]
    )
    pseudocolor_requested = any(keyword in text for keyword in ["伪彩", "伪彩色", "可视化", "人工检查", "展示", "weicaise", "pseudocolor"])
    strict_grayscale_training = any(keyword in text for keyword in ["灰度训练", "保持灰度", "物理灰度", "标准sar灰度", "standard grayscale"])
    traditional_requested = any(keyword in text for keyword in ["传统", "基础增广", "基础增强", "traditional augmentation", "basic augmentation"])
    training_excluded = any(
        keyword in text
        for keyword in [
            "不训练",
            "不要训练",
            "无需训练",
            "不用训练",
            "不需要训练",
            "no training",
            "without training",
        ]
    )

    def add(delta: float, reason: str) -> None:
        adjustments.append({"delta": delta, "reason": reason})

    if EXECUTABLE_RECIPE_TASKS.get(name) == task:
        add(0.35, "Candidate skill exactly matches the recognized executable task.")
    elif task in set(EXECUTABLE_RECIPE_TASKS.values()) and name in EXECUTABLE_RECIPE_TASKS:
        add(-0.25, "Candidate skill does not match the recognized executable task.")
    if task == "classification" and name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"} and "physics_interpretable_simulation" not in goals:
        add(-0.22, "RaySAR is treated as a physics simulation skill, not a default classification augmentation skill.")
    if physics_simulation_excluded and name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill", "ModelToPOVSceneCompilerSkill"}:
        add(-0.6, "The request explicitly excludes RaySAR/physics simulation.")

    if speed == "high":
        if name == "TraditionalAugmentationSkill":
            add(0.22, "User/data constraints prioritize speed; traditional augmentation is fastest.")
        if name in {"DiffusionLoRAGenerationSkill", "GeoDiffSARSkill", "BackgroundGenerationSkill"}:
            add(-0.18, "High speed priority penalizes expensive generation.")
        if name == "GaussianSplattingCompletionSkill":
            add(0.12, "Gaussian splatting is the lightweight sparse-view option.")
        if name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
            add(-0.08, "RaySAR is physically interpretable but slower than simple augmentation.")
    if quality == "high":
        if name in {"DiffusionLoRAGenerationSkill", "GeoDiffSARSkill"}:
            add(0.16, "High-quality request favors diffusion or physical-prior generation.")
        if name == "TraditionalAugmentationSkill":
            add(-0.04, "Traditional augmentation is conservative but limited for high-quality synthesis.")
    if gpu == "limited" and name in {"DiffusionLoRAGenerationSkill", "GeoDiffSARSkill", "BackgroundGenerationSkill"}:
        add(-0.22, "Limited GPU budget penalizes heavy generation.")
    if gpu == "limited" and name == "GaussianSplattingCompletionSkill":
        add(0.12, "Low-memory request favors the lightweight Gaussian splatting option.")
    if gpu == "available" and name == "DiffusionLoRAGenerationSkill":
        add(0.08, "Available GPU makes LoRA generation more feasible.")
    if speed == "high" and name == "GANImageToImageSkill":
        add(0.1, "GAN is a fast learned generator for simple target appearance variation.")
    if traditional_requested:
        if name == "TraditionalAugmentationSkill":
            add(0.32, "The request explicitly asks for traditional/basic augmentation.")
        elif name in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill", "StyleTransferSkill"}:
            add(-0.16, "The request explicitly asks for traditional/basic augmentation rather than learned generation.")
    if training_excluded:
        if name == "TraditionalAugmentationSkill":
            add(0.22, "The request excludes model training; conservative deterministic augmentation remains feasible.")
        if name in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill"}:
            add(-0.58, "The request excludes model training, which makes train-from-data generators unsuitable without a provided checkpoint.")
    if "complete_sparse_azimuth" in goals:
        if name == "GeoDiffSARSkill":
            add(0.3, "Sparse azimuth completion with high quality is GeoDiff-SAR's core use case.")
            if physical_prior_available:
                add(0.12, "A 3D/physical prior is available or explicitly mentioned, which GeoDiff-SAR requires.")
            else:
                add(-0.16, "GeoDiff-SAR should not be preferred without a 3D/geometric/physical prior.")
        if name == "GaussianSplattingCompletionSkill":
            add(0.24, "Sparse azimuth completion with speed/VRAM constraints favors Gaussian splatting.")
            if speed == "high" or gpu == "limited" or any(keyword in text for keyword in ["轻量", "低显存", "快速"]):
                add(0.12, "The request emphasizes speed or low VRAM, matching Gaussian splatting's main advantage.")
        if name == "TraditionalAugmentationSkill":
            add(-0.18, "Traditional augmentation cannot create true unseen viewpoints.")
        if name == "RaySARSweepSynthesisSkill":
            add(0.12, "Azimuth sweep simulation can add controlled target aspect coverage from a 3D model.")
    if "physics_interpretable_simulation" in goals:
        if name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
            add(0.35, "Physics-interpretable simulation explicitly favors RaySAR.")
        if name in {"GANImageToImageSkill", "DiffusionLoRAGenerationSkill"}:
            add(-0.08, "Data-driven generation is less interpretable than RaySAR for this request.")
    if raysar_geometry.get("azimuth_sweep") or raysar_geometry.get("azimuth_values"):
        if name == "RaySARSweepSynthesisSkill":
            add(0.48, "The request contains explicit azimuth sweep/list geometry, so the sweep skill is preferred.")
        if name == "RaySARSynthesisSkill":
            add(-0.28, "Single-view RaySAR is less aligned than the sweep skill for explicit multi-view geometry.")
    if "domain_adaptation" in goals:
        if name == "StyleTransferSkill":
            add(0.3, "Cross-domain or payload adaptation explicitly favors style/feature transfer.")
    if "generate_backgrounds" in goals or task == "background_generation":
        if name == "BackgroundGenerationSkill":
            add(0.42, "The request explicitly asks for SAR background generation.")
        elif name in {"TraditionalAugmentationSkill", "GANImageToImageSkill", "DiffusionLoRAGenerationSkill"}:
            add(-0.1, "Background generation is better handled by the dedicated packaged background model.")
    if "target_background_composition" in goals or task == "target_background_composition":
        if name == "TargetBackgroundCompositionSkill":
            add(0.48, "The request explicitly asks for target/background or image composition.")
        elif name == "BackgroundGenerationSkill":
            add(0.08, "Background generation can provide scene assets before composition.")
        elif name in {"TraditionalAugmentationSkill", "PseudocolorSkill"}:
            add(-0.12, "Simple transforms do not perform target/background composition.")
    if pseudocolor_requested:
        if name == "PseudocolorSkill":
            add(0.42, "The request explicitly asks for pseudocolor/visualization.")
        elif name in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill", "StyleTransferSkill"} and task == "pseudocolor_transform":
            add(-0.2, "Pseudocolor is a direct feature/visual transform request, not learned generation.")
    elif name == "PseudocolorSkill":
        add(-0.24, "Pseudocolor is not a default SAR augmentation unless the user asks for visualization or RGB feature ablation.")
    if strict_grayscale_training and name == "PseudocolorSkill":
        add(-0.34, "The request emphasizes grayscale/physical intensity preservation, which conflicts with pseudocolor.")
    if radiometric_preservation:
        if name == "SARPreprocessSkill":
            add(-0.28, "Radiometric/amplitude preservation request penalizes aggressive preprocessing.")
        if name == "PseudocolorSkill":
            add(-0.18, "Radiometric/amplitude preservation also disfavors pseudocolor transforms.")
    if "polarization_conditioned_generation" in goals:
        if name == "DiffusionLoRAGenerationSkill":
            add(0.18, "Polarization-conditioned target generation aligns with metadata-caption LoRA.")
        if name == "GANImageToImageSkill":
            add(-0.06, "GAN wrapper has weaker explicit metadata conditioning than LoRA.")
    if caption_evidence.get("usable_for_lora") and name == "DiffusionLoRAGenerationSkill":
        add(0.1, "MetadataCaptionSkill evidence shows captions can be staged for LoRA training.")
    elif name == "DiffusionLoRAGenerationSkill" and caption_evidence and not caption_evidence.get("usable_for_lora"):
        add(-0.1, "Caption evidence does not show usable captions for LoRA training.")
    if balance_deficit > 0 and name in {"TraditionalAugmentationSkill", "DiffusionLoRAGenerationSkill", "GANImageToImageSkill"}:
        add(0.08 if name == "DiffusionLoRAGenerationSkill" else 0.05, "DatasetBalancingSkill found actionable class/metadata deficits.")
    if "polarization" in balance_fields and name == "DiffusionLoRAGenerationSkill":
        add(0.04, "Balance evidence includes polarization bins, which LoRA captions can condition on.")
    if preprocess_evidence.get("recommended_before_generation") and name in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill"}:
        add(0.02, "SARPreprocessSkill evidence can stabilize image representation before learned generation.")
    if name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
        if not physical_prior_available:
            add(-0.12, "RaySAR needs a POV scene, Contributions.txt, or 3D model; none is evident in the request.")
        if speed == "high":
            add(-0.08, "RaySAR is physically interpretable but not the fast large-batch option.")
    if name == "RaySARSweepSynthesisSkill" and intent.get("model_file") and (
        raysar_geometry.get("azimuth_sweep") or raysar_geometry.get("azimuth_values")
    ):
        add(0.14, "A 3D model plus explicit angle sweep makes RaySAR sweep executable and well-aligned.")
    if constraints.get("needs_downstream_evidence") and name == "ClassificationEvaluationSkill":
        add(0.22, "User asks for downstream evidence; evaluator should be part of the plan.")

    alignment_issues = (multi_dataset_profile.get("alignment") or {}).get("issues") or []
    if alignment_issues and name in {"StyleTransferSkill", "ClassificationEvaluationSkill"}:
        add(-0.04, "Auxiliary dataset alignment has warnings; use evaluator/profiler reports before strong claims.")

    policy_stats = ((memory_context.get("policy_summary") or {}).get("skill_stats") or {}).get(name) or {}
    if policy_stats.get("classification_improved_runs"):
        add(0.12, "Policy memory has downstream-improved runs for this skill on similar data.")
    if policy_stats.get("classification_regressed_runs"):
        add(-0.14, "Policy memory has downstream-regressed runs for this skill on similar data.")
    global_prior = global_skill_prior(name=name, memory_context=memory_context)
    if global_prior:
        adjustment = safe_float(global_prior.get("planner_adjustment")) or 0.0
        if adjustment > 0:
            add(min(0.1, adjustment), "Global policy memory has positive evidence for this skill.")
        elif adjustment < 0:
            add(max(-0.1, adjustment), "Global policy memory has risk or weak evidence for this skill.")
    param_policy = top_param_policy(name=name, memory_context=memory_context)
    if param_policy:
        score = float(param_policy.get("policy_score") or 0.0)
        if score > 0:
            add(min(0.08, score / 10.0), "Policy memory has a favorable parameter pattern for this skill.")
        elif score < 0:
            add(max(-0.08, score / 10.0), "Policy memory has weak parameter outcomes for this skill.")

    rationale = " ".join(item["reason"] for item in adjustments[:4]) or "Ranked from dataset needs and skill utility priors."
    return adjustments, rationale


def default_params_for_skill(
    *,
    name: str,
    intent: dict[str, Any],
    constraints: dict[str, Any],
    multi_dataset_profile: dict[str, Any],
    memory_context: dict[str, Any] | None = None,
    planning_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_images = int(((multi_dataset_profile.get("datasets") or {}).get("primary") or {}).get("num_images") or 0)
    requested_count = safe_int(intent.get("target_count") or constraints.get("target_count"))
    explicit_epochs = safe_int(intent.get("training_epochs") or constraints.get("training_epochs")) is not None
    protected_keys = {"target_count"} if requested_count is not None else set()
    if explicit_epochs:
        protected_keys.add("training_epochs")
    count = requested_count or evidence_target_count(planning_evidence, total_images, constraints) or default_target_count(total_images, constraints)
    epochs = safe_int(intent.get("training_epochs") or constraints.get("training_epochs"))
    if epochs is None:
        epochs = default_epochs(constraints)
    if name == "TraditionalAugmentationSkill":
        params = {
            "target_count": count,
            "multiplier": 2 if constraints.get("speed_priority") != "high" else 1,
            "policy": "fast_sar_label_preserving",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "DiffusionLoRAGenerationSkill":
        params = {
            "target_count": count,
            "training_epochs": epochs,
            "model_family": intent.get("model_family") or constraints.get("model_family") or "flux",
            "auto_caption_from_metadata": True,
            "quality_mode": constraints.get("quality_priority"),
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "GeoDiffSARSkill":
        params = {
            "target_count": count,
            "training_epochs": epochs,
            "model_file": intent.get("model_file"),
            "target_azimuths": (intent.get("raysar_geometry") or {}).get("azimuth_values"),
            "azimuth_sweep": (intent.get("raysar_geometry") or {}).get("azimuth_sweep"),
            "depressions": (intent.get("raysar_geometry") or {}).get("depressions"),
            "quality_mode": constraints.get("quality_priority") or "high",
            "policy": "geodiff_physical_prior_sparse_azimuth_completion",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "GaussianSplattingCompletionSkill":
        params = {
            "target_count": count,
            "model_file": intent.get("model_file"),
            "target_azimuths": (intent.get("raysar_geometry") or {}).get("azimuth_values"),
            "azimuth_sweep": (intent.get("raysar_geometry") or {}).get("azimuth_sweep"),
            "render_resolution": safe_int(intent.get("width")) or 128,
            "project_root": "myproject/SAR GS V1",
            "policy": "lightweight_sparse_azimuth_completion",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "GANImageToImageSkill":
        params = {
            "target_count": count,
            "training_epochs": epochs if constraints.get("speed_priority") != "high" else min(epochs, 5),
            "variant": "gpu" if constraints.get("gpu_budget") == "available" else "cpu",
            "policy": "fast_simple_target_generation",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "StyleTransferSkill":
        params = {
            "style_strength_policy": "conservative_structure_preserving",
            "needs_content_source": bool(intent.get("content_source")),
            "needs_style_source": bool(intent.get("style_source")),
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "PseudocolorSkill":
        params = {
            "target_count": count,
            "colormap": intent.get("colormap") or constraints.get("colormap") or "sar",
            "preserve_tree": True,
            "policy": "visualization_or_explicit_rgb_ablation_only",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "BackgroundGenerationSkill":
        params = {
            "target_count": count,
            "num_images": count,
            "scene_prompt": intent.get("scene_prompt") or intent.get("_raw_text") or "SAR background scene",
            "width": safe_int(intent.get("width")) or 512,
            "height": safe_int(intent.get("height")) or 512,
            "policy": "direct_segment_controlled_background_generation_no_user_training",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "TargetBackgroundCompositionSkill":
        params = {
            "target_count": count,
            "target_dir": intent.get("target_source") or intent.get("dataset_source"),
            "background_dir": intent.get("background_source") or intent.get("background_dir"),
            "mask_dir": intent.get("mask_source") or intent.get("mask_dir"),
            "blend_mode": intent.get("blend_mode") or "feather",
            "placement_policy": intent.get("placement_policy") or "random",
            "policy": "sar_intensity_matched_target_background_fusion",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "RaySARSynthesisSkill":
        params = {
            "pov_scene": intent.get("pov_scene") or intent.get("dataset_source"),
            "model_file": intent.get("model_file"),
            "parameters_file": intent.get("parameters_file"),
            "contributions_txt": intent.get("contributions_txt"),
            "width": safe_int(intent.get("width")) or 512,
            "height": safe_int(intent.get("height")) or 512,
            "raysar_geometry": intent.get("raysar_geometry") or {},
            "postprocess": True,
            "policy": "physics_interpretable_with_domain_gap_warning",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "RaySARSweepSynthesisSkill":
        params = {
            "model_file": intent.get("model_file"),
            "parameters_file": intent.get("parameters_file"),
            "width": safe_int(intent.get("width")) or 512,
            "height": safe_int(intent.get("height")) or 512,
            "raysar_geometry": intent.get("raysar_geometry") or {},
            "policy": "physics_interpretable_azimuth_sweep_with_domain_gap_warning",
        }
        return apply_memory_param_recommendations(name, params, memory_context, protected_keys=protected_keys)
    if name == "ClassificationEvaluationSkill":
        return {
            "primary_metric": "acc1_article",
            "split_policy": "stable_train_val_split",
            "include_augmented_in_validation": False,
        }
    return {"target_count": count}


def top_param_policy(name: str, memory_context: dict[str, Any] | None) -> dict[str, Any] | None:
    summary = (memory_context or {}).get("policy_summary") or {}
    rows = (summary.get("skill_param_stats") or {}).get(name) or []
    if not rows:
        global_learning = (memory_context or {}).get("global_policy_learning") or {}
        rows = (global_learning.get("recommended_params_by_skill") or {}).get(name) or []
    if not rows:
        return None
    best = rows[0]
    if not isinstance(best, dict):
        return None
    return best


def global_skill_prior(name: str, memory_context: dict[str, Any] | None) -> dict[str, Any] | None:
    global_learning = (memory_context or {}).get("global_policy_learning") or {}
    priors = global_learning.get("skill_priors") or {}
    prior = priors.get(name)
    return prior if isinstance(prior, dict) else None


def apply_memory_param_recommendations(
    name: str,
    params: dict[str, Any],
    memory_context: dict[str, Any] | None,
    protected_keys: set[str] | None = None,
) -> dict[str, Any]:
    protected_keys = protected_keys or set()
    policy = top_param_policy(name=name, memory_context=memory_context)
    if not policy:
        return params
    recommended = policy.get("params") if isinstance(policy.get("params"), dict) else {}
    if not recommended:
        return params
    allowed_by_skill = {
        "TraditionalAugmentationSkill": {"target_count", "multiplier"},
        "DiffusionLoRAGenerationSkill": {"target_count", "training_epochs", "model_family"},
        "GeoDiffSARSkill": {"target_count", "training_epochs"},
        "GaussianSplattingCompletionSkill": {"target_count"},
        "GANImageToImageSkill": {"target_count", "training_epochs", "variant"},
        "StyleTransferSkill": set(),
        "PseudocolorSkill": {"colormap"},
        "BackgroundGenerationSkill": {"target_count", "num_images"},
        "TargetBackgroundCompositionSkill": {"target_count"},
        "RaySARSynthesisSkill": {"width", "height"},
        "RaySARSweepSynthesisSkill": {"width", "height"},
    }
    allowed = allowed_by_skill.get(name, set())
    merged = dict(params)
    applied = {}
    for key in allowed:
        if key in protected_keys:
            continue
        value = recommended.get(key)
        if value in (None, "", {}, []):
            continue
        if key in {"target_count", "training_epochs", "multiplier", "num_images", "width", "height"}:
            parsed = safe_int(value)
            if parsed is None or parsed <= 0:
                continue
            value = parsed
        if key == "model_family" and str(value).lower() not in {"flux", "sd3", "sdxl"}:
            continue
        if key == "variant" and str(value).lower() not in {"gpu", "cpu"}:
            continue
        merged[key] = value
        applied[key] = value
    if applied:
        merged["memory_recommended_params"] = {
            "source": "policy_memory",
            "signature": policy.get("signature"),
            "policy_score": policy.get("policy_score"),
            "applied": applied,
        }
    return merged


def recipe_intent_updates(name: str, intent: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    recipe_task = EXECUTABLE_RECIPE_TASKS.get(name)
    if not recipe_task:
        return {}
    updates = {
        "task": recipe_task,
        "target_count": params.get("target_count"),
        "multiplier": params.get("multiplier"),
        "training_epochs": params.get("training_epochs"),
        "model_family": params.get("model_family"),
        "target_azimuths": params.get("target_azimuths"),
        "azimuth_sweep": params.get("azimuth_sweep"),
        "depressions": params.get("depressions"),
        "colormap": params.get("colormap"),
        "preserve_tree": params.get("preserve_tree"),
    }
    if name == "GaussianSplattingCompletionSkill":
        updates.update(
            {
                "project_root": params.get("project_root"),
                "render_resolution": params.get("render_resolution"),
                "target_azimuths": params.get("target_azimuths"),
                "azimuth_sweep": params.get("azimuth_sweep"),
            }
        )
    for key in (
        "dataset_source",
        "content_source",
        "style_source",
        "colormap",
        "preserve_tree",
        "scene_prompt",
        "target_source",
        "background_source",
        "mask_source",
        "blend_mode",
        "placement_policy",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
        "width",
        "height",
        "raysar_geometry",
        "filters",
        "exclude_filters",
    ):
        if intent.get(key) not in (None, "", {}, []):
            updates[key] = intent.get(key)
    return {key: value for key, value in updates.items() if value not in (None, "", {}, [])}


def pipeline_for_skill(name: str, recipe_task: str | None, include_evaluator: bool) -> list[dict[str, Any]]:
    if name == "TraditionalAugmentationSkill":
        steps = [
            "inspect_inputs",
            "run_traditional_augmentation",
            "evaluate_outputs",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "DiffusionLoRAGenerationSkill":
        steps = [
            "inspect_inputs",
            "run_diffusion_lora",
            "evaluate_outputs",
            "evaluate_distribution",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "GeoDiffSARSkill":
        steps = [
            "inspect_inputs",
            "run_geodiff_sar",
            "evaluate_outputs",
            "evaluate_distribution",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "GaussianSplattingCompletionSkill":
        steps = ["inspect_inputs", "run_gaussian_splatting_completion", "evaluate_outputs", "evaluate_distribution", "evaluate_sar_artifacts", "duplicate_check", "repair_policy", "export_dataset"]
    elif name == "GANImageToImageSkill":
        steps = [
            "inspect_inputs",
            "run_gan_generation",
            "evaluate_outputs",
            "evaluate_distribution",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "StyleTransferSkill":
        steps = [
            "inspect_inputs",
            "select_content",
            "run_style_transfer",
            "evaluate_outputs",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "PseudocolorSkill":
        steps = ["inspect_inputs", "run_pseudocolor", "evaluate_outputs", "duplicate_check", "export_dataset"]
    elif name == "BackgroundGenerationSkill":
        steps = [
            "inspect_inputs",
            "run_background_generation",
            "evaluate_outputs",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "TargetBackgroundCompositionSkill":
        steps = [
            "inspect_inputs",
            "run_target_background_composition",
            "evaluate_outputs",
            "evaluate_distribution",
            "evaluate_sar_artifacts",
            "duplicate_check",
            "repair_policy",
            "export_dataset",
        ]
    elif name == "RaySARSynthesisSkill":
        steps = ["inspect_inputs", "run_raysar_synthesis", "evaluate_outputs", "evaluate_sar_artifacts", "repair_policy", "export_dataset"]
    elif name == "RaySARSweepSynthesisSkill":
        steps = ["inspect_inputs", "run_raysar_sweep", "evaluate_outputs", "evaluate_sar_artifacts", "repair_policy", "export_dataset"]
    elif name == "ClassificationEvaluationSkill":
        steps = ["inspect_inputs", "leakage_check", "duplicate_check", "evaluate_classification"]
    else:
        steps = ["planned_skill_adapter", "quality_evaluation", "downstream_evaluation"]
    if include_evaluator and name not in {"ClassificationEvaluationSkill"}:
        steps.extend(["leakage_check_after_export", "duplicate_check_after_export", "evaluate_classification_after_export"])
    return [{"id": step, "status": "planned" if recipe_task is None else "executable"} for step in steps]


def select_plan(ranked: list[dict[str, Any]], prefer_executable: bool = True) -> dict[str, Any] | None:
    if prefer_executable:
        for item in ranked:
            if item.get("execution_policy", {}).get("executable"):
                return item
    return ranked[0] if ranked else None


def plan_tie_breaker(item: dict[str, Any], intent: dict[str, Any]) -> float:
    skill = item.get("skill")
    geometry = intent.get("raysar_geometry") or {}
    if skill == "RaySARSweepSynthesisSkill" and (geometry.get("azimuth_sweep") or geometry.get("azimuth_values")):
        return 0.2
    if skill == "RaySARSynthesisSkill" and (geometry.get("azimuth_sweep") or geometry.get("azimuth_values")):
        return -0.2
    if item.get("recipe_task") == intent.get("task"):
        return 0.05
    return 0.0


def should_auto_apply_plan(intent_spec: dict[str, Any], augmentation_plan: dict[str, Any]) -> bool:
    intent = intent_spec.get("intent") or {}
    task = intent.get("task")
    if task not in {"augmentation_or_generation", "classification", "unknown"}:
        return augmentation_plan.get("selected_recipe_task") == task
    text = str(intent_spec.get("raw_text") or "")
    if not any(keyword in text for keyword in ["增广", "增强", "扩增", "生成", "augmentation", "augment"]):
        return False
    selected_task = augmentation_plan.get("selected_recipe_task")
    return selected_task in {
        "traditional_augmentation",
        "diffusion_lora_generation",
        "geodiff_sar_generation",
        "gaussian_splatting_completion",
        "gan_generation",
        "style_transfer",
        "pseudocolor_transform",
        "background_generation",
        "target_background_composition",
        "raysar_synthesis",
        "gaussian_splatting_completion",
    }


def apply_selected_plan_to_intent(intent_spec: dict[str, Any], augmentation_plan: dict[str, Any]) -> dict[str, Any]:
    import copy

    selected = None
    for item in augmentation_plan.get("ranked_plans") or []:
        if item.get("plan_id") == augmentation_plan.get("selected_plan_id"):
            selected = item
            break
    if not selected:
        return intent_spec
    effective = copy.deepcopy(intent_spec)
    intent = effective.setdefault("intent", {})
    updates = selected.get("recipe_intent_updates") or {}
    applied = []
    for key, value in updates.items():
        if value in (None, "", {}, []):
            continue
        intent[key] = value
        applied.append({"field": key, "value": value})
    effective["augmentation_planner_notes"] = {
        "source": AUGMENTATION_PLAN_VERSION,
        "selected_plan_id": selected.get("plan_id"),
        "selected_skill": selected.get("skill"),
        "selected_recipe_task": selected.get("recipe_task"),
        "applied_updates": applied,
        "rationale": selected.get("rationale"),
    }
    return effective


def is_augmentation_or_evaluator_skill(name: str) -> bool:
    excluded = {
        "DatasetProfileReportSkill",
        "FileSelectionSkill",
        "ExportDatasetSkill",
        "RepairPolicySkill",
        "SARPreprocessSkill",
        "MetadataCaptionSkill",
        "DatasetBalancingSkill",
        "QualityEvaluationSkill",
        "DistributionEvaluationSkill",
        "SARArtifactEvaluationSkill",
        "ClassificationEvaluationSkill",
        "PerMetadataSliceEvaluationSkill",
        "LeakageCheckSkill",
        "DuplicateNearDuplicateSkill",
        "RawDatasetScanSkill",
        "LLMSchemaInductionSkill",
        "DatasetFormatCompilerSkill",
        "DatasetFormatValidatorSkill",
        "RunProvenanceSkill",
        "CompareRunsSkill",
    }
    return name.endswith("Skill") and name not in excluded


def default_target_count(total_images: int, constraints: dict[str, Any]) -> int:
    if constraints.get("speed_priority") == "high":
        return max(10, min(50, max(1, total_images // 20)))
    if constraints.get("quality_priority") == "high":
        return max(20, min(200, max(1, total_images // 10)))
    return max(20, min(100, max(1, total_images // 10)))


def evidence_target_count(
    planning_evidence: dict[str, Any] | None,
    total_images: int,
    constraints: dict[str, Any],
) -> int | None:
    balance = ((planning_evidence or {}).get("summary") or {}).get("dataset_balance") or {}
    deficit = safe_int(balance.get("total_recommended_additions"))
    if deficit is None or deficit <= 0:
        return None
    cap = default_target_count(total_images, constraints)
    if constraints.get("speed_priority") == "high":
        return max(1, min(deficit, cap))
    return max(1, min(deficit, max(cap, 200)))


def default_epochs(constraints: dict[str, Any]) -> int:
    if constraints.get("speed_priority") == "high":
        return 5
    if constraints.get("quality_priority") == "high" or constraints.get("time_budget") == "relaxed":
        return 25
    return 10


def expected_benefits(name: str) -> list[str]:
    mapping = {
        "TraditionalAugmentationSkill": ["fast regularization", "robustness baseline", "cheap comparison"],
        "DiffusionLoRAGenerationSkill": ["higher target appearance diversity", "metadata-conditioned generation", "sample expansion"],
        "GANImageToImageSkill": ["fast simple-target generation", "cheap learned baseline", "appearance variation"],
        "StyleTransferSkill": ["cross-domain or payload adaptation", "feature/style diversity"],
        "PseudocolorSkill": ["fast visualization", "manual inspection", "explicit RGB feature ablation"],
        "BackgroundGenerationSkill": ["background diversity", "scene asset generation", "future composition support"],
        "TargetBackgroundCompositionSkill": ["target/background scene synthesis", "background diversity use", "composition manifest and masks"],
        "RaySARSynthesisSkill": ["physical interpretability", "scattering-layer maps", "controlled geometry simulation"],
        "RaySARSweepSynthesisSkill": ["physical interpretability", "controlled multi-view target coverage", "scattering-layer maps by azimuth"],
        "GeoDiffSARSkill": ["high-quality sparse azimuth completion", "physical-prior consistency"],
        "GaussianSplattingCompletionSkill": ["fast sparse-view completion", "low-memory novel-view expansion"],
        "GaussianSplattingCompletionSkill": ["fast sparse-view completion", "low-memory novel-view expansion"],
        "ClassificationEvaluationSkill": ["downstream evidence", "policy memory calibration"],
    }
    return mapping.get(name, ["candidate augmentation benefit"])


def risks(name: str) -> list[str]:
    mapping = {
        "TraditionalAugmentationSkill": ["limited true geometry diversity", "over-aggressive transforms can damage SAR structure"],
        "DiffusionLoRAGenerationSkill": ["label drift", "overfit captions", "runtime and GPU cost"],
        "GANImageToImageSkill": ["mode collapse", "limited complex structure control", "label drift"],
        "StyleTransferSkill": ["over-transfer", "target structure damage"],
        "PseudocolorSkill": ["color domain shift", "not physically new SAR data", "should not be default grayscale training augmentation"],
        "BackgroundGenerationSkill": ["generated backgrounds may not match target domain", "not useful for pure target-chip classification without composition"],
        "TargetBackgroundCompositionSkill": ["auto-mask errors", "target/background mismatch", "detection annotations require validation"],
        "RaySARSynthesisSkill": ["requires RaySAR-compatible 3D scene", "slow simulation", "real-synthetic domain gap"],
        "RaySARSweepSynthesisSkill": ["requires a valid 3D model", "runtime scales with view count", "real-synthetic domain gap"],
        "GeoDiffSARSkill": ["requires 3D/physical prior", "high runtime and integration cost"],
        "GaussianSplattingCompletionSkill": ["weaker SAR physics than GeoDiff-SAR", "reconstruction artifacts"],
        "GaussianSplattingCompletionSkill": ["weaker SAR physics than GeoDiff-SAR", "reconstruction artifacts"],
        "ClassificationEvaluationSkill": ["expensive", "needs leakage-safe split"],
    }
    return mapping.get(name, ["requires validation"])


def render_augmentation_plan_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Augmentation Plan",
        "",
        f"- Selected plan: `{report.get('selected_plan_id')}`",
        f"- Selected skill: `{report.get('selected_skill')}`",
        f"- Selected recipe task: `{report.get('selected_recipe_task')}`",
        f"- Planning evidence: `{report.get('planning_evidence_summary')}`",
        "",
        "## Ranked Plans",
        "",
    ]
    for idx, item in enumerate(report.get("ranked_plans") or [], start=1):
        lines.extend(
            [
                f"### {idx}. {item.get('plan_id')}",
                "",
                f"- Skill: `{item.get('skill')}` ({item.get('status')})",
                f"- Planner score: {item.get('planner_score')}",
                f"- Recipe task: `{item.get('recipe_task')}`",
                f"- Params: `{item.get('params')}`",
                f"- Rationale: {item.get('rationale')}",
                f"- Expected benefits: `{item.get('expected_benefits')}`",
                f"- Risks: `{item.get('risks')}`",
                "",
            ]
        )
    return "\n".join(lines)


def safe_plan_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "augmentation_plan"


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


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

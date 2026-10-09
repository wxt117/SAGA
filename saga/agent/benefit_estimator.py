from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saga.agent.skill_registry import built_in_skill_specs
from saga.core.config import save_json, save_text


SKILL_EFFECT_CARD_VERSION = "saga_skill_effect_cards_v1"
DATASET_NEED_PROFILE_VERSION = "saga_dataset_need_profile_v1"
PLAN_UTILITY_VERSION = "saga_plan_utility_v1"


@dataclass
class SkillEffectCard:
    name: str
    status: str
    main_goal: str
    supported_tasks: list[str] = field(default_factory=list)
    input_requirements: dict[str, Any] = field(default_factory=dict)
    required_metadata: list[str] = field(default_factory=list)
    addresses_dataset_needs: dict[str, str] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    risks: dict[str, str] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    best_use_cases: list[str] = field(default_factory=list)
    bad_use_cases: list[str] = field(default_factory=list)
    controllable_params: list[str] = field(default_factory=list)
    expected_benefit_prior: dict[str, str] = field(default_factory=dict)
    evaluation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def built_in_skill_effect_cards() -> list[SkillEffectCard]:
    registry = {spec.name: spec for spec in built_in_skill_specs()}

    def status(name: str) -> str:
        return registry.get(name).status if name in registry else "planned"

    def cost(name: str, runtime: str = "medium", gpu_required: bool = False) -> dict[str, Any]:
        if name in registry:
            return dict(registry[name].cost)
        return {"runtime": runtime, "gpu_required": gpu_required}

    return [
        SkillEffectCard(
            name="DatasetProfileReportSkill",
            status=status("DatasetProfileReportSkill"),
            main_goal="Record raw, semantic, and validated dataset profile artifacts for traceability.",
            supported_tasks=["all"],
            input_requirements={"dataset_profile": True, "validated_profile": False},
            addresses_dataset_needs={"metadata_completeness": "medium", "traceability": "very_high"},
            strengths=["Makes planning evidence explicit.", "Keeps user data interpretation auditable."],
            weaknesses=["Does not change the data distribution by itself."],
            risks={"false_confidence": "low"},
            cost=cost("DatasetProfileReportSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"reproducibility": "very_high", "downstream_accuracy": "none"},
            evaluation=["artifact_presence", "profile_consistency"],
        ),
        SkillEffectCard(
            name="FileSelectionSkill",
            status=status("FileSelectionSkill"),
            main_goal="Select samples by validated metadata filters before applying a downstream skill.",
            supported_tasks=["classification", "style_transfer", "object_detection", "segmentation"],
            input_requirements={"content_images": True, "validated_dataset_config": True},
            required_metadata=["filter_fields"],
            addresses_dataset_needs={"metadata_completeness": "medium", "controlled_subset_selection": "very_high"},
            strengths=["Prevents natural-language filters from being applied to unvalidated metadata."],
            weaknesses=["Cannot recover missing metadata."],
            risks={"empty_selection": "medium"},
            cost=cost("FileSelectionSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"reproducibility": "high", "downstream_accuracy": "none"},
            evaluation=["selected_count", "filter_field_coverage"],
        ),
        SkillEffectCard(
            name="TraditionalAugmentationSkill",
            status=status("TraditionalAugmentationSkill"),
            main_goal="Apply deterministic SAR-aware image transforms as low-cost regularization.",
            supported_tasks=["classification", "object_detection", "segmentation"],
            input_requirements={"images": True},
            addresses_dataset_needs={
                "small_sample_regularization": "medium",
                "robustness": "medium",
                "sample_count": "low",
                "class_balance": "low",
                "speed_priority": "very_high",
            },
            strengths=["Easy to operate.", "Very fast.", "Good baseline for users with the simplest augmentation request."],
            weaknesses=[
                "Limited diversity.",
                "Does not create new target geometry, new scene semantics, or true viewpoint coverage.",
                "Repeated transforms can create near-duplicates rather than useful new evidence.",
            ],
            risks={"unrealistic_sar_transform": "medium", "directionality_break": "medium", "label_preservation": "medium"},
            cost=cost("TraditionalAugmentationSkill", runtime="low", gpu_required=False),
            best_use_cases=["Users need the simplest augmentation.", "Users require the fastest turnaround.", "Small classification datasets.", "Need a conservative baseline."],
            bad_use_cases=["Sparse azimuth completion where new poses are required.", "Requests for new target shapes or new scene semantics."],
            controllable_params=["flip", "rotate", "crop", "noise", "speckle", "intensity_scale"],
            expected_benefit_prior={"in_domain_accuracy": "low_to_medium", "robustness": "medium"},
            evaluation=["image_quality", "sar_artifacts", "duplicate_rate", "distribution_proxy"],
        ),
        SkillEffectCard(
            name="SARPreprocessSkill",
            status=status("SARPreprocessSkill"),
            main_goal="Normalize SAR intensity and image representation before generation or evaluation.",
            supported_tasks=["all"],
            input_requirements={"images": True},
            addresses_dataset_needs={"image_quality": "high", "metadata_completeness": "low"},
            strengths=["Reduces avoidable failures from bit-depth and dynamic-range mismatch."],
            weaknesses=["Can remove meaningful amplitude cues if over-normalized.", "Not appropriate when absolute radiometry must be preserved."],
            risks={"amplitude_distortion": "medium", "scatterer_suppression": "medium", "radiometric_distribution_shift": "medium"},
            cost=cost("SARPreprocessSkill", runtime="low", gpu_required=False),
            best_use_cases=["16-bit TIFF or inconsistent dynamic range.", "Before LoRA, style transfer, GAN, or evaluator pipelines."],
            bad_use_cases=["Absolute radiometric calibration must be preserved.", "Images are already standardized and the user wants unchanged grayscale statistics."],
            controllable_params=["normalization", "percentile_clip", "bit_depth", "resize"],
            expected_benefit_prior={"pipeline_stability": "high", "downstream_accuracy": "low_to_medium"},
            evaluation=["dynamic_range", "black_white_ratio"],
        ),
        SkillEffectCard(
            name="MetadataCaptionSkill",
            status=status("MetadataCaptionSkill"),
            main_goal="Turn validated SAR metadata into stable text captions for LoRA/diffusion training.",
            supported_tasks=["classification", "diffusion_lora_generation", "target_generation"],
            input_requirements={"images": True, "validated_dataset_config": False},
            required_metadata=["class"],
            addresses_dataset_needs={"caption_readiness": "very_high", "metadata_conditioned_generation": "high", "traceability": "high"},
            strengths=["Makes metadata-conditioned generation reproducible.", "Avoids ad hoc caption text."],
            weaknesses=["Only as good as validated metadata coverage."],
            risks={"wrong_caption_semantics": "medium"},
            cost=cost("MetadataCaptionSkill", runtime="low", gpu_required=False),
            best_use_cases=["LoRA training when txt captions are missing.", "Polarization/band/resolution conditioned generation."],
            bad_use_cases=["Metadata fields are missing or unvalidated.", "The generation task is intentionally unconditional."],
            controllable_params=["template", "mode", "overwrite"],
            expected_benefit_prior={"generation_control": "high", "pipeline_stability": "high"},
            evaluation=["caption_coverage", "metadata_field_coverage"],
        ),
        SkillEffectCard(
            name="DatasetBalancingSkill",
            status=status("DatasetBalancingSkill"),
            main_goal="Estimate class and metadata deficits before choosing an augmentation strategy.",
            supported_tasks=["classification", "target_generation"],
            input_requirements={"images": True},
            addresses_dataset_needs={"class_balance": "very_high", "sample_count": "high", "metadata_conditioned_generation": "medium"},
            strengths=["Turns vague augmentation requests into target counts.", "Supports class/polarization/azimuth-aware planning."],
            weaknesses=["Does not generate samples by itself."],
            risks={"over_balancing_small_bins": "medium"},
            cost=cost("DatasetBalancingSkill", runtime="low", gpu_required=False),
            best_use_cases=["Before LoRA/GAN/traditional augmentation.", "When class or metadata coverage is uneven."],
            bad_use_cases=["No labels or validated metadata are available.", "Bins are too sparse to estimate a meaningful deficit.", "The user goal is not distribution filling."],
            controllable_params=["fields", "target_per_bin", "max_multiplier"],
            expected_benefit_prior={"planner_decision_quality": "very_high", "downstream_accuracy": "medium"},
            evaluation=["deficit_count", "per_bin_coverage"],
        ),
        SkillEffectCard(
            name="PseudocolorSkill",
            status=status("PseudocolorSkill"),
            main_goal="Create lightweight pseudocolor SAR feature variants for visualization, manual inspection, or explicit classifier ablations.",
            supported_tasks=["visualization", "feature_transform", "classification"],
            input_requirements={"images": True},
            addresses_dataset_needs={"visualization": "very_high", "appearance_diversity": "low", "image_quality": "low"},
            strengths=["Very fast.", "Simple to evaluate as an ablation.", "Uses existing weicaise idea without heavy dependencies."],
            weaknesses=["Not physically new SAR data.", "May create color cues that do not transfer.", "Should not be a default for standard grayscale SAR training."],
            risks={"spurious_color_cues": "medium", "color_domain_shift": "high"},
            cost=cost("PseudocolorSkill", runtime="low", gpu_required=False),
            best_use_cases=["Visualization packs.", "Manual inspection.", "Classifier ablation when the model explicitly accepts RGB/pseudocolor inputs."],
            bad_use_cases=["Standard grayscale SAR training.", "The user wants to preserve physical grayscale meaning."],
            controllable_params=["colormap"],
            expected_benefit_prior={"classification_accuracy": "low", "explainability": "medium"},
            evaluation=["image_quality", "duplicate_rate", "downstream_classification_only_if_requested"],
        ),
        SkillEffectCard(
            name="DiffusionLoRAGenerationSkill",
            status=status("DiffusionLoRAGenerationSkill"),
            main_goal="Train text-caption diffusion LoRA and generate high-quality target-level SAR images for sample expansion.",
            supported_tasks=["classification", "diffusion_lora_generation", "target_generation"],
            input_requirements={"paired_image_txt_captions": True, "gpu": True},
            required_metadata=["class"],
            addresses_dataset_needs={
                "sample_count": "high",
                "class_balance": "medium",
                "appearance_diversity": "high",
                "caption_readiness": "very_high",
                "metadata_conditioned_generation": "high",
                "azimuth_coverage": "low",
            },
            strengths=["High quality.", "Works well when users have enough VRAM and no strict time limit.", "Uses existing text-caption generation projects."],
            weaknesses=["Medium speed.", "Weak pose control without ControlNet/physical priors.", "Can overfit small captioned sets."],
            risks={"label_drift": "medium", "structural_distortion": "medium", "domain_overfit": "medium", "sar_physics_instability": "medium", "mode_collapse": "medium"},
            cost=cost("DiffusionLoRAGenerationSkill", runtime="high", gpu_required=True),
            best_use_cases=["High-quality generation is more important than speed.", "Users have enough GPU memory.", "Paired or metadata-captionable image datasets.", "Need more target-like samples."],
            bad_use_cases=["Users need the fastest method.", "Very small datasets.", "Wrong or unvalidated captions.", "Strict sparse azimuth completion without pose control.", "Strict physical/geometric consistency without GeoDiff/ControlNet priors."],
            controllable_params=["model_family", "target_count", "prompt", "steps", "guidance", "seed"],
            expected_benefit_prior={
                "classification_accuracy": "medium",
                "minority_class_recall": "medium",
                "appearance_diversity": "high",
                "azimuth_generalization": "low",
            },
            evaluation=["image_quality", "distribution_proxy", "sar_artifacts", "duplicate_rate", "caption_consistency", "downstream_classification_if_claiming_task_gain"],
        ),
        SkillEffectCard(
            name="GANImageToImageSkill",
            status=status("GANImageToImageSkill"),
            main_goal="Fast GAN image-to-image generation for simple target or appearance variation.",
            supported_tasks=["classification", "generation_dataset"],
            input_requirements={"images": True, "gpu": True},
            addresses_dataset_needs={"sample_count": "medium", "appearance_diversity": "medium"},
            strengths=["Fast inference.", "Suitable for simple datasets and simple target structures.", "Useful as a simple generation baseline."],
            weaknesses=["Usually limited to simple targets and image-to-image mappings.", "Less suitable for complex structure or high-fidelity controllable generation."],
            risks={"mode_collapse": "high", "label_drift": "medium", "training_instability": "medium"},
            cost=cost("GANImageToImageSkill", runtime="medium", gpu_required=True),
            best_use_cases=["MSTAR-like simple targets.", "Simple user datasets.", "Fast image-to-image generation.", "Cheap comparison against diffusion methods."],
            bad_use_cases=["Complex text or multi-condition control.", "High-fidelity SAR generation.", "Fine-grained pose completion or physically controlled generation."],
            controllable_params=["translation_strength", "checkpoint", "seed"],
            expected_benefit_prior={"classification_accuracy": "low_to_medium", "diversity": "medium"},
            evaluation=["image_quality", "duplicate_rate", "downstream_classification"],
        ),
        SkillEffectCard(
            name="StyleTransferSkill",
            status=status("StyleTransferSkill"),
            main_goal="Transfer SAR feature/style characteristics across payloads or domains.",
            supported_tasks=["classification", "domain_adaptation", "style_transfer"],
            input_requirements={"content_images": True, "style_images": True, "gpu": True},
            addresses_dataset_needs={"domain_shift": "high", "payload_shift": "high", "appearance_diversity": "medium"},
            strengths=["Directly addresses cross-platform/cross-payload feature transfer.", "Already wrapped as an executable skill."],
            weaknesses=["May damage target structure.", "Not a substitute for true viewpoint completion."],
            risks={"target_structure_damage": "medium", "over_transfer": "high", "label_preservation": "medium", "label_pollution": "medium"},
            cost=cost("StyleTransferSkill", runtime="medium", gpu_required=True),
            best_use_cases=["Users explicitly need cross-platform or cross-payload transfer.", "Style examples are available."],
            bad_use_cases=["Content and style domains have incompatible semantics.", "The task needs geometry changes.", "When class geometry must remain strictly unchanged and no structure evaluator is available."],
            controllable_params=["style_strength", "content_weight", "preprocess_mode"],
            expected_benefit_prior={
                "cross_domain_generalization": "high",
                "in_domain_accuracy": "low_to_medium",
                "azimuth_generalization": "low",
            },
            evaluation=["structure_preservation", "style_strength", "downstream_cross_domain_test"],
        ),
        SkillEffectCard(
            name="GeoDiffSARSkill",
            status=status("GeoDiffSARSkill"),
            main_goal="Use diffusion plus 3D/physical priors, LoRA, and ControlNet for high-quality sparse-azimuth target completion.",
            supported_tasks=["classification", "sparse_azimuth_completion", "target_generation"],
            input_requirements={"target_images": True, "azimuth_metadata": True, "physical_prior": True, "gpu": True},
            required_metadata=["class", "azimuth_deg"],
            addresses_dataset_needs={
                "azimuth_coverage": "very_high",
                "geometric_consistency": "high",
                "sample_count": "medium",
                "appearance_diversity": "medium",
            },
            strengths=["Best aligned with high-quality sparse azimuth completion.", "Uses task-specific physical/geometric priors."],
            weaknesses=["Complex adapter.", "Requires 3D model/physical prior assets.", "Expensive in time and VRAM."],
            risks={"physical_prior_mismatch": "medium", "implementation_complexity": "high", "runtime_cost": "high"},
            cost={**cost("GeoDiffSARSkill", runtime="high", gpu_required=True), "recommended_vram_gb": "32+"},
            best_use_cases=["User explicitly requests sparse azimuth completion.", "A 3D model or physical prior is available.", "Quality matters more than time or VRAM."],
            bad_use_cases=["No azimuth metadata.", "No 3D/physical prior.", "Complex large scenes.", "User needs fast or low-memory augmentation."],
            controllable_params=["target_azimuths", "control_strength", "lora_weight", "guidance", "seed"],
            expected_benefit_prior={
                "azimuth_generalization": "very_high",
                "classification_accuracy": "high",
                "cross_domain_generalization": "medium",
            },
            evaluation=["azimuth_coverage", "target_structure_consistency", "downstream_classification_by_azimuth"],
        ),
        SkillEffectCard(
            name="GaussianSplattingCompletionSkill",
            status=status("GaussianSplattingCompletionSkill"),
            main_goal="Lightweight sparse-view target reconstruction and novel-view completion.",
            supported_tasks=["classification", "sparse_azimuth_completion", "target_generation"],
            input_requirements={"sparse_views": True, "gpu": True, "project_root": "myproject/SAR GS V1"},
            required_metadata=["class", "azimuth_deg"],
            addresses_dataset_needs={"azimuth_coverage": "high", "sample_count": "medium", "speed_value": "very_high"},
            strengths=["Lightweight.", "Very fast.", "Low VRAM compared with GeoDiff-SAR.", "Useful before heavier GeoDiff-SAR refinement."],
            weaknesses=["Sparse-angle completion ability is weaker than GeoDiff-SAR.", "May miss SAR scattering physics.", "Quality depends on sparse-view reconstruction."],
            risks={"reconstruction_artifact": "medium", "scattering_physics_mismatch": "medium"},
            cost={**cost("GaussianSplattingCompletionSkill", runtime="medium", gpu_required=True), "typical_vram_gb": 2},
            best_use_cases=["Users need fast, lightweight sparse-angle completion.", "Sparse views are available.", "VRAM is limited."],
            bad_use_cases=["Too few usable views.", "Non-rigid targets.", "No view metadata or unusable sparse-view geometry."],
            controllable_params=["target_views", "render_resolution", "regularization"],
            expected_benefit_prior={"azimuth_generalization": "high", "speed_value": "very_high"},
            evaluation=["azimuth_coverage", "reconstruction_artifacts", "downstream_classification_by_azimuth"],
        ),
        SkillEffectCard(
            name="ModelToPOVSceneCompilerSkill",
            status=status("ModelToPOVSceneCompilerSkill"),
            main_goal="Convert common 3D target assets into RaySAR-ready POV-Ray scenes.",
            supported_tasks=["physics_simulation", "target_generation", "raysar_synthesis"],
            input_requirements={"model_file": True, "simulation_geometry": True},
            required_metadata=[],
            addresses_dataset_needs={"geometric_consistency": "high", "physics_interpretability": "high"},
            strengths=["Deterministic.", "No GPU required.", "Lets users provide models instead of hand-written POV scenes."],
            weaknesses=["Material model is a coarse reflectivity proxy.", "Large meshes can make downstream RaySAR rendering slow."],
            risks={"domain_gap": "medium", "material_mismatch": "medium", "mesh_scale_error": "medium"},
            cost=cost("ModelToPOVSceneCompilerSkill", runtime="low_to_medium", gpu_required=False),
            best_use_cases=["User has OBJ/STL/PLY/GLB target assets.", "Need RaySAR physical simulation without manual POV-Ray authoring."],
            bad_use_cases=["No 3D model or scene parameters are available.", "Need fully electromagnetic material modeling."],
            controllable_params=["incidence_angle_deg", "depression_angle_deg", "azimuth_deg", "target_extent_m", "sensor_plane_m"],
            expected_benefit_prior={"ray_sar_usability": "high", "classification_accuracy": "indirect"},
            evaluation=["scene_compile_success", "ray_sar_render_success", "domain_gap"],
        ),
        SkillEffectCard(
            name="RaySARSynthesisSkill",
            status=status("RaySARSynthesisSkill"),
            main_goal="Generate physically interpretable SAR reflection maps from RaySAR/POV-Ray 3D scenes.",
            supported_tasks=["physics_simulation", "raysar_synthesis", "physical_analysis"],
            input_requirements={"pov_scene_or_compiled_scene": True, "ray_sar_parameters": True, "adapted_povray_or_contributions": True},
            required_metadata=[],
            addresses_dataset_needs={"geometric_consistency": "high", "azimuth_coverage": "medium", "appearance_diversity": "medium"},
            strengths=["Physics-based and interpretable.", "Can expose single/double/all-bounce reflection layers.", "Useful when 3D scene assets exist."],
            weaknesses=["Needs RaySAR-compatible scenes and adapted POV-Ray output.", "Not a direct replacement for data-driven high-fidelity image generation."],
            risks={"scene_asset_mismatch": "high", "domain_gap": "high", "integration_complexity": "high"},
            cost=cost("RaySARSynthesisSkill", runtime="medium_to_high", gpu_required=False),
            best_use_cases=["Synthetic SAR ablations with known 3D geometry.", "Scattering-layer analysis.", "Pose/view sweeps from compiled or hand-authored POV scenes."],
            bad_use_cases=["User only has ordinary SAR images and no 3D/POV scene.", "Need quick generic target augmentation without physical assets."],
            controllable_params=["pov_scene", "compiled_scene", "azimuth_angle", "bounce_level", "range_direction", "pixel_spacing", "clip_mode"],
            expected_benefit_prior={"geometric_interpretability": "high", "physics_analysis": "high", "classification_accuracy": "none"},
            evaluation=["render_success", "image_quality_with_physics_map_tolerance", "sar_artifacts", "domain_gap"],
        ),
        SkillEffectCard(
            name="RaySARSweepSynthesisSkill",
            status=status("RaySARSweepSynthesisSkill"),
            main_goal="Generate physically interpretable multi-view SAR reflection maps from a 3D model.",
            supported_tasks=["physics_simulation", "raysar_synthesis", "physical_analysis", "sparse_azimuth_completion"],
            input_requirements={"model_file": True, "azimuth_sweep_or_values": True, "simulation_geometry": True},
            required_metadata=[],
            addresses_dataset_needs={
                "geometric_consistency": "high",
                "azimuth_coverage": "high",
                "appearance_diversity": "medium",
            },
            strengths=[
                "Directly supports natural-language view sweeps such as 0 to 350 every 10 degrees.",
                "No GPU required.",
                "Keeps per-view compile/render provenance and exports caption sidecars.",
            ],
            weaknesses=[
                "Slow for many views.",
                "Physical interpretability does not remove real-synthetic domain gap.",
                "Material model is still a coarse RaySAR/POV-Ray proxy.",
            ],
            risks={"domain_gap": "medium", "mesh_scale_error": "medium", "long_runtime": "high"},
            cost=cost("RaySARSweepSynthesisSkill", runtime="high", gpu_required=False),
            best_use_cases=[
                "User has a 3D model and asks for a controlled azimuth/aspect sweep.",
                "Sparse-view SAR target augmentation needs interpretable physical simulation.",
                "Need scattering-layer maps over many target poses.",
            ],
            bad_use_cases=[
                "No 3D model or view geometry is available.",
                "User needs very fast generic image augmentation.",
                "User needs photorealistic learned generation rather than interpretable simulation.",
            ],
            controllable_params=["azimuth_sweep", "azimuth_values", "incidence_angle_deg", "target_extent_m", "width", "height"],
            expected_benefit_prior={
                "geometric_interpretability": "very_high",
                "azimuth_physics_coverage": "medium_to_high",
                "classification_accuracy": "none",
            },
            evaluation=["render_success_rate", "image_quality_with_physics_map_tolerance", "sar_artifacts", "per_angle_coverage", "domain_gap"],
        ),
        SkillEffectCard(
            name="ClassRebalanceAugmentationSkill",
            status=status("ClassRebalanceAugmentationSkill"),
            main_goal="Map minority-class deficits to concrete augmentation targets.",
            supported_tasks=["classification", "target_generation"],
            input_requirements={"balance_plan": True},
            addresses_dataset_needs={"class_balance": "very_high", "sample_count": "high"},
            strengths=["Keeps generation focused on downstream imbalance instead of arbitrary sample count."],
            weaknesses=["Policy skill only until connected to concrete generators."],
            risks={"minority_overfit": "medium"},
            cost=cost("ClassRebalanceAugmentationSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"minority_class_recall": "high", "classification_accuracy": "medium"},
            evaluation=["per_class_recall", "macro_f1"],
        ),
        SkillEffectCard(
            name="AzimuthCoverageCompletionSkill",
            status=status("AzimuthCoverageCompletionSkill"),
            main_goal="Plan missing azimuth/view coverage for SAR target recognition.",
            supported_tasks=["classification", "sparse_azimuth_completion"],
            input_requirements={"azimuth_metadata": True},
            required_metadata=["azimuth_deg"],
            addresses_dataset_needs={"azimuth_coverage": "very_high", "sample_count": "medium"},
            strengths=["SAR-specific target augmentation policy.", "Can route to GeoDiff-SAR or Gaussian splatting depending on quality/speed constraints."],
            weaknesses=["Requires reliable angle metadata."],
            risks={"wrong_pose_semantics": "high"},
            cost=cost("AzimuthCoverageCompletionSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"azimuth_generalization": "very_high", "classification_accuracy": "high"},
            evaluation=["per_azimuth_accuracy", "azimuth_coverage"],
        ),
        SkillEffectCard(
            name="PolarizationConditionedGenerationSkill",
            status=status("PolarizationConditionedGenerationSkill"),
            main_goal="Plan polarization-aware generation or filtering for full-polarization SAR datasets.",
            supported_tasks=["classification", "diffusion_lora_generation", "target_generation"],
            input_requirements={"polarization_metadata": True},
            required_metadata=["polarization"],
            addresses_dataset_needs={"metadata_conditioned_generation": "very_high", "class_balance": "medium"},
            strengths=["Turns requests like HH/HV/VV or non-pauli generation into explicit conditions."],
            weaknesses=["Needs validated polarization parsing."],
            risks={"condition_label_mismatch": "medium"},
            cost=cost("PolarizationConditionedGenerationSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"conditioned_generation_control": "high"},
            evaluation=["per_polarization_accuracy", "caption_consistency"],
        ),
        SkillEffectCard(
            name="HardSampleMiningSkill",
            status=status("HardSampleMiningSkill"),
            main_goal="Use downstream classifier errors to select the next augmentation targets.",
            supported_tasks=["classification", "classification_evaluation"],
            input_requirements={"per_sample_predictions": True},
            addresses_dataset_needs={"downstream_evidence": "high", "class_balance": "medium"},
            strengths=["Closes the loop from evaluator evidence back into planning."],
            weaknesses=["Requires a previous benchmark run."],
            risks={"overfitting_to_validation": "medium"},
            cost=cost("HardSampleMiningSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"repair_signal": "high", "policy_calibration": "high"},
            evaluation=["error_rate_by_class", "error_rate_by_metadata"],
        ),
        SkillEffectCard(
            name="BackgroundGenerationSkill",
            status=status("BackgroundGenerationSkill"),
            main_goal="Generate SAR background images directly when users need backgrounds; SAGA does not ask users to train background models.",
            supported_tasks=["object_detection", "background_generation", "target_background_composition"],
            input_requirements={"control_condition": True, "gpu": True},
            addresses_dataset_needs={"background_diversity": "very_high", "scene_diversity": "high", "domain_shift": "medium"},
            strengths=["Can increase scene diversity without requiring users to train background models.", "Useful as a direct background asset generator."],
            weaknesses=["Not central for target-only classification.", "Needs composition policy to affect detection data."],
            risks={"background_artifact": "medium", "scene_label_mismatch": "medium"},
            cost=cost("BackgroundGenerationSkill", runtime="high", gpu_required=True),
            best_use_cases=["Detection datasets with limited backgrounds.", "Scene composition workflows."],
            bad_use_cases=["Pure target classification without scene context."],
            controllable_params=["segment_condition", "prompt", "scene_type", "seed"],
            expected_benefit_prior={"detection_recall": "medium_to_high", "classification_accuracy": "low"},
            evaluation=["background_quality", "composition_compatibility", "downstream_detection"],
        ),
        SkillEffectCard(
            name="TargetBackgroundCompositionSkill",
            status=status("TargetBackgroundCompositionSkill"),
            main_goal="Fuse SAR target chips into background scenes when the user explicitly needs synthetic scene/data composition.",
            supported_tasks=["object_detection", "classification", "target_background_composition"],
            input_requirements={"target_bank": True, "background_bank": True, "masks": False},
            addresses_dataset_needs={"small_object_recall": "high", "background_diversity": "high", "class_balance": "medium"},
            strengths=["Directly supports target-background composition.", "CPU-only and deterministic.", "Supports feather, Laplacian pyramid, and Poisson-style blending."],
            weaknesses=["Auto masks are heuristic.", "Detection annotations and placement policy still need validation."],
            risks={"annotation_error": "high", "target_background_mismatch": "medium"},
            cost=cost("TargetBackgroundCompositionSkill", runtime="medium", gpu_required=False),
            best_use_cases=["Users explicitly request composition.", "Detection and scene-level augmentation.", "Background bank exists or can be generated."],
            bad_use_cases=["No reliable target masks or placement rules."],
            controllable_params=["placement_policy", "blend_mode", "mask_mode", "target_count", "scale_range"],
            expected_benefit_prior={"small_object_recall": "high", "detection_map": "medium_to_high"},
            evaluation=["image_quality", "composition_quality", "annotation_validity", "downstream_detection"],
        ),
        SkillEffectCard(
            name="ClassificationEvaluationSkill",
            status=status("ClassificationEvaluationSkill"),
            main_goal="Train/evaluate a classification model to measure downstream benefit of augmented data.",
            supported_tasks=["classification", "classification_evaluation"],
            input_requirements={"train_dataset": True, "val_or_test_dataset": True, "gpu": True},
            addresses_dataset_needs={"downstream_evidence": "very_high", "policy_calibration": "very_high"},
            strengths=["Turns SAGA from generation workflow into task-benefit decision system."],
            weaknesses=["Expensive.", "Needs controlled splits and reproducible training config."],
            risks={"benchmark_leakage": "high", "overfitting_to_validation": "medium"},
            cost=cost("ClassificationEvaluationSkill", runtime="high", gpu_required=True),
            best_use_cases=["Comparing augmentation recipes.", "Calibrating policy memory."],
            bad_use_cases=["No held-out validation/test split."],
            controllable_params=["model", "epochs", "split_policy", "metrics"],
            expected_benefit_prior={"decision_quality": "very_high"},
            evaluation=["accuracy", "macro_f1", "per_class_recall", "confusion_matrix"],
        ),
        SkillEffectCard(
            name="PerMetadataSliceEvaluationSkill",
            status=status("PerMetadataSliceEvaluationSkill"),
            main_goal="Expose whether augmentation helps or hurts specific SAR metadata slices.",
            supported_tasks=["classification", "classification_evaluation"],
            input_requirements={"validated_dataset_config": True},
            addresses_dataset_needs={"downstream_evidence": "very_high", "metadata_conditioned_generation": "medium"},
            strengths=["Catches regressions hidden by aggregate accuracy.", "Especially useful for azimuth, polarization, band, and resolution slices."],
            weaknesses=["Needs per-sample predictions for true slice accuracy."],
            risks={"small_slice_noise": "medium"},
            cost=cost("PerMetadataSliceEvaluationSkill", runtime="low", gpu_required=False),
            best_use_cases=["After classification evaluation.", "Before claiming angle/polarization generalization."],
            controllable_params=["fields", "predictions_csv"],
            expected_benefit_prior={"decision_quality": "very_high", "paper_defensibility": "high"},
            evaluation=["per_slice_accuracy", "per_slice_count"],
        ),
        SkillEffectCard(
            name="LeakageCheckSkill",
            status=status("LeakageCheckSkill"),
            main_goal="Prevent benchmark leakage between original, augmented, and validation datasets.",
            supported_tasks=["classification", "classification_evaluation", "all"],
            input_requirements={"baseline_dataset": True},
            addresses_dataset_needs={"downstream_evidence": "very_high", "traceability": "high"},
            strengths=["Cheap and important for credible downstream claims."],
            weaknesses=["Near-duplicate semantic leakage still needs perceptual checks."],
            risks={"false_negative_for_heavily_transformed_duplicates": "medium"},
            cost=cost("LeakageCheckSkill", runtime="low", gpu_required=False),
            best_use_cases=["Before downstream evaluator.", "Before promoting recipe to memory."],
            expected_benefit_prior={"benchmark_validity": "very_high"},
            evaluation=["hash_overlap", "stem_overlap"],
        ),
        SkillEffectCard(
            name="DuplicateNearDuplicateSkill",
            status=status("DuplicateNearDuplicateSkill"),
            main_goal="Detect memorization and repeated generated samples before export or evaluation.",
            supported_tasks=["classification", "target_generation", "all"],
            input_requirements={"generated_images": True},
            addresses_dataset_needs={"downstream_evidence": "medium", "appearance_diversity": "high", "image_quality": "medium"},
            strengths=["Catches mode collapse and overfit-like generated batches."],
            weaknesses=["Perceptual hashes are a proxy, not a semantic SAR duplicate detector."],
            risks={"false_positive_for_similar_targets": "medium"},
            cost=cost("DuplicateNearDuplicateSkill", runtime="low", gpu_required=False),
            best_use_cases=["After GAN/LoRA generation.", "Before export and classification evaluation."],
            controllable_params=["hamming_threshold", "hash_size"],
            expected_benefit_prior={"quality_screening": "high", "decision_quality": "medium"},
            evaluation=["near_duplicate_pair_count"],
        ),
        SkillEffectCard(
            name="ObjectDetectionEvaluationSkill",
            status=status("ObjectDetectionEvaluationSkill"),
            main_goal="Evaluate detection recipes with mAP, recall, and class-specific AP.",
            supported_tasks=["object_detection"],
            input_requirements={"detection_dataset": True, "val_or_test_dataset": True, "gpu": True},
            addresses_dataset_needs={"downstream_evidence": "very_high", "small_object_recall": "very_high"},
            strengths=["Needed before claiming detection/background synthesis gains."],
            weaknesses=["Requires stable annotations and training config."],
            risks={"benchmark_leakage": "high"},
            cost=cost("ObjectDetectionEvaluationSkill", runtime="high", gpu_required=True),
            expected_benefit_prior={"decision_quality": "very_high"},
            evaluation=["mAP", "AP_per_class", "small_object_recall"],
        ),
        SkillEffectCard(
            name="SegmentationEvaluationSkill",
            status=status("SegmentationEvaluationSkill"),
            main_goal="Evaluate segmentation augmentation recipes with mask-aware metrics.",
            supported_tasks=["segmentation"],
            input_requirements={"segmentation_dataset": True, "val_or_test_dataset": True, "gpu": True},
            addresses_dataset_needs={"downstream_evidence": "very_high"},
            strengths=["Needed before claiming segmentation gains."],
            weaknesses=["Requires mask consistency and stable benchmark."],
            risks={"mask_misalignment": "high"},
            cost=cost("SegmentationEvaluationSkill", runtime="high", gpu_required=True),
            expected_benefit_prior={"decision_quality": "very_high"},
            evaluation=["mIoU", "Dice", "boundary_f1"],
        ),
        SkillEffectCard(
            name="QualityEvaluationSkill",
            status=status("QualityEvaluationSkill"),
            main_goal="Catch obvious generated image failures before export or downstream training.",
            supported_tasks=["all"],
            input_requirements={"generated_images": True},
            addresses_dataset_needs={"image_quality": "very_high", "downstream_evidence": "low"},
            strengths=["Cheap observer gate.", "Good for bounded repair triggers."],
            weaknesses=["Not a task-performance metric."],
            risks={"false_accept": "medium"},
            cost=cost("QualityEvaluationSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"pipeline_stability": "high"},
            evaluation=["black_ratio", "white_ratio", "dynamic_range", "image_count"],
        ),
        SkillEffectCard(
            name="DistributionEvaluationSkill",
            status=status("DistributionEvaluationSkill"),
            main_goal="Compare generated samples against the reference image distribution with quick FID-style metrics.",
            supported_tasks=["all"],
            input_requirements={"generated_images": True, "reference_images": True},
            addresses_dataset_needs={"image_quality": "medium", "downstream_evidence": "medium", "traceability": "high"},
            strengths=["Fast numerical proxy after generation.", "Uses sampled reference/generated images.", "Can run by default before expensive downstream task evaluation."],
            weaknesses=[
                "FID-style metrics are not direct task-performance evidence.",
                "Small sample counts make absolute scores noisy.",
                "For style transfer or intentional domain shift, the score describes shift magnitude rather than pass/fail quality.",
            ],
            risks={"false_confidence": "medium", "metric_misinterpretation": "medium"},
            cost=cost("DistributionEvaluationSkill", runtime="medium", gpu_required=False),
            best_use_cases=["Default post-generation validation.", "Users want quick numeric evidence without downstream training."],
            bad_use_cases=["Making final claims about classifier/detector improvement.", "Using distribution closeness as a hard gate for cross-domain/style-transfer outputs.", "Very small reference sets."],
            controllable_params=["reference_sample_limit", "generated_sample_limit", "standard_fid", "image_size"],
            expected_benefit_prior={"quality_screening": "high", "decision_quality": "medium"},
            evaluation=["fid_pytorch", "sar_fid_lite", "histogram_jsd", "mmd_rbf", "generated_diversity"],
        ),
        SkillEffectCard(
            name="SARArtifactEvaluationSkill",
            status=status("SARArtifactEvaluationSkill"),
            main_goal="Catch SAR-specific visual artifacts before export or downstream training.",
            supported_tasks=["all"],
            input_requirements={"generated_images": True},
            addresses_dataset_needs={"image_quality": "very_high", "sar_artifact_screening": "very_high"},
            strengths=["Detects stripe energy, smooth background gradients, target compactness, fragmentation, and centering."],
            weaknesses=["Heuristic observer only; not a downstream task metric."],
            risks={"false_accept": "medium", "false_reject": "medium"},
            cost=cost("SARArtifactEvaluationSkill", runtime="low", gpu_required=False),
            best_use_cases=["Post-generation SAR quality screening.", "Repair triggers before exporting generated samples."],
            controllable_params=["sample_limit", "image_size", "artifact_thresholds"],
            expected_benefit_prior={"quality_screening": "high", "repair_signal": "high"},
            evaluation=["stripe_score", "gradient_score", "target_compactness", "target_fragment_count", "target_center_offset"],
        ),
        SkillEffectCard(
            name="RepairPolicySkill",
            status=status("RepairPolicySkill"),
            main_goal="Convert observer triggers into bounded repair suggestions.",
            supported_tasks=["all"],
            input_requirements={"quality_report": True},
            addresses_dataset_needs={"pipeline_stability": "high"},
            strengths=["Avoids open-ended ReAct-style uncontrolled retries."],
            weaknesses=["Current MVP records repair policy but does not auto-rerun."],
            risks={"under_repair": "medium"},
            cost=cost("RepairPolicySkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"reproducibility": "high"},
            evaluation=["trigger_resolution", "repair_trace"],
        ),
        SkillEffectCard(
            name="ExportDatasetSkill",
            status=status("ExportDatasetSkill"),
            main_goal="Package generated data with provenance, manifest, and dataset card.",
            supported_tasks=["all"],
            input_requirements={"generated_outputs": True},
            addresses_dataset_needs={"traceability": "very_high"},
            strengths=["Makes generated data usable and auditable."],
            weaknesses=["Does not judge downstream value."],
            risks={"exporting_bad_samples_without_quality_gate": "medium"},
            cost=cost("ExportDatasetSkill", runtime="low", gpu_required=False),
            expected_benefit_prior={"reproducibility": "very_high"},
            evaluation=["manifest_validity", "provenance_completeness"],
        ),
    ]


def skill_effect_cards_payload() -> dict[str, Any]:
    cards = [card.to_dict() for card in built_in_skill_effect_cards()]
    return {
        "schema_version": SKILL_EFFECT_CARD_VERSION,
        "summary": {
            "total": len(cards),
            "executable": len([card for card in cards if card["status"] == "executable"]),
            "planned": len([card for card in cards if card["status"] == "planned"]),
        },
        "cards": cards,
    }


def build_benefit_context(
    raw_profile: dict[str, Any],
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    intent_spec: dict[str, Any],
    memory_context: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    need_profile = build_dataset_need_profile(
        raw_profile=raw_profile,
        validated_profile=validated_profile,
        bridge_report=bridge_report,
        intent_spec=intent_spec,
    )
    skill_cards = skill_effect_cards_payload()
    skill_scores = score_skill_cards(
        cards=skill_cards["cards"],
        need_profile=need_profile,
        intent_spec=intent_spec,
        bridge_report=bridge_report,
        memory_context=memory_context or {},
    )
    context = {
        "schema_version": "saga_benefit_context_v1",
        "dataset_need_profile": need_profile,
        "skill_effect_cards": skill_cards,
        "skill_utility": skill_scores,
        "memory_context": memory_context or {},
    }
    if output_dir:
        write_benefit_context(output_dir, context)
    return context


def build_dataset_need_profile(
    raw_profile: dict[str, Any],
    validated_profile: dict[str, Any],
    bridge_report: dict[str, Any],
    intent_spec: dict[str, Any],
) -> dict[str, Any]:
    intent = intent_spec.get("intent", {})
    task = intent.get("task") or "classification"
    total_images = int(raw_profile.get("images", {}).get("total_images") or 0)
    sidecar_counts = raw_profile.get("sidecars", {}).get("same_stem_sidecar_counts") or {}
    txt_sidecars = int(sidecar_counts.get(".txt") or sidecar_counts.get("txt") or 0)
    caption_ratio = ratio(txt_sidecars, total_images)
    validation = bridge_report.get("validation") or {}
    coverage = validation.get("field_coverage") or validated_profile.get("field_coverage") or {}
    request_text = str(intent_spec.get("raw_text") or "").lower()
    image_probe = raw_profile.get("images", {}).get("probe_summary", {})
    mode_counts = image_probe.get("mode_counts") or {}
    size_counts = image_probe.get("size_counts") or {}

    needs: dict[str, Any] = {}
    add_need(
        needs,
        "sample_count",
        sample_count_severity(total_images),
        [f"total_images={total_images}", "classification/generation tasks benefit from enough target diversity"],
    )
    add_need(
        needs,
        "caption_readiness",
        1.0 - caption_ratio,
        [f"txt_sidecar_ratio={caption_ratio:.3f}", "text-caption generation needs paired image/txt data"],
        inverse=True,
    )
    azimuth_coverage = max(field_coverage(coverage, "azimuth_deg"), field_coverage(coverage, "azimuth_angle"), field_coverage(coverage, "azimuth"))
    add_need(
        needs,
        "azimuth_coverage",
        1.0 - azimuth_coverage,
        [f"azimuth_field_coverage={azimuth_coverage:.3f}", "pose/azimuth completion needs validated angle metadata"],
    )
    depression_coverage = max(
        field_coverage(coverage, "depression_angle_deg"),
        field_coverage(coverage, "depression_angle"),
        field_coverage(coverage, "incidence_angle_deg"),
    )
    add_need(
        needs,
        "view_angle_coverage",
        1.0 - depression_coverage,
        [f"depression_or_incidence_coverage={depression_coverage:.3f}"],
    )
    band_coverage = field_coverage(coverage, "band")
    polarization_coverage = field_coverage(coverage, "polarization")
    resolution_coverage = max(field_coverage(coverage, "resolution_m"), field_coverage(coverage, "resolution"))
    metadata_missing = []
    for name, value in {
        "azimuth": azimuth_coverage,
        "depression_or_incidence": depression_coverage,
        "band": band_coverage,
        "polarization": polarization_coverage,
        "resolution": resolution_coverage,
    }.items():
        if value < 0.5:
            metadata_missing.append(name)
    add_need(
        needs,
        "metadata_completeness",
        clamp(len(metadata_missing) / 5),
        [f"low_coverage_fields={metadata_missing}"],
    )
    goals = set(intent.get("goals") or [])
    if "polarization_conditioned_generation" in goals or "极化" in request_text or "polarization" in request_text:
        add_need(
            needs,
            "metadata_conditioned_generation",
            0.9 if polarization_coverage >= 0.95 else 0.65,
            [
                "request asks for polarization-aware generation",
                f"polarization_field_coverage={polarization_coverage:.3f}",
            ],
        )
    else:
        add_need(
            needs,
            "metadata_conditioned_generation",
            0.2,
            ["no explicit metadata-conditioned generation request"],
        )
    domain_keywords = ["style", "风格", "迁移", "跨", "载荷", "payload", "domain"]
    domain_shift = 0.75 if any(keyword in request_text for keyword in domain_keywords) or task == "style_transfer" else 0.25
    add_need(
        needs,
        "domain_shift",
        domain_shift,
        ["request/task indicates style transfer or cross-domain adaptation"] if domain_shift > 0.5 else ["no explicit domain-shift request"],
    )
    physics_excluded = excludes_physics_simulation_request(request_text)
    physics_simulation_need = (
        0.05
        if physics_excluded
        else 0.85
        if task == "raysar_synthesis" or any(keyword in request_text for keyword in ["raysar", "povray", "pov-ray", "物理仿真", "射线追踪"])
        else 0.15
    )
    add_need(
        needs,
        "geometric_consistency",
        physics_simulation_need,
        ["request explicitly excludes physics/RaySAR simulation"]
        if physics_excluded
        else ["request asks for physics/interpretable model-based simulation"]
        if physics_simulation_need > 0.5
        else ["no explicit model-based simulation request"],
    )
    detection_task = task in {"object_detection", "segmentation", "target_background_composition", "background_generation"}
    add_need(
        needs,
        "background_diversity",
        0.65 if detection_task else 0.15,
        ["scene/background diversity matters for detection or composition"] if detection_task else ["target-only classification/generation has weaker background need"],
    )
    add_need(
        needs,
        "class_balance",
        0.35,
        ["class distribution is not fully estimated in the current profiler"],
    )
    add_need(
        needs,
        "image_quality",
        image_quality_need(mode_counts=mode_counts, size_counts=size_counts),
        [f"image_modes={mode_counts}", f"image_sizes={size_counts}"],
    )
    visualization_need = 0.8 if any(keyword in request_text for keyword in ["伪彩", "可视化", "人工检查", "展示", "pseudocolor", "weicaise"]) else 0.1
    add_need(
        needs,
        "visualization",
        visualization_need,
        ["request explicitly asks for pseudocolor/visualization"] if visualization_need > 0.5 else ["no explicit visualization request"],
    )
    downstream_requested = (
        task == "classification_evaluation"
        or "downstream_benefit_validation" in goals
        or any(keyword in request_text for keyword in ["下游", "准确率", "识别率", "分类评估", "分类收益", "classification benchmark", "train classifier"])
    )
    add_need(
        needs,
        "downstream_evidence",
        0.85 if downstream_requested else 0.25,
        ["user explicitly asks for downstream classification evidence"]
        if downstream_requested
        else ["default policy uses lightweight probes; downstream evaluator is reserved for explicit task-gain claims"],
    )
    return {
        "schema_version": DATASET_NEED_PROFILE_VERSION,
        "task": task,
        "total_images": total_images,
        "profile_level": bridge_report.get("profile_level_after_validation") or validated_profile.get("profile_level"),
        "bridge_valid": bool(bridge_report.get("valid")),
        "needs": needs,
        "top_needs": top_needs(needs),
        "notes": [
            "This is a prior-based need profile, not downstream experimental evidence.",
            "Severity is in [0, 1]; higher means the dataset likely needs that capability more.",
        ],
    }


def score_skill_cards(
    cards: list[dict[str, Any]],
    need_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    bridge_report: dict[str, Any],
    memory_context: dict[str, Any],
) -> dict[str, Any]:
    task = need_profile.get("task") or intent_spec.get("intent", {}).get("task") or "classification"
    scored = [score_skill_card(card, need_profile, task, bridge_report, memory_context, intent_spec) for card in cards]
    scored = sorted(scored, key=lambda item: item["utility_score"], reverse=True)
    return {
        "schema_version": "saga_skill_utility_v1",
        "task": task,
        "ranked_skills": scored,
        "top_recommendations": scored[:8],
    }


def score_skill_card(
    card: dict[str, Any],
    need_profile: dict[str, Any],
    task: str,
    bridge_report: dict[str, Any],
    memory_context: dict[str, Any],
    intent_spec: dict[str, Any],
) -> dict[str, Any]:
    needs = need_profile.get("needs") or {}
    addresses = card.get("addresses_dataset_needs") or {}
    matched = []
    need_match = 0.0
    weight_sum = 0.0
    for need, strength in addresses.items():
        severity = float(needs.get(need, {}).get("severity", 0.0))
        strength_value = qualitative_to_score(strength)
        contribution = severity * strength_value
        if contribution > 0:
            matched.append(
                {
                    "need": need,
                    "severity": round(severity, 3),
                    "strength": strength,
                    "contribution": round(contribution, 3),
                }
            )
        need_match += contribution
        weight_sum += strength_value
    need_match = clamp(need_match / max(weight_sum, 1.0))
    task_alignment = task_alignment_score(card, task)
    metadata_readiness = metadata_readiness_score(card, bridge_report, need_profile)
    request_alignment = explicit_request_alignment_score(card, intent_spec, task)
    benefit_prior = average_qualitative(card.get("expected_benefit_prior") or {})
    risk = average_qualitative(card.get("risks") or {})
    cost_penalty = cost_score(card.get("cost") or {})
    implementation_penalty = implementation_penalty_score(str(card.get("status") or "planned"))
    memory_success = memory_success_score(memory_context)
    memory_outcome = memory_outcome_score(card, memory_context)
    utility = (
        0.25 * need_match
        + 0.18 * metadata_readiness
        + 0.16 * task_alignment
        + 0.18 * request_alignment
        + 0.15 * benefit_prior
        + 0.06 * memory_success
        + 0.10 * memory_outcome
        - 0.10 * risk
        - 0.05 * cost_penalty
        - implementation_penalty
    )
    return {
        "skill": card.get("name"),
        "status": card.get("status"),
        "utility_score": round(clamp(utility), 3),
        "need_match": round(need_match, 3),
        "metadata_readiness": round(metadata_readiness, 3),
        "task_alignment": round(task_alignment, 3),
        "request_alignment": round(request_alignment, 3),
        "benefit_prior": round(benefit_prior, 3),
        "risk_penalty": round(risk, 3),
        "cost_penalty": round(cost_penalty, 3),
        "implementation_penalty": round(implementation_penalty, 3),
        "memory_success": round(memory_success, 3),
        "memory_outcome": round(memory_outcome, 3),
        "matched_needs": sorted(matched, key=lambda item: item["contribution"], reverse=True),
        "verdict": skill_verdict(card, utility),
    }


def evaluate_plan_utility(
    candidate: dict[str, Any],
    benefit_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not benefit_context:
        return {
            "schema_version": PLAN_UTILITY_VERSION,
            "plan_id": candidate.get("plan_id"),
            "utility_score": 0.5,
            "matched_needs": [],
            "skill_scores": [],
            "verdict": "benefit_context_unavailable",
        }
    skill_rank = {
        item["skill"]: item
        for item in benefit_context.get("skill_utility", {}).get("ranked_skills", [])
    }
    pipeline = candidate.get("pipeline") or []
    plan_skill_scores = []
    matched_needs: dict[str, dict[str, Any]] = {}
    for step in pipeline:
        skill_name = step.get("skill")
        score = skill_rank.get(skill_name)
        if not score:
            continue
        plan_skill_scores.append(
            {
                "step_id": step.get("id"),
                "skill": skill_name,
                "utility_score": score.get("utility_score"),
                "status": score.get("status"),
                "verdict": score.get("verdict"),
            }
        )
        for match in score.get("matched_needs", []):
            need = match["need"]
            existing = matched_needs.get(need)
            if existing is None or match.get("contribution", 0) > existing.get("contribution", 0):
                matched_needs[need] = {
                    "need": need,
                    "severity": match.get("severity"),
                    "covered_by": [skill_name],
                    "contribution": match.get("contribution"),
                }
            elif skill_name not in existing["covered_by"]:
                existing["covered_by"].append(skill_name)
    if plan_skill_scores:
        executable_bonus = 0.03 * len([item for item in plan_skill_scores if item.get("status") == "executable"])
        planned_penalty = 0.04 * len([item for item in plan_skill_scores if item.get("status") == "planned"])
        utility = sum(float(item.get("utility_score") or 0) for item in plan_skill_scores) / len(plan_skill_scores)
        utility = clamp(utility + executable_bonus - planned_penalty)
    else:
        utility = 0.25
    return {
        "schema_version": PLAN_UTILITY_VERSION,
        "plan_id": candidate.get("plan_id"),
        "task": candidate.get("task"),
        "utility_score": round(utility, 3),
        "matched_needs": sorted(matched_needs.values(), key=lambda item: item.get("contribution", 0), reverse=True),
        "skill_scores": plan_skill_scores,
        "top_dataset_needs": benefit_context.get("dataset_need_profile", {}).get("top_needs", []),
        "verdict": plan_utility_verdict(utility, plan_skill_scores),
    }


def write_benefit_context(output_dir: str | Path, context: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    save_json(path / "skill_effect_cards.json", context["skill_effect_cards"])
    save_json(path / "dataset_need_profile.json", context["dataset_need_profile"])
    save_json(path / "skill_utility.json", context["skill_utility"])
    save_text(path / "skill_effect_cards.md", render_skill_effect_cards_markdown(context["skill_effect_cards"]))
    save_text(path / "dataset_need_profile.md", render_dataset_need_profile_markdown(context["dataset_need_profile"]))
    save_text(path / "skill_utility.md", render_skill_utility_markdown(context["skill_utility"]))


def write_plan_utility_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    save_json(path / "plan_utility_report.json", report)
    save_text(path / "plan_utility_report.md", render_plan_utility_markdown(report))


def render_skill_effect_cards_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SAGA Skill Effect Cards",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Total cards: {payload.get('summary', {}).get('total')}",
        f"- Executable: {payload.get('summary', {}).get('executable')}",
        f"- Planned: {payload.get('summary', {}).get('planned')}",
        "",
    ]
    for card in payload.get("cards", []):
        lines.extend(
            [
                f"## {card.get('name')}",
                "",
                f"- Status: `{card.get('status')}`",
                f"- Main goal: {card.get('main_goal')}",
                f"- Supported tasks: `{card.get('supported_tasks')}`",
                f"- Addresses needs: `{card.get('addresses_dataset_needs')}`",
                f"- Expected benefit prior: `{card.get('expected_benefit_prior')}`",
                f"- Cost: `{card.get('cost')}`",
                f"- Risks: `{card.get('risks')}`",
                "",
            ]
        )
    return "\n".join(lines)


def render_dataset_need_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# SAGA Dataset Need Profile",
        "",
        f"- Task: `{profile.get('task')}`",
        f"- Total images: {profile.get('total_images')}",
        f"- Bridge valid: {profile.get('bridge_valid')}",
        f"- Profile level: {profile.get('profile_level')}",
        "",
        "## Top Needs",
        "",
    ]
    for item in profile.get("top_needs", []):
        lines.append(f"- `{item.get('need')}` severity={item.get('severity')}: {item.get('summary')}")
    lines.extend(["", "## All Needs", ""])
    for name, item in profile.get("needs", {}).items():
        lines.append(f"- `{name}` severity={item.get('severity')} evidence=`{item.get('evidence')}`")
    lines.append("")
    return "\n".join(lines)


def render_skill_utility_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Skill Utility",
        "",
        f"- Task: `{report.get('task')}`",
        "",
    ]
    for idx, item in enumerate(report.get("ranked_skills", []), start=1):
        lines.extend(
            [
                f"## {idx}. {item.get('skill')}",
                "",
                f"- Status: `{item.get('status')}`",
                f"- Utility score: {item.get('utility_score')}",
                f"- Need match: {item.get('need_match')}",
                f"- Metadata readiness: {item.get('metadata_readiness')}",
                f"- Task alignment: {item.get('task_alignment')}",
                f"- Request alignment: {item.get('request_alignment')}",
                f"- Benefit prior: {item.get('benefit_prior')}",
                f"- Memory success: {item.get('memory_success')}",
                f"- Memory outcome: {item.get('memory_outcome')}",
                f"- Risk penalty: {item.get('risk_penalty')}",
                f"- Cost penalty: {item.get('cost_penalty')}",
                f"- Verdict: {item.get('verdict')}",
                "",
            ]
        )
    return "\n".join(lines)


def render_plan_utility_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Plan Utility Report",
        "",
        f"- Schema: `{report.get('schema_version')}`",
        f"- Selected plan: `{report.get('selected_plan_id')}`",
        f"- Selected utility: {report.get('selected_utility_score')}",
        "",
        "## Ranked Plans",
        "",
    ]
    for idx, item in enumerate(report.get("ranked_plans", []), start=1):
        lines.extend(
            [
                f"## {idx}. {item.get('plan_id')}",
                "",
                f"- Task: `{item.get('task')}`",
                f"- Utility score: {item.get('utility_score')}",
                f"- Verdict: {item.get('verdict')}",
                f"- Matched needs: `{item.get('matched_needs')}`",
                f"- Skill scores: `{item.get('skill_scores')}`",
                "",
            ]
        )
    return "\n".join(lines)


def add_need(
    needs: dict[str, Any],
    name: str,
    severity: float,
    evidence: list[str],
    inverse: bool = False,
) -> None:
    severity = clamp(severity)
    if inverse:
        summary = "readiness is high" if severity < 0.35 else "readiness is weak"
    elif severity >= 0.7:
        summary = "high priority"
    elif severity >= 0.4:
        summary = "medium priority"
    else:
        summary = "low priority"
    needs[name] = {
        "severity": round(severity, 3),
        "summary": summary,
        "evidence": evidence,
    }


def top_needs(needs: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    rows = [{"need": name, **value} for name, value in needs.items()]
    return sorted(rows, key=lambda item: item["severity"], reverse=True)[:limit]


def ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return clamp(part / total)


def sample_count_severity(total_images: int) -> float:
    if total_images <= 0:
        return 1.0
    if total_images < 50:
        return 0.9
    if total_images < 200:
        return 0.7
    if total_images < 1000:
        return 0.45
    return 0.2


def field_coverage(coverage: dict[str, Any], name: str) -> float:
    try:
        return clamp(float(coverage.get(name, 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def image_quality_need(mode_counts: dict[str, Any], size_counts: dict[str, Any]) -> float:
    severity = 0.2
    if len(size_counts) > 4:
        severity += 0.25
    if not mode_counts:
        severity += 0.2
    return clamp(severity)


def qualitative_to_score(value: Any) -> float:
    text = str(value).lower()
    table = {
        "none": 0.0,
        "very_low": 0.1,
        "low": 0.25,
        "low_to_medium": 0.35,
        "medium": 0.55,
        "medium_to_high": 0.7,
        "high": 0.8,
        "very_high": 1.0,
    }
    return table.get(text, 0.5)


def average_qualitative(mapping: dict[str, Any]) -> float:
    if not mapping:
        return 0.0
    values = [qualitative_to_score(value) for value in mapping.values()]
    return sum(values) / len(values)


def explicit_request_alignment_score(card: dict[str, Any], intent_spec: dict[str, Any], task: str) -> float:
    """Estimate whether the user explicitly asked for this augmentation family."""

    name = str(card.get("name") or "")
    text = str(intent_spec.get("raw_text") or "").lower()
    goals = set(intent_spec.get("intent", {}).get("goals") or [])

    if name == "DiffusionLoRAGenerationSkill":
        if task == "diffusion_lora_generation" or any(keyword in text for keyword in ["lora", "lo-ra", "扩散模型", "diffusion"]):
            return 1.0
        if "生成" in text or "增广" in text:
            return 0.65
    if name == "TraditionalAugmentationSkill":
        if any(keyword in text for keyword in ["传统", "翻转", "旋转", "裁剪", "speckle", "noise", "噪声", "最快", "速度最快", "简单增广"]):
            return 1.0
    if name == "GANImageToImageSkill":
        if any(keyword in text for keyword in ["gan", "图生图", "简单目标", "mstar"]):
            return 1.0
    if name == "GeoDiffSARSkill":
        if any(keyword in text for keyword in ["geodiff", "controlnet", "3d", "三维", "物理先验"]) or (
            "complete_sparse_azimuth" in goals and any(keyword in text for keyword in ["高质量", "不计时间", "不计显存"])
        ):
            return 1.0
    if name == "GaussianSplattingCompletionSkill":
        if task == "gaussian_splatting_completion" or (
            any(keyword in text for keyword in ["高斯", "泼溅", "splatting", "轻量", "低显存"]) and "complete_sparse_azimuth" in goals
        ):
            return 1.0
    if name == "RaySARSynthesisSkill":
        if excludes_physics_simulation_request(text):
            return 0.0
        if any(keyword in text for keyword in ["raysar", "pov-ray", "povray", ".pov", "物理仿真", "射线追踪"]):
            return 1.0
    if name == "RaySARSweepSynthesisSkill":
        if excludes_physics_simulation_request(text):
            return 0.0
        has_raysar = any(keyword in text for keyword in ["raysar", "pov-ray", "povray", ".obj", ".stl", ".ply", ".glb", "物理仿真", "射线追踪"])
        has_sweep = any(keyword in text for keyword in ["方位角从", "每隔", "间隔", "步长", "sweep", "azimuth"])
        if has_raysar and has_sweep:
            return 1.0
        if has_raysar:
            return 0.75
    if name == "StyleTransferSkill":
        if task == "style_transfer" or any(keyword in text for keyword in ["风格迁移", "特征迁移", "跨平台", "跨载荷", "style transfer"]):
            return 1.0
    if name == "PseudocolorSkill":
        if any(keyword in text for keyword in ["伪彩", "伪彩色", "可视化", "人工检查", "展示", "weicaise", "pseudocolor"]):
            return 1.0
        if any(keyword in text for keyword in ["保持灰度", "物理灰度", "灰度训练", "辐射定标", "绝对幅度"]):
            return 0.0
        return 0.1
    if name == "BackgroundGenerationSkill":
        if "背景生成" in text or ("背景" in text and "生成" in text):
            return 1.0
    if name == "TargetBackgroundCompositionSkill":
        if any(keyword in text for keyword in ["合成", "融合", "图像融合", "目标和场景", "目标-场景", "目标和背景", "目标-背景", "composition", "blending"]):
            return 1.0
    if name in {"ClassificationEvaluationSkill", "ObjectDetectionEvaluationSkill", "SegmentationEvaluationSkill"}:
        if any(keyword in text for keyword in ["评估", "下游", "accuracy", "map", "miou", "收益"]):
            return 0.8
    if name == "QualityEvaluationSkill":
        if any(keyword in text for keyword in ["质量", "伪影", "黑图", "白图"]):
            return 0.6
        return 0.25
    if name == "DistributionEvaluationSkill":
        if any(keyword in text for keyword in ["fid", "kid", "mmd", "分布", "数值指标", "指标"]):
            return 0.85
        return 0.35
    if name == "SARArtifactEvaluationSkill":
        if any(keyword in text for keyword in ["伪影", "条纹", "背景梯度", "散射中心", "目标结构"]):
            return 0.85
        return 0.4
    if name in {"DatasetProfileReportSkill", "FileSelectionSkill", "ExportDatasetSkill", "RepairPolicySkill", "SARPreprocessSkill"}:
        return 0.25
    return 0.0


def excludes_physics_simulation_request(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
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
        ]
    ) or any(keyword in lowered for keyword in ["no raysar", "without raysar", "no physics simulation", "without physics simulation"])


def task_alignment_score(card: dict[str, Any], task: str) -> float:
    name = str(card.get("name") or "")
    supported = set(card.get("supported_tasks") or [])
    if task == "classification" and name in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
        return 0.15
    if "all" in supported:
        return 0.7
    if task in supported:
        return 1.0
    if task == "diffusion_lora_generation" and "classification" in supported:
        return 0.75
    if task == "traditional_augmentation" and "classification" in supported:
        return 0.8
    if task == "style_transfer" and "domain_adaptation" in supported:
        return 0.85
    if task == "raysar_synthesis" and "physics_simulation" in supported:
        return 1.0
    if task in {"classification", "augmentation_or_generation"} and "target_generation" in supported:
        return 0.75
    return 0.25


def metadata_readiness_score(card: dict[str, Any], bridge_report: dict[str, Any], need_profile: dict[str, Any]) -> float:
    required = list(card.get("required_metadata") or [])
    if not required:
        base = 0.75 if bridge_report.get("valid") else 0.45
        return apply_input_readiness_adjustments(base, card, bridge_report, need_profile)
    coverage = (bridge_report.get("validation") or {}).get("field_coverage") or {}
    scores = []
    for field in required:
        if field == "filter_fields":
            scores.append(1.0 if bridge_report.get("valid") else 0.0)
        elif field == "azimuth_deg":
            scores.append(max(field_coverage(coverage, "azimuth_deg"), field_coverage(coverage, "azimuth_angle"), field_coverage(coverage, "azimuth")))
        else:
            scores.append(field_coverage(coverage, field))
    base = sum(scores) / len(scores) if scores else 0.5
    return apply_input_readiness_adjustments(base, card, bridge_report, need_profile)


def apply_input_readiness_adjustments(
    base: float,
    card: dict[str, Any],
    bridge_report: dict[str, Any],
    need_profile: dict[str, Any],
) -> float:
    requirements = card.get("input_requirements") or {}
    needs = need_profile.get("needs") or {}
    score = base
    if requirements.get("paired_image_txt_captions"):
        caption_missing = float(needs.get("caption_readiness", {}).get("severity", 1.0))
        if caption_missing >= 0.95 and bridge_report.get("valid"):
            # LoRA can still run by staging metadata-derived captions when a
            # DatasetFormatSpec has been validated. This is weaker than real
            # human captions, but it should not be treated as impossible.
            score *= 0.65
        else:
            score *= 1.0 - caption_missing
    if requirements.get("azimuth_metadata"):
        azimuth_missing = float(needs.get("azimuth_coverage", {}).get("severity", 1.0))
        score *= 1.0 - azimuth_missing
    if requirements.get("physical_prior"):
        score *= 0.35
    if requirements.get("val_or_test_dataset"):
        score *= 0.35
    if requirements.get("validated_dataset_config") and not bridge_report.get("valid"):
        score *= 0.2
    return clamp(score)


def cost_score(cost: dict[str, Any]) -> float:
    runtime = str(cost.get("runtime") or "medium").lower()
    score = {"low": 0.15, "medium": 0.5, "high": 0.85}.get(runtime, 0.5)
    if cost.get("gpu_required"):
        score += 0.1
    return clamp(score)


def implementation_penalty_score(status: str) -> float:
    if status == "executable":
        return 0.0
    if status == "experimental":
        return 0.08
    if status == "planned":
        return 0.18
    return 0.25


def memory_success_score(memory_context: dict[str, Any]) -> float:
    matches = memory_context.get("matches") or []
    if not matches:
        return 0.0
    score = 0.0
    for item in matches[:5]:
        entry = item.get("entry") or item
        execution = entry.get("execution") or {}
        planner = entry.get("planner") or {}
        if execution.get("status") in {"succeeded", "dry_run"}:
            score += 0.15
        if planner.get("verification_status") == "passed":
            score += 0.1
    return clamp(score)


def memory_outcome_score(card: dict[str, Any], memory_context: dict[str, Any]) -> float:
    matches = memory_context.get("matches") or []
    if not matches:
        return 0.0
    skill_name = str(card.get("name") or "")
    weighted_score = 0.0
    weight_sum = 0.0
    for item in matches[:8]:
        selected_skills = set(item.get("selected_skills") or [])
        if skill_name not in selected_skills:
            continue
        weight = min(float(item.get("score") or 0) / 100.0, 1.0)
        if weight <= 0:
            weight = 0.1
        local = 0.0
        execution = item.get("execution") or {}
        quality = item.get("quality") or {}
        sar_artifacts = item.get("sar_artifacts") or {}
        classification = ((item.get("outcomes") or {}).get("classification") or {})

        if execution.get("status") == "succeeded":
            local += 0.15
        elif execution.get("status") == "dry_run":
            local += 0.04
        elif execution.get("status") == "failed":
            local -= 0.25

        quality_triggers = int(quality.get("trigger_count") or 0)
        sar_triggers = int(sar_artifacts.get("trigger_count") or 0)
        if quality_triggers == 0:
            local += 0.08
        else:
            local -= min(0.24, 0.06 * quality_triggers)
        if sar_triggers == 0:
            local += 0.08
        else:
            local -= min(0.24, 0.06 * sar_triggers)

        if classification.get("available"):
            verdict = classification.get("verdict")
            delta = safe_float(classification.get("delta")) or 0.0
            if verdict == "improved":
                local += 0.45 + min(abs(delta), 0.2)
            elif verdict == "regressed":
                local -= 0.45 + min(abs(delta), 0.2)
            elif verdict == "unchanged":
                local += 0.05
        weighted_score += clamp_signed(local) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return 0.0
    return clamp_signed(weighted_score / weight_sum)


def skill_verdict(card: dict[str, Any], utility: float) -> str:
    status = card.get("status")
    if utility >= 0.65 and status == "executable":
        return "high_potential_executable"
    if utility >= 0.65:
        return "high_potential_but_not_executable"
    if utility >= 0.4 and status == "executable":
        return "moderate_potential_executable"
    if utility >= 0.4:
        return "moderate_potential_planned"
    return "low_priority_for_current_profile"


def plan_utility_verdict(utility: float, plan_skill_scores: list[dict[str, Any]]) -> str:
    if not plan_skill_scores:
        return "no_registered_skill_evidence"
    planned = [item for item in plan_skill_scores if item.get("status") == "planned"]
    if utility >= 0.65 and planned:
        return "high_potential_contains_planned_skills"
    if utility >= 0.65:
        return "high_potential_executable_plan"
    if utility >= 0.4:
        return "moderate_potential_plan"
    return "low_potential_for_current_needs"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

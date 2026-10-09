from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text


SKILL_REGISTRY_VERSION = "saga_skill_registry_v1"


@dataclass
class SkillSpec:
    name: str
    status: str
    description: str
    task_types: list[str] = field(default_factory=list)
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def built_in_skill_specs() -> list[SkillSpec]:
    return [
        SkillSpec(
            name="DatasetProfileReportSkill",
            status="executable",
            description="Record dataset profile and format-bridge artifacts for provenance.",
            task_types=["all"],
            input_contract={"dataset_profile_path": "json", "validated_profile_path": "yaml_or_json"},
            output_contract={"profile_artifacts": "paths"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"default_dry_run": False, "requires_run_flag": False},
        ),
        SkillSpec(
            name="FileSelectionSkill",
            status="executable",
            description="Select image files using validated metadata filters.",
            task_types=["style_transfer", "classification", "object_detection", "segmentation"],
            input_contract={"content_root": "image_directory", "dataset_config": "validated_dataset_config"},
            output_contract={"selected_content": "image_directory", "selection_manifest": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"requires_validated_metadata_for_filters": True},
        ),
        SkillSpec(
            name="StyleTransferSkill",
            status="executable",
            description="Run wrapped SAR feature/style transfer using myproject/stytransfer.",
            task_types=["style_transfer", "domain_adaptation"],
            input_contract={"content": "image_or_directory", "style": "image_or_directory"},
            output_contract={"generated_images": "image_directory", "run_report": "json"},
            cost={"gpu_required": True, "runtime": "medium"},
            safety={"default_dry_run": True, "requires_run_flag": True},
            limitations=["Quality depends heavily on upstream style-transfer model and preprocessing."],
        ),
        SkillSpec(
            name="DiffusionLoRAGenerationSkill",
            status="executable",
            description="Train text-caption LoRA and infer augmented SAR images through qinglong/sd-scripts.",
            task_types=["diffusion_lora_generation", "generation_dataset", "classification"],
            input_contract={
                "dataset_root": "paired_image_txt_caption_dataset_or_validated_metadata_dataset",
                "dataset_config": "optional_validated_dataset_config_for_auto_caption_staging",
            },
            output_contract={"generated_images": "image_directory", "lora_weights": "safetensors"},
            cost={"gpu_required": True, "runtime": "high"},
            safety={"default_dry_run": True, "requires_run_flag": True},
            limitations=[
                "No ControlNet.",
                "No GeoDiff-SAR physical prior.",
                "If no txt captions exist, SAGA needs validated metadata to stage metadata-derived captions.",
            ],
        ),
        SkillSpec(
            name="QualityEvaluationSkill",
            status="executable",
            description="Evaluate generated image count, readability, and simple image statistics.",
            task_types=["all"],
            input_contract={"input_dir": "image_directory"},
            output_contract={"quality_report": "json", "triggers": "list"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"can_gate_export": True},
            limitations=["Not a downstream task-performance evaluator."],
        ),
        SkillSpec(
            name="DistributionEvaluationSkill",
            status="executable",
            description="Compare generated and reference image distributions with FID-style lightweight SAR metrics and optional pytorch_fid.",
            task_types=["all"],
            input_contract={"generated_dir": "image_directory", "reference_dir": "image_directory"},
            output_contract={"distribution_report": "json", "metrics": "fid_like_mmd_histogram_diversity"},
            cost={"gpu_required": False, "runtime": "low_to_medium"},
            safety={"default_sampled": True, "does_not_claim_downstream_gain": True},
            limitations=["FID-style metrics are proxies and do not replace downstream task evaluation."],
        ),
        SkillSpec(
            name="SARArtifactEvaluationSkill",
            status="executable",
            description="Evaluate SAR-specific visual artifacts such as stripes, smooth gradients, target compactness, fragmentation, and centering.",
            task_types=["all"],
            input_contract={"input_dir": "image_directory"},
            output_contract={"sar_artifact_report": "json", "triggers": "list"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"can_trigger_bounded_repair": True},
            limitations=["Heuristic SAR artifact metrics are observer triggers, not downstream performance proof."],
        ),
        SkillSpec(
            name="RepairPolicySkill",
            status="executable",
            description="Create bounded repair suggestions without automatically rerunning expensive skills.",
            task_types=["all"],
            input_contract={"quality_report": "json"},
            output_contract={"repair_policy": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"auto_rerun": False, "bounded": True},
        ),
        SkillSpec(
            name="ExportDatasetSkill",
            status="executable",
            description="Package generated outputs with manifest, provenance, dataset card, and optional caption sidecars.",
            task_types=["all"],
            input_contract={"source_dir": "image_directory"},
            output_contract={"augmented_dataset": "directory", "manifest": "jsonl", "caption_sidecars": "optional_txt_sidecars"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"can_require_quality_pass": True},
        ),
        SkillSpec(
            name="TraditionalAugmentationSkill",
            status="executable",
            description="Fast deterministic SAR-aware image augmentation baseline.",
            task_types=["classification", "object_detection", "segmentation"],
            input_contract={"dataset_root": "image_dataset", "dataset_config": "optional_validated_dataset_config"},
            output_contract={"augmented_images": "image_directory", "run_report": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"label_preserving_only": True, "default_dry_run": True},
            limitations=["Does not synthesize new target geometry or true unseen poses."],
        ),
        SkillSpec(
            name="SARPreprocessSkill",
            status="executable",
            description="Normalize SAR intensity, bit depth, size, and dynamic range before generation or evaluation.",
            task_types=["all"],
            input_contract={"images": "image_directory"},
            output_contract={"normalized_images": "image_directory"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"label_preserving_only": True, "default_dry_run": True},
            limitations=["Can distort amplitude cues if normalization is too aggressive."],
        ),
        SkillSpec(
            name="MetadataCaptionSkill",
            status="executable",
            description="Compile validated metadata into image/txt caption pairs for LoRA/diffusion training.",
            task_types=["classification", "diffusion_lora_generation", "target_generation"],
            input_contract={"dataset_root": "image_dataset", "dataset_config": "optional_validated_dataset_config"},
            output_contract={"captioned_dataset": "image_txt_pair_directory", "caption_manifest": "jsonl"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"requires_validated_metadata_for_semantic_claims": True, "default_dry_run": True},
            limitations=["Caption quality is bounded by metadata coverage and DatasetFormatSpec validation."],
        ),
        SkillSpec(
            name="DatasetBalancingSkill",
            status="executable",
            description="Analyze class/metadata bins and produce target-oriented augmentation deficits.",
            task_types=["classification", "target_generation", "diffusion_lora_generation", "traditional_augmentation"],
            input_contract={"dataset_root": "image_dataset", "dataset_config": "optional_validated_dataset_config"},
            output_contract={"balance_plan": "jsonl", "deficit_report": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"does_not_generate_images": True, "planner_evidence_only": True},
        ),
        SkillSpec(
            name="PseudocolorSkill",
            status="executable",
            description="Apply lightweight SAR pseudocolor visualization/feature transform using the weicaise technique.",
            task_types=["classification", "feature_transform", "visualization"],
            input_contract={"input_dir": "image_directory"},
            output_contract={"pseudocolor_images": "rgb_image_directory"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"label_preserving_transform": True, "default_dry_run": True},
            limitations=["Useful as feature/visual transform; validate downstream before treating as true SAR-domain augmentation."],
        ),
        SkillSpec(
            name="ClassificationEvaluationSkill",
            status="executable",
            description="SAGA standard downstream classification evaluator: adapts datasets into the referenced timm multi-label project, runs baseline-vs-augmented training/validation, and reports comparable metrics.",
            task_types=["classification", "classification_evaluation"],
            input_contract={
                "baseline_dataset": "classification_dataset_or_saga_export",
                "augmented_dataset": "optional_classification_dataset_or_saga_export",
                "val_dataset": "optional_heldout_validation_dataset",
            },
            output_contract={"benchmark_dataset": "project_ready_csv_dataset", "metrics": "baseline_augmented_delta", "commands": "shell_commands"},
            cost={"gpu_required": True, "runtime": "high"},
            safety={"default_dry_run": True, "requires_run_flag": True, "requires_controlled_split": True},
            limitations=[
                "Uses the referenced project head names article/color/gender; SAGA maps SAR class to article by default.",
                "Real downstream claims require running the generated train/validation protocol on a stable held-out split.",
            ],
        ),
        SkillSpec(
            name="ObjectDetectionEvaluationSkill",
            status="planned",
            description="Future downstream object detection evaluator.",
            task_types=["object_detection"],
            input_contract={"dataset": "detection_dataset"},
            output_contract={"metrics": "map_recall_ap_per_class"},
            cost={"gpu_required": True, "runtime": "high"},
        ),
        SkillSpec(
            name="PerMetadataSliceEvaluationSkill",
            status="executable",
            description="Evaluate or summarize downstream results by class, azimuth, polarization, band, resolution, and other metadata slices.",
            task_types=["classification", "classification_evaluation"],
            input_contract={"dataset_config": "validated_dataset_config", "predictions_csv": "optional_per_sample_predictions"},
            output_contract={"slice_report": "json", "slice_rows": "jsonl"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"does_not_claim_gain_without_predictions": True},
        ),
        SkillSpec(
            name="LeakageCheckSkill",
            status="executable",
            description="Check train/augmented/validation sets for exact hash overlap and suspicious same-stem overlap.",
            task_types=["classification", "classification_evaluation", "all"],
            input_contract={"baseline_dataset": "image_directory", "augmented_dataset": "optional_image_directory", "val_dataset": "optional_image_directory"},
            output_contract={"leakage_report": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"benchmark_guard": True},
        ),
        SkillSpec(
            name="DuplicateNearDuplicateSkill",
            status="executable",
            description="Detect duplicate and near-duplicate generated images with perceptual hashing.",
            task_types=["classification", "target_generation", "all"],
            input_contract={"input_dir": "image_directory"},
            output_contract={"duplicate_report": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"overfit_guard": True},
        ),
        SkillSpec(
            name="SegmentationEvaluationSkill",
            status="planned",
            description="Future downstream segmentation evaluator.",
            task_types=["segmentation"],
            input_contract={"dataset": "segmentation_dataset"},
            output_contract={"metrics": "miou_dice"},
            cost={"gpu_required": True, "runtime": "high"},
        ),
        SkillSpec(
            name="GANImageToImageSkill",
            status="executable",
            description="Wrap myproject/dcgan or myproject/dcgan-cpu for fast simple-target GAN generation.",
            task_types=["gan_generation", "generation_dataset", "classification"],
            input_contract={"dataset_root": "simple_target_image_directory", "parameters_xml": "generated_or_user_xml"},
            output_contract={"generated_images": "image_directory", "gan_report": "json"},
            cost={"gpu_required": True, "runtime": "medium"},
            safety={"default_dry_run": True, "requires_run_flag": True},
            limitations=["Best for simple target distributions; not a physically controlled generator."],
        ),
        SkillSpec(
            name="ClassRebalanceAugmentationSkill",
            status="planned",
            description="Task-oriented policy skill that maps class deficits to concrete augmentation recipes.",
            task_types=["classification", "target_generation"],
            input_contract={"balance_plan": "jsonl"},
            output_contract={"recipe_intent_updates": "dict"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="AzimuthCoverageCompletionSkill",
            status="planned",
            description="Task-oriented policy skill for sparse azimuth coverage completion using GeoDiff/Gaussian/LoRA options.",
            task_types=["classification", "sparse_azimuth_completion"],
            input_contract={"validated_azimuth_metadata": "required"},
            output_contract={"target_azimuth_plan": "json"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="PolarizationConditionedGenerationSkill",
            status="planned",
            description="Task-oriented policy skill that turns polarization deficits into metadata-conditioned generation requests.",
            task_types=["classification", "diffusion_lora_generation", "target_generation"],
            input_contract={"validated_polarization_metadata": "required"},
            output_contract={"polarization_generation_plan": "json"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="ResolutionBandConditionedGenerationSkill",
            status="planned",
            description="Task-oriented policy skill for resolution/band-conditioned SAR target augmentation.",
            task_types=["classification", "target_generation"],
            input_contract={"resolution_or_band_metadata": "validated_or_user_confirmed"},
            output_contract={"conditioned_generation_plan": "json"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="HardSampleMiningSkill",
            status="planned",
            description="Use downstream errors to select hard classes, views, and metadata slices for the next augmentation round.",
            task_types=["classification", "classification_evaluation"],
            input_contract={"per_sample_predictions": "csv_or_jsonl"},
            output_contract={"hard_sample_plan": "json"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="OODDomainExpansionSkill",
            status="planned",
            description="Plan cross-domain style/generation expansion when payload, platform, scene, or collection-domain shift is detected.",
            task_types=["classification", "domain_adaptation"],
            input_contract={"source_domain_profile": "json", "target_domain_profile": "optional_json"},
            output_contract={"domain_expansion_plan": "json"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
        SkillSpec(
            name="TargetBackgroundCompositionSkill",
            status="executable",
            description="SAR-aware target/background image fusion using mask extraction, intensity matching, feather/laplacian/Poisson blending, and composition manifests.",
            task_types=["object_detection", "classification", "target_background_composition"],
            input_contract={"target_dir": "target_image_bank", "background_dir": "background_image_bank", "mask_dir": "optional_target_masks"},
            output_contract={"composed_images": "image_directory", "composition_masks": "image_directory", "composition_manifest": "jsonl"},
            cost={"gpu_required": False, "runtime": "medium"},
            safety={"default_dry_run": True, "requires_run_flag": False, "annotation_validation_required_for_detection": True},
            limitations=[
                "Auto masks are heuristic and should be checked for cluttered target chips.",
                "Classification benefit is indirect unless composed images match the training protocol.",
                "Detection labels from generated boxes should be validated before downstream claims.",
            ],
        ),
        SkillSpec(
            name="BackgroundGenerationSkill",
            status="executable",
            description="Generate SAR background scenes directly with the packaged segmentation-controlled FLUX ControlNet model and trained weights.",
            task_types=["background_generation", "target_background_composition"],
            input_contract={"scene_prompt": "natural_language_scene_description", "segment_control_bank": "packaged_control_index"},
            output_contract={
                "background_images": "image_directory",
                "selected_controls": "segmentation_control_maps",
                "prompt_file": "txt",
                "run_report": "json",
            },
            cost={"gpu_required": True, "runtime": "high"},
            safety={"default_dry_run": True, "requires_run_flag": True, "does_not_train_user_model": True},
            limitations=[
                "This skill generates background/scene assets, not default target-chip classification augmentation.",
                "It uses the packaged trained weights under myproject/scene_gen_segment_large and does not train user background models.",
                "Backgrounds need composition/detection or task-specific evaluation before claiming downstream benefit.",
            ],
        ),
        SkillSpec(
            name="GeoDiffSARSkill",
            status="executable",
            description="Composite GeoDiff-SAR physical-prior sparse-azimuth completion: real-image GEFM extraction, FLUX ControlNet training, 3D-model GEFM rendering, and ControlNet inference.",
            task_types=["geodiff_sar_generation", "sparse_azimuth_completion", "target_generation", "classification"],
            input_contract={
                "dataset_root": "real SAR image/txt caption dataset used to train ControlNet and extract real-image GEFM conditions",
                "model_file": "3D model used by raytracing to render inference-time GEFM conditions",
                "target_azimuths_or_sweep": "explicit target azimuth list or sweep; defaults to a 30-degree full sweep",
            },
            output_contract={
                "completed_views": "generated image directory",
                "train_gefm_conditions": "real-image GEFM condition maps",
                "inference_gefm_conditions": "3D-model GEFM condition maps",
                "controlnet_weights": "trained or user-provided ControlNet safetensors",
                "run_report": "json",
            },
            cost={"gpu_required": True, "runtime": "very_high", "recommended_vram_gb": "24+"},
            safety={"default_dry_run": True, "requires_run_flag": True, "requires_3d_prior": True, "domain_gap_warning": True},
            limitations=[
                "Requires both real SAR training images and a compatible 3D/physical prior for full train+infer workflow.",
                "The wrapper is deterministic, but the underlying diffusion training/inference is expensive and stochastic.",
                "ControlNet/GEFM quality must be checked before claiming sparse-angle or downstream classification benefit.",
            ],
        ),
        SkillSpec(
            name="GaussianSplattingCompletionSkill",
            status="executable",
            description="Lightweight sparse-azimuth completion using the SAR Gaussian splatting implementation under myproject/SAR GS V1.",
            task_types=["gaussian_splatting_completion", "sparse_azimuth_completion", "target_generation", "classification"],
            input_contract={
                "target_images": "sparse_azimuth_dataset",
                "azimuth_metadata": "required_for_view_completion",
                "sar_gs_project": "myproject/SAR GS V1",
            },
            output_contract={"completed_views": "image_directory", "gaussian_model": "ply_or_npz", "run_report": "json"},
            cost={"gpu_required": True, "runtime": "medium", "typical_vram_gb": 2},
            safety={"default_dry_run": True, "requires_run_flag": True, "bounded_view_request": True},
            limitations=[
                "Completion ability is weaker than GeoDiff-SAR for large-angle extrapolation.",
                "Requires usable sparse views and azimuth metadata.",
                "Rendered samples still require artifact, duplicate, and downstream checks before benefit claims.",
            ],
        ),
        SkillSpec(
            name="ModelToPOVSceneCompilerSkill",
            status="executable",
            description="Compile OBJ/STL/PLY/GLB/point-cloud 3D target assets into RaySAR-compatible POV-Ray scenes with camera/light/header.",
            task_types=["raysar_synthesis", "physics_simulation", "target_generation", "sparse_azimuth_completion"],
            input_contract={
                "model_file": "OBJ/STL/PLY/GLB/GLTF/OFF/DAE/3DS mesh or xyz/txt point cloud",
                "simulation_geometry": "incidence/depression angle, azimuth/aspect, target scale, sensor plane, coordinate axes",
            },
            output_contract={"compiled_scene": "RaySAR-compatible .pov scene", "scene_compile_report": "json"},
            cost={"gpu_required": False, "runtime": "low_to_medium"},
            safety={"deterministic": True, "dry_run_writes_scene": True, "domain_gap_warning": True},
            limitations=[
                "Material settings are coarse POV-Ray/RaySAR proxies, not full EM material models.",
                "Very large meshes can make POV-Ray rendering slow; simplify or raise mesh budgets explicitly.",
            ],
        ),
        SkillSpec(
            name="RaySARSynthesisSkill",
            status="executable",
            description="Physics-based SAR simulation adapter around RaySAR/POV-Ray contribution rendering and Python map post-processing.",
            task_types=["raysar_synthesis", "physics_simulation", "target_generation", "sparse_azimuth_completion"],
            input_contract={
                "pov_scene": "RaySAR-compatible .pov scene with target geometry; SAGA can auto-enable SAR_Output_Data and SAR_Intersection",
                "simulation_parameters": "parameters.txt or parameter override dict with pixel spacing, bounce level, range direction, and angle settings",
                "adapted_povray": "optional compiled RaySAR POV-Ray executable for rendering contributions",
                "contributions_txt": "optional existing contributions.txt for postprocess-only mode",
            },
            output_contract={"reflection_maps": "Single_Bounce/Double_Bounce/All_Reflections tif maps", "ray_sar_report": "json"},
            cost={"gpu_required": False, "runtime": "medium_to_high"},
            safety={"requires_physical_scene_assets": True, "postprocess_only_without_adapted_povray": True, "domain_gap_warning": True},
            limitations=[
                "Primary executable input is a RaySAR-compatible .pov scene; common 3D mesh assets should be compiled first by ModelToPOVSceneCompilerSkill.",
                "Full synthesis needs adapted POV-Ray; SAGA can compile the bundled RaySAR source on Linux when dependencies are present.",
                "Synthetic maps are physically interpretable but can have a real-to-synthetic domain gap.",
                "RaySAR is registered as a physics simulation skill, not a default classification augmentation skill.",
            ],
        ),
        SkillSpec(
            name="RaySARSweepSynthesisSkill",
            status="executable",
            description="Multi-view RaySAR simulation from a 3D model by sweeping azimuth/aspect angles.",
            task_types=["raysar_synthesis", "physics_simulation", "target_generation", "sparse_azimuth_completion"],
            input_contract={
                "model_file": "OBJ/STL/PLY/GLB/GLTF/OFF/DAE/3DS mesh or xyz/txt point cloud",
                "azimuth_sweep": "start/stop/step sweep or explicit azimuth_values",
                "simulation_geometry": "incidence/depression angle, target scale, sensor plane, coordinate axes",
                "adapted_povray": "optional compiled RaySAR POV-Ray executable",
            },
            output_contract={
                "sweep_maps": "aggregated Single/Double/All reflection maps renamed by azimuth",
                "caption_sidecars": "txt sidecars containing RaySAR simulation metadata",
                "sweep_report": "json",
            },
            cost={"gpu_required": False, "runtime": "high"},
            safety={"requires_physical_scene_assets": True, "bounded_max_views": True, "domain_gap_warning": True},
            limitations=[
                "Requires a valid 3D model and simulation geometry; it does not infer real material EM properties.",
                "Rendering time grows linearly with requested view count.",
                "Synthetic physical maps should not be treated as classification training data unless the user explicitly requests that experiment.",
            ],
        ),
        SkillSpec(
            name="RawDatasetScanSkill",
            status="executable",
            description="Deterministically scan file structure, image headers, sidecars, path tokens, and naming patterns.",
            task_types=["dataset_profiling", "all"],
            input_contract={"dataset_root": "directory"},
            output_contract={"raw_profile": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"llm_not_required": True},
        ),
        SkillSpec(
            name="LLMSchemaInductionSkill",
            status="executable",
            description="Use an LLM to induce candidate dataset schema from raw profile summaries and user format hints.",
            task_types=["dataset_profiling", "format_understanding"],
            input_contract={"raw_profile": "json", "user_description": "text"},
            output_contract={"candidate_dataset_format_spec": "yaml_or_json"},
            cost={"gpu_required": False, "runtime": "medium"},
            safety={"proposal_only": True, "requires_validator": True},
        ),
        SkillSpec(
            name="DatasetFormatCompilerSkill",
            status="executable",
            description="Compile user/LLM format hints into a DatasetFormatSpec artifact.",
            task_types=["dataset_profiling", "format_bridge"],
            input_contract={"format_hints": "json", "raw_profile": "json"},
            output_contract={"dataset_format_spec": "yaml"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"deterministic": True},
        ),
        SkillSpec(
            name="DatasetFormatValidatorSkill",
            status="executable",
            description="Validate DatasetFormatSpec field coverage and required metadata before execution.",
            task_types=["dataset_profiling", "format_bridge"],
            input_contract={"dataset_format_spec": "yaml", "dataset_root": "directory"},
            output_contract={"validation_report": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"execution_gate": True},
        ),
        SkillSpec(
            name="RunProvenanceSkill",
            status="executable",
            description="Record environment, git, config, and runtime provenance for reproducibility.",
            task_types=["all"],
            input_contract={"run_dir": "directory"},
            output_contract={"run_provenance": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"reproducibility": True},
        ),
        SkillSpec(
            name="BenefitEvidenceReportSkill",
            status="executable",
            description="Unify observer, duplicate/leakage, export, and downstream evaluator results into an evidence-level report.",
            task_types=["all"],
            input_contract={"recipe_execution": "json", "agent_state": "json"},
            output_contract={"benefit_evidence": "json_markdown"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"does_not_claim_downstream_gain_without_evaluator": True},
        ),
        SkillSpec(
            name="PlanCriticSkill",
            status="executable",
            description="Bounded critic that checks task-skill mismatch, evaluator gaps, format risks, and unsupported benefit claims.",
            task_types=["all"],
            input_contract={"augmentation_plan": "json", "recipe": "yaml", "format_bridge": "json"},
            output_contract={"plan_critic": "json_markdown"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"verification_only": True},
        ),
        SkillSpec(
            name="CandidateRecipePilotSkill",
            status="executable",
            description="Compile top-k planner candidates into small pilot recipes and optionally execute dry-run pilots for comparison.",
            task_types=["all"],
            input_contract={"augmentation_plan": "json", "validated_profile": "json", "intent_spec": "json"},
            output_contract={"candidate_recipes": "directory", "pilot_report": "json"},
            cost={"gpu_required": False, "runtime": "low_to_medium"},
            safety={"defaults_to_dry_run": True, "bounded_top_k": True},
        ),
        SkillSpec(
            name="BoundedAutoRepairSkill",
            status="executable",
            description="Create a whitelist-bounded repair recipe and optionally execute the revised recipe for a limited trial.",
            task_types=["all"],
            input_contract={"recipe": "yaml", "recipe_execution": "json"},
            output_contract={"auto_repair_report": "json", "revised_recipe": "optional_yaml"},
            cost={"gpu_required": False, "runtime": "low_to_high_if_execute_revised"},
            safety={"bounded": True, "requires_execute_revised_flag": True},
        ),
        SkillSpec(
            name="CompareRunsSkill",
            status="executable",
            description="Compare multiple SAGA runs by planner choice, observers, export, and evaluator outcomes.",
            task_types=["all"],
            input_contract={"run_dirs": "list"},
            output_contract={"run_comparison": "json"},
            cost={"gpu_required": False, "runtime": "low"},
            safety={"does_not_mutate_runs": True},
        ),
        SkillSpec(
            name="DatasetCardSkill",
            status="planned",
            description="Generate a dataset card for exported augmented datasets.",
            task_types=["all"],
            input_contract={"export_report": "json", "recipe": "yaml"},
            output_contract={"dataset_card": "markdown"},
            cost={"gpu_required": False, "runtime": "low"},
        ),
    ]


def get_skill_registry() -> dict[str, SkillSpec]:
    return {spec.name: spec for spec in built_in_skill_specs()}


def skill_registry_payload() -> dict[str, Any]:
    specs = [spec.to_dict() for spec in built_in_skill_specs()]
    return {
        "schema_version": SKILL_REGISTRY_VERSION,
        "skills": specs,
        "summary": {
            "total": len(specs),
            "executable": len([spec for spec in specs if spec["status"] == "executable"]),
            "planned": len([spec for spec in specs if spec["status"] == "planned"]),
        },
    }


def skill_registry_for_prompt() -> list[dict[str, Any]]:
    rows = []
    for spec in built_in_skill_specs():
        rows.append(
            {
                "name": spec.name,
                "status": spec.status,
                "description": spec.description,
                "task_types": spec.task_types,
                "input_contract": spec.input_contract,
                "output_contract": spec.output_contract,
                "cost": spec.cost,
                "safety": spec.safety,
                "limitations": spec.limitations,
            }
        )
    return rows


def executable_skill_names() -> set[str]:
    return {spec.name for spec in built_in_skill_specs() if spec.status == "executable"}


def planned_skill_names() -> set[str]:
    return {spec.name for spec in built_in_skill_specs() if spec.status == "planned"}


def save_skill_registry(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir).expanduser().resolve()
    payload = skill_registry_payload()
    save_json(path / "skill_registry.json", payload)
    save_text(path / "skill_registry.md", render_skill_registry_markdown(payload))
    return payload


def render_skill_registry_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SAGA Skill Registry",
        "",
        f"- Schema: `{payload.get('schema_version')}`",
        f"- Total skills: {payload.get('summary', {}).get('total')}",
        f"- Executable: {payload.get('summary', {}).get('executable')}",
        f"- Planned: {payload.get('summary', {}).get('planned')}",
        "",
    ]
    for spec in payload.get("skills", []):
        lines.extend(
            [
                f"## {spec.get('name')}",
                "",
                f"- Status: {spec.get('status')}",
                f"- Task types: `{spec.get('task_types')}`",
                f"- GPU required: {spec.get('cost', {}).get('gpu_required')}",
                f"- Runtime: {spec.get('cost', {}).get('runtime')}",
                f"- Description: {spec.get('description')}",
                "",
            ]
        )
    return "\n".join(lines)

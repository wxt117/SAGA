# SAGA Skill Catalog Draft

This catalog lists the skills SAGA currently knows how to reason about. A skill may be executable, planned, or purely evaluative. The benefit-aware planner uses the corresponding `SkillEffectCard` to estimate whether a skill is likely to help a dataset before downstream experiments are available.

## Executable Skills

### DatasetProfileReportSkill

Records dataset profile, bridge, and validation artifacts for provenance. It is a support skill, not an augmentation method.

### FileSelectionSkill

Selects images using validated metadata filters. It is useful when a user says, for example, "only use depression angle 20" and the field has passed DatasetFormatSpec validation.

### StyleTransferSkill

Runs the wrapped SAR feature/style transfer code from `myproject/stytransfer`. It is currently the main executable domain/payload adaptation skill.

Best for: explicit cross-platform, cross-payload, or cross-domain feature transfer when both content and style examples are available.

Main risk: over-transfer can damage target structure, so it should be followed by quality and structure-preservation checks.

### DiffusionLoRAGenerationSkill

Wraps text-caption LoRA training and inference from the qinglong trainer project. It is currently the main executable deep generative augmentation skill for paired image/txt datasets or validated metadata datasets that can be staged into captions.

Best for: high-quality target-level sample expansion when the user has enough GPU memory and no strict speed requirement.

Current capability: if `.txt` captions are missing, SAGA can create a temporary `metadata_caption_dataset/` from a validated `DatasetFormatSpec`, including metadata such as class, azimuth, band, resolution, and polarization when available.

Main limits: medium-to-slow runtime, weak pose control without ControlNet or physical priors, and possible overfit or label drift on small datasets.

### QualityEvaluationSkill

Checks generated output count, image readability, grayscale dynamic range, and black/white saturation ratios. It is an observer gate, not a downstream task metric.

### DistributionEvaluationSkill

Compares generated samples against the reference training image distribution with fast numerical metrics. It uses sampled `pytorch_fid` when available, plus lightweight SAR-oriented proxy metrics such as `sar_fid_lite`, `histogram_jsd`, `mmd_rbf`, generated diversity, and nearest-reference distance.

Best for: default post-generation validation when the user wants quick numerical evidence and does not want to spend time on downstream task training.

Main limits: these metrics can catch obvious distribution mismatch, but they do not prove downstream classifier or detector improvement.

### RepairPolicySkill

Turns observer triggers into bounded repair suggestions. For supported skills such as `DiffusionLoRAGenerationSkill`, it can also create a whitelisted parameter-repair plan and a revised recipe artifact. It does not automatically rerun expensive skills.

Current LoRA parameter whitelist: `training_epochs`, `max_train_samples`, `prompt_count`, `inference_steps`, `inference_guidance`, `inference_cfg_scale`, and `lora_multiplier`.

### ExportDatasetSkill

Packages generated outputs into an auditable augmented dataset with manifest, dataset card, provenance, and optional image-stem `.txt` caption sidecars for text-caption diffusion training. LoRA generation recipes enable caption sidecar export by default.

## Executable Augmentation And Evaluation Skills

### TraditionalAugmentationSkill

Low-cost SAR-aware deterministic transforms for classification, detection, and segmentation baselines. This skill is now executable through `saga traditional-augment` and `agent-run` recipes.

Candidate operations: flip, rotation, crop, intensity perturbation, speckle/noise, and other SAR-aware conservative transforms.

Best for: simplest augmentation requests, fastest turnaround, low-risk baselines, and users without GPU time.

Main limits: limited diversity and no true new target geometry or viewpoint coverage.

Direct CLI use supports validated metadata filters:

```bash
conda run -n sd3 python -m saga traditional-augment \
  --dataset exampledataset/车辆数据-全极化/ZJGC-X \
  --dataset-config runs/agent_run/<run>/validated_dataset.yaml \
  --exclude-polarization pauli \
  --target-count 8 \
  --output runs/traditional_aug/example \
  --run
```

The skill records a reproducible `augmentation_plan.json` containing every selected source, operation, parameter, and deterministic seed. `traditional_aug_run.json` stores the same run's summary and plan digest.

### SARArtifactEvaluationSkill

SAR-specific observer for generated or augmented images. It checks stripe energy, smooth background gradients, bright target compactness, target fragmentation, and target centeredness.

Best for: catching visual SAR artifacts that generic black/white/dynamic-range checks miss.

Main limits: heuristic quality observer; it does not prove downstream task improvement.

## Planned Augmentation And Generation Skills

### SARPreprocessSkill

SAR normalization and preprocessing before generation, style transfer, or evaluation.

### GANImageToImageSkill

Fast image-to-image generation for simple target appearance variation.

Best for: simple targets, simple datasets, MSTAR-like data, and fast image-to-image variation.

Main limits: less suitable for complex structures or controllable high-fidelity generation; mode collapse and label drift remain risks.

### GeoDiffSARSkill

GeoDiff-SAR sparse-azimuth completion with diffusion, LoRA, ControlNet, and 3D/physical priors.

Best for: high-quality sparse azimuth completion when the user has 3D/physical priors and quality matters more than time or VRAM.

Current capability: executable through `saga geodiff-sar` and `agent-run` recipes. The wrapper composes `myproject/geodiff_components/preprocess3` for extracting real-image GEFM condition maps, `myproject/geodiff_components/raytracing` for computing GEFM from 3D models at inference angles, and `myproject/qinglong_trainer_29b` for FLUX ControlNet training/inference.

Main limits: complex workflow, high runtime and GPU memory cost, and dependence on the correctness of priors and GEFM condition maps.

### GaussianSplattingCompletionSkill

Lightweight sparse-view target reconstruction and novel-view completion.

Best for: fast, lightweight sparse azimuth completion when VRAM is limited or when GeoDiff-SAR is too expensive.

Main limits: weaker angle-completion capability than GeoDiff-SAR and possible mismatch with SAR scattering physics.

### BackgroundGenerationSkill

Executable controlled SAR background generation using `myproject/scene_gen_segment_large`.

Policy: SAGA should not ask ordinary users to train background generators. If the user needs backgrounds, SAGA calls the packaged segmentation-controlled FLUX ControlNet generator and its trained weights directly.

Best for: explicit SAR background/scene generation requests, detection and scene-composition workflows where background diversity matters, and later target-background synthesis.

Main limits: generated backgrounds are scene assets, not default target-chip classification augmentation; downstream benefit should be checked after composition or task-specific use.

### TargetBackgroundCompositionSkill

Executable SAR-aware target/background image fusion for scene composition. The implementation is deterministic and CPU-friendly: it can infer simple target masks, match target intensity to the background ROI, and blend with `feather`, `laplacian`, `poisson`, or `hard` modes. Poisson mode uses OpenCV `seamlessClone` when `cv2` is available and falls back to feather blending otherwise.

Best for: explicit target-background, target-scene, or image-fusion requests, especially when the user has a target chip bank and a background/scene bank.

Outputs: composed images, auto/provided masks, preview strips, caption sidecars, `composition_plan.json`, `composition_manifest.jsonl`, and a skill report.

Main limits: auto masks are heuristic; detection boxes/masks and downstream benefit claims require quality, distribution, SAR artifact, and annotation checks. For SAR target chips, `feather` or `laplacian` is the default safer policy; Poisson can over-smooth or alter scatterer contrast.

Direct CLI use:

```bash
conda run -n sd3 python -m saga compose-target-background \
  --targets exampledataset/targets \
  --backgrounds exampledataset/backgrounds \
  --output runs/composition/example \
  --target-count 10 \
  --blend-mode feather \
  --run
```

Natural-language `agent-run` can also select it:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "把 exampledataset/targets 作为目标图，exampledataset/backgrounds 作为背景图，做目标和背景图像融合，生成10张，使用羽化融合" \
  --output runs/composition_agent/example \
  --run
```

## Downstream Evaluators

### ClassificationEvaluationSkill

SAGA standard downstream classification evaluator for measuring whether augmentation improves recognition performance. It references `myproject/PyTorch-Image-Models-Multi-Label-Classification-main`, but no longer assumes the user dataset already has that project's CSV schema.

The evaluator now has four stages:

```text
source dataset or SAGA export
  -> SourceAdapter
  -> project-ready benchmark dataset
  -> baseline-vs-augmented train/validation commands
  -> standard SAGA classification_evaluation.json
```

Supported sources include:

- the reference project's `all.csv/train.csv/val.csv` format
- a SAGA exported augmented dataset with `manifest.jsonl`
- a plain image folder or class-folder dataset
- an optional validated SAGA dataset config for semantic labels/metadata

For the referenced project, SAGA writes `all.csv`, `train.csv`, and `val.csv` with columns `image_path,gender,article,color`. By default, SAR class is mapped to `article`, polarization is mapped to `color`, and domain/source is mapped to `gender`. Baseline and augmented runs share the same held-out validation split; augmented samples are added to the training split only unless explicitly configured otherwise.

Default command:

```bash
conda run -n sd3 python -m saga classification-eval \
  --baseline-dataset exampledataset/车辆数据-全极化/ZJGC-X \
  --augmented-dataset runs/agent_run/<run>/augmented_dataset \
  --output runs/classification_eval/example \
  --config configs/skills/classification_evaluation.yaml \
  --dry-run
```

The generated `classification_evaluation_commands.sh` trains a baseline model and an augmented model, locates each best checkpoint directory, validates both on the same held-out split, and writes comparable metrics. The project `validate.py` has been patched to include `acc1_color`, `acc1_gender`, and `acc1_article` in structured CSV output; SAGA uses `acc1_article` as the default primary metric.

### ObjectDetectionEvaluationSkill

Detection training/evaluation loop for mAP, AP per class, small-object recall, and background/generalization gains.

### SegmentationEvaluationSkill

Segmentation training/evaluation loop for mIoU, Dice, boundary quality, and mask consistency.

## Feedback Needed

For each skill, SAGA still needs expert priors from the user:

```yaml
name:
status:
main_goal:
supported_tasks:
input_requirements:
required_metadata:
addresses_dataset_needs:
strengths:
weaknesses:
risks:
cost:
best_use_cases:
bad_use_cases:
controllable_params:
expected_benefit_prior:
evaluation:
```

The current defaults are deliberately conservative. They should be corrected with project-specific knowledge before SAGA uses them for serious recipe comparison.

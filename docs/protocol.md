# SAGA Protocol Draft

SAGA is a downstream-task-oriented SAR data generation and augmentation decision system. The protocol defines the shared objects used by the LLM planner, augmentation skills, generation models, composition tools, evaluators, and reports.

This draft is intentionally model-agnostic. GAN, diffusion, GeoDiff-SAR, style transfer, segment-controlled background generation, Gaussian splatting completion, and future methods should all plug into the same request, asset, recipe, and provenance contracts.

The first implementation target is classification-oriented target augmentation and target generation. Detection, segmentation, background generation, and scene composition should use the same core objects later, but they do not need to be fully implemented in the first version.

SAGA fixes its internal normalized sample format, not the user's input dataset format. User input layouts should be described by a configurable `DatasetFormatSpec`, which can be written by a user, adapted from a template, or inferred by an LLM from dataset samples and user-provided text.

## Design Principles

- LLMs plan, route, explain, and summarize. SAR image recognition and quality scoring are handled by SAR-specific tools, encoders, detectors, segmenters, statistical analyzers, and validation modules.
- Every generated sample must be traceable to its source data, skill, model, parameters, and random seed.
- Skills should exchange structured artifacts rather than informal file paths whenever possible.
- The same protocol must support offline dataset generation, online training-time augmentation, and downstream benchmarking.
- User-provided validation and test data must not be used to build target banks, background banks, style banks, or generative training assets unless explicitly allowed.

## Top-Level Request

`SagaRequest` is the top-level object passed to the agent.

```yaml
request_id: user_aircraft_sparse_azimuth_cls_v1
task:
  type: classification
  target_classes: [aircraft]
  downstream_model: optional
  primary_metric: accuracy

goals:
  - complete_sparse_azimuth
  - improve_classification
  - adapt_to_new_sensor

dataset:
  config: configs/datasets/user_aircraft.yaml

constraints:
  max_augmented_ratio: 3.0
  preserve_annotations: true
  allow_traditional_aug: true
  allow_gan: true
  allow_diffusion: true
  allow_geodiff_sar: true
  allow_style_transfer: true
  allow_background_generation: false
  allow_gaussian_splatting: true
  allow_scene_composition: false
  use_train_split_only_for_asset_banks: true

preferences:
  speed: balanced       # fast | balanced | quality
  quality: high         # draft | standard | high
  explainability: high
  output_format: saga_pair   # saga_pair | imagefolder | coco | yolo | dota | native
```

## Dataset Config

`SagaDatasetConfig` describes the user's dataset before SAGA normalizes it.

```yaml
name: user_sar_dataset
root: /path/to/dataset
format: custom
format_spec: configs/format_specs/user_dataset_format.yaml
task: classification
splits:
  train: train
  val: val
  test: test
image:
  domain: intensity       # unknown | complex_slc | amplitude | intensity | log_intensity | db
  channels: 1
  bit_depth: unknown
sar:
  sensor: unknown
  platform: unknown
  polarization: VV
  resolution_m: 1.0
  incidence_angle_range: unknown
  look_direction: unknown
classes:
  - id: 0
    name: aircraft
metadata_fields:
  has_target_azimuth: false
  has_instance_mask: false
  has_shadow_mask: false
label:
  source: filename           # txt_sidecar | filename | directory | mixed
  txt_schema: auto           # saga_kv_v1 | caption_v1 | auto
  filename_schema: dataset_format_spec
  pairing: numeric_stem      # used only when source is txt_sidecar
  require_contiguous_index: true
```

When fields are unknown, SAGA should record that uncertainty rather than silently inventing metadata.

## User Input Dataset Layouts

SAGA should support both generation-project image/text pairs and user-uploaded datasets where class and SAR parameters are encoded in paths, filenames, sidecar text, or a user description. These layouts should not be hard-coded in the core parser. They should be represented with `DatasetFormatSpec`.

Recommended first-wave formats:

- `saga_pair`: numeric image/text pairs, such as `1.png` + `1.txt`
- `custom`: metadata parsed through a user-specific `DatasetFormatSpec`
- `mixed`: text sidecars when available, filename parsing otherwise

The full strategy is documented in [data_format_strategy.md](data_format_strategy.md).

### Numeric Image/Text Pairs

Your generation-oriented datasets can use a simple numeric pair layout:

```text
dataset_root/
  train/
    1.png
    1.txt
    2.png
    2.txt
    3.png
    3.txt
  val/
    1.png
    1.txt
  test/
    1.png
    1.txt
```

Rules:

- image and text files must share the same numeric stem
- files should be sorted by integer value, not string value
- valid image suffixes: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`
- each image should have exactly one sidecar `.txt`
- missing indices can be allowed later, but the first version should warn or fail when `require_contiguous_index` is true
- split directories are optional; if no split exists, SAGA can treat the whole folder as `train` or create a split by config

Example split-free layout:

```text
dataset_root/
  1.png
  1.txt
  2.png
  2.txt
```

Config:

```yaml
name: generated_aircraft_targets
root: /path/to/dataset_root
format: saga_pair
task: classification
splits:
  train: .
label:
  source: txt_sidecar
  txt_schema: auto
  pairing: numeric_stem
  require_contiguous_index: true
```

### Filename Metadata Datasets

Some user datasets have no `.txt` labels. The label comes from directory names and filename tokens.

Example:

```text
exampledataset/B747/海况0级_聚焦后_波音747级民航客机(NULL民航客机)_850_20_0_0.5_X.tif
```

Possible parsed metadata:

```yaml
class: B747
target_name_cn: 波音747级民航客机
sea_state: 0
processing: focused
range_or_param_1: 850
incidence_angle_deg: 20
azimuth_deg: 0
resolution_m: 0.5
band: X
```

Example:

```text
exampledataset/车辆数据-全极化/d7(装甲工程车）/Ku/inci-20-azim-345.0000-AHH.png
```

Possible parsed metadata:

```yaml
class: d7
class_description: 装甲工程车
band: Ku
incidence_angle_deg: 20
azimuth_deg: 345.0
polarization: AHH
```

For filename datasets, the parser should store both parsed fields and the original relative path so the mapping remains auditable.

## TXT Label Standard

SAGA should support two text styles:

1. `caption_v1`: existing free-form generation captions, such as `"SAR image, 0.5m, Ku-band, aircraft, azimuth 45 deg, VV polarization"`.
2. `saga_kv_v1`: a stable key-value format that is easier to parse, validate, and generate.

The recommended first standard is `saga_kv_v1`.

```text
schema: saga_kv_v1
task: classification
class: aircraft
caption: SAR image, 0.5m, Ku-band, aircraft, azimuth 45 deg, VV polarization
resolution_m: 0.5
band: Ku
polarization: VV
azimuth_deg: 45
incidence_angle_deg: unknown
sensor: unknown
image_domain: intensity
source: generated
method: geodiff_sar
```

Required fields for classification MVP:

- `schema`
- `task`
- `class`

Strongly recommended SAR/generation fields:

- `caption`
- `resolution_m`
- `band`
- `polarization`
- `azimuth_deg`
- `incidence_angle_deg`
- `sensor`
- `image_domain`
- `source`
- `method`

Allowed unknown value:

```text
unknown
```

Why keep `caption` even in key-value labels:

- it preserves compatibility with existing text-conditioned generation scripts
- it gives diffusion models a direct prompt-like input
- it allows SAGA to parse structured fields while retaining the original generative description

Existing free-form captions can still be ingested:

```text
SAR image, 0.5m, Ku-band, aircraft target, azimuth 45 deg, VV polarization
```

When `txt_schema: auto`, SAGA should:

1. parse key-value labels if a `schema:` line is present
2. otherwise treat the txt as `caption_v1`
3. extract known fields using conservative patterns
4. mark unparsed fields as `unknown`
5. keep the original text as `caption`

## Normalized Dataset Object

Internally, datasets are represented as `SagaDataset`.

```yaml
dataset_id: user_sar_dataset:v1
name: user_sar_dataset
task: classification
format: saga_native
splits:
  train:
    image_count: 1200
    sample_count: 1200
  val:
    image_count: 300
    sample_count: 300
classes:
  - id: 0
    name: aircraft
metadata:
  image_domain: intensity
  polarization: VV
  resolution_m: 1.0
artifacts:
  native_annotations: .saga/datasets/user_sar_dataset/annotations.jsonl
  image_index: .saga/datasets/user_sar_dataset/images.jsonl
```

## Sample Object

`SagaSample` is the unit processed by augmentation, generation, validation, and export.

```yaml
sample_id: train_000123
split: train
image:
  path: images/train/000123.png
  width: 1024
  height: 1024
  domain: intensity
label:
  class_id: 0
  class_name: aircraft
  caption: SAR image, 0.5m, Ku-band, aircraft, azimuth 45 deg, VV polarization
  attributes:
    target_azimuth_deg: 45
    synthetic: false
metadata:
  sensor: unknown
  polarization: VV
  resolution_m: 1.0
  incidence_angle_deg: null
  band: Ku
  original_relpath: B747/海况0级_聚焦后_波音747级民航客机(NULL民航客机)_850_20_0_0.5_X.tif
provenance:
  source: user
  synthetic: false
```

## Asset Object

`SagaAsset` represents reusable target, background, style, layout, model output, or feature assets. In the classification-first MVP, most assets are whole target images or generated target candidates rather than cropped detection instances.

```yaml
asset_id: target_aircraft_000321
asset_type: target        # target | background | style | layout | feature | generated_candidate
source:
  dataset_id: user_sar_dataset:v1
  sample_id: train_000123
  split: train
data:
  image_path: banks/targets/aircraft/target_aircraft_000321.png
  txt_path: banks/targets/aircraft/target_aircraft_000321.txt
  mask_path: null
  annotation_path: null
metadata:
  class_name: aircraft
  target_azimuth_deg: 90
  sensor: unknown
  polarization: VV
  resolution_m: 1.0
  background_type: runway
quality:
  score: 0.86
  warnings: []
provenance:
  created_by_skill: TargetBankBuilderSkill
  created_at: null
  recipe_id: null
```

## Skill Interface

Every skill should expose a structured capability description and a stable run contract.

```python
class SagaSkill:
    name: str
    version: str

    def describe(self) -> dict:
        """Return capabilities, accepted inputs, produced outputs, and constraints."""

    def can_handle(self, request: dict, context: dict) -> float:
        """Return a score in [0, 1] used by the planner."""

    def run(self, inputs: dict, params: dict, context: dict) -> dict:
        """Execute the skill and return structured artifact references."""
```

Recommended skill output:

```yaml
skill: GeoDiffSARSkill
version: 0.1.0
status: succeeded
artifacts:
  generated_assets: banks/generated/geodiff_sar/assets.jsonl
  preview_dir: runs/previews/geodiff_sar
summary:
  requested_count: 500
  generated_count: 500
  accepted_count: 430
  rejected_count: 70
warnings:
  - target_azimuth metadata inferred for 38 percent of source targets
metrics:
  mean_quality_score: 0.82
```

## Core Skill Families

### DatasetDiagnosisSkill

Purpose: normalize the dataset, inspect basic data health, infer missing high-level properties when possible, and generate a diagnosis used by the planner.

Inputs:

- `SagaDatasetConfig`
- optional user task and goals

Outputs:

- `dataset_diagnosis.json`
- normalized dataset index
- warnings and missing metadata report

For classification-first datasets, diagnosis focuses on:

- class distribution
- image/text pair validity
- filename metadata parse validity
- numeric order consistency
- caption and key-value field coverage
- azimuth distribution
- resolution, band, polarization, and sensor coverage when available
- duplicate or near-duplicate images
- possible domain clusters from image statistics or SAR encoder embeddings

### AssetBankSkill

Purpose: build and manage target, background, style, layout, and generated candidate banks.

Subskills:

- `TargetBankBuilderSkill`
- `BackgroundBankBuilderSkill`
- `StyleBankBuilderSkill`
- `AssetRetrievalSkill`

For the classification MVP, `TargetBankBuilderSkill` can simply normalize whole image/text pairs into a class-organized target bank.

### TraditionalAugSkill

Purpose: SAR-aware lightweight augmentation with annotation synchronization.

Examples:

- geometric transforms with task-specific limits
- intensity and log-intensity jitter
- speckle noise
- resolution degradation
- crop, resize, mosaic, mixup, cutout

For classification, this skill returns transformed image/text pairs and preserves the class label unless a method explicitly changes semantic content.

### GANImageTranslationSkill

Purpose: fast image-to-image translation for simple target or domain transformations.

Expected constraints:

- must preserve annotation geometry unless explicitly marked otherwise
- should not be used for strict sparse-azimuth physical completion

### DiffusionGenerationSkill

Purpose: generic diffusion-based target, image, or inpainting candidate generation.

Expected outputs should usually enter `generated_candidate` banks before being accepted into training data.

### GeoDiffSARSkill

Purpose: SAR target sparse-azimuth completion using diffusion, 3D physical priors, LoRA, and ControlNet.

Primary use cases:

- target azimuth gap filling
- physically constrained target generation
- high-quality target candidate expansion

### StyleTransferSkill

Purpose: cross-payload, cross-sensor, or cross-domain feature/style transfer while preserving target geometry and annotations.

### SegmentControlledBackgroundGenerationSkill

Purpose: generate SAR backgrounds from scene type, segment map, layout template, and optional domain condition.

Recommended control levels:

- `type_controlled`: scene type only
- `segment_controlled`: segment map or layout template
- `condition_controlled`: segment map plus sensor, polarization, resolution, or style condition

### GaussianSplattingCompletionSkill

Purpose: lightweight and fast sparse-azimuth target completion, especially for bulk candidate generation.

### SceneCompositionSkill

Purpose: compose target assets and background assets into training images with valid annotations.

Minimum expected features:

- allowed-region placement
- overlap avoidance
- bbox and mask update
- local intensity matching
- edge feathering or blending
- provenance record

## Existing Method Adapter Card

Each existing generation or transfer project should be described by an adapter card before it is wrapped by SAGA. This is the format you can fill in around each codebase.

```yaml
method_name: GeoDiff-SAR
adapter_name: geodiff_sar
task_type:
  - classification_target_generation
  - sparse_azimuth_completion
current_entrypoint: /path/to/run_geodiff_sar.py
environment: sd3
execution:
  type: python_script        # python_script | shell_script | python_api
  command_template: >
    conda run -n sd3 python /path/to/run_geodiff_sar.py
    --input {input_dir}
    --output {output_dir}
    --class {class_name}
    --azimuth {azimuth_deg}
input_requirements:
  required:
    - image
    - txt_label
    - class
    - azimuth_deg
  optional:
    - resolution_m
    - band
    - polarization
    - incidence_angle_deg
    - sensor
outputs:
  image_pattern: "{index}.png"
  txt_pattern: "{index}.txt"
  metadata_format: saga_kv_v1
checkpoint_paths:
  base_model: /path/to/base_model
  lora: /path/to/lora
  controlnet: /path/to/controlnet
runtime:
  gpu_required: true
  expected_speed: medium
limitations:
  - requires target azimuth in txt label
  - output quality depends on 3D prior availability
```

For a script that only accepts captions, the adapter can map structured fields into a caption string:

```yaml
prompt_template: "SAR image, {resolution_m}m, {band}-band, {class}, azimuth {azimuth_deg} deg, {polarization} polarization"
```

## Recipe Object

`SagaRecipe` is the executable augmentation plan.

```yaml
recipe_id: aircraft_sparse_azimuth_cls_saga_v1
created_from_request: user_aircraft_sparse_azimuth_cls_v1
goals:
  - complete_sparse_azimuth
  - improve_classification

pipeline:
  - step_id: diagnose
    skill: DatasetDiagnosisSkill
    params: {}

  - step_id: build_target_bank
    skill: TargetBankBuilderSkill
    params:
      split: train
      mode: whole_image_pair

  - step_id: complete_azimuth_fast
    skill: GaussianSplattingCompletionSkill
    params:
      desired_azimuth_step_deg: 15
      candidate_count: 2000

  - step_id: complete_azimuth_quality
    skill: GeoDiffSARSkill
    params:
      desired_azimuth_step_deg: 15
      candidate_count: 500
      lora_profile: aircraft_lora_default
      controlnet_profile: geometry_control_default

  - step_id: traditional_aug
    skill: TraditionalAugSkill
    params:
      rotate_deg: [-10, 10]
      speckle_noise_p: 0.3
      intensity_jitter_p: 0.3

filters:
  - name: LabelValidationSkill
  - name: DataQualityEvaluationSkill

output:
  augmented_ratio: 3.0
  format: saga_pair
  root: datasets/user_aircraft_cls_saga_v1
```

## Provenance Record

Every synthetic sample and asset should write one JSONL record.

```json
{
  "sample_id": "saga_000001",
  "synthetic": true,
  "output_image": "datasets/user_aircraft_cls_saga_v1/train/100001.png",
  "output_label": "datasets/user_aircraft_cls_saga_v1/train/100001.txt",
  "recipe_id": "aircraft_sparse_azimuth_cls_saga_v1",
  "pipeline_steps": [
    {
      "skill": "GeoDiffSARSkill",
      "model_id": "geodiff_sar_aircraft_v1",
      "source_assets": ["target_aircraft_000321"],
      "params": {
        "target_azimuth_deg": 45
      },
      "seed": 20260515
    },
  ],
  "quality": {
    "accepted": true,
    "score": 0.84,
    "warnings": []
  }
}
```

## Agent Output

The planner should return structured outputs, not just natural language.

```yaml
status: ready_to_run
diagnosis_path: runs/diagnosis/user_aircraft/report.json
recipe_path: runs/recipes/aircraft_sparse_azimuth_cls_saga_v1.yaml
selected_skills:
  - DatasetDiagnosisSkill
  - TargetBankBuilderSkill
  - GaussianSplattingCompletionSkill
  - GeoDiffSARSkill
  - TraditionalAugSkill
warnings:
  - incidence_angle metadata is missing
  - target azimuth labels are incomplete; azimuth completion may need inferred labels
explanation: >
  The dataset appears to have sparse aircraft azimuth coverage.
  The recipe therefore combines image/text pair diagnosis, target bank normalization,
  azimuth completion, and conservative SAR-aware traditional augmentation.
```

## Minimum CLI Contract

All Python commands should run inside the `sd3` environment.

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成1张747 SAR增广图像" \
  --output runs/agent_run/lora_747_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
  --dry-run \
  --no-execute
```

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成1张747 SAR增广图像" \
  --output runs/agent_run/lora_747_llm_intent_planner_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
  --diffusion-lora-config configs/skills/diffusion_lora_smoke.yaml \
  --use-llm-intent \
  --use-llm-planner \
  --llm-config configs/llm/deepseek_v4pro.yaml \
  --dry-run \
  --no-execute
```

```bash
conda run -n sd3 python -m saga execute-recipe \
  --recipe runs/agent_run/lora_747_dry/recipe.yaml \
  --output runs/agent_run/lora_747_dry/execution_check \
  --dry-run
```

```bash
conda run -n sd3 python -m saga memory list \
  --memory-dir runs/memory \
  --limit 10
```

## Open Fields To Confirm

The protocol needs project-specific details for the classification MVP:

- whether numeric pair indices must be contiguous or can have gaps
- whether split directories are required, optional, or created by SAGA
- first `txt` label schema and caption parsing rules
- first target classes and whether class-specific policies are required
- available metadata: azimuth angle, incidence angle, sensor, polarization, resolution, band, orbit/look direction
- current script interfaces for GAN, diffusion, GeoDiff-SAR, style transfer, and Gaussian splatting
- expected model artifact layout and checkpoint registry format
- preferred output format for first MVP: `saga_pair` is currently recommended
- whether downstream classification evaluation should be included in MVP-1 or delayed

# SAGA Architecture Draft

SAGA is a SAR data generation and augmentation decision system optimized for downstream task gains. It should not be built as a loose collection of augmentation scripts. It should be built as an agent-orchestrated system with explicit data contracts, pluggable skills, model registries, asset banks, provenance tracking, and evaluation hooks.

The first implementation target is classification-oriented SAR target augmentation and generation. Most existing generation projects naturally produce target-level image/text pairs, so SAGA should first make that workflow reliable before expanding into detection, background synthesis, and scene composition.

SAGA should not hard-code user input dataset layouts. It should use a fixed internal sample format and a configurable `DatasetFormatSpec` for each input dataset. The spec can be inferred by an LLM from an inspection report plus the user's natural-language explanation.

The agent architecture should be treated as seven explicit layers:

```text
User Request / Dataset
        |
        v
1. Dataset Profiler
        |
        v
2. Task & Intent Recognizer
        |
        v
3. LLM / Rule Hybrid Planner
        |
        v
4. Recipe Generator
        |
        v
5. Skill Executor
        |
        v
6. Observer / Evaluator
        |
        v
7. Repair / Replan / Memory
        |
        v
Augmented Dataset + Report
```

The current implementation now covers the first executable spine across all seven layers. `Dataset Profiler` must observe before interpretation: deterministic code scans the raw dataset, the intent recognizer extracts user goals and format hints from natural language, LLM-facing modules may propose semantic mappings from compact summaries, and only deterministic validation can promote those mappings into trusted ingestion contracts.

The first Plan-then-Execute runtime is documented in [agent_runtime.md](agent_runtime.md). It introduces `agent-run` for natural-language dry-runs and `execute-recipe` for executing serialized `SagaRecipe` DAGs. The MVP runtime now includes an optional LLM intent recognizer, a hybrid multi-plan LLM planner, deterministic rule guardrails, a formal `SkillRegistry`, prior-based `SkillEffectCards`, `DatasetNeedProfile`, `PlanUtilityReport`, `PlanVerificationReport`, deterministic observer gates (`QualityEvaluationSkill` and `DistributionEvaluationSkill`), a bounded repair layer (`RepairPolicySkill` plus whitelisted parameter repair for LoRA recipes), a traceable export layer (`ExportDatasetSkill`), and a policy memory index, so execution results can be judged, packaged, and recorded before downstream task evaluation without giving the LLM free control over reruns.

Current implemented spine:

```text
User request
  -> rule IntentSpec + optional LLM intent proposal
  -> RawDatasetProfile
  -> DatasetFormatSpec compiler + validator
  -> SkillRegistry + memory retrieval
  -> DatasetNeedProfile + SkillEffectCards
  -> LLM multi-plan proposal + benefit-aware deterministic plan ranking
  -> planner guardrail + PlanVerificationReport
  -> rule Recipe Generator
  -> deterministic Skill Executor
  -> Observer / RepairPolicy / Export
  -> Policy Memory record
```

## System Positioning

SAGA answers four questions for a user-provided SAR dataset:

1. What is missing or weak in this dataset?
2. Which augmentation and generation skills should be used under the user's constraints?
3. How should generated targets, backgrounds, styles, and scenes be composed into valid training data?
4. What evidence suggests that the generated data is useful for the downstream task?

The LLM component is responsible for planning, routing, explaining, and report writing. SAR image understanding should be handled by dedicated SAR tools and models.

```text
User dataset + user goal
  |
  v
Dataset diagnosis and asset extraction
  |
  v
LLM planner + rule-based safeguards
  |
  v
Executable SagaRecipe
  |
  v
Skill execution: augmentation, generation, transfer, composition
  |
  v
Quality gates and provenance tracking
  |
  v
Augmented classification dataset + report + optional downstream benchmark
```

## Main Components

```text
saga/
  agent/
    intent_recognizer.py
    llm_intent.py
    llm_planner.py
    plan_ranker.py
    plan_verifier.py
    recipe_generator.py
    runtime.py
    continuation.py
    skill_registry.py

  core/
    protocol.py
    config.py
    recipe.py

  data/
    profile.py
    format_bridge.py
    discovery.py
    loader.py

  skills/
    style_transfer/
    diffusion_lora/

  executor/
    recipe_executor.py

  observer/
    quality.py
    repair_policy.py

  exporter/
    dataset_exporter.py

  memory/
    policy_bank.py

  cli.py
```

The research skills remain pluggable. GeoDiff-SAR, Gaussian splatting completion, background generation, GAN image-to-image, traditional augmentation, and downstream task evaluators should be added behind the same skill contract rather than hard-wired into the agent loop.

## Layered Architecture

### 1. Data Layer

Responsibilities:

- read user datasets
- convert annotations into SAGA native representation
- protect validation and test splits from train asset leakage
- normalize image and SAR metadata
- expose samples to skills

Primary objects:

- `SagaDatasetConfig`
- `SagaDataset`
- `SagaSample`

Supported first-wave formats should likely be:

- SAGA numeric image/text pairs: `1.png` + `1.txt`
- caption-style text labels
- key-value text labels
- image-folder classification if needed

Detection formats such as COCO, YOLO, DOTA, segmentation masks, and SAR-specific raw formats should remain planned extensions, but they do not need to be first-wave implementation targets.

Classification pair layout:

```text
dataset_root/
  train/
    1.png
    1.txt
    2.png
    2.txt
  val/
    1.png
    1.txt
```

The `.txt` file is the label and parameter sidecar. It can be either a free-form generation caption or the structured `saga_kv_v1` format defined in [protocol.md](protocol.md).

SAGA should also support filename-driven datasets where parameters are encoded in paths, but these rules should be supplied through `DatasetFormatSpec`, not hard-coded into core ingestion. See [data_format_strategy.md](data_format_strategy.md).

### 2. Diagnosis Layer

Responsibilities:

- inspect class distribution
- inspect target size, aspect ratio, and density
- inspect missing metadata
- find invalid image/text pairs
- find duplicate or near-duplicate images
- infer rough visual domains using statistics and SAR encoder embeddings if available
- identify likely gaps: sparse azimuth, class imbalance, limited backgrounds, cross-sensor domain shift

The diagnosis layer should produce structured JSON that the planner can consume. The LLM can summarize this diagnosis, but should not be the only mechanism doing visual recognition.

Example diagnosis output:

```yaml
dataset_id: user_aircraft:v1
summary:
  image_count: 1200
  sample_count: 1200
  classes:
    aircraft: 1200
issues:
  - type: sparse_azimuth
    severity: high
    evidence: "azimuth labels are available in txt sidecars; observed angles concentrate around 0, 90, and 180 degrees"
  - type: label_schema_mixed
    severity: medium
    evidence: "78 percent of txt files use free-form captions and 22 percent use key-value labels"
missing_metadata:
  - incidence_angle
  - look_direction
recommendations:
  - normalize image/text pairs
  - use sparse azimuth completion
  - preserve original captions while adding structured fields
```

### 3. Asset Layer

Responsibilities:

- build target banks from train split only
- build background banks from valid background regions
- build style/domain banks for cross-payload transfer
- store generated candidates before they become accepted training samples
- provide retrieval APIs for composition and generation

Asset types:

- `target`
- `background`
- `style`
- `layout`
- `feature`
- `generated_candidate`

The asset layer is essential because SAGA's generation skills do not directly dump images into the final training set. They produce assets and candidates that later pass through composition and validation.

For the classification MVP, target assets can be whole image/text pairs. A detection-style crop, bbox, or mask is optional and should not be required.

### 4. Skill Layer

Each skill should be independently runnable and registered with the skill registry.

First-class skills:

- `DatasetDiagnosisSkill`
- `TargetBankBuilderSkill`
- `TraditionalAugSkill`
- `GANImageTranslationSkill`
- `DiffusionGenerationSkill`
- `GeoDiffSARSkill`
- `StyleTransferSkill`
- `GaussianSplattingCompletionSkill`
- `LabelValidationSkill`
- `DataQualityEvaluationSkill`

Detection and background skills should remain planned extensions:

- `BackgroundBankBuilderSkill`
- `SegmentControlledBackgroundGenerationSkill`
- `SceneCompositionSkill`
- `AnnotationValidationSkill`

Skill selection is performed by the planner, but skills themselves must declare capabilities and constraints.

Example capability declaration:

```yaml
name: GeoDiffSARSkill
version: 0.1.0
inputs:
  required:
    - target_assets
    - desired_azimuths
  optional:
    - lora_profile
    - controlnet_profile
    - model_3d_prior
outputs:
  - generated_candidate_assets
best_for:
  - complete_sparse_azimuth
  - target_generation
constraints:
  requires_target_class: true
  requires_model_checkpoint: true
  requires_txt_azimuth: true
```

### 5. Model Adapter Layer

The model adapter layer wraps existing scripts and checkpoints behind stable interfaces. This avoids forcing every research script to be rewritten immediately.

Recommended adapter contract:

```python
class ModelAdapter:
    model_id: str

    def prepare(self, context: dict) -> None:
        ...

    def run(self, inputs: dict, params: dict, output_dir: str) -> dict:
        ...
```

Adapters needed for your current methods:

- GAN image-to-image adapter
- diffusion generation adapter
- GeoDiff-SAR adapter
- style transfer adapter
- segment-controlled background diffusion adapter
- Gaussian splatting completion adapter

The first implementation can invoke existing scripts via subprocess as long as inputs and outputs are normalized into SAGA artifacts.

The current `myproject/stytransfer` code can be wrapped as the first concrete style transfer adapter. Its entrypoint is:

```bash
conda run -n sd3 python myproject/stytransfer/style_transfer_general.py \
  --cnt {content_path_or_dir} \
  --sty {style_path_or_dir} \
  --output-dir {output_dir} \
  --model-path {model_path} \
  --device {device} \
  --image-size {image_size} \
  --steps {steps} \
  --lr {lr} \
  --weight {content_weight} \
  --seed {seed} \
  --self-layers {self_layers}
```

Script behavior observed from the code:

- `--cnt` supports a single image or a directory of content images.
- `--sty` supports a single image or a directory of style images.
- supported input suffixes are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, and `.tiff`.
- output names are built as `{content_stem}_styliedby_{style_stem}.png`.
- if content input is a directory, relative subdirectories are preserved under the output directory.
- default model is `runwayml/stable-diffusion-v1-5`; offline use should provide `--model-path`.

Recommended adapter card:

```yaml
method_name: stytransfer
adapter_name: style_transfer_general
task_type:
  - cross_payload_style_transfer
  - feature_transfer
current_entrypoint: myproject/stytransfer/style_transfer_general.py
environment: sd3
execution:
  type: python_script
  command_template: >
    conda run -n sd3 python myproject/stytransfer/style_transfer_general.py
    --cnt {content}
    --sty {style}
    --output-dir {output_dir}
    --model-path {model_path}
    --device {device}
    --image-size {image_size}
    --steps {steps}
    --lr {lr}
    --weight {content_weight}
    --seed {seed}
    --self-layers {self_layers}
input_requirements:
  required:
    - content_images
    - style_images
  optional:
    - model_path
    - device
    - image_size
    - steps
    - lr
    - content_weight
    - self_layers
outputs:
  image_pattern: "{content_stem}_styliedby_{style_stem}.png"
  label_policy: preserve_source_label_and_add_style_metadata
runtime:
  gpu_required: recommended
  expected_speed: slow_to_medium
limitations:
  - output is png even when input is tif
  - sidecar txt labels are not automatically created by the script
  - SAGA adapter must copy or generate metadata sidecars after transfer
```

Each existing method should provide an adapter card:

```yaml
method_name: GeoDiff-SAR
adapter_name: geodiff_sar
task_type:
  - classification_target_generation
  - sparse_azimuth_completion
current_entrypoint: /path/to/run_geodiff_sar.py
environment: sd3
execution:
  type: python_script
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
outputs:
  image_pattern: "{index}.png"
  txt_pattern: "{index}.txt"
  metadata_format: saga_kv_v1
```

### 6. Planner Layer

The planner combines rule-based safeguards with LLM reasoning.

Rule-based logic should handle high-risk constraints:

- never use val or test splits for asset banks by default
- do not silently overwrite user captions or sidecar txt labels
- do not run GeoDiff-SAR if required checkpoints or 3D priors are unavailable
- do not use style transfer as a substitute for true target azimuth completion
- require quality gates before generated candidates enter the final dataset

LLM planning should handle:

- translating user goals into augmentation objectives
- explaining tradeoffs between speed, quality, and controllability
- proposing candidate skill combinations from diagnosis results and the `SkillRegistry`
- drafting high-level candidate plans, not executable recipes
- writing warnings when metadata is missing

The implemented planner uses this boundary:

```text
LLM proposes candidate_plans
Rule ranker selects a candidate
Rule guardrail validates task, paths, filters, model family, and skill support
Plan verifier checks SkillRegistry + bridge validity + clarification status
Rule recipe generator compiles the executable SagaRecipe
```

This keeps SAGA agentic without making it stochastic at execution time. The LLM is a semantic planner; the rule layer is the executor contract owner.

Planner examples:

```text
Goal: complete sparse azimuth
Diagnosis: txt labels contain azimuth fields but the azimuth distribution has large gaps
Plan:
  normalize image/text pairs -> target bank -> Gaussian splatting bulk candidates -> GeoDiff-SAR high-quality candidates -> label validation
```

```text
Goal: cross-payload adaptation
Diagnosis: train and target domains differ in embedding statistics
Plan:
  style bank -> style transfer -> label validation -> optional traditional augmentation -> downstream benchmark
```

```text
Goal: improve small-target detection
Diagnosis: small objects dominate and background diversity is low
Plan:
  target bank -> segment-controlled background generation -> scene composition -> conservative traditional augmentation
```

This detection-oriented example is deferred until the detection/background stage.

### 7. Execution Layer

The executor runs a recipe step by step.

Responsibilities:

- resolve datasets, assets, model checkpoints, and output directories
- call skills with structured inputs and params
- write step logs
- write provenance JSONL
- stop or continue based on step severity
- collect artifacts for the report

Execution directory:

```text
runs/agent_run/<run_id>/
  agent_state.json
  agent_run.md
  recipe.yaml
  recipe.json
  raw_profile.json
  dataset_profile.json
  format_bridge.json
  validated_dataset_profile.json
  skill_registry.json
  memory_retrieval.json
  planner_proposal.json
  planner_guardrail.json
  plan_verification.json
  execution/
    recipe_execution.json
    steps/
  recipe_artifacts/
  augmented_dataset/
```

### 8. Quality Gate Layer

The quality gate decides whether generated candidates become final training data.

Minimum checks for MVP:

- image/text pair validity
- label parse validity
- class label consistency
- required field coverage
- azimuth field validity when sparse-azimuth completion is requested
- duplicate detection
- train/val/test leakage check
- basic image intensity sanity check

Later checks:

- SAR statistical consistency
- target-background compatibility
- shadow and bright scattering structure consistency
- SAR encoder feature distance
- detector or classifier confidence
- human review queue

### 9. Composition Layer

Scene composition is the junction between target generation and background generation.

This layer is not required for the classification MVP, but it should remain in the architecture because it becomes central for detection and background-controlled augmentation.

MVP requirements:

- take target assets and background assets
- place targets in allowed segment regions
- avoid excessive overlap
- update COCO/YOLO annotations
- apply local intensity matching
- apply simple edge blending
- record provenance

Later extensions:

- learned blending
- shadow-aware placement
- scattering-aware harmonization
- multi-target layout generation
- road, runway, water, and harbor-specific placement policies

### 10. Reporting Layer

The report should be useful to both a researcher and an engineer.

Required report sections:

- dataset diagnosis
- selected recipe and rationale
- generated asset summary
- accepted and rejected sample counts
- key warnings
- representative previews
- provenance summary
- optional downstream benchmark results

The report can be Markdown first, then HTML later.

## Recommended MVP

The current MVP should be understood as the agent spine, not a complete augmentation library:

```text
MVP-1 Agent Spine
  - flexible raw dataset profiling
  - DatasetFormatSpec compiler + deterministic validator
  - rule IntentSpec + optional LLM intent recognizer
  - SkillRegistry
  - SkillEffectCards and DatasetNeedProfile
  - LLM multi-plan proposal
  - benefit-aware deterministic plan ranking and PlanVerificationReport
  - SagaRecipe generation
  - execute-recipe runtime
  - observer / bounded repair / export / policy memory
```

Then harden classification-target augmentation:

```text
MVP-2 Classification Skills
  - traditional SAR-aware augmentation
  - diffusion LoRA generation
  - style transfer
  - GAN image-to-image adapter
  - classification data quality evaluator
  - downstream classification benchmark hook
```

Then integrate heavier research methods:

```text
MVP-3 Research Skills
  - GeoDiff-SAR adapter
  - segment-controlled background generation adapter
  - Gaussian splatting adapter
```

Then add decision quality:

```text
MVP-4 Decision Quality
  - task-performance evaluation
  - recipe comparison
  - automatic policy retrieval from memory
  - bounded replan with parameter edits
```

## First Supported Workflows

### Sparse Azimuth Target Completion

```text
Input:
  SAR target classification dataset with numeric image/text pairs and azimuth in txt labels

Skills:
  DatasetDiagnosisSkill
  TargetBankBuilderSkill
  GaussianSplattingCompletionSkill
  GeoDiffSARSkill
  LabelValidationSkill

Output:
  expanded target assets
  generated classification image/text pairs
  azimuth coverage report
```

### Cross-Payload Style Adaptation

```text
Input:
  source payload dataset and target payload style examples

Skills:
  DatasetDiagnosisSkill
  StyleBankBuilderSkill
  StyleTransferSkill
  LabelValidationSkill

Output:
  style-transferred training data
  domain shift report
```

### Segment-Controlled Background Expansion

```text
Input:
  target detection dataset with limited backgrounds

Skills:
  DatasetDiagnosisSkill
  BackgroundBankBuilderSkill
  SegmentControlledBackgroundGenerationSkill
  SceneCompositionSkill

Output:
  generated background bank
  composed target-background training data
```

This workflow is deferred until the detection/background stage.

## Agent Autonomy Boundaries

SAGA can autonomously:

- inspect dataset structure and annotations
- inspect numeric image/text pair consistency
- compute statistics
- run registered SAR models
- infer likely data gaps
- select a recipe under configured constraints
- generate candidate data
- filter candidates using validators
- produce reports

SAGA should ask for user input or emit explicit warnings when:

- required model checkpoints are missing
- metadata required by a requested skill is absent
- the task cannot be inferred from the dataset
- class names are ambiguous
- the requested generation would likely violate annotation preservation
- a method needs target azimuth labels but none are available

SAGA should not claim that an LLM alone has reliably recognized SAR targets. Any image-level conclusion should cite the tool or model that produced it.

## Configuration Strategy

Use YAML for user-facing configuration and Pydantic or dataclasses for internal validation.

Configuration groups:

```text
configs/
  datasets/
  requests/
  recipes/
  models/
  skills/
```

Example model registry:

```yaml
models:
  geodiff_sar_aircraft_v1:
    type: geodiff_sar
    checkpoint: /path/to/checkpoint.safetensors
    lora_profiles:
      aircraft_default: /path/to/lora.safetensors
    controlnet_profiles:
      geometry_default: /path/to/controlnet
    runner: scripts/run_geodiff_sar.py

  background_segment_diffusion_v1:
    type: background_diffusion
    checkpoint: /path/to/background_model
    runner: scripts/run_background_generation.py
```

## CLI Strategy

All Python commands should be run in the `sd3` environment.

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成1张747 SAR增广图像" \
  --output runs/agent_run/lora_747_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
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
conda run -n sd3 python -m saga agent-continue \
  --run-dir runs/agent_run/lora_747_llm_planner_fallback_dry \
  --answer "使用提示词 SAR image,Boeing 747,X-band,20 degree depression angle。" \
  --output runs/agent_run/lora_747_continue_dry \
  --dry-run \
  --no-execute
```

## What We Need From Current Methods

For each existing method, SAGA needs a small adapter card:

```yaml
method_name: GeoDiff-SAR
adapter_name: geodiff_sar
task_type:
  - classification_target_generation
  - sparse_azimuth_completion
current_entrypoint: /path/to/run_geodiff_sar.py
environment: sd3
execution:
  type: python_script
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
  expected_speed: medium
  gpu_required: true
limitations:
  - requires target azimuth in txt label
```

This lets us wrap research code without prematurely redesigning it.

## Near-Term Decisions

The next engineering decisions are now:

- stabilize the LLM planner prompt and schema against more user phrasings
- add regression fixtures for style transfer and diffusion LoRA agent runs
- define the formal skill card schema used by `SkillRegistry`
- add a classification evaluator that can measure downstream benefit
- decide which skills are allowed in automatic bounded repair
- keep GeoDiff-SAR as a later complex skill behind a clean adapter boundary

## Suggested Next Step

Next code focus:

```text
1. make the planner schema stricter and easier to test
2. add fixture-based tests for agent-run / bridge-format / execute-recipe
3. add downstream classification evaluation as the first task-benefit evaluator
4. convert each existing method into a skill card plus a thin deterministic adapter
```

After that, each existing model can be connected through an adapter without changing the system contract.

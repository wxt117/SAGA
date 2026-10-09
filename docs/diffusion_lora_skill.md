# SAGA DiffusionLoRAGenerationSkill

`DiffusionLoRAGenerationSkill` wraps `myproject/qinglong_trainer_29b` as a SAGA skill for text-caption SAR image generation and augmentation.

This skill is intentionally narrow:

- It supports image + `.txt` caption LoRA training.
- It supports Flux, SD3/SD3.5, and SDXL command generation through qinglong/sd-scripts adapters.
- It supports LoRA inference command generation and optional execution.
- It does not invoke ControlNet.
- It does not implement GeoDiff-SAR, 3D priors, pose completion, or sparse-azimuth physical reasoning.

## Why This Is A Skill

The qinglong project already contains powerful training and inference scripts, but those scripts are research-engineering entry points with many local assumptions. SAGA adds a stable agent boundary around them:

```text
user dataset with images + txt captions
        |
        v
CaptionDatasetProfile
        |
        v
lora_dataset_config.toml
        |
        v
train_command.sh + inference_commands.sh
        |
        v
QualityEvaluationSkill + ExportDatasetSkill
```

For non-captioned datasets, SAGA can use the agent-level format bridge first:

```text
user dataset without txt captions
        |
        v
RawDatasetProfile + FormatHint
        |
        v
validated DatasetFormatSpec
        |
        v
metadata_caption_dataset/
        |
        v
CaptionDatasetProfile
        |
        v
LoRA train/inference commands
```

This path is only allowed after deterministic validation. The LLM or user may suggest that a filename field means `polarization`, `azimuth`, `band`, etc., but the skill receives those fields only through a validated `compiled_dataset.yaml`.

The first version is command-level and artifact-level by design. It lets SAGA plan, inspect, validate, and reproduce a LoRA augmentation run before spending GPU time.

## Expected Dataset

The dataset root should contain paired images and same-stem `.txt` captions:

```text
747/
  1.png
  1.txt
  2.png
  2.txt
```

Caption example:

```text
SAR image,Boeing 747,X-band,HIgh,20,0
```

The skill checks:

- image count
- `.txt` sidecar count
- image/txt pairing
- empty captions
- image readability, size, and mode on a sample
- common caption tokens
- generation-only flags such as `--cn` or `--d` accidentally mixed into captions

If captions contain generation-only flags, SAGA uses sanitized prompts and prepares a cleaned dataset copy for training.

Alternatively, the agent recipe may pass:

```yaml
dataset_config: runs/.../compiled_dataset.yaml
filters: {}
exclude_filters:
  polarization:
    - pauli
auto_caption_from_metadata: true
```

In that case, the skill stages a temporary paired image/txt dataset under `recipe_artifacts/diffusion_lora/metadata_caption_dataset/`. Example generated captions:

```text
SAR image, ZJGC-X, hh polarization
SAR image, ZJGC-X, hv polarization
SAR image, ZJGC-X, vh polarization
SAR image, ZJGC-X, vv polarization
```

This is useful for datasets such as `D7_135_hv.png`, where the suffix token has been validated as polarization and Pauli composites should be excluded from LoRA training.

## Bounded Parameter Repair

After a LoRA run, SAGA can inspect `QualityEvaluationSkill` and `DistributionEvaluationSkill` reports and produce a bounded repair plan. This is not free-form LLM tuning. The repair planner maps metric triggers to a small whitelist of parameters:

- `training_epochs`
- `max_train_samples`
- `prompt_count`
- `inference_steps`
- `inference_guidance`
- `inference_cfg_scale`
- `lora_multiplier`

The output is a `parameter_repair_plan.json` plus an optional `revised_recipe.yaml`. SAGA does not automatically launch the revised recipe; the user or a later bounded repair loop must execute it explicitly.

```bash
conda run -n sd3 python -m saga repair-parameters \
  --recipe runs/agent_run/zjgc_pol_lora_25ep_10gen_real/recipe.yaml \
  --execution runs/agent_run/zjgc_pol_lora_25ep_10gen_real/execution/recipe_execution.json \
  --output runs/agent_run/zjgc_pol_lora_25ep_10gen_real/parameter_repair_check
```

If no hard observer trigger is raised, SAGA records diagnostic warnings rather than changing parameters. For example, a high standard FID on only 10 generated SAR images is treated as weak evidence and does not by itself trigger a rerun.

## Direct Dry Run

Always start with dry-run:

```bash
conda run -n sd3 python -m saga diffusion-lora \
  --dataset myproject/qinglong_trainer_29b/train/dsine/747 \
  --output runs/diffusion_lora/747_dry \
  --config configs/skills/diffusion_lora.yaml \
  --model-family flux \
  --target-count 12 \
  --dry-run
```

Outputs:

```text
runs/diffusion_lora/747_dry/
  caption_dataset_profile.json
  caption_dataset_profile.md
  lora_dataset_config.toml
  prompt_plan.txt
  train_command.sh
  inference_commands.sh
  diffusion_lora_run.json
  diffusion_lora_run.md
```

## Agent Run

The natural-language path is also supported:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成20张747 SAR增广图像" \
  --output runs/agent_run/lora_747_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
  --dry-run
```

This produces a full SAGA recipe:

```text
inspect_inputs
run_diffusion_lora
evaluate_outputs
repair_policy
export_dataset
```

The recipe is still dry-run by default. The observer and export layers write reports without requiring generated images to exist.

## Real Run

To actually run training and inference, use `--run` explicitly:

```bash
conda run -n sd3 python -m saga diffusion-lora \
  --dataset myproject/qinglong_trainer_29b/train/dsine/747 \
  --output runs/diffusion_lora/747_real \
  --config configs/skills/diffusion_lora.yaml \
  --model-family flux \
  --target-count 20 \
  --run
```

The expected LoRA weight path is:

```text
myproject/qinglong_trainer_29b/output/saga_lora_747_flux.safetensors
```

Inference commands write each prompt into a separate subdirectory under `generated_images/000001`, `generated_images/000002`, and so on. This avoids filename collisions in qinglong's minimal inference scripts, which use timestamp-based output names.

## Quick Real-Run Validation

For quick end-to-end validation with FID-style metrics, use:

```bash
CUDA_VISIBLE_DEVICES=2 \
XFORMERS_FORCE_DISABLE_TRITON=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_CUDNN_V8_API_ENABLED=1 \
conda run -n sd3 python -m saga agent-run \
  --request "请用lora控制扩散模型生成，数据集就用exampledataset/车辆数据-全极化/ZJGC-X。用户要求有极化方式的生成，数据集中后面的字母代表极化方式，pauli字母不作为数据集，最后需求一个50张的增广数据。验证使用FID等快速数值指标，不跑耗时下游任务。" \
  --output runs/agent_run/zjgc_pol_lora_quick_real \
  --diffusion-lora-config configs/skills/diffusion_lora_quick_real.yaml \
  --run \
  --no-memory
```

The quick config uses:

```yaml
max_train_samples: 64
stage_balance_fields: [polarization]
target_count: 50
max_train_epochs: 1
inference_steps: 8
```

This validates the whole SAGA path quickly. It is not intended as final-quality LoRA training.

## Config

Default config:

```text
configs/skills/diffusion_lora.yaml
```

Important fields:

- `adapter.project_root`: qinglong project root
- `adapter.conda_env`: must remain `sd3` in this workspace
- `adapter.default_model_family`: `flux`, `sd3`, or `sdxl`
- `adapter.max_train_epochs`: default training epochs for the preset
- `adapter.pretrained_models`: base checkpoints
- `adapter.clip_l`, `adapter.clip_g`, `adapter.t5xxl`, `adapter.ae`: text encoder and VAE/AE paths
- `adapter.train_scripts`: qinglong train scripts
- `adapter.inference_scripts`: qinglong inference scripts
- `adapter.network_modules`: LoRA module per model family
- `adapter.optimizer_type`: default is `AdamW8bit`; `Prodigy` should only be used with compatible single-learning-rate settings
- `target_count`: number of inference commands to plan by default
- `max_train_samples`: optional cap for staged training samples
- `stage_balance_fields`: metadata fields used for balanced staged sampling, for example `polarization`
- `stage_dataset`: force copying/sanitizing dataset before training

Natural-language requests can also override run-level values. For example, "训练25轮，生成10张" becomes:

```yaml
training_epochs: 25
target_count: 10
```

Those values are applied by the agent recipe and passed into the skill; ordinary users should not need to edit YAML for common requests.

Flux is usable out of the box with the local paths currently present in the project. SD3/SDXL adapters are wired, but their base checkpoint paths must be filled in before real execution.

## Smoke Test Result

The current Flux smoke configuration has been tested on a 12-sample paired caption dataset:

```bash
CUDA_VISIBLE_DEVICES=2 \
XFORMERS_FORCE_DISABLE_TRITON=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_CUDNN_V8_API_ENABLED=1 \
conda run -n sd3 python -m saga diffusion-lora \
  --dataset runs/diffusion_lora/smoke_747_dataset \
  --output runs/diffusion_lora/747_smoke_real3 \
  --config configs/skills/diffusion_lora_smoke.yaml \
  --model-family flux \
  --target-count 1 \
  --output-name saga_lora_747_flux_smoke3 \
  --run
```

Observed artifacts:

```text
myproject/qinglong_trainer_29b/output/saga_lora_747_flux_smoke3.safetensors
runs/diffusion_lora/747_smoke_real3/generated_images/000001/20260517_094954.png
runs/diffusion_lora/747_smoke_real3/diffusion_lora_run.json
```

The smoke run completed one epoch over 12 samples and one 8-step inference image. Earlier Prodigy smoke attempts failed because Prodigy rejected different learning rates across parameter groups; this is why the default smoke config now uses `AdamW8bit`.

## Skill Boundary

Use this skill for:

- SAR target image augmentation from text-caption datasets
- SAR target image augmentation from validated metadata datasets that can be caption-staged
- fast LoRA training/inference workflow planning
- converting a user's image/txt dataset into qinglong-compatible training artifacts

Do not use this skill for:

- ControlNet conditional generation
- pose-map controlled generation
- GeoDiff-SAR sparse azimuth completion
- 3D-model physical-prior generation
- target-background composition

Those should be separate SAGA skills so the agent can choose them deliberately and evaluators can judge them with the right criteria.

# SAGA Agent Runtime

SAGA now has a minimal Plan-then-Execute runtime. It connects user language, dataset profiling, format validation, recipe generation, and deterministic execution reports.

```text
natural-language request
        |
        v
IntentSpec + FormatHint
        |
        v
RawDatasetProfile
        |
        v
DatasetFormatSpec bridge + validation
        |
        v
SkillRegistry + memory retrieval
        |
        v
DatasetNeedProfile + SkillEffectCards
        |
        v
LLM intent/planner proposals + rule guardrails
        |
        v
benefit-aware multi-plan ranking + PlanVerificationReport
        |
        v
SagaRecipe DAG
        |
        v
execute-recipe runtime
        |
        v
Observer / Evaluator + bounded repair policy
```

## agent-run

`agent-run` is the high-level entry point. It defaults to dry-run for safety; use `--run` only when you want expensive skills such as style transfer to execute.

```bash
conda run -n sd3 python -m saga agent-run \
  --request "把exampledataset/2_Boeing707-客机作为指导图，将707的特征迁移到exampledataset/B747的数据上，只迁移747下视角为20的数据，数据命名中850后面的数字为下视角、方位角、分辨率、波段" \
  --output runs/agent_run/b747_707_dry
```

For text-caption diffusion LoRA augmentation, the same entry point can generate a LoRA recipe:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成20张747 SAR增广图像" \
  --output runs/agent_run/lora_747_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
  --dry-run
```

If the user dataset has no `.txt` captions but has metadata that can be validated from filenames or paths, SAGA can stage a temporary caption dataset for LoRA training. For example, this request says that the filename suffix letters are polarization labels and that `pauli` should be excluded:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "请用lora控制扩散模型生成，数据集就用exampledataset/车辆数据-全极化/ZJGC-X。用户要求有极化方式的生成，数据集中后面的字母代表极化方式，pauli字母不作为数据集，最后需求一个50张的增广数据。" \
  --output runs/agent_run/zjgc_pol_lora_dry \
  --diffusion-lora-config configs/skills/diffusion_lora_smoke.yaml \
  --use-llm-intent \
  --use-llm-planner \
  --llm-config configs/llm/deepseek_v4pro.yaml \
  --dry-run
```

The flow is:

```text
natural-language suffix hint
        -> filename_suffix_field FormatHint
        -> suffix_polarization DatasetFormatSpec rule
        -> deterministic validation of polarization coverage
        -> exclude_filters: {polarization: [pauli]}
        -> metadata_caption_dataset/
        -> LoRA train/inference commands
```

Generated captions are conservative metadata captions such as:

```text
SAR image, ZJGC-X, hh polarization
SAR image, ZJGC-X, hv polarization
```

This is weaker than expert-written captions, but it lets SAGA run a controlled LoRA augmentation recipe when the dataset has validated semantic metadata and no caption sidecars.

Outputs:

```text
runs/agent_run/b747_707_dry/
  intent_spec.json
  format_hints.json
  raw_profile.json
  dataset_profile.json
  compiled_format_spec.yaml
  compiled_dataset.yaml
  format_validation.json
  format_bridge.json
  validated_dataset_profile.json
  dataset_need_profile.json
  skill_effect_cards.json
  skill_utility.json
  recipe.yaml
  recipe.json
  agent_state.json
  agent_run.md
  execution/
    recipe_execution.json
    recipe_execution.md
    steps/
      inspect_inputs.json
      select_content.json
      run_style_transfer.json
      evaluate_outputs.json
      repair_policy.json
    observer/
      evaluate_outputs/
        quality_evaluation.json
        quality_evaluation.md
      repair_policy/
        repair_policy.json
        repair_policy.md
  augmented_dataset/
    images/
    manifest.jsonl
    dataset_card.json
    provenance.json
    export_report.json
    export_report.md
  agent_state.json
```

With `--use-llm-intent` and `--use-llm-planner`, the run directory also contains the LLM-facing and guardrail artifacts:

```text
intent_llm_proposal.json
intent_guardrail.json
intent_effective_spec.json
intent_merged_format_hints.json
intent_llm_report.md
skill_registry.json
skill_registry.md
memory_retrieval.json
candidate_plans.json
plan_ranking.json
plan_ranking.md
plan_utility_report.json
plan_utility_report.md
planner_proposal.json
planner_guardrail.json
planner_clarification.json
plan_verification.json
plan_verification.md
effective_intent_spec.json
```

The key safety rule is:

```text
LLM or user hints can create candidate semantics.
Only bridge-format validation can make those semantics executable.
Only planner guardrails can let an LLM proposal influence intent/recipe fields.
Only a SagaRecipe can be executed.
```

## Hybrid LLM Planner

`agent-run` defaults to the deterministic rule planner. To let the LLM participate in high-level planning, enable the hybrid planner:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用myproject/qinglong_trainer_29b/train/dsine/747训练一个Flux LoRA，并生成20张747 SAR增广图像" \
  --output runs/agent_run/lora_747_llm_planner_dry \
  --dataset-root myproject/qinglong_trainer_29b/train/dsine/747 \
  --use-llm-planner \
  --llm-config configs/llm/deepseek_v4pro.yaml \
  --dry-run
```

The LLM planner does not directly emit an executable recipe. It writes:

```text
planner_proposal.json
planner_guardrail.json
planner_clarification.json
effective_intent_spec.json
planner_report.md
```

The boundary is:

```text
LLM Planner Proposal
  -> deterministic task/skill/path/filter guardrails
  -> effective IntentSpec
  -> rule Recipe Generator
  -> execute-recipe
```

This keeps SAGA in a controlled Plan-then-Execute form: the LLM can propose goals, strategy, missing validations, and candidate skills, but executable recipes are still compiled from validated intent, validated dataset format, and the registered skill interface.

The guardrail assigns one of these decisions:

```text
accept_with_guardrails
accept_with_clarification_warnings
needs_clarification
fallback_to_rule_planner
```

Dry-runs still write recipes and reports so the user can inspect the plan. Real execution is blocked when the planner guardrail is invalid, when clarification is required, or when the `PlanVerificationReport` is not `passed`. This prevents an uncertain LLM proposal from spending GPU time before the user has resolved the ambiguity.

## LLM Intent Recognizer

The optional intent recognizer lets the LLM read the user's natural-language request before profiling and planning:

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

The recognizer may propose intent updates and format hints, but the rule guardrail decides which fields can be applied. This is important for non-standard user requests: the LLM can understand flexible phrasing, while deterministic code still owns path checks, supported task names, allowed model families, numeric target counts, and filter normalization.

Filters and exclusions are separate. Equality filters select samples:

```yaml
filters:
  depression_angle_deg: 20
```

Exclusion filters remove samples:

```yaml
exclude_filters:
  polarization:
    - pauli
```

If an LLM writes a value such as `exclude(pauli)` inside normal `filters`, guardrails normalize it into `exclude_filters` before recipe generation.

## Multi-Plan Ranking

The planner prompt now asks the LLM for `candidate_plans`. A single-plan response is still accepted and wrapped into one candidate for compatibility. SAGA then ranks the candidates with deterministic scoring:

- prefer executable tasks and registered executable skills
- prefer plans that match the rule-recognized intent
- penalize unsupported skill names
- penalize plans that need clarification
- account for bridge validation and retrieved memory
- account for dataset needs and prior skill utility
- record high-cost GPU steps without treating them as invalid

The selected candidate becomes `planner_proposal.json`. The complete candidate set and ranking are written to `candidate_plans.json`, `plan_ranking.json`, and `plan_ranking.md`.

## Benefit-Aware Planning

SAGA now creates a prior-based benefit context before LLM planning:

```text
validated dataset profile
        |
        v
DatasetNeedProfile
        |
        v
SkillEffectCards
        |
        v
SkillUtility + PlanUtilityReport
```

`DatasetNeedProfile` estimates what the dataset likely needs, such as sample expansion, azimuth coverage, view-angle metadata, domain adaptation, background diversity, image-quality cleanup, or downstream evidence. `SkillEffectCards` describe each skill's expected strengths, weaknesses, risks, cost, input requirements, metadata requirements, and prior downstream benefit. These are expert priors, not experimental proof.

The current benefit model also tracks explicit request alignment. For example, if a user explicitly asks for LoRA/diffusion generation, `DiffusionLoRAGenerationSkill` receives a request-alignment boost over generic support skills such as quality evaluation. If a user asks for polarization-aware generation and the DatasetFormatSpec validates `polarization`, the need profile records `metadata_conditioned_generation`.

## Default Fast Generation Metrics

For generation recipes, SAGA now defaults to two post-generation observer layers:

```text
QualityEvaluationSkill
  -> count/readability/black-white ratio/dynamic range

DistributionEvaluationSkill
  -> sampled pytorch_fid when available
  -> sar_fid_lite
  -> histogram_jsd
  -> mmd_rbf
  -> generated_diversity
  -> nearest_reference_distance

SARArtifactEvaluationSkill
  -> stripe_score
  -> gradient_score
  -> target_compactness
  -> target_fragment_count
  -> target_center_offset
```

These metrics are intentionally quick proxies. They are useful when the user wants numerical validation without spending time on downstream classification or detection training. They do not prove downstream task improvement. When the user asks for task benefit evidence, SAGA should still route to downstream evaluators such as `ClassificationEvaluationSkill`.

`ClassificationEvaluationSkill` is the current standard downstream evaluator for classification. It adapts arbitrary classification-like sources or SAGA exports into a project-ready benchmark dataset, keeps a shared held-out validation split, trains a baseline model and an augmented model, then reports comparable metrics and metric deltas. The referenced project uses three heads named `article`, `color`, and `gender`; SAGA maps SAR class to `article`, polarization to `color`, and domain/source to `gender` by default.

`SARArtifactEvaluationSkill` is a SAR-specific observer for visual artifacts that generic image statistics often miss, such as stripes, smooth generated backgrounds, diffuse scattering centers, fragmented target responses, or off-center targets. It provides repair/export triggers, not final downstream proof.

For short real-run validation, use:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "请用lora控制扩散模型生成，数据集就用exampledataset/车辆数据-全极化/ZJGC-X。用户要求有极化方式的生成，数据集中后面的字母代表极化方式，pauli字母不作为数据集，最后需求一个50张的增广数据。验证使用FID等快速数值指标，不跑耗时下游任务。" \
  --output runs/agent_run/zjgc_pol_lora_quick_real \
  --diffusion-lora-config configs/skills/diffusion_lora_quick_real.yaml \
  --run \
  --no-memory
```

`configs/skills/diffusion_lora_quick_real.yaml` uses one Flux LoRA epoch over a balanced 64-sample metadata-caption subset, then still generates the requested 50 images. It is for pipeline validation, not final-quality training.

The current skill list is summarized in [skill_catalog.md](skill_catalog.md). The JSON version used by each run is written as `skill_effect_cards.json`.

The planner uses this context to answer a different question from the guardrail:

```text
Guardrail: can this plan run safely?
Benefit estimator: is this plan likely worth running for this dataset?
```

This is the first step from "can plan and verify" toward "can choose augmentation strategies by likely downstream value." True benefit still requires downstream evaluators such as `ClassificationEvaluationSkill`.

## Plan Verification

After guardrail validation, SAGA writes a `PlanVerificationReport`:

```text
planner_proposal.json
planner_guardrail.json
format_bridge.json
skill_registry.json
        |
        v
plan_verification.json / plan_verification.md
```

This report answers whether the selected plan can become an executable recipe. It checks skill registry status, unsupported/planned skills, clarification requirements, bridge validity, and high-cost GPU steps. Real `agent-run --run` execution is blocked unless the report status is `passed`.

## DatasetFormatSpec Bridge

The bridge is the handoff between flexible user data and deterministic execution. It has two jobs:

```text
RawDatasetProfile + IntentSpec + FormatHint
        |
        v
DatasetFormatSpec Compiler
        |
        v
Format Validator
        |
        v
validated_dataset_profile.json + compiled_dataset.yaml
```

The compiler turns observed structure and user/LLM hints into a concrete `DatasetFormatSpec`. The validator then tests that spec over real files. Only a validated spec can produce `compiled_dataset.yaml`, which is what downstream skills use for filtering and loading. This is why SAGA can accept many external dataset layouts without letting the LLM invent unverified metadata.

## Clarification Continuation

If a planner or intent guardrail asks a question, the user can continue the run without starting from scratch:

```bash
conda run -n sd3 python -m saga agent-continue \
  --run-dir runs/agent_run/lora_747_llm_planner_fallback_dry \
  --answer "使用提示词 SAR image,Boeing 747,X-band,20 degree depression angle。" \
  --output runs/agent_run/lora_747_continue_dry \
  --diffusion-lora-config configs/skills/diffusion_lora_smoke.yaml \
  --dry-run \
  --no-execute
```

`agent-continue` reads `agent_state.json`, existing clarification artifacts, and the user's answer. It creates a continued request and runs the same profiling, bridge, recipe, and optional LLM pipeline into a new output directory. The original run remains immutable.

## execute-recipe

`execute-recipe` runs an existing recipe. Always test with `--dry-run` first:

```bash
conda run -n sd3 python -m saga execute-recipe \
  --recipe runs/agent_run/b747_707_dry/recipe.yaml \
  --output runs/agent_run/b747_707_dry/execution_check \
  --dry-run
```

When dry-run is used:

- file selection reports how many samples would be staged
- style transfer writes the exact command that would run
- output evaluation reports existing images separately from new generated images
- bounded repair policy records suggested actions and may write a whitelisted parameter-repair recipe; it is not applied automatically
- dataset export writes an export plan and manifest, but does not materialize generated images
- no expensive model execution is started

To actually run a recipe:

```bash
conda run -n sd3 python -m saga execute-recipe \
  --recipe runs/agent_run/b747_707_dry/recipe.yaml \
  --output runs/agent_run/b747_707_real
```

For high-level natural-language runs, `agent-run` still defaults to dry-run. Use `--run` explicitly:

```bash
conda run -n sd3 python -m saga agent-run \
  --request "..." \
  --output runs/agent_run/real_run \
  --run
```

## Recipe DAG

The current `SagaRecipe` schema is intentionally simple and serializable:

```yaml
schema_version: saga_recipe_v1
recipe_id: style_transfer_B747_707_depression_angle_deg_20
task: style_transfer
inputs:
  content_source: /path/to/B747
  style_source: /path/to/2_Boeing707-客机
  dataset_config: /path/to/compiled_dataset.yaml
  format_spec: /path/to/compiled_format_spec.yaml
filters:
  depression_angle_deg: 20
pipeline:
  - id: inspect_inputs
    skill: DatasetProfileReportSkill
  - id: select_content
    skill: FileSelectionSkill
    depends_on: [inspect_inputs]
  - id: run_style_transfer
    skill: StyleTransferSkill
    depends_on: [select_content]
  - id: evaluate_outputs
    skill: QualityEvaluationSkill
    depends_on: [run_style_transfer]
    params:
      expected_from_step: select_content
  - id: repair_policy
    skill: RepairPolicySkill
    depends_on: [evaluate_outputs]
    params:
      max_trials: 3
      auto_rerun: false
  - id: export_dataset
    skill: ExportDatasetSkill
    depends_on: [repair_policy]
    params:
      source_from_step: run_style_transfer
      quality_from_step: evaluate_outputs
      repair_from_step: repair_policy
```

This gives SAGA a reproducible contract between planning and execution. The observer layer is deliberately deterministic: it checks output count, image readability, simple grayscale dynamic range, and black/white saturation ratios. The repair layer is bounded: it records triggers, suggested actions, and when possible a whitelisted parameter-repair plan with a revised recipe artifact. It still does not rerun expensive skills automatically. The export layer packages generated outputs as a traceable candidate dataset; it does not claim downstream improvement until a task evaluator confirms that.

For diffusion LoRA generation, the recipe shape is:

```yaml
task: diffusion_lora_generation
pipeline:
  - id: inspect_inputs
    skill: DatasetProfileReportSkill
  - id: run_diffusion_lora
    skill: DiffusionLoRAGenerationSkill
  - id: evaluate_outputs
    skill: QualityEvaluationSkill
  - id: evaluate_sar_artifacts
    skill: SARArtifactEvaluationSkill
  - id: repair_policy
    skill: RepairPolicySkill
  - id: export_dataset
    skill: ExportDatasetSkill
```

`DiffusionLoRAGenerationSkill` is a text-caption LoRA wrapper only. ControlNet and GeoDiff-SAR are intentionally separate future skills.

For LoRA recipes, `RepairPolicySkill` can now attach a `parameter_repair_plan` based on quality and distribution triggers. The plan may patch only bounded, whitelisted parameters such as `training_epochs`, `max_train_samples`, `prompt_count`, `inference_steps`, `inference_guidance`, and `lora_multiplier`. The output is a reviewable `revised_recipe.yaml`; SAGA does not launch the revised recipe unless the user explicitly executes it.

```bash
conda run -n sd3 python -m saga repair-parameters \
  --recipe runs/agent_run/zjgc_pol_lora_25ep_10gen_real/recipe.yaml \
  --execution runs/agent_run/zjgc_pol_lora_25ep_10gen_real/execution/recipe_execution.json \
  --output runs/agent_run/zjgc_pol_lora_25ep_10gen_real/parameter_repair_check
```

Typical outputs:

```text
parameter_repair_plan.json
parameter_repair_plan.md
revised_recipe.yaml   # only when a bounded patch is proposed
```

## evaluate-output

`evaluate-output` lets you run the observer layer on an existing output directory without running a full agent recipe:

```bash
conda run -n sd3 python -m saga evaluate-output \
  --input runs/agent_run/b747_707_real/recipe_artifacts/style_transfer_output \
  --output runs/eval/b747_707_real \
  --expected-count 36
```

Outputs:

```text
runs/eval/b747_707_real/
  quality_evaluation.json
  quality_evaluation.md
  repair_policy/
    repair_policy.json
    repair_policy.md
    parameter_repair/
      parameter_repair_plan.json
      parameter_repair_plan.md
```

The evaluator is not a task-performance metric. It is an observer gate that catches obvious execution and image-quality failures before a generated dataset is exported or used for downstream training.

## export-dataset

`export-dataset` packages an existing generated-output directory into SAGA's internal augmented dataset artifact:

```bash
conda run -n sd3 python -m saga export-dataset \
  --source runs/agent_run/b747_707_real/recipe_artifacts/style_transfer_output \
  --output runs/export/b747_707_candidate \
  --recipe-id style_transfer_b747_707 \
  --task style_transfer \
  --mode copy
```

Outputs:

```text
runs/export/b747_707_candidate/
  images/
  manifest.jsonl
  dataset_card.json
  provenance.json
  export_report.json
  export_report.md
```

If generated images have source captions and the exported dataset should be reusable as an image/txt caption dataset, add `--write-caption-sidecars`. SAGA LoRA recipes set the same export option automatically:

```bash
conda run -n sd3 python -m saga export-dataset \
  --source runs/agent_run/zjgcx_lora/recipe_artifacts/diffusion_lora/generated_images \
  --output runs/export/zjgcx_lora_captioned \
  --recipe-id diffusion_lora_zjgcx \
  --task diffusion_lora_generation \
  --write-caption-sidecars
```

For a cheap preflight, use:

```bash
conda run -n sd3 python -m saga export-dataset \
  --source runs/agent_run/b747_707_real/recipe_artifacts/style_transfer_output \
  --output runs/export/b747_707_candidate_dry \
  --recipe-id style_transfer_b747_707 \
  --task style_transfer \
  --dry-run
```

To use the observer as a hard gate, pass a quality report:

```bash
conda run -n sd3 python -m saga export-dataset \
  --source runs/agent_run/b747_707_real/recipe_artifacts/style_transfer_output \
  --output runs/export/b747_707_candidate \
  --recipe-id style_transfer_b747_707 \
  --task style_transfer \
  --quality-report runs/eval/b747_707_real/quality_evaluation.json \
  --require-quality-pass
```

## Policy Memory

`agent-run` retrieves similar previous runs before planner execution and records a compact run summary into `runs/memory` after it writes the run report. This is the first version of the recipe/policy bank:

```text
runs/memory/
  policy_memory.jsonl
  policy_memory_latest.json
  policy_memory.md
```

You can also record an existing run:

```bash
conda run -n sd3 python -m saga memory record \
  --run runs/agent_run/b747_707_dry \
  --memory-dir runs/memory
```

List recent entries:

```bash
conda run -n sd3 python -m saga memory list \
  --memory-dir runs/memory \
  --limit 10
```

The retrieved context is saved as `memory_retrieval.json` and is shown to the LLM planner as compact evidence, not as an instruction that must be followed. The memory entry stores a dataset signature, validated bridge fields, recipe id, planner decision, plan verification status, execution status, observer triggers, repair status, and export status. This gives SAGA the substrate for a policy/recipe bank without making the LLM an unconstrained recommender.

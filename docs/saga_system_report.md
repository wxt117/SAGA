# SAGA 系统报告

版本：当前工程实现状态
日期：2026-05-20
环境约定：所有 Python 运行、测试和依赖安装默认使用 `sd3` Conda 环境

## 1. 报告摘要

SAGA 是一个面向 SAR 数据生成与增广的智能决策系统，全称可理解为 SAR Augmentation and Generation Agent。它的目标不是简单包装一批脚本，也不是一个通用聊天式 ReAct agent，而是一个面向下游任务收益的 SAR 数据生成、增广、评估、修复与记忆闭环系统。

当前 SAGA 的核心能力已经从“根据命令调用某个增广工具”扩展为：

1. 接收用户自然语言请求和任意风格的数据集目录。
2. 通过确定性程序扫描数据集的客观结构。
3. 结合用户说明和可选 LLM，对非标准命名和标签格式进行语义归纳。
4. 将外部非固定格式编译成内部确定的 `DatasetFormatSpec` 和 `compiled_dataset.yaml`。
5. 根据数据画像、用户目标、资源约束、skill 先验、历史 memory 和轻量证据选择合适的增广策略。
6. 生成可执行的 `SagaRecipe` DAG。
7. 通过确定性 `Skill Executor` 执行增广、生成、评估、修复、导出。
8. 用 Observer 和 BenefitEvidence 将“生成成功”和“下游收益成立”严格区分。
9. 将运行结果写入 policy memory，为后续类似数据集提供策略参考。

当前系统已经具备完整主干闭环：

```text
User Request + Dataset
  -> Dataset Profiler
  -> Intent Recognizer
  -> Format Bridge
  -> Planning Evidence
  -> Benefit-aware Planner
  -> Recipe Generator
  -> Skill Executor
  -> Observer / Evaluator
  -> Repair Policy
  -> Export Dataset
  -> Benefit Evidence
  -> Policy Memory
```

截至当前实现，Skill Registry 中注册了 45 个 skill，其中 35 个为 executable，10 个为 planned。主要可执行 skill 已覆盖传统增广、LoRA 扩散生成、GAN 生成、特征迁移、背景生成、目标背景融合、GeoDiff-SAR、RaySAR 仿真、SAR 预处理、metadata caption、分类评估、质量评估、分布评估、SAR 伪影评估、泄漏检查、近重复检查、导出、修复和 memory 相关流程。当前明确尚未完成的关键生成类 skill 是 Gaussian Splatting 稀疏方位角补全。

## 2. 系统定位

SAGA 的定位是：

```text
面向下游任务收益的 SAR 数据生成与增广决策系统
```

它回答四类问题：

1. 用户给的数据集是什么结构，能否被可靠读取。
2. 当前数据集缺什么，例如样本数量、类别平衡、极化覆盖、方位角覆盖、跨载荷域差异、背景不足等。
3. 在用户的目标、资源、速度、质量要求下，哪种增广或生成方法最合适。
4. 生成结果是否可以被接受，是否有证据支持后续用于下游任务。

因此，SAGA 不应被理解为：

```text
自然语言 -> 某个脚本
```

而应被理解为：

```text
自然语言 + 数据集 -> 数据画像 -> 策略判断 -> 可执行 recipe -> 结果观察 -> 修复/记忆
```

这种设计的关键价值是：SAGA 不把 LLM 放在“自由执行工具”的位置，而是让 LLM 参与自然语言理解、schema 归纳和高层规划；事实扫描、格式验证、工具执行、质量门控、导出和 memory 写入全部由确定性程序完成。

## 3. 总体架构

SAGA 当前采用七层结构：

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

这七层可以进一步拆成当前代码中的主要模块：

```text
saga/
  agent/
    intent_recognizer.py
    llm_intent.py
    llm_planner.py
    augmentation_planner.py
    benefit_estimator.py
    planning_evidence.py
    plan_verifier.py
    plan_critic.py
    recipe_generator.py
    runtime.py
    skill_registry.py
    candidate_pilot.py
    auto_repair.py

  data/
    profile.py
    format_bridge.py
    format_builder.py
    format_spec.py
    format_validation.py
    loader.py
    discovery.py
    path_patterns.py

  core/
    protocol.py
    recipe.py
    skill_report.py
    evaluator_report.py
    provenance.py
    saga_config.py

  executor/
    recipe_executor.py

  observer/
    quality.py
    distribution.py
    sar_artifacts.py
    repair_policy.py
    parameter_repair.py

  exporter/
    dataset_exporter.py

  memory/
    policy_bank.py

  skills/
    traditional_augmentation/
    diffusion_lora/
    gan_generation/
    style_transfer/
    geodiff_sar/
    background_generation/
    target_background_composition/
    raysar/
    raysar_sweep/
    model_to_pov_scene/
    sar_preprocess/
    metadata_caption/
    dataset_balancing/
    pseudocolor/
    classification_evaluation/
    evaluation_utils/
    label_validation/
    diagnosis/

  cli.py
```

## 4. SAGA 不是简单 ReAct 或普通 Plan-then-Execute

当前 SAGA 的 agent 形式更准确地说是：

```text
Plan-then-Execute 主干
+ Deterministic Dataset Profiler
+ LLM/Rule Hybrid Intent and Planner
+ Benefit-aware Skill Ranking
+ Recipe DAG
+ Observer/Evaluator
+ Bounded Repair Loop
+ Policy Memory
```

与普通 ReAct 的区别：

1. SAGA 不让 LLM 在循环中随意调用工具。
2. SAGA 不让 LLM 直接读取整个数据集并下结论。
3. SAGA 不让 LLM 判断事实是否成立，事实必须由 profiler、validator、executor、observer 验证。
4. SAGA 的执行单元不是临时 tool call，而是可保存、可复现、可重跑的 `SagaRecipe`。
5. SAGA 的修复不是开放式反复试错，而是有边界的 repair policy，最多尝试 K 次，只允许修改白名单参数。

与普通 Plan-then-Execute 的区别：

1. SAGA 在 plan 前加入数据画像和格式桥接。
2. SAGA 的 planner 不是只生成步骤，而是判断 skill 对当前数据集的潜在收益。
3. SAGA 有 SkillEffectCard 和 DatasetNeedProfile，用任务收益逻辑选择增广方法。
4. SAGA 在执行后生成 BenefitEvidence，严格区分轻量质量证据和下游任务收益证据。
5. SAGA 将历史运行写入 memory，下一次遇到类似数据集时可检索历史 recipe 和结果。

## 5. 数据集理解与格式桥接

### 5.1 核心原则

SAGA 不要求用户数据集外部格式固定。用户可以上传各种目录结构、文件命名方式、sidecar 文本、caption、标准格式或自定义格式。

但在正式执行增广前，SAGA 内部必须得到一个确定的读取协议：

```text
DatasetFormatSpec + compiled_dataset.yaml
```

因此，SAGA 采用如下原则：

```text
外部格式可以不确定
内部格式必须确定
```

### 5.2 Raw Dataset Profiler

`Dataset Profiler` 的第一步是 raw scan。它只观察低层客观事实，不解释语义。

它会扫描：

1. 文件总数、目录总数。
2. 图像后缀统计。
3. 图像尺寸、模式、位深、动态范围。
4. sidecar 文件数量，例如 `.txt`、`.json`、`.xml`、`.csv`。
5. 文件名中的数字 token 和字符串 token。
6. 路径深度、父目录分布、路径 pattern。
7. 标准格式候选，例如 image-folder、YOLO、COCO、VOC、mask folder 等。
8. 示例路径和样本摘要。

Raw profiler 的输出包括：

```text
raw_profile.json
llm_context.json
dataset_profile.json
clarification_questions.json
profile_report.md
```

`llm_context.json` 是给 LLM 使用的压缩摘要。它不会把整个数据集交给 LLM，只给出统计、样例和可观察结构。

### 5.3 渐进式数据画像

SAGA 将数据画像分为多个 level：

```text
Level 0: Raw filesystem and image facts only.
Level 1: Format candidates or user semantic hints exist, but no validated DatasetFormatSpec yet.
Level 2: A DatasetFormatSpec has been validated and semantic fields are trusted.
Level 3: Task-ready profile with validated fields and downstream diagnosis.
```

这意味着如果用户不给格式说明，而数据集又不是标准格式，SAGA 不会强行猜测。它会停在 Level 0 或 Level 1，并生成 clarification questions。

### 5.4 Intent 和 FormatHint

用户自然语言会被解析成两类信息：

```text
IntentSpec: 用户想做什么
FormatHint: 用户如何解释数据格式
```

例如用户说：

```text
后面的字母代表极化方式，pauli 不作为数据集。
```

SAGA 可以得到：

```json
{
  "intent": {
    "task": "diffusion_lora_generation",
    "exclude_filters": {
      "polarization": ["pauli"]
    }
  },
  "format_hints": [
    {
      "type": "filename_suffix_field",
      "field": "polarization"
    }
  ]
}
```

如果启用 LLM intent recognizer，LLM 可提出更丰富的 intent 和 format hint，但仍然必须经过规则合并和 guardrail。

### 5.5 DatasetFormatSpec Compiler + Validator

桥接层由两个部分组成：

```text
DatasetFormatSpec Compiler
DatasetFormatSpec Validator
```

Compiler 的作用是将用户说明、LLM 候选解释、文件名启发式规则转成统一的读取规则。例如：

```yaml
format_spec:
  name: ZJGC_X_compiled_format
  version: saga_format_spec_v1
  rules:
    - name: suffix_polarization
      target: relative_path
      pattern: "^(?:.*/)?(?P<target_name>.+?)_(?P<polarization>[A-Za-z]+)\\.[^.]+$"
      fields:
        target_name: target_name
        polarization: polarization
```

Validator 的作用是确定这个 spec 是否真的能读当前数据集：

1. 样本是否能匹配规则。
2. 必要字段是否有覆盖率。
3. filter 和 exclude_filter 是否能作用于真实字段。
4. class、polarization、azimuth、band、resolution 等字段是否能被解析。
5. 解析失败时生成 validation report，而不是继续执行危险计划。

桥接层输出：

```text
compiled_format_spec.yaml
compiled_dataset.yaml
format_bridge.json
format_bridge.md
validated_dataset_profile.json
```

这一步的意义是把“用户任意格式”变成“SAGA 内部可执行格式”，同时避免 LLM 直接猜错格式后进入训练。

## 6. Agent Runtime 逻辑

当前 `agent-run` 的主流程由 `saga/agent/runtime.py` 驱动。

完整逻辑如下：

```text
1. 读取 saga config
2. 保存 run provenance
3. 保存 skill registry
4. 规则 Intent Recognizer 解析自然语言
5. 可选 LLM Intent Recognizer 生成补充 intent 和 format hints
6. 推断 dataset root
7. 运行 Raw Dataset Profiler
8. 运行 Format Bridge 编译并验证 DatasetFormatSpec
9. 构建 multi dataset profile
10. 构建 planning evidence
11. 检索 policy memory
12. 构建 benefit context
13. 运行 deterministic augmentation planner
14. 可选 LLM planner 生成高层 plan proposal
15. 运行 plan verifier 和 guardrail
16. 生成 SagaRecipe
17. 生成 candidate pilot recipes
18. 运行 plan critic
19. 如未被阻断，执行 recipe
20. 生成 benefit evidence report
21. 再次运行 plan critic，检查执行后 claim 风险
22. 保存 agent_state 和 agent_run.md
23. 写入 policy memory
```

典型命令：

```bash
conda run -n sd3 python -m saga agent-run \
  --request "用 exampledataset/车辆数据-全极化/ZJGC-X 训练LoRA做有极化方式的扩散模型生成，后面的字母代表极化方式，pauli不作为数据集，生成10张，训练25轮。只做轻量质量、分布和SAR伪影评估，不训练分类器。" \
  --output runs/real_experiment_lora_zjgcx_25ep_10gen \
  --diffusion-lora-config configs/skills/diffusion_lora_25ep_10gen.yaml \
  --saga-config configs/saga.yaml \
  --run
```

## 7. Planner 设计

### 7.1 Planner 的输入

SAGA planner 不只看自然语言，还看结构化证据：

```text
IntentSpec
RequestConstraints
RawDatasetProfile
ValidatedDatasetProfile
MultiDatasetProfile
PlanningEvidence
SkillRegistry
SkillEffectCards
DatasetNeedProfile
PolicyMemory
```

### 7.2 RequestConstraints

系统会从请求中抽取约束，例如：

1. 是否要求速度快。
2. 是否要求高质量。
3. 是否允许大显存和长时间训练。
4. 是否要求物理可解释。
5. 是否要求下游任务收益证据。
6. 是否明确禁止某类方法，例如不做 RaySAR。

这些约束会影响 skill 排名。例如：

```text
速度优先 -> TraditionalAugmentationSkill 加分，Diffusion/GeoDiff 扣分
高质量优先 -> DiffusionLoRAGenerationSkill 或 GeoDiffSARSkill 加分
有 3D 模型和稀疏方位角补全 -> GeoDiffSARSkill 或 RaySAR/Gaussian 类方法加分
跨载荷/跨平台迁移 -> StyleTransferSkill 加分
只要背景 -> BackgroundGenerationSkill 加分
物理仿真 -> RaySARSynthesisSkill 或 RaySARSweepSynthesisSkill 加分
```

### 7.3 SkillEffectCards

`SkillEffectCard` 是 SAGA 的任务特化知识库。它描述每个 skill 的：

1. 主要目标。
2. 适合任务。
3. 输入要求。
4. 需要的 metadata。
5. 能解决哪些 dataset needs。
6. 优点和缺点。
7. 风险。
8. 计算成本。
9. 适合和不适合的场景。
10. 可控参数。
11. 预期收益先验。
12. 推荐评估方式。

这使 planner 不只是“识别关键词调用工具”，而是根据数据集问题和 skill 先验做策略选择。

### 7.4 DatasetNeedProfile

`DatasetNeedProfile` 估计当前数据集的需求，例如：

```text
sample_count
class_balance
caption_readiness
metadata_conditioned_generation
azimuth_coverage
polarization_coverage
image_quality
domain_shift
speed_priority
physical_prior_need
```

它来自 raw profile、validated profile、format bridge、planning evidence 和 user constraints。

### 7.5 Deterministic Augmentation Planner

`augmentation_planner.py` 会计算候选计划：

```text
skill utility score
+ task match adjustment
+ request constraint adjustment
+ planning evidence adjustment
+ memory adjustment
+ multi dataset alignment adjustment
```

输出：

```text
augmentation_plan.json
augmentation_plan.md
selected_skill
selected_recipe_task
ranked_plans
candidate params
expected_benefits
risks
execution_policy
```

### 7.6 LLM Planner 的位置

LLM planner 是可选模块，默认配置中未强制启用。它的角色是提出高层规划建议，而不是直接执行。

LLM planner 的边界：

1. 可以读取用户请求、raw profile 摘要、validated profile、memory、skill registry、benefit context。
2. 可以提出 plan proposal。
3. 不可以绕过 skill registry。
4. 不可以直接执行命令。
5. 不可以跳过 `PlanVerifier`、`PlanCritic` 和 deterministic recipe generation。

因此当前结构是：

```text
LLM proposes.
Rules verify.
Recipe Generator compiles.
Executor runs.
Observers judge.
Repair is bounded.
```

## 8. Recipe DAG 与执行器

### 8.1 SagaRecipe

SAGA 的核心中间产物是 `SagaRecipe`。它是一个可保存、可审计、可执行的 DAG。

一个 recipe 包含：

```text
recipe_id
task
planner
inputs
pipeline
outputs
repair_policy
metadata
```

一个典型 LoRA recipe pipeline：

```text
inspect_inputs
  -> run_diffusion_lora
  -> evaluate_outputs
  -> evaluate_distribution
  -> evaluate_sar_artifacts
  -> repair_policy
  -> export_dataset
```

### 8.2 Skill Executor

`recipe_executor.py` 是确定性执行层。它负责：

1. 拓扑排序 recipe steps。
2. 检查依赖是否完成。
3. 调用对应 skill wrapper。
4. 为每一步写入 started report 和 final report。
5. 持续更新 `recipe_execution_status.json`。
6. 生成统一的 `recipe_execution.json` 和 `recipe_execution.md`。
7. 为每一步附加 `SkillRunReport`。
8. 验证 `SkillRunReport` 协议一致性。

执行器支持：

```text
dry-run
real run
stop_on_error
step-level checkpoint artifacts
status tracking
uniform report validation
```

### 8.3 SkillRunReport 协议

每个 step 都会被包装成统一的 `SkillRunReport`：

```json
{
  "schema_version": "saga_skill_run_report_v1",
  "step_id": "run_diffusion_lora",
  "skill": "DiffusionLoRAGenerationSkill",
  "status": "succeeded",
  "dry_run": false,
  "message": "...",
  "elapsed_seconds": 123.4,
  "inputs": {},
  "outputs": {},
  "metrics": {},
  "artifacts": {},
  "issues": [],
  "warnings": [],
  "observer": {
    "trigger_count": 0,
    "triggers": []
  },
  "provenance": {}
}
```

合法状态包括：

```text
succeeded
failed
blocked
warning
triggered
not_triggered
dry_run
skipped
unknown
```

其中 `not_triggered` 用于 RepairPolicy 等 observer 类步骤，表示没有触发修复，不再被当作未知状态。

## 9. Observer、Evaluator 与 Benefit Evidence

### 9.1 Observer 的定位

SAGA 明确区分三类证据：

```text
生成是否成功
生成图是否基本可用
生成数据是否提升下游任务
```

质量检查和分布检查只能说明“可作为候选数据”，不能直接宣称提升分类、检测或分割。

### 9.2 当前可执行 Observer

当前主要 observer 包括：

1. `QualityEvaluationSkill`
   - 检查图像数量、可读性、尺寸、模式、黑白像素比例、亮度动态范围。

2. `DistributionEvaluationSkill`
   - 使用 `sar_fid_lite`、histogram L1、JSD、MMD、diversity、nearest reference distance、可选 pytorch FID。

3. `SARArtifactEvaluationSkill`
   - 检查条纹、平滑背景梯度、目标面积比例、目标 compactness、目标碎片数量、目标中心偏移。

4. `LeakageCheckSkill`
   - 检查 baseline、augmented、validation 之间的 hash overlap 和同 stem 可疑重叠。

5. `DuplicateNearDuplicateSkill`
   - 用 perceptual hashing 检查生成图的重复和近重复。

6. `ClassificationEvaluationSkill`
   - 需要显式请求或配置，默认不自动训练分类器。

### 9.3 BenefitEvidence 等级

`BenefitEvidenceReportSkill` 输出 evidence level：

```text
Level 0: planned_only / dry_run / failed
Level 1: generated and basic quality screened
Level 2: lightweight probes passed
Level 3: metadata or slice coverage evidence improved
Level 4: downstream task evaluator reports improvement and data gates pass
```

只有 Level 4 且分类/检测/分割 evaluator 给出 improvement，才允许声明下游任务收益。

这样可以避免常见问题：

```text
生成了很多图 -> 直接宣称提升准确率
```

SAGA 的报告会明确写：

```text
Downstream claim allowed: False
```

除非真的跑了下游 evaluator 并通过泄漏和重复检查。

## 10. Repair 与 Replan

SAGA 的修复不是开放式 ReAct，而是 bounded repair。

当前 repair 策略：

1. Observer 触发质量、分布、SAR 伪影等问题。
2. `RepairPolicySkill` 收集触发器。
3. `parameter_repair.py` 按白名单生成参数修复计划。
4. 修复计划可以写出 revised recipe。
5. expensive skill 不会自动重跑，除非用户显式执行 `repair-run` 或 `auto-repair-run`。

LoRA 修复参数白名单包括：

```text
training_epochs
max_train_samples
prompt_count
inference_steps
inference_guidance
inference_cfg_scale
lora_multiplier
```

RepairPolicy 的原则：

```text
最多 K 次
只改白名单参数
每次修复必须有 observer trigger
每次修复必须记录原因
失败时保留原始结果或拒绝坏样本
```

## 11. Memory 与策略学习

SAGA 的 memory 不是聊天记忆，而是 policy memory。

它记录：

1. 数据集画像。
2. 任务类型。
3. 选择的 skill 和 recipe。
4. 执行状态。
5. observer 结果。
6. export 结果。
7. downstream evaluator 结果。
8. benefit evidence level。
9. 风险和下一步建议。

下一次遇到类似数据集时，planner 可以检索历史 memory：

```text
similar dataset profile
similar task
similar metadata fields
similar skill choice
previous evidence level
previous warnings/failures
```

这使 SAGA 可以逐步从“先验规则”变成“经验策略”。

当前配置中 memory 默认启用：

```yaml
memory:
  enabled: true
  dir: runs/memory
  retrieval_limit: 5
```

## 12. Skill 体系

### 12.1 当前 Skill Registry 统计

当前注册：

```text
Total skills: 45
Executable: 35
Planned: 10
```

### 12.2 支撑与协议类 Skill

| Skill | 状态 | 作用 |
|---|---:|---|
| DatasetProfileReportSkill | executable | 记录数据画像和格式桥接结果 |
| RawDatasetScanSkill | executable | 确定性扫描文件、图像、sidecar、路径 token |
| LLMSchemaInductionSkill | executable | 让 LLM 基于摘要和用户说明提出 schema 候选 |
| DatasetFormatCompilerSkill | executable | 编译 DatasetFormatSpec |
| DatasetFormatValidatorSkill | executable | 验证 DatasetFormatSpec |
| RunProvenanceSkill | executable | 记录环境、配置和运行 provenance |
| PlanCriticSkill | executable | 检查计划风险和 claim 风险 |
| CandidateRecipePilotSkill | executable | 为 top-k 候选计划生成 pilot recipe |
| BoundedAutoRepairSkill | executable | 生成有界修复 recipe |
| CompareRunsSkill | executable | 比较多轮 SAGA run |
| BenefitEvidenceReportSkill | executable | 统一生成收益证据报告 |
| DatasetCardSkill | planned | 后续生成更完整数据集卡片 |

### 12.3 数据准备与规划证据 Skill

| Skill | 状态 | 作用 |
|---|---:|---|
| SARPreprocessSkill | executable | SAR 强度、位深、尺寸、动态范围归一化 |
| MetadataCaptionSkill | executable | 将 validated metadata 编译成 image/txt caption 对 |
| DatasetBalancingSkill | executable | 分析类别和 metadata bin 的缺口 |
| FileSelectionSkill | executable | 基于 validated metadata 做文件筛选 |

补充说明：工程中存在 `saga/skills/label_validation/` 模块，用于标签检查相关能力，但它当前没有作为独立 SkillRegistry 条目列入上述 45 个注册 skill 中。

### 12.4 生成与增广 Skill

| Skill | 状态 | 适用场景 |
|---|---:|---|
| TraditionalAugmentationSkill | executable | 最快、最简单、低风险 SAR-aware 传统增广 |
| GANImageToImageSkill | executable | 简单目标、简单外观变化、速度较快 |
| DiffusionLoRAGenerationSkill | executable | 高质量目标级生成，适合有 GPU 和时间的场景 |
| GeoDiffSARSkill | executable | 扩散模型 + 3D/物理先验 + ControlNet 的稀疏方位角补全 |
| StyleTransferSkill | executable | 跨载荷、跨平台、跨域特征迁移 |
| BackgroundGenerationSkill | executable | 使用已训练背景生成模型直接生成 SAR 背景 |
| TargetBackgroundCompositionSkill | executable | 目标和背景融合，支持 feather/laplacian/poisson/hard |
| PseudocolorSkill | executable | 伪彩色特征变换和可视化 |
| ModelToPOVSceneCompilerSkill | executable | 3D 模型到 RaySAR POV scene |
| RaySARSynthesisSkill | executable | RaySAR 物理仿真单场景成像 |
| RaySARSweepSynthesisSkill | executable | RaySAR 多方位角 sweep 成像 |
| GaussianSplattingCompletionSkill | planned | 轻量稀疏方位角补全，当前尚未接入 |

### 12.5 任务导向策略 Skill

| Skill | 状态 | 作用 |
|---|---:|---|
| ClassRebalanceAugmentationSkill | planned | 将类别缺口转成增广 recipe |
| AzimuthCoverageCompletionSkill | planned | 方位角覆盖补全策略 |
| PolarizationConditionedGenerationSkill | planned | 极化条件生成策略 |
| ResolutionBandConditionedGenerationSkill | planned | 分辨率、波段条件生成策略 |
| HardSampleMiningSkill | planned | 基于下游错误挖 hard samples |
| OODDomainExpansionSkill | planned | OOD 和跨域扩展策略 |

这些 planned skill 目前更多表现为 planner 中的策略逻辑和 SkillEffectCard，而不是独立可执行脚本。

### 12.6 评估类 Skill

| Skill | 状态 | 作用 |
|---|---:|---|
| QualityEvaluationSkill | executable | 基础图像质量检查 |
| DistributionEvaluationSkill | executable | 轻量分布/FID/MMD/多样性检查 |
| SARArtifactEvaluationSkill | executable | SAR 伪影和目标结构启发式检查 |
| LeakageCheckSkill | executable | 数据泄漏检查 |
| DuplicateNearDuplicateSkill | executable | 重复/近重复检查 |
| ClassificationEvaluationSkill | executable | 分类任务 baseline-vs-augmented 下游评估 |
| PerMetadataSliceEvaluationSkill | executable | 按 metadata slice 汇总结果 |
| ObjectDetectionEvaluationSkill | planned | 检测任务评估，当前不作为重点 |
| SegmentationEvaluationSkill | planned | 分割任务评估，当前不作为重点 |

## 13. 主要可执行工作流

### 13.1 LoRA 扩散模型生成

适用场景：

1. 用户希望高质量目标级 SAR 图像生成。
2. 用户有足够显存和时间。
3. 数据集有 `.txt` caption，或文件名/sidecar 中有可验证 metadata。
4. 用户希望按 class、polarization、band、resolution 等条件生成。

当前流程：

```text
dataset_root
  -> Raw Profiler
  -> Format Bridge
  -> MetadataCaptionSkill evidence
  -> DatasetBalancingSkill evidence
  -> DiffusionLoRAGenerationSkill
  -> QualityEvaluationSkill
  -> DistributionEvaluationSkill
  -> SARArtifactEvaluationSkill
  -> RepairPolicySkill
  -> ExportDatasetSkill
```

如果原始数据没有 `.txt`，但 format bridge 验证出了 metadata，LoRA skill 会自动 staging：

```text
metadata_caption_dataset/
  D7_0_hh.png
  D7_0_hh.txt
  D7_0_hv.png
  D7_0_hv.txt
```

导出时 LoRA recipe 默认写 caption sidecars，使最终增广数据集可直接复用为 image/txt caption dataset。

### 13.2 传统增广

适用场景：

1. 用户要求最快。
2. 用户只需要基础增强。
3. 数据集较小，需要一个保守 baseline。
4. 没有 GPU 或不希望训练生成模型。

支持 SAR-aware 的翻转、旋转、裁剪、强度扰动、speckle/noise 等。

### 13.3 GAN 图生图

适用场景：

1. 简单目标。
2. 简单外观变化。
3. MSTAR-like 数据。
4. 用户需要速度快于扩散模型。

局限是模式崩塌、结构控制弱、复杂目标效果不如扩散。

### 13.4 特征迁移

适用场景：

1. 用户明确提出跨平台、跨载荷、跨域适配。
2. 有 content 数据和 style/reference 数据。
3. 目标结构变化不大，主要是风格或传感器特征变化。

风险是过迁移导致目标结构损坏，因此默认接质量和 SAR artifact observer。

### 13.5 GeoDiff-SAR

GeoDiff-SAR 是高质量稀疏方位角补全 skill。当前 wrapper 将三部分接入：

1. `myproject/geodiff_components/preprocess3`
   - 从真实图像中提取 GEFM 条件图，用于训练。
2. `myproject/geodiff_components/raytracing`
   - 从 3D 模型中计算推理所需 GEFM 条件图。
3. `myproject/qinglong_trainer_29b`
   - 使用 ControlNet/LoRA 训练和推理。

适合：

```text
用户有 3D 模型
用户要求稀疏方位角补全
用户质量要求高
用户不敏感于时间和显存成本
```

### 13.6 RaySAR 物理仿真

RaySAR 在 SAGA 中被定义为物理仿真 skill，不作为默认分类增广。

当前分两层：

1. `ModelToPOVSceneCompilerSkill`
   - 将 OBJ/STL/PLY/GLB/点云等模型编译成 RaySAR 兼容 `.pov` scene。

2. `RaySARSynthesisSkill` / `RaySARSweepSynthesisSkill`
   - 使用 RaySAR/POV-Ray 执行 SAR 仿真和多视角 sweep。

优势：

```text
物理可解释性高
适合仿真、物理分析、3D 模型到 SAR 映射
```

局限：

```text
速度慢
存在 sim-to-real domain gap
需要模型和几何参数
不能默认当作分类任务增广
```

### 13.7 背景生成与目标背景融合

背景生成 skill 使用 `myproject/scene_gen_segment_large` 中已经训练好的模型，不要求用户训练背景生成器。

目标背景融合 skill 支持：

```text
auto/provided masks
intensity matching
feather
laplacian
poisson
hard
manifest
preview
caption sidecars
```

它适合后续检测和场景合成，但对于分类任务只是辅助，不是默认重点。

## 14. 输出结构与产物

一次完整 `agent-run` 目录通常包含：

```text
run_dir/
  saga_config_effective.json
  run_provenance.json
  skill_registry.json
  skill_registry.md

  intent_spec.json
  intent_effective_spec.json
  format_hints.json
  request_constraints.json

  raw_profile.json
  llm_context.json
  dataset_profile.json
  profile_report.md

  compiled_format_spec.yaml
  compiled_dataset.yaml
  format_bridge.json
  format_bridge.md
  validated_dataset_profile.json

  multi_dataset_profile.json
  planning_evidence/
    planning_evidence.json
    sar_preprocess/
    metadata_caption/
    dataset_balance/

  memory_retrieval.json
  dataset_need_profile.json
  skill_effect_cards.json
  skill_utility.json
  augmentation_plan.json
  augmentation_plan.md

  candidate_pilot/
  plan_critic.json
  plan_critic.md

  recipe.yaml

  execution/
    recipe_execution_status.json
    recipe_execution.json
    recipe_execution.md
    steps/
    observer/

  recipe_artifacts/
    ...

  augmented_dataset/
    images/
    manifest.jsonl
    dataset_card.json
    provenance.json
    export_report.json
    export_report.md

  benefit_evidence.json
  benefit_evidence.md
  agent_state.json
  agent_run.md
```

## 15. 配置体系

主配置文件：

```text
configs/saga.yaml
```

当前配置重点：

```yaml
planner:
  auto_apply_selected_augmentation_plan: true
  candidate_plan_limit: 8
  prefer_executable_skills: true

llm:
  config: configs/llm/deepseek_v4pro.yaml
  use_intent_by_default: true
  use_planner_by_default: true

execution:
  default_dry_run: true
  record_provenance: true
  max_repair_trials: 3

evaluation_policy:
  default_benefit_probe_mode: lightweight_metrics
  classification_evaluation_requires_explicit_request: true

memory:
  enabled: true
  dir: runs/memory
  retrieval_limit: 5

hardware:
  preferred_conda_env: sd3
```

LLM 配置支持写在：

```text
configs/llm/deepseek_v4pro.yaml
configs/llm/deepseek_v4flash.yaml
```

报告中不展示 API key。实际工程中建议优先通过环境变量 `DEEPSEEK_API_KEY` 注入，配置文件可以保留字段但应避免提交真实密钥。

## 16. CLI 命令

当前 CLI 已包含完整主干和主要 skill：

```text
diagnose
inspect-format
profile-dataset
infer-format
understand-format
build-format-spec
validate-format
bridge-format
agent-run
agent-continue
execute-recipe
smoke-matrix
evaluate-output
evaluate-distribution
evaluate-sar-artifacts
repair-parameters
repair-run
export-dataset
memory
explain-plan
compare-runs
benefit-evidence
plan-critic
pilot-recipes
auto-repair-run
plan
style-transfer
diffusion-lora
geodiff-sar
traditional-augment
sar-preprocess
metadata-caption
dataset-balance
gan-generate
background-generate
compose-target-background
pseudocolor
model-to-pov
raysar-synthesis
raysar-sweep
metadata-slice-eval
leakage-check
duplicate-check
classification-eval
agent-command
```

所有 Python 命令建议使用：

```bash
conda run -n sd3 python -m saga ...
```

## 17. 当前真实实验验证

当前已完成一轮真实 LoRA 实验，不是 dry-run。

用户需求：

```text
用 exampledataset/车辆数据-全极化/ZJGC-X 训练 LoRA 做有极化方式的扩散模型生成，
后面的字母代表极化方式，pauli 不作为数据集，
生成 10 张，训练 25 轮。
只做轻量质量、分布和 SAR 伪影评估，不训练分类器。
```

SAGA 自动完成：

1. 自然语言 intent 识别。
2. 数据集 raw profile。
3. 文件名后缀极化字段解析。
4. `pauli` 排除。
5. 自动 staging metadata-caption dataset。
6. Flux LoRA 训练。
7. 10 张推理生成。
8. 质量评估、分布评估、SAR 伪影评估。
9. repair policy 判断无触发。
10. 导出 augmented dataset。
11. 生成 benefit evidence。

实验结果：

```text
Execution status: succeeded
Training: 1600/1600 steps
Final average loss: about 0.311
Generated images: 10
Exported images: 10
Quality observer: passed
Distribution observer: passed
SAR artifact observer: passed
Repair policy: not_triggered
Evidence level: 3
Downstream claim allowed: False
```

关键目录：

```text
runs/real_experiment_lora_zjgcx_25ep_10gen/
  recipe.yaml
  execution/recipe_execution.json
  augmented_dataset/
  benefit_evidence.md
```

该实验说明 SAGA 当前已经具备自然语言到真实生成任务执行的闭环能力。

同时，SAGA 没有把这次结果夸大为“分类准确率提升”，因为用户明确不训练分类器，BenefitEvidence 正确给出：

```text
Downstream claim allowed: False
```

这是系统可靠性的关键表现。

## 18. 当前成熟度判断

当前 SAGA 已经不是一个简单脚本调度器，而是具备以下 agent 能力：

1. 非固定数据格式的渐进式理解。
2. LLM 与规则混合的 intent 和 format 解析。
3. DatasetFormatSpec 编译和验证门控。
4. 面向任务收益的 skill 排名。
5. 规划证据和 memory 参与决策。
6. Recipe DAG 中间产物。
7. 确定性执行器和 step-level report。
8. Observer/Evaluator 质量门控。
9. 有界 repair 和 replan。
10. Augmented dataset 标准导出。
11. BenefitEvidence 证据等级。
12. Policy memory 闭环。

因此，它已经具有区别于普通 ReAct 或普通 Plan-then-Execute 的任务特化 agent 结构。

## 19. 当前不足

### 19.1 Gaussian Splatting skill 尚未接入

这是当前明确缺口。它应作为轻量稀疏方位角补全方法，与 GeoDiff-SAR 形成互补：

```text
GeoDiff-SAR: 高质量，高成本，强物理/扩散先验
Gaussian Splatting: 轻量，速度快，补全能力弱于 GeoDiff
```

### 19.2 LLM planner 已默认开启，但仍受规则护栏约束

当前配置中：

```yaml
use_intent_by_default: true
use_planner_by_default: true
```

这意味着 `agent-run` 在加载 `configs/saga.yaml` 时会默认启用 LLM intent recognizer 和 LLM planner。LLM 仍然只负责高层意图解析、格式解释和计划建议；最终是否可执行仍由 deterministic guardrail、DatasetFormatSpec validator、Recipe Generator 和 PlanVerifier 共同约束。

当前需要继续积累更多真实任务测试，重点观察：

```text
自然语言意图是否稳定映射到正确 skill
DatasetFormatSpec 是否能覆盖更多非标准命名和标签格式
LLM planner proposal 是否能被规则 planner 正确吸收
PlanCritic 是否能及时阻止高风险或低收益计划
```

### 19.3 部分 planned policy skill 还未实体化

例如：

```text
ClassRebalanceAugmentationSkill
AzimuthCoverageCompletionSkill
PolarizationConditionedGenerationSkill
ResolutionBandConditionedGenerationSkill
HardSampleMiningSkill
OODDomainExpansionSkill
```

目前这些逻辑部分存在于 planner 和 SkillEffectCards 中，但还不是独立可执行 skill。

### 19.4 检测和分割暂非重点

检测和分割 evaluator 仍是 planned。考虑到当前 SAR 增广重点是目标级分类，这个优先级是合理的，但后续如果做场景合成和目标背景融合，就需要补检测格式和 bbox/mask 验证。

### 19.5 轻量指标不能替代下游实验

FID、MMD、SAR artifact、duplicate check 可以作为快速 probe，但不能证明分类或检测提升。SAGA 已在 BenefitEvidence 中做了限制，但论文和报告中也必须明确。

### 19.6 外部 skill 依赖仍较重

LoRA、GeoDiff-SAR、背景生成、RaySAR、分类评估都依赖外部项目、权重或系统环境。SAGA 当前通过 wrapper 和 config 管理这些依赖，但仍需补充更完整的 dependency check 和 install report。

## 20. 是否需要多 agent 框架

当前 SAGA 不需要立即改成复杂多 agent 框架。原因：

1. 当前执行链路需要强一致性和可复现性。
2. 数据格式验证、recipe 编译、执行、评估之间存在严格依赖。
3. 多 agent 如果没有清晰边界，容易引入不可复现的自由协商。

更合理的方向是保持单主控 agent，并在内部形成多个模块化角色：

```text
Profiler module
Intent module
Planner module
Verifier/Critic module
Executor module
Observer module
Repair module
Memory module
Reporter module
```

当未来任务变大时，可以引入受控并行子 agent，但建议只用于：

1. 多候选 recipe 的并行 dry-run。
2. 多指标评估的并行计算。
3. 多数据集 profile 的并行扫描。
4. 文档和报告生成。

不建议让多个 agent 同时修改同一个 recipe 或自由调用 heavy skill。

## 21. 推荐下一步

### P0：接入 Gaussian Splatting skill

这是当前 skill 层最明显缺口。需要定义：

```text
输入：稀疏方位角目标图、方位角 metadata、可选目标类别
输出：补全视角图像、角度 caption、run report
适用：用户要求快速稀疏方位角补全，显存/时间有限
评估：角度覆盖、质量、SAR artifact、重复率、可选分类 slice
```

### P1：开启更多 LLM planner 真实场景测试

在 guardrail 下测试：

```text
use_llm_intent
use_llm_planner
deepseek_v4pro
```

重点看 LLM planner 是否能提出比规则 planner 更合理的候选策略，但不能绕过 deterministic verifier。

### P2：强化 SkillEffectCard 和 policy memory

继续填充每个 skill 的适用/不适用表，尤其是：

```text
传统增广
GAN
LoRA diffusion
GeoDiff-SAR
Gaussian Splatting
Style Transfer
Background Generation
RaySAR
Target-Background Composition
```

### P3：完善 downstream evaluator 的轻量替代策略

默认不训练分类器是合理的，但可以加入更便宜的 probe：

1. embedding diversity。
2. 训练集/生成集最近邻可视化。
3. metadata slice coverage。
4. low-cost feature separability。
5. SAR-specific scatterer statistics。

### P4：完善报告和论文叙述

SAGA 的论文叙述重点应放在：

```text
不是又一个 SAR augmentation collection
而是 SAR dataset-conditioned augmentation decision agent
```

核心创新可表达为：

1. Progressive dataset profiling for arbitrary SAR user datasets.
2. LLM-assisted but validator-gated schema induction.
3. Benefit-aware augmentation planning with SAR skill priors.
4. Executable recipe DAG with deterministic skill runtime.
5. Observer-driven bounded repair and evidence-level reporting.
6. Policy memory for iterative augmentation strategy learning.

## 22. 总结

当前 SAGA 已经形成一个相对完整的 SAR 数据增广 agent 框架。它的核心不是“LLM 会调用工具”，而是将 SAR 数据集理解、格式桥接、任务意图识别、技能选择、可执行 recipe、确定性执行、质量观察、有界修复、收益证据和策略记忆串成闭环。

目前系统已经能完成真实 LoRA 生成实验，并能正确处理非标准数据命名中的极化信息、排除指定类别、自动生成 caption、训练生成模型、执行轻量评估、导出标准增广数据集，同时避免未经下游评估就声称任务收益。

从工程成熟度看，SAGA 当前已经具备一个生产原型所需的主干结构。接下来最值得投入的是：

```text
1. 接入 Gaussian Splatting skill
2. 继续打磨 LLM planner
3. 强化 memory 到 benefit policy learning
4. 丰富轻量收益 probe
5. 在更多真实 SAR 数据集上做对比实验
```

如果这些完成，SAGA 就可以从“可执行的 SAR 增广 agent 原型”进一步推进到“能根据数据集状态主动选择最有可能提升下游任务收益的 SAR 增广决策系统”。

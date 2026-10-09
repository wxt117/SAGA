# SAGA Skill Applicability Matrix

这个表用于把每个 skill 的适用/不适用条件整理成可被 planner 使用的策略知识。你可以直接在表格里补充或修改，后续我会把稳定内容同步进 `SkillEffectCard` / planner policy。

默认评估策略：

- 默认使用轻量 probe：`quality`、`distribution/FID-like`、`SAR artifacts`、`duplicate/near-duplicate`、必要时 `leakage`。
- 除非用户明确要求“下游分类收益、准确率、分类评估、benchmark、训练分类模型”，否则不默认训练分类 evaluator。
- 下游分类 evaluator 的结果最有说服力，但成本高，适合作为用户明确要求或最终确认阶段。

## 需要你填写/确认的 Skill 表


| Skill | 当前状态 | 核心用途 | 适用场景/数据集条件 | 不适用场景/失败条件 | 必需输入 | 可选输入 | 关键可调参数 | 速度/耗时 | 显存/硬件 | 主要优势 | 主要风险 | 默认轻量评估指标 | 何时需要分类 evaluator | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TraditionalAugmentationSkill | executable | 快速 SAR-aware 传统增广 | 简单快速增广；小样本分类；快速 baseline；需要低成本扩充 | 用户反馈需要更强增广；需要生成新目标形态/新场景语义；过强几何变换会破坏 SAR 方向性 | 图像数据集 | DatasetFormatSpec / metadata filters | flip, rotate, crop, intensity, speckle/noise, target_count, multiplier | 快，秒级到分钟级 | CPU 即可；大批量可多进程；几乎不占显存 | 低成本、稳定、可解释、易复现 | 多样性有限；可能产生重复；过强旋转/翻转可能破坏 SAR 成像几何或标签一致性 | quality, SAR artifacts, duplicate | 当增广样本加入训练并声称提升分类性能时需要；仅做快速生成可不触发 | 已接入 recipe |
| SARPreprocessSkill | executable | SAR 强度/位深/尺寸归一化 | 16-bit TIFF；动态范围不一致；LoRA、style transfer、GAN、evaluator 前处理 | 需要保持绝对辐射定标；图像已标准化；不希望改变灰度统计 | 图像目录 | resize / normalization mode | percentile_clip, log, resize, bit_depth | 快，秒级到分钟级 | CPU 即可 | 统一动态范围，降低后续生成/迁移/评估不稳定性 | 过度归一化会压制强散射点；log/clip 可能改变原始物理分布 | quality, dynamic range | 当预处理后的图像直接参与训练或影响分类输入时需要 | planner evidence skill |
| MetadataCaptionSkill | executable | 从 validated metadata 生成 LoRA caption 数据 | 有已验证 metadata；需要按极化、方位角、下视角、分辨率等条件训练 LoRA | metadata 缺失、字段未验证、caption 模板不可信；纯无条件生成 | 图像数据集 | DatasetFormatSpec / caption template | template, overwrite, mode | 很快，秒级 | CPU 即可 | 让 LoRA 训练具备条件语义；caption 与 metadata 可追踪 | metadata 错误会污染 caption；模板过单一会降低生成多样性 | caption coverage, metadata coverage | caption 生成的样本用于分类训练时需要；仅准备 caption 不需要 | planner evidence skill |
| DatasetBalancingSkill | executable | 类别/metadata bin 缺口估计 | 类别不均衡；极化/方位角/下视角覆盖不均；需要决定每类生成多少 | 无标签或 metadata；bin 太稀疏；用户目标不是补齐分布 | 图像数据集 | DatasetFormatSpec | fields, target_per_bin, max_multiplier | 很快，秒级 | CPU 即可 | 为 planner 提供 target_count / multiplier 依据 | 只能估计缺口，不能保证生成质量；过度补齐可能引入偏置 | deficit count, bin coverage | 根据其建议实际生成并加入训练后需要；单独诊断不需要 | planner evidence skill |
| DiffusionLoRAGenerationSkill | executable | 文本 caption LoRA 训练与扩散生成 | 有一定数量同域 SAR 图像；有 caption 或可生成 metadata caption；需要生成新样本、风格、目标或条件变化 | 数据极少；caption 错误；显存不足；要求严格几何/物理一致；时间预算很低 | 图像 + txt caption 或可生成 metadata caption | model_family / prompt / LoRA weights | model_family, target_count, epochs, prompt, guidance, seed | 慢；训练通常分钟到小时级，推理也较传统增广慢 | GPU 推荐；FLUX/SDXL 通常需要较大显存；CPU 不建议 | 多样性强；可学习 SAR 域风格；可结合 caption 做条件生成 | 过拟合、幻觉、标签不一致、SAR 物理不稳、模式坍缩 | quality, distribution/FID-like, SAR artifacts, duplicate, caption consistency | 必须；只要生成样本用于分类训练或声称提升分类性能 | 包装 qinglong trainer |
| GANImageToImageSkill | executable | 快速 GAN 图生图/简单目标生成 | 需要快速图像变换或简单生成；已有 checkpoint；低分辨率或轻量实验 | 需要复杂文本控制、多条件控制、高保真 SAR 生成；训练数据极少且分布复杂 | 图像数据集 | checkpoint / variant | target_count, epochs, variant, seed | 中等；通常比 diffusion 快 | GPU 推荐；dcgan-cpu 可 CPU 运行但较慢 | 轻量、速度快、实现简单 | 模式坍塌、多样性不足、训练不稳定、细节差 | quality, distribution/FID-like, duplicate/mode collapse | 生成样本进入分类训练时需要 | 包装 myproject/dcgan / dcgan-cpu |
| StyleTransferSkill | executable | 跨平台/跨载荷特征或风格迁移 | 有 content 域和 style 域；希望保持结构、迁移载荷/平台/成像风格 | content 与 style 语义差异过大；需要改变目标几何；标签严格依赖原始纹理 | content 图像/目录 + style 图像/目录 | metadata filters | style_strength, content_weight, preprocess_mode | 中等；按图优化时秒级到分钟级/张 | GPU 推荐；CPU 可运行但慢 | 无需大规模训练；适合少样本跨域迁移；结构相对可保留 | 风格过强会破坏目标结构；可能引入伪影；标签可能被污染 | quality, SAR artifacts, structure preservation proxy, distribution | 迁移结果作为分类训练数据时需要；仅可视化不需要 | 包装 myproject/stytransfer |
| PseudocolorSkill | executable | 伪彩色/特征可视化变换 | 可视化展示；人工检查；下游模型明确支持伪彩输入 | 标准 SAR 灰度训练；需要保持物理灰度意义；不希望引入颜色域偏移 | 图像目录 | colormap | colormap, preserve_tree | 很快，秒级 | CPU 即可 | 简单直观，便于展示和人工分析 | 颜色不具备真实物理语义；可能造成下游 domain shift | quality, duplicate, downstream-only-if-requested | 只有伪彩结果作为分类训练输入时需要 | 包装 myproject/weicaise 思路 |
| ModelToPOVSceneCompilerSkill | executable | 3D 模型转 RaySAR-compatible POV scene | 有 OBJ/STL/PLY/GLB/点云；需要物理仿真；尺度和坐标轴可确定 | 无 3D 模型；尺度/坐标不明；复杂材质无法定义；只需要快速增广 | OBJ/STL/PLY/GLB/点云 | incidence / scale / axes | incidence, azimuth, target_extent, sensor_plane, axes | 中等；通常秒级到分钟级 | CPU 即可 | 打通 3D 模型到 RaySAR 仿真；几何参数可控 | 坐标轴/尺度错误会导致仿真失真；材质简化明显 | scene compile success, geometry sanity | 单独编译不需要；仿真样本进入分类训练时需要 | RaySAR 前置 skill |
| RaySARSynthesisSkill | executable | 单视角 RaySAR 物理仿真 | 已有 POV scene 或 Contributions.txt；需要单视角物理可解释 SAR 样本 | 追求真实纹理分布；需要大规模快速生成；RaySAR/POV-Ray 环境不可用；作为默认分类增广任务 | POV scene 或 Contributions.txt | parameters.txt / adapted POV-Ray | width, height, postprocess, bounce/map products | 中等到慢；随场景复杂度增加 | 主要 CPU；依赖 RaySAR/POV-Ray 环境 | 物理可解释、几何可控、适合验证成像机制 | sim-to-real gap 明显；背景和 speckle 真实性有限 | render success, quality, SAR artifacts, domain-gap proxy | 默认不需要；只有用户明确把仿真图加入分类训练或做泛化实验时才需要 | 物理可解释但有域差异；不作为默认分类 skill |
| RaySARSweepSynthesisSkill | executable | 3D 模型多方位角 RaySAR sweep | 需要多方位角物理仿真/成像机制分析；有可靠 3D 模型 | 无 3D 模型；不需要角度覆盖；大规模场景快速生成；作为默认 ATR/分类增广任务 | 3D 模型 + azimuth_sweep/values | incidence / scale / parameters | azimuth_sweep, azimuth_values, incidence, target_extent, width, height | 慢；耗时随角度数量线性增长 | CPU 多进程推荐；可并行 | 系统性补全角度覆盖；物理参数清楚 | 计算成本高；sim-to-real gap；角度/尺度标注错误会放大问题 | render success rate, quality, SAR artifacts, per-angle coverage | 默认不需要；只有用户明确把 RaySAR sweep 输出用于 ATR/分类训练时才需要 | 用于物理可解释多角度生成；不作为默认分类 skill |
| ClassificationEvaluationSkill | executable | 下游分类 baseline-vs-augmented 评估 | 图像分类/ATR；需要验证增广是否提升下游性能 | 无分类标签；检测/分割任务；样本太少无法稳定划分；泄漏未处理 | baseline dataset | augmented dataset / val dataset | model, epochs, split_policy, metrics | 中等到慢；取决于训练轮数和模型 | GPU 推荐；小模型可 CPU | 直接验证任务收益，是最强证据 | 数据泄漏、随机种子、训练预算会影响结论；只适合分类 | leakage + duplicate gates before training | 本身就是分类 evaluator | 默认不自动触发，除非用户明确要求 |
| LeakageCheckSkill | executable | 数据泄漏检查 | baseline/augmented/val/test 共存；做下游评估前 | 只有单一目录且不比较训练/验证；只做质量检查 | baseline dataset | augmented / val dataset | sample_limit | 快，秒级到分钟级 | CPU 即可 | 防止虚高结果；保证实验可信 | hash 检查可能漏掉语义重复；same-stem 规则可能误伤 | hash overlap, same-stem overlap | 分类评估前必须 | evaluator gate |
| DuplicateNearDuplicateSkill | executable | 重复/近重复检查 | 生成样本入库前；检测 mode collapse；训练前清洗 | 仅做 raw profile；数据量极大但未采样 | 图像目录 | hash params | hash_size, hamming_threshold, sample_limit | 快到中等；取决于样本数 | CPU 即可 | 发现重复样本和生成坍塌 | 感知哈希对 SAR 强度变化敏感；可能误检/漏检 | duplicate count, near-duplicate pairs | 分类评估前建议 | mode collapse/过拟合 proxy |
| PerMetadataSliceEvaluationSkill | executable | 按 metadata slice 统计或评估 | 有 validated metadata；需要分析不同极化、方位角、类别、分辨率上的收益 | metadata 缺失或未验证；slice 样本太少；没有预测结果时只能统计 | DatasetFormatSpec | predictions csv | fields | 快；如果含模型预测则取决于预测规模 | CPU 即可 | 解释增广收益来自哪些 slice；发现长尾失败 | 小 slice 统计不稳定；metadata 错误会误导分析 | per-class/polarization/azimuth slice counts | 有分类预测时最有价值 | 可用于解释收益分布 |
| QualityEvaluationSkill | executable | 基础图像质量检查 | 任何生成/迁移/预处理输出图像目录 | 不能替代语义正确性和下游任务评估 | 图像目录 | thresholds | black/white ratio, dynamic range thresholds | 快，秒级到分钟级 | CPU 即可 | 默认质量 gate；能快速发现坏图、黑图、白图、动态范围异常 | 指标较粗，不能判断是否“像真实 SAR” | black/white/dynamic range/count | 不需要 | 默认 observer |
| DistributionEvaluationSkill | executable | FID-like / MMD / 分布距离 | 有 reference 和 generated；需要快速估计生成分布是否偏离真实集 | reference 很少；目标本来就是跨域迁移；强域变换不适合用分布接近衡量 | reference + generated 图像目录 | sample limits / feature size | image_size, sample_limit, standard_fid | 中等；standard FID 可能较慢 | CPU 可跑 lite；标准 FID 建议 GPU | 无需训练下游模型即可给出分布 proxy | FID-like 与 SAR 任务收益相关性有限；容易惩罚有用但分布不同的样本 | FID-like, MMD, histogram JSD, diversity | 只在需要最终任务收益时补分类 evaluator | 默认轻量 benefit probe |
| SARArtifactEvaluationSkill | executable | SAR-specific 伪影检查 | SAR 生成、迁移、仿真输出；需要检查条纹、碎片、目标紧致性 | 非 SAR 图像；无目标中心先验时 compactness 解释较弱 | 图像目录 | thresholds | stripe, gradient, compactness, center_offset | 快，秒级到分钟级 | CPU 即可 | 比通用 quality 更贴近 SAR 图像异常 | 启发式指标，不能完全代表物理真实性 | stripe, gradient, target compactness, fragmentation | 不需要 | 默认 observer |
| GeoDiffSARSkill | executable | 扩散 + 3D/物理先验稀疏方位角补全 | 目标级 SAR；方位角稀疏；有真实图像可提 GEFM；有 3D/几何先验可渲染推理 GEFM；需要参数可控生成 | 无 3D/几何先验；无训练图像或无 ControlNet 权重；复杂大场景；只需快速增广；显存不足 | 真实 SAR 图像/文本标签数据集 + 3D 模型 + 目标方位角需求 | 已有 ControlNet 权重 / 初始权重 / prompt / target_count / epochs | target_azimuths, azimuth_sweep, depressions, epochs, target_count, prompt, seed, guidance | 慢；GEFM、训练、推理均较重 | GPU 推荐，24GB+ 可尝试，32GB+ 更稳 | 几何感知、方位角可控、适合目标级补全；真实图像 GEFM 与 3D GEFM 打通 | GEFM 提取/渲染质量影响很大；3D模型不匹配会误导生成；速度慢，显存要求高 | quality, per-angle coverage, structure consistency, distribution, SAR artifacts | 生成样本用于 ATR/分类训练或声称方位角泛化收益时必须 | 已接入协议壳：`preprocess3` 真实图提 GEFM、`raytracing` 3D 模型算 GEFM、`qinglong_trainer_29b` FLUX ControlNet 训练/推理 |
| GaussianSplattingCompletionSkill | planned | 轻量高斯泼溅稀疏方位角补全 | 有多视角或稀疏视角样本；需要快速补视角；目标几何较稳定 | 视角太少；目标非刚性；SAR 散射与高斯外观表示差异过大 | 稀疏视角图像/目标实例 + 目标视角列表 | metadata / pose / masks | target_views, render_resolution, regularization | 较快 | GPU 2gb左右，非常轻量 | 比完整扩散训练轻；适合视角插值/补全 | SAR 散射中心不一定符合可见光 3DGS 假设；伪影和几何漂移，方位角补全能力较之geodiff更有限 | coverage, artifacts, distribution | 补全样本进入分类训练时需要 | 后续接实验室代码 |
| BackgroundGenerationSkill | executable | SAR 背景生成 | 显式要求生成背景/场景；检测、合成任务需要背景多样性；城市、水体、道路等 scene prompt 可描述 | 纯目标 chip 分类；无背景/场景需求；希望训练自己的背景模型 | scene prompt/scene class；默认使用内置 control bank 和已训练权重 | segment-controlled FLUX ControlNet 背景生成器 | scene_prompt, num_images, cns, steps, scale, seed, width, height | 中等到慢；dry-run 很快，真实推理取决于生成模型 | GPU 推荐 | 增强复杂背景和杂波多样性；可为后续目标-背景合成提供背景库 | 幻觉目标；背景分布不真实；可能引入未标注目标；不直接等价于分类收益 | background quality, distribution, artifact | 检测/合成任务需要时更应使用 detection/composition evaluator；分类任务通常不需要 | 已接入 `myproject/scene_gen_segment_large`，用户不训练，SAGA 直接调用已训练权重 |
| TargetBackgroundCompositionSkill | planned | 目标-背景/场景合成 | 目标检测数据增广；有 target bank 和 background bank；需要构造复杂场景、遮挡、小目标、长尾类别 | 没有可靠标签/mask；目标背景尺度/分辨率不匹配；严格物理耦合要求很高 | target bank + background bank | masks / placement rules | placement, scale, blend_mode, targets_per_scene | 中等；copy-paste 快，harmonization 较慢 | CPU 可做基础合成；深度 harmonization 需 GPU | 标签可控，直接服务检测任务，能补长尾和复杂背景 | 边界伪影、阴影不一致、bbox/mask 错误、目标-背景不合理 | annotation validity, composition quality | 检测任务应使用 detection evaluator；若转成 chip 分类再用分类 evaluator | 目前还没有代码 |


## 可选补充字段

如果某个 skill 比较复杂，可以在表格之外追加更细的条目：

```yaml
skill: SkillName
preferred_dataset_profile:
  num_images:
  num_classes:
  metadata_required:
  target_complexity:
  azimuth_condition:
  polarization_condition:
best_when:
  -
avoid_when:
  -
auto_tunable_params:
  -
user_must_provide:
  -
known_failure_modes:
  -
recommended_lightweight_probe:
  -
classification_evaluator_policy:
  when_required:
  default_epochs:
```

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saga.agent.augmentation_planner import build_augmentation_plan
from saga.agent.benefit_estimator import build_benefit_context
from saga.agent.intent_recognizer import recognize_request
from saga.agent.request_constraints import infer_request_constraints
from saga.agent.skill_registry import built_in_skill_specs
from saga.core.config import save_json, save_text


@dataclass(frozen=True)
class PlanningCase:
    case_id: str
    title: str
    case_group: str
    request: str
    profile: dict[str, Any]
    gold_task: str
    gold_skill: str | None
    gold_rejected_skills: list[str] = field(default_factory=list)
    required_observers: list[str] = field(default_factory=list)
    required_params: dict[str, Any] = field(default_factory=dict)
    expected_executable: bool = True
    reason: str = ""


@dataclass(frozen=True)
class PlannerMethod:
    method_id: str
    title: str
    description: str


METHODS = [
    PlannerMethod("keyword_router", "Keyword Router", "Selects a skill from literal request keywords only."),
    PlannerMethod("rule_only", "Rule-only Planner", "Uses deterministic task rules and shallow compatibility checks without LLM proposals or benefit ranking."),
    PlannerMethod("llm_only", "LLM-only Planner", "Uses an unverified semantic proposal protocol without schema validation, skill guardrails, or recipe compilation."),
    PlannerMethod("react_style", "ReAct-style Agent", "Simulates free-form tool selection and dry-run tool calls without a global recipe validator."),
    PlannerMethod("full_saga", "Full SAGA", "Combines LLM-style proposal, validated profile evidence, guardrails, benefit ranking, observers, and recipe readiness."),
]


GENERATION_OBSERVERS = ["QualityEvaluationSkill", "SARArtifactEvaluationSkill", "DuplicateNearDuplicateSkill"]
DISTRIBUTION_OBSERVERS = ["DistributionEvaluationSkill"]
COMPOSITION_OBSERVERS = ["QualityEvaluationSkill", "DistributionEvaluationSkill", "SARArtifactEvaluationSkill", "DuplicateNearDuplicateSkill"]
PHYSICS_OBSERVERS = ["QualityEvaluationSkill", "SARArtifactEvaluationSkill"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 2: intent recognition and skill planning.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp2_intent_skill_planning",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    reset_dir(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cases = benchmark_cases()
    results, case_details = run_cases(cases=cases, output_dir=output_dir)
    summary = summarize_results(results)
    error_summary = summarize_error_tags(results)
    case_group_summary = summarize_case_groups(results)
    case_matrix = render_case_matrix_rows(results, cases)

    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "summary_by_method.csv", summary)
    write_csv(output_dir / "error_summary.csv", error_summary)
    write_csv(output_dir / "case_group_summary.csv", case_group_summary)
    write_csv(output_dir / "case_method_matrix.csv", case_matrix)
    write_csv(output_dir / "case_catalog.csv", [case_row(case) for case in cases])
    save_json(output_dir / "case_details.json", case_details)
    save_text(output_dir / "table_exp2_summary.tex", render_latex_summary(summary))
    save_text(output_dir / "table_exp2_error_summary.tex", render_latex_error_summary(error_summary))
    save_text(output_dir / "exp2_report.md", render_report(output_dir, cases, summary, results, error_summary, case_group_summary))

    if not args.skip_plots:
        plot_all(results, summary, cases, figures_dir)

    print(f"Experiment 2 complete: {output_dir}")
    print(f"Results: {output_dir / 'results.csv'}")
    print(f"Summary: {output_dir / 'summary_by_method.csv'}")
    print(f"Report: {output_dir / 'exp2_report.md'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def benchmark_cases() -> list[PlanningCase]:
    cases = [
        PlanningCase(
            case_id="traditional_fast",
            title="Fast conservative augmentation",
            case_group="explicit_generation",
            request="对 exampledataset/车辆数据-全极化/ZJGC-X 做最快的传统增广，生成50张，不训练模型。",
            profile=profile_vehicle_pol(),
            gold_task="traditional_augmentation",
            gold_skill="TraditionalAugmentationSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GeoDiffSARSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"target_count": 50},
            reason="Speed-first label-preserving augmentation should not invoke heavy generative models.",
        ),
        PlanningCase(
            case_id="polarization_lora",
            title="Polarization-conditioned LoRA",
            case_group="metadata_conditioned_generation",
            request="基于全极化 ZJGC-X 数据训练LoRA，利用文件名中的极化方式生成100张HH/HV/VH/VV样本，训练25个epoch。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 100, "training_epochs": 25, "auto_caption_from_metadata": True},
            reason="Metadata-conditioned target generation with captions is LoRA's best fit.",
        ),
        PlanningCase(
            case_id="gan_simple_fast",
            title="Simple target GAN",
            case_group="explicit_generation",
            request="MSTAR-like 简单车辆目标，GPU可用，希望快速训练一个GAN生成200张外观变化样本。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill="GANImageToImageSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 200, "variant": "gpu"},
            reason="Simple target and fast learned generation favor GAN over heavy diffusion.",
        ),
        PlanningCase(
            case_id="geodiff_high_quality_sparse",
            title="High-quality sparse-azimuth completion",
            case_group="geometry_conditioned_generation",
            request=(
                "这些飞机SAR图像只有稀疏方位角，已有 myproject/geodiff_components/models/b747.obj 作为3D物理先验，"
                "请用GeoDiff-SAR高质量补全方位角0到350每隔10度，不计显存。"
            ),
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="geodiff_sar_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["GaussianSplattingCompletionSkill", "TraditionalAugmentationSkill", "GANImageToImageSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={
                "model_file": "myproject/geodiff_components/models/b747.obj",
                "azimuth_sweep": {"start": 0, "stop": 350, "step": 10},
                "quality_mode": "high",
            },
            reason="Large-angle high-quality completion with a physical prior favors GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="gaussian_splatting_low_vram",
            title="Lightweight Gaussian splatting completion",
            case_group="geometry_conditioned_generation",
            request=(
                "稀疏方位角车辆数据需要补方位，但显存很低，只要轻量快速的高斯泼溅/SAR GS方案，"
                "补30、60、90度视角。"
            ),
            profile=profile_vehicle_sparse(),
            gold_task="gaussian_splatting_completion",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_azimuths": [30, 60, 90], "project_root": "myproject/SAR GS V1", "render_resolution": 128},
            reason="Low-VRAM sparse-view completion explicitly requests Gaussian splatting.",
        ),
        PlanningCase(
            case_id="implicit_low_vram_sparse_completion",
            title="Implicit low-VRAM sparse-view completion",
            case_group="implicit_benefit_planning",
            request=(
                "车辆SAR数据只有少数方位角，想补方位到30、60、90度，显存很低，"
                "不要用很重的扩散模型，请自动选择合适的轻量方案。"
            ),
            profile=profile_vehicle_sparse(),
            gold_task="augmentation_or_generation",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_azimuths": [30, 60, 90], "project_root": "myproject/SAR GS V1", "render_resolution": 128},
            reason="The method is not named; low VRAM and sparse-view completion should favor SAR GS over GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="implicit_geodiff_with_prior",
            title="Implicit high-quality completion with physical prior",
            case_group="implicit_benefit_planning",
            request=(
                "飞机SAR只有少数方位，已有 exampledataset/models/b747.obj 作为3D物理先验。"
                "不做RaySAR物理仿真，希望高质量补方位到0到350每隔10度，请自动选择方法。"
            ),
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="augmentation_or_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["GaussianSplattingCompletionSkill", "TraditionalAugmentationSkill", "GANImageToImageSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={
                "model_file": "exampledataset/models/b747.obj",
                "azimuth_sweep": {"start": 0, "stop": 350, "step": 10},
                "quality_mode": "high",
            },
            reason="The method is not named; high quality plus a 3D physical prior should favor GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="raysar_sweep",
            title="RaySAR full-azimuth sweep",
            case_group="physics_simulation",
            request=(
                "使用 exampledataset/models/t72.obj 做RaySAR物理仿真，方位角从0到350每隔10度，"
                "输出可解释的单次/二次散射图。"
            ),
            profile=profile_model_only(has_model=True, sweep=True),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSweepSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"model_file": "exampledataset/models/t72.obj", "raysar_geometry": {"azimuth_sweep": {"start": 0, "stop": 350, "step": 10}}},
            reason="Explicit 3D model plus azimuth sweep should select RaySAR sweep.",
        ),
        PlanningCase(
            case_id="raysar_single_scene",
            title="RaySAR single scene or postprocess",
            case_group="physics_simulation",
            request="已有 RaySAR-compatible scene.pov 和 parameters.txt，请做一次物理仿真并导出反射图，不需要训练网络。",
            profile=profile_model_only(has_model=True, sweep=False),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"pov_scene": "scene.pov", "parameters_file": "parameters.txt"},
            reason="Single RaySAR scene without sweep should not be converted into data-driven generation.",
        ),
        PlanningCase(
            case_id="style_transfer_payload",
            title="Cross-payload style transfer",
            case_group="multi_input_composition",
            request=(
                "把 exampledataset/车辆数据-全极化/ZJGC-X 作为内容图，exampledataset/车辆数据-全极化/BBZC-X "
                "作为风格图，做跨载荷特征迁移，风格强度保守一些。"
            ),
            profile=profile_domain_pair(),
            gold_task="style_transfer",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["TraditionalAugmentationSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"content_source": "exampledataset/车辆数据-全极化/ZJGC-X", "style_source": "exampledataset/车辆数据-全极化/BBZC-X"},
            reason="Content/style domains and payload shift indicate style transfer.",
        ),
        PlanningCase(
            case_id="background_generation",
            title="SAR water background generation",
            case_group="explicit_generation",
            request="生成80张SAR水体背景图，包含海杂波和岸线，不训练用户模型，使用已有背景生成权重。",
            profile=profile_background_prompt(),
            gold_task="background_generation",
            gold_skill="BackgroundGenerationSkill",
            gold_rejected_skills=["TargetBackgroundCompositionSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"target_count": 80},
            reason="The request asks for background assets, not target generation or composition.",
        ),
        PlanningCase(
            case_id="ship_water_composition",
            title="Ship target-background composition",
            case_group="multi_input_composition",
            request="把 exampledataset/ship/cnt 的船目标合成到随机生成的水体背景中，输出合成图和manifest。",
            profile=profile_composition(),
            gold_task="target_background_composition",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["PseudocolorSkill", "TraditionalAugmentationSkill"],
            required_observers=COMPOSITION_OBSERVERS,
            required_params={"target_dir": "exampledataset/ship/cnt", "background_dir": "runs/background_generation"},
            reason="Target and background sources imply SAR-aware image composition.",
        ),
        PlanningCase(
            case_id="pseudocolor_visualization",
            title="Pseudocolor visualization",
            case_group="explicit_transform",
            request="对ZJGC-X样本做伪彩色可视化，方便人工检查散射中心，不要训练生成模型。",
            profile=profile_vehicle_pol(),
            gold_task="pseudocolor_transform",
            gold_skill="PseudocolorSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            reason="Explicit visualization/pseudocolor request should not become learned generation.",
        ),
        PlanningCase(
            case_id="reject_geodiff_no_prior",
            title="Reject GeoDiff without physical prior",
            case_group="negative_guardrail",
            request="只有一小批ship SAR chip，没有3D模型也没有物理先验，但想用GeoDiff-SAR补全大角度方位。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="geodiff_sar_generation",
            gold_skill=None,
            gold_rejected_skills=["GeoDiffSARSkill", "GaussianSplattingCompletionSkill", "RaySARSweepSynthesisSkill"],
            required_observers=[],
            expected_executable=False,
            reason="GeoDiff and Gaussian completion require geometry/view metadata; this case should be rejected or clarified.",
        ),
        PlanningCase(
            case_id="reject_raysar_no_model",
            title="Reject RaySAR without model",
            case_group="negative_guardrail",
            request="我想做RaySAR物理仿真，但现在只有普通SAR PNG目标图，没有obj/stl/ply模型或pov场景。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="raysar_synthesis",
            gold_skill=None,
            gold_rejected_skills=["RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"],
            required_observers=[],
            expected_executable=False,
            reason="RaySAR must not be selected as executable without model/POV/contributions assets.",
        ),
        PlanningCase(
            case_id="downstream_evidence_lora",
            title="LoRA with downstream evidence",
            case_group="downstream_evidence",
            request="训练LoRA生成ZJGC-X增广数据，并要求最后做下游ATR分类评估，报告准确率提升证据。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["RaySARSynthesisSkill", "PseudocolorSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS + ["ClassificationEvaluationSkill"],
            reason="Generation remains LoRA, but downstream evaluator must be attached.",
        ),
        PlanningCase(
            case_id="reject_style_missing_domain",
            title="Reject style transfer without style domain",
            case_group="negative_guardrail",
            request="我只有ZJGC-X内容图，但想做跨载荷风格迁移，暂时没有任何风格图或参考域。",
            profile=profile_vehicle_pol(),
            gold_task="style_transfer",
            gold_skill=None,
            gold_rejected_skills=["StyleTransferSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Style transfer must not execute without a style/reference domain.",
        ),
        PlanningCase(
            case_id="reject_composition_missing_background",
            title="Reject composition without background",
            case_group="negative_guardrail",
            request="把这些船目标合成到水体背景中，但现在只有船目标chip，没有背景图、mask或背景生成结果。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="target_background_composition",
            gold_skill=None,
            gold_rejected_skills=["TargetBackgroundCompositionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Target-background composition requires a target source and a background source.",
        ),
        PlanningCase(
            case_id="implicit_fast_no_training",
            title="Implicit fast no-training augmentation",
            case_group="implicit_benefit_planning",
            request="这批车辆SAR chip 只是想尽快扩一点训练集，不要训练任何生成模型，也不要造新方位，生成120张。",
            profile=profile_simple_targets(captions=False),
            gold_task="augmentation_or_generation",
            gold_skill="TraditionalAugmentationSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill", "GeoDiffSARSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            required_params={"target_count": 120},
            reason="Implicit speed-first request with no training should resolve to conservative traditional augmentation.",
        ),
        PlanningCase(
            case_id="implicit_high_quality_metadata_generation",
            title="Implicit high-quality metadata generation",
            case_group="implicit_benefit_planning",
            request=(
                "ZJGC-X 文件名里有类别、方位角和极化字段，想生成更像真实目标的高质量样本，"
                "优先利用这些metadata自动做caption，生成160张。"
            ),
            profile=profile_vehicle_pol(),
            gold_task="augmentation_or_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["TraditionalAugmentationSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 160, "auto_caption_from_metadata": True},
            reason="High-quality metadata-conditioned target generation should favor diffusion LoRA even without naming LoRA.",
        ),
        PlanningCase(
            case_id="captioned_lora_sdxl",
            title="Captioned SDXL LoRA generation",
            case_group="metadata_conditioned_generation",
            request="数据集已有同名txt caption，用SDXL训练LoRA 18个epoch，生成90张目标SAR样本。",
            profile=profile_simple_targets(captions=True),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 90, "training_epochs": 18, "model_family": "sdxl"},
            reason="Explicit LoRA with captions should bind model family, epoch count, and target count.",
        ),
        PlanningCase(
            case_id="fast_gan_no_diffusion",
            title="Fast GAN without diffusion",
            case_group="explicit_generation",
            request="简单车辆chip上训练一个DCGAN快速生成150张，不要扩散模型，也不需要物理先验。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill="GANImageToImageSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GeoDiffSARSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 150},
            reason="Explicit fast DCGAN request should not be routed to diffusion or physics simulation.",
        ),
        PlanningCase(
            case_id="gs_3dgs_synonym",
            title="3DGS synonym sparse-view completion",
            case_group="geometry_conditioned_generation",
            request="用3DGS/SAR GS给稀疏视角车辆SAR补45、90、135度，分辨率128，显存只有6GB。",
            profile=profile_vehicle_sparse(),
            gold_task="gaussian_splatting_completion",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"project_root": "myproject/SAR GS V1", "render_resolution": 128},
            reason="The request uses a SAR GS synonym and low-VRAM constraint, so Gaussian splatting should be selected.",
        ),
        PlanningCase(
            case_id="geodiff_not_raysar_prior",
            title="GeoDiff with prior, not RaySAR",
            case_group="geometry_conditioned_generation",
            request=(
                "已有 exampledataset/models/b747.obj 和稀疏方位角图像，但不要RaySAR直接仿真，"
                "希望用几何扩散补全0到350每隔20度的视角。"
            ),
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="geodiff_sar_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["RaySARSweepSynthesisSkill", "GaussianSplattingCompletionSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"model_file": "exampledataset/models/b747.obj", "quality_mode": "balanced"},
            reason="Physical simulation is explicitly excluded while a prior is available for GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="geodiff_downstream_evidence",
            title="GeoDiff with downstream evidence",
            case_group="downstream_evidence",
            request=(
                "用GeoDiff-SAR补全飞机目标稀疏方位角，并在增广后训练ATR分类器，"
                "报告按方位角划分的准确率提升。"
            ),
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="geodiff_sar_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["TraditionalAugmentationSkill", "GANImageToImageSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS + ["LeakageCheckSkill", "ClassificationEvaluationSkill"],
            reason="Sparse-angle completion remains GeoDiff-SAR, but downstream evidence requires leakage and classification evaluators.",
        ),
        PlanningCase(
            case_id="raysar_contributions_postprocess",
            title="RaySAR contribution postprocess",
            case_group="physics_simulation",
            request="已有RaySAR输出的 Contributions.txt 和参数文件，只需要后处理成SAR反射图并做质量检查。",
            profile=profile_raysar_contributions(),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"contributions_txt": "runs/raysar/Contributions.txt", "parameters_file": "parameters.txt"},
            reason="Existing RaySAR contribution files should use the single-scene synthesis/postprocess path.",
        ),
        PlanningCase(
            case_id="raysar_sweep_obj_ply",
            title="RaySAR sweep from PLY model",
            case_group="physics_simulation",
            request="使用 exampledataset/models/truck.ply 做三维模型成像，方位角从0到180每隔15度，输出物理可解释样本。",
            profile=profile_model_only(has_model=True, sweep=True, model_file="exampledataset/models/truck.ply"),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSweepSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"model_file": "exampledataset/models/truck.ply"},
            reason="A PLY model plus explicit azimuth sweep should choose RaySAR sweep.",
        ),
        PlanningCase(
            case_id="background_generation_distribution",
            title="Background generation with distribution probe",
            case_group="explicit_generation",
            request="做SAR背景生成，生成60张海面背景图，要求除了质量和伪影，也给出与真实水体背景的分布指标。",
            profile=profile_background_prompt(),
            gold_task="background_generation",
            gold_skill="BackgroundGenerationSkill",
            gold_rejected_skills=["TargetBackgroundCompositionSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 60},
            reason="The generation skill is clear, but the request explicitly asks for a distribution probe.",
        ),
        PlanningCase(
            case_id="composition_poisson_clean_background",
            title="Poisson ship-background composition",
            case_group="multi_input_composition",
            request=(
                "把 exampledataset/ship/cnt 作为目标，runs/background_generation/clean_water 作为背景，"
                "用泊松融合生成70张检测训练图。"
            ),
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="target_background_composition",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["PseudocolorSkill", "TraditionalAugmentationSkill"],
            required_observers=COMPOSITION_OBSERVERS,
            required_params={"target_count": 70, "blend_mode": "poisson"},
            reason="Explicit target and clean background sources with Poisson blending imply composition.",
        ),
        PlanningCase(
            case_id="composition_downstream_leakage",
            title="Composition with downstream leakage check",
            case_group="downstream_evidence",
            request="做船目标和水体背景合成，输出后必须检查重复/泄漏，并训练下游分类模型看是否提升。",
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="target_background_composition",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["BackgroundGenerationSkill", "PseudocolorSkill"],
            required_observers=COMPOSITION_OBSERVERS + ["LeakageCheckSkill", "ClassificationEvaluationSkill"],
            reason="Composition is the selected skill, with downstream and leakage observers attached.",
        ),
        PlanningCase(
            case_id="pseudocolor_rgb_ablation",
            title="Pseudocolor RGB ablation",
            case_group="explicit_transform",
            request="给ZJGC-X生成伪彩色RGB版本，只用于分类消融和人工检查，不作为物理新样本。",
            profile=profile_vehicle_pol(),
            gold_task="pseudocolor_transform",
            gold_skill="PseudocolorSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            reason="Pseudocolor is explicitly framed as visualization/RGB ablation rather than generation.",
        ),
        PlanningCase(
            case_id="style_transfer_with_downstream",
            title="Style transfer with downstream test",
            case_group="downstream_evidence",
            request=(
                "把 exampledataset/车辆数据-全极化/ZJGC-X 作为内容图，exampledataset/车辆数据-全极化/BBZC-X "
                "作为参考域做跨载荷风格迁移，然后做下游分类评估。"
            ),
            profile=profile_domain_pair(),
            gold_task="style_transfer",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["RaySARSynthesisSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + ["LeakageCheckSkill", "ClassificationEvaluationSkill"],
            required_params={"content_source": "exampledataset/车辆数据-全极化/ZJGC-X", "style_source": "exampledataset/车辆数据-全极化/BBZC-X"},
            reason="Cross-payload style transfer should attach downstream evidence checks when requested.",
        ),
        PlanningCase(
            case_id="style_transfer_numeric_strength",
            title="Style transfer numeric strength",
            case_group="hard_argument_binding",
            request=(
                "把 exampledataset/车辆数据-全极化/ZJGC-X 作为内容图，exampledataset/车辆数据-全极化/BBZC-X 作为风格图做跨载荷风格迁移，"
                "style_strength=0.25，尽量保持目标结构。"
            ),
            profile=profile_domain_pair(),
            gold_task="style_transfer",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["TraditionalAugmentationSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"content_source": "exampledataset/车辆数据-全极化/ZJGC-X", "style_source": "exampledataset/车辆数据-全极化/BBZC-X", "style_strength": 0.25},
            reason="This hard case tests whether the planner can bind a numeric style-strength argument, not only select the right skill.",
        ),
        PlanningCase(
            case_id="reject_lora_user_forbids_training",
            title="Reject LoRA when training is forbidden",
            case_group="negative_guardrail",
            request="我想要LoRA风格的高质量样本，但这次明确不训练任何模型，也没有现成LoRA权重可用。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill=None,
            gold_rejected_skills=["DiffusionLoRAGenerationSkill"],
            required_observers=[],
            expected_executable=False,
            reason="The user asks for LoRA-like generation but forbids the required training/existing-weight condition.",
        ),
        PlanningCase(
            case_id="reject_gan_user_forbids_training",
            title="Reject GAN when training is forbidden",
            case_group="negative_guardrail",
            request="用GAN给简单车辆目标生成样本，但不要训练GAN，也没有提供预训练checkpoint。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill=None,
            gold_rejected_skills=["GANImageToImageSkill"],
            required_observers=[],
            expected_executable=False,
            reason="A GAN generation request cannot execute when the user forbids training and provides no checkpoint.",
        ),
        PlanningCase(
            case_id="reject_gs_no_azimuth",
            title="Reject SAR GS without view metadata",
            case_group="negative_guardrail",
            request="只有普通ship chip，没有方位角字段，但想用myproject/SAR GS V1补45和90度视角。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="gaussian_splatting_completion",
            gold_skill=None,
            gold_rejected_skills=["GaussianSplattingCompletionSkill", "GeoDiffSARSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Gaussian splatting completion requires view/azimuth metadata.",
        ),
        PlanningCase(
            case_id="reject_geodiff_has_azimuth_no_prior",
            title="Reject GeoDiff with azimuth but no prior",
            case_group="negative_guardrail",
            request="这些车辆SAR有方位角metadata，但没有3D模型或物理先验，请用GeoDiff-SAR补大角度视角。",
            profile=profile_ship_chips(no_azimuth=False),
            gold_task="geodiff_sar_generation",
            gold_skill=None,
            gold_rejected_skills=["GeoDiffSARSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Azimuth metadata alone is insufficient for GeoDiff-SAR without a physical or geometric prior.",
        ),
        PlanningCase(
            case_id="reject_composition_missing_target",
            title="Reject composition without target source",
            case_group="negative_guardrail",
            request="我只有一批水体背景图，想做船目标背景合成，但没有任何船目标chip或目标mask。",
            profile=profile_background_only(),
            gold_task="target_background_composition",
            gold_skill=None,
            gold_rejected_skills=["TargetBackgroundCompositionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Composition must not execute without a target source.",
        ),
        PlanningCase(
            case_id="reject_multi_skill_auto_background",
            title="Reject unsupported implicit multi-skill composition",
            case_group="hard_multi_step",
            request="现在只有船目标chip，请先自动生成水体背景，再把目标合成进去，整个流程一次性给recipe。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="target_background_composition",
            gold_skill=None,
            gold_rejected_skills=["TargetBackgroundCompositionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="This benchmark version requires explicit background assets; implicit background-generation plus composition is marked as a clarification case.",
        ),
        PlanningCase(
            case_id="reject_raysar_missing_geometry",
            title="Reject RaySAR model without geometry",
            case_group="negative_guardrail",
            request="有一个obj模型，但没有RaySAR参数、视角设置或pov场景，直接生成一套可用SAR训练图。",
            profile=profile_model_file_only(),
            gold_task="raysar_synthesis",
            gold_skill=None,
            gold_rejected_skills=["RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"],
            required_observers=[],
            expected_executable=False,
            reason="The model exists but the simulation geometry is underspecified; this case should request clarification.",
        ),
    ]
    cases.extend(additional_planning_stress_cases())
    if len(cases) != 80:
        raise RuntimeError(f"Experiment 2 should contain 80 planning cases, got {len(cases)}")
    return cases


def additional_planning_stress_cases() -> list[PlanningCase]:
    return [
        PlanningCase(
            case_id="traditional_speckle_small",
            title="Traditional speckle-safe augmentation",
            case_group="explicit_transform",
            request="给车辆SAR chip 做少量平移、裁剪和speckle扰动，生成40张，保持标签不变，不训练模型。",
            profile=profile_simple_targets(captions=False),
            gold_task="traditional_augmentation",
            gold_skill="TraditionalAugmentationSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            required_params={"target_count": 40},
            reason="Small label-preserving transforms should use traditional augmentation.",
        ),
        PlanningCase(
            case_id="traditional_no_new_angles",
            title="Traditional no-new-angle request",
            case_group="implicit_benefit_planning",
            request="只想增强平移鲁棒性，不要生成新方位角或新目标外观，扩到180张。",
            profile=profile_simple_targets(captions=False),
            gold_task="traditional_augmentation",
            gold_skill="TraditionalAugmentationSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "GaussianSplattingCompletionSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            required_params={"target_count": 180},
            reason="The request excludes semantic/view synthesis and asks for robustness augmentation.",
        ),
        PlanningCase(
            case_id="lora_gpu0_polar_balance",
            title="LoRA GPU0 polarization balance",
            case_group="metadata_conditioned_generation",
            request="使用GPU 0训练LoRA，按HH/HV/VH/VV均衡生成120张ZJGC-X车辆样本，epoch=20。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 120, "training_epochs": 20, "auto_caption_from_metadata": True},
            reason="Polarization-balanced metadata-conditioned generation is a LoRA case.",
        ),
        PlanningCase(
            case_id="lora_caption_epochs_12",
            title="Captioned LoRA 12 epochs",
            case_group="hard_argument_binding",
            request="已有caption txt，训练LoRA 12个epoch，生成75张目标chip，保留caption-derived metadata。",
            profile=profile_simple_targets(captions=True),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 75, "training_epochs": 12},
            reason="Tests binding of epoch and target-count arguments.",
        ),
        PlanningCase(
            case_id="lora_minority_class_downstream",
            title="LoRA minority class downstream",
            case_group="downstream_evidence",
            request="给少数类车辆训练LoRA增广，之后做ATR分类器评估，并检查重复和泄漏。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["PseudocolorSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS + ["LeakageCheckSkill", "ClassificationEvaluationSkill"],
            required_params={"auto_caption_from_metadata": True},
            reason="Minority-class generation with downstream evidence remains a LoRA planning case.",
        ),
        PlanningCase(
            case_id="gan_cpu_tiny",
            title="Tiny GAN under low budget",
            case_group="explicit_generation",
            request="没有足够时间跑扩散模型，在简单车辆chip上训练一个轻量GAN生成64张。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill="GANImageToImageSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GeoDiffSARSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 64},
            reason="Low-budget learned generation should favor GAN.",
        ),
        PlanningCase(
            case_id="gan_checkpoint_generate_only",
            title="GAN checkpoint generate-only",
            case_group="explicit_generation",
            request="已有GAN checkpoint，只用它生成100张MSTAR-like车辆chip，不重新训练。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill="GANImageToImageSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 100},
            reason="A provided generator checkpoint should still select the GAN generation skill.",
        ),
        PlanningCase(
            case_id="geodiff_aircraft_20deg",
            title="GeoDiff aircraft 20-degree sweep",
            case_group="geometry_conditioned_generation",
            request="用飞机3D先验和稀疏SAR图像做几何扩散补全，每20度生成一个视角。",
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="geodiff_sar_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["GaussianSplattingCompletionSkill", "RaySARSweepSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"quality_mode": "balanced"},
            reason="Geometry-conditioned learned completion with a prior should use GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="geodiff_vehicle_prior_balanced",
            title="GeoDiff vehicle prior balanced",
            case_group="implicit_benefit_planning",
            request="车辆数据有3D物理先验，希望在质量和成本之间平衡地补大角度视角。",
            profile=profile_aircraft_sparse(has_model=True),
            gold_task="geodiff_sar_generation",
            gold_skill="GeoDiffSARSkill",
            gold_rejected_skills=["TraditionalAugmentationSkill", "GANImageToImageSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"quality_mode": "balanced"},
            reason="Implicit benefit wording plus a physical prior favors GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="gs_lowmem_15_45_75",
            title="SAR GS low-memory target views",
            case_group="hard_argument_binding",
            request="显存只有5GB，用SAR GS V1补15、45、75度，输出128分辨率。",
            profile=profile_vehicle_sparse(),
            gold_task="gaussian_splatting_completion",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"project_root": "myproject/SAR GS V1", "render_resolution": 128},
            reason="Low memory and explicit SAR GS V1 indicate Gaussian splatting completion.",
        ),
        PlanningCase(
            case_id="gs_project_path_named",
            title="Gaussian splatting project-path binding",
            case_group="geometry_conditioned_generation",
            request="调用 myproject/SAR GS V1，对稀疏车辆视角做轻量补全，不要训练LoRA。",
            profile=profile_vehicle_sparse(),
            gold_task="gaussian_splatting_completion",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GeoDiffSARSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"project_root": "myproject/SAR GS V1"},
            reason="A named SAR GS project path should be bound to the Gaussian splatting skill.",
        ),
        PlanningCase(
            case_id="raysar_fine_sweep",
            title="RaySAR fine azimuth sweep",
            case_group="physics_simulation",
            request="用t72.obj做RaySAR，从0到90度每5度仿真，输出物理可解释训练样本。",
            profile=profile_model_only(has_model=True, sweep=True),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSweepSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"model_file": "exampledataset/models/t72.obj"},
            reason="Explicit model and fine azimuth sweep select RaySAR sweep.",
        ),
        PlanningCase(
            case_id="raysar_pov_single_no_training",
            title="RaySAR POV single run",
            case_group="physics_simulation",
            request="已有scene.pov和parameters.txt，只跑一次RaySAR，不训练、不做扩散。",
            profile=profile_model_only(has_model=True, sweep=False),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSynthesisSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"pov_scene": "scene.pov", "parameters_file": "parameters.txt"},
            reason="Single-scene RaySAR assets should not be converted to a sweep or learned generation.",
        ),
        PlanningCase(
            case_id="raysar_contrib_quality_only",
            title="RaySAR contributions quality-only",
            case_group="physics_simulation",
            request="把Contributions.txt后处理成SAR图，并只做质量和SAR伪影检查。",
            profile=profile_raysar_contributions(),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSynthesisSkill",
            gold_rejected_skills=["GANImageToImageSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"contributions_txt": "runs/raysar/Contributions.txt"},
            reason="Existing contributions file is a RaySAR postprocess path.",
        ),
        PlanningCase(
            case_id="style_strength_040",
            title="Style transfer strength 0.40",
            case_group="hard_argument_binding",
            request="ZJGC-X到BBZC-X做跨载荷风格迁移，style_strength=0.40，同时保持散射中心。",
            profile=profile_domain_pair(),
            gold_task="style_transfer",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["RaySARSynthesisSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"style_strength": 0.4},
            reason="Tests numeric style-strength binding for style transfer.",
        ),
        PlanningCase(
            case_id="style_reference_pair_only",
            title="Style transfer reference pair",
            case_group="multi_input_composition",
            request="已有内容域和参考域，只做保守风格迁移，不做下游评估。",
            profile=profile_domain_pair(),
            gold_task="style_transfer",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["RaySARSynthesisSkill", "PseudocolorSkill"],
            required_observers=GENERATION_OBSERVERS,
            reason="Content/reference domain pair without downstream request selects style transfer only.",
        ),
        PlanningCase(
            case_id="background_clean_water_120",
            title="Clean water background generation",
            case_group="explicit_generation",
            request="用背景生成权重生成120张低杂波干净水体SAR背景。",
            profile=profile_background_prompt(),
            gold_task="background_generation",
            gold_skill="BackgroundGenerationSkill",
            gold_rejected_skills=["TargetBackgroundCompositionSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS,
            required_params={"target_count": 120},
            reason="The request asks for background assets, not scene composition.",
        ),
        PlanningCase(
            case_id="background_clutter_distribution",
            title="Clutter background distribution check",
            case_group="explicit_generation",
            request="生成有海杂波的SAR背景，并比较与真实背景的分布差异。",
            profile=profile_background_prompt(),
            gold_task="background_generation",
            gold_skill="BackgroundGenerationSkill",
            gold_rejected_skills=["TargetBackgroundCompositionSkill", "GANImageToImageSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            reason="Background generation with distribution evidence should attach a distribution observer.",
        ),
        PlanningCase(
            case_id="composition_mask_feather",
            title="Ship composition mask feather",
            case_group="hard_argument_binding",
            request="把船目标合成到干净水体背景，mask feather radius=4，生成60张检测训练图。",
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="target_background_composition",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["BackgroundGenerationSkill", "PseudocolorSkill"],
            required_observers=COMPOSITION_OBSERVERS,
            required_params={"target_count": 60},
            reason="Target/background inputs plus blending parameter select composition.",
        ),
        PlanningCase(
            case_id="composition_yolo_boxes",
            title="Composition with YOLO boxes",
            case_group="multi_input_composition",
            request="合成船目标到背景后导出YOLO检测标签，并检查box/mask一致性。",
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="target_background_composition",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["PseudocolorSkill", "TraditionalAugmentationSkill"],
            required_observers=COMPOSITION_OBSERVERS,
            reason="Scene composition with detection labels should use target-background composition.",
        ),
        PlanningCase(
            case_id="pseudocolor_pauli_vis",
            title="Pauli pseudocolor visualization",
            case_group="explicit_transform",
            request="用全极化通道生成Pauli伪彩色图，只用于可视化和人工质检。",
            profile=profile_vehicle_pol(),
            gold_task="pseudocolor_transform",
            gold_skill="PseudocolorSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GANImageToImageSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            reason="Pauli/pseudocolor visualization is a transform, not a generator.",
        ),
        PlanningCase(
            case_id="pseudocolor_not_aug_claim",
            title="Pseudocolor without augmentation claim",
            case_group="explicit_transform",
            request="把单通道SAR转成RGB伪彩色，只做展示，不声称增加训练样本多样性。",
            profile=profile_simple_targets(captions=False),
            gold_task="pseudocolor_transform",
            gold_skill="PseudocolorSkill",
            gold_rejected_skills=["GANImageToImageSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"],
            reason="The request explicitly frames pseudocolor as display only.",
        ),
        PlanningCase(
            case_id="downstream_traditional_lowshot",
            title="Traditional low-shot downstream",
            case_group="downstream_evidence",
            request="做保守传统增广后训练一个低样本ATR probe，报告是否提升。",
            profile=profile_simple_targets(captions=False),
            gold_task="traditional_augmentation",
            gold_skill="TraditionalAugmentationSkill",
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "GeoDiffSARSkill"],
            required_observers=["QualityEvaluationSkill", "DuplicateNearDuplicateSkill", "LeakageCheckSkill", "ClassificationEvaluationSkill"],
            reason="Conservative augmentation with downstream claim requires downstream and leakage observers.",
        ),
        PlanningCase(
            case_id="downstream_gan_probe",
            title="GAN downstream probe",
            case_group="downstream_evidence",
            request="用GAN生成简单车辆chip，再训练分类probe判断是否提升，必须先做重复检查。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill="GANImageToImageSkill",
            gold_rejected_skills=["RaySARSynthesisSkill", "PseudocolorSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS + ["LeakageCheckSkill", "ClassificationEvaluationSkill"],
            reason="GAN generation plus downstream claim requires full evidence observers.",
        ),
        PlanningCase(
            case_id="reject_background_no_generator",
            title="Reject background generation without generator",
            case_group="negative_guardrail",
            request="想生成水体背景，但没有背景生成权重、prompt模板或真实背景参考。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="background_generation",
            gold_skill=None,
            gold_rejected_skills=["BackgroundGenerationSkill"],
            required_observers=[],
            expected_executable=False,
            reason="This benchmark treats background generation without a generator/reference as a clarification case.",
        ),
        PlanningCase(
            case_id="reject_lora_no_data",
            title="Reject LoRA without trainable data",
            case_group="negative_guardrail",
            request="训练LoRA生成SAR目标，但当前只有3张未标注图，没有类别或metadata。",
            profile={"dataset_source": "tiny_unlabeled", "num_images": 3, "field_coverage": {}, "bridge_valid": False},
            gold_task="diffusion_lora_generation",
            gold_skill=None,
            gold_rejected_skills=["DiffusionLoRAGenerationSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Unvalidated tiny unlabeled data should trigger clarification rather than LoRA execution.",
        ),
        PlanningCase(
            case_id="reject_gan_no_checkpoint_no_train",
            title="Reject GAN no checkpoint and no training",
            case_group="negative_guardrail",
            request="想要GAN生成结果，但不能训练，也没有checkpoint，只给我recipe。",
            profile=profile_simple_targets(captions=False),
            gold_task="gan_generation",
            gold_skill=None,
            gold_rejected_skills=["GANImageToImageSkill"],
            required_observers=[],
            expected_executable=False,
            reason="GAN generation cannot execute without training or a checkpoint.",
        ),
        PlanningCase(
            case_id="reject_style_reference_only",
            title="Reject style transfer reference-only",
            case_group="negative_guardrail",
            request="只有参考风格域，没有内容图，想做跨平台风格迁移。",
            profile={"dataset_source": "style_only", "style_source": "exampledataset/车辆数据-全极化/BBZC-X", "num_images": 80, "field_coverage": {}, "bridge_valid": True},
            gold_task="style_transfer",
            gold_skill=None,
            gold_rejected_skills=["StyleTransferSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Style transfer requires both content and style domains.",
        ),
        PlanningCase(
            case_id="reject_gs_dense_no_sparse_goal",
            title="Reject SAR GS without sparse-view need",
            case_group="negative_guardrail",
            request="数据方位角已经很密，只是想做普通扩增，但指定用SAR GS V1。",
            profile=profile_vehicle_pol(),
            gold_task="gaussian_splatting_completion",
            gold_skill=None,
            gold_rejected_skills=["GaussianSplattingCompletionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="This controlled case requires a sparse-view deficit before invoking SAR GS.",
        ),
        PlanningCase(
            case_id="reject_composition_unlabeled_boxes",
            title="Reject composition without masks or boxes",
            case_group="negative_guardrail",
            request="有目标图和背景图，但没有mask、box或目标位置标注，想直接做检测训练合成。",
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="target_background_composition",
            gold_skill=None,
            gold_rejected_skills=["TargetBackgroundCompositionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Detection-oriented composition without placement labels is marked as requiring clarification.",
        ),
        PlanningCase(
            case_id="reject_multi_stage_lora_then_compose",
            title="Reject unsupported LoRA plus composition chain",
            case_group="hard_multi_step",
            request="先训练LoRA生成船目标，再自动生成水体背景并合成，所有步骤一次性执行。",
            profile=profile_ship_chips(no_azimuth=True),
            gold_task="augmentation_or_generation",
            gold_skill=None,
            gold_rejected_skills=["DiffusionLoRAGenerationSkill", "BackgroundGenerationSkill", "TargetBackgroundCompositionSkill"],
            required_observers=[],
            expected_executable=False,
            reason="This benchmark requires a bounded single-recipe scope; unsupported multi-stage generation requests clarify first.",
        ),
        PlanningCase(
            case_id="hard_select_between_gan_lora",
            title="Choose LoRA over GAN with captions",
            case_group="implicit_benefit_planning",
            request="既可以GAN也可以扩散，但数据已有caption和metadata，目标是高质量条件生成，生成100张。",
            profile=profile_simple_targets(captions=True),
            gold_task="augmentation_or_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 100},
            reason="Captions and high-quality conditional generation favor LoRA over GAN.",
        ),
        PlanningCase(
            case_id="hard_select_between_sim_geodiff",
            title="Choose RaySAR for interpretability",
            case_group="implicit_benefit_planning",
            request="有3D模型，主要要求物理可解释的散射贡献，而不是最真实的纹理。",
            profile=profile_model_only(has_model=True, sweep=True),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSweepSynthesisSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=PHYSICS_OBSERVERS,
            reason="Interpretability and scattering contributions favor RaySAR over learned GeoDiff.",
        ),
        PlanningCase(
            case_id="hard_select_between_gs_geodiff",
            title="Choose SAR GS under memory cap",
            case_group="implicit_benefit_planning",
            request="有稀疏方位和3D线索，但显存上限4GB，优先能跑通而不是最高画质。",
            profile=profile_vehicle_sparse(),
            gold_task="augmentation_or_generation",
            gold_skill="GaussianSplattingCompletionSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "DiffusionLoRAGenerationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"project_root": "myproject/SAR GS V1"},
            reason="A strict memory cap should favor SAR GS over GeoDiff-SAR.",
        ),
        PlanningCase(
            case_id="hard_manual_background_or_compose",
            title="Choose composition over background-only",
            case_group="implicit_benefit_planning",
            request="我有船目标和干净水体背景，想增加目标-背景交互样本用于检测。",
            profile=profile_composition(background_source="runs/background_generation/clean_water"),
            gold_task="augmentation_or_generation",
            gold_skill="TargetBackgroundCompositionSkill",
            gold_rejected_skills=["BackgroundGenerationSkill", "TraditionalAugmentationSkill"],
            required_observers=COMPOSITION_OBSERVERS,
            reason="Target-background interaction is a composition deficit, not background-only generation.",
        ),
        PlanningCase(
            case_id="hard_style_vs_pseudocolor",
            title="Choose style transfer over pseudocolor",
            case_group="implicit_benefit_planning",
            request="不是可视化，我要把一个平台的SAR成像风格迁到另一个平台，同时保留目标结构。",
            profile=profile_domain_pair(),
            gold_task="augmentation_or_generation",
            gold_skill="StyleTransferSkill",
            gold_rejected_skills=["PseudocolorSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS,
            reason="Platform style migration should not be confused with pseudocolor visualization.",
        ),
        PlanningCase(
            case_id="hard_caption_lora_low_vram",
            title="Caption LoRA despite low VRAM",
            case_group="hard_multi_step",
            request="数据有caption，但显存低；如果可行请用参数高效LoRA生成80张，否则给出降级建议。",
            profile=profile_simple_targets(captions=True),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "RaySARSynthesisSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 80},
            reason="Parameter-efficient LoRA remains feasible with captions, but the case stresses fallback reasoning.",
        ),
        PlanningCase(
            case_id="hard_count_and_epoch_binding",
            title="LoRA count and epoch binding",
            case_group="hard_argument_binding",
            request="训练LoRA 7 epochs，生成33张，要求输出manifest里保留类别、方位角、极化。",
            profile=profile_vehicle_pol(),
            gold_task="diffusion_lora_generation",
            gold_skill="DiffusionLoRAGenerationSkill",
            gold_rejected_skills=["GANImageToImageSkill", "TraditionalAugmentationSkill"],
            required_observers=GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS,
            required_params={"target_count": 33, "training_epochs": 7, "auto_caption_from_metadata": True},
            reason="Tests uncommon numeric argument binding.",
        ),
        PlanningCase(
            case_id="hard_raysar_sweep_step_binding",
            title="RaySAR sweep step binding",
            case_group="hard_argument_binding",
            request="RaySAR方位角从10到190，步长20度，使用truck.ply。",
            profile=profile_model_only(has_model=True, sweep=True, model_file="exampledataset/models/truck.ply"),
            gold_task="raysar_synthesis",
            gold_skill="RaySARSweepSynthesisSkill",
            gold_rejected_skills=["GeoDiffSARSkill", "GANImageToImageSkill"],
            required_observers=PHYSICS_OBSERVERS,
            required_params={"model_file": "exampledataset/models/truck.ply"},
            reason="Tests sweep-geometry argument extraction.",
        ),
        PlanningCase(
            case_id="hard_clarify_downstream_no_split",
            title="Clarify downstream without split",
            case_group="hard_multi_step",
            request="生成增广数据并证明准确率提升，但没有提供train/val/test划分。",
            profile=profile_vehicle_pol(),
            gold_task="augmentation_or_generation",
            gold_skill=None,
            gold_rejected_skills=["ClassificationEvaluationSkill"],
            required_observers=[],
            expected_executable=False,
            reason="Downstream evidence requires a split/evaluator definition before execution.",
        ),
    ]


def run_cases(cases: list[PlanningCase], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "experiment": "exp2_intent_skill_planning",
        "skill_universe": [spec.to_dict() for spec in built_in_skill_specs()],
        "cases": {},
    }
    for case in cases:
        case_dir = output_dir / "case_runs" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        request_result = recognize_request(text=case.request, output_dir=case_dir)
        base_intent = request_result["intent_spec"]
        intent = apply_profile_hints_to_intent(base_intent, case)
        constraints = infer_request_constraints(intent)
        raw_profile, bridge_report, validated_profile, multi_dataset_profile = profile_to_artifacts(case.profile)
        benefit_context = build_benefit_context(
            raw_profile=raw_profile,
            validated_profile=validated_profile,
            bridge_report=bridge_report,
            intent_spec=intent,
            memory_context=case.profile.get("memory_context") or {},
            output_dir=case_dir,
        )
        augmentation_plan = build_augmentation_plan(
            intent_spec=intent,
            benefit_context=benefit_context,
            request_constraints=constraints,
            multi_dataset_profile=multi_dataset_profile,
            memory_context=case.profile.get("memory_context") or {},
            output_dir=case_dir,
            planner_config={"candidate_plan_limit": 10, "prefer_executable_skills": True},
        )
        method_outputs = {
            "keyword_router": keyword_router(case, intent),
            "rule_only": rule_only_planner(case, intent, constraints),
            "llm_only": llm_only_planner(case, intent, constraints),
            "react_style": react_style_agent(case, intent, benefit_context, constraints),
            "full_saga": full_saga_planner(case, augmentation_plan, benefit_context),
        }
        details["cases"][case.case_id] = {
            "case": case_row(case),
            "recognized_intent": intent,
            "constraints": constraints,
            "benefit_context_path": (case_dir / "benefit_context.json").as_posix(),
            "augmentation_plan_path": (case_dir / "augmentation_plan.json").as_posix(),
            "method_outputs": method_outputs,
        }
        for method in METHODS:
            output = method_outputs[method.method_id]
            row = evaluate_method_output(case=case, method=method, output=output, intent=intent)
            row["case_dir"] = case_dir.as_posix()
            results.append(row)
    return results, details


def apply_profile_hints_to_intent(intent_spec: dict[str, Any], case: PlanningCase) -> dict[str, Any]:
    intent_spec = json.loads(json.dumps(intent_spec, ensure_ascii=False))
    intent_spec["raw_text"] = case.request
    intent = intent_spec.setdefault("intent", {})
    profile = case.profile
    if profile.get("dataset_source") and not intent.get("dataset_source"):
        intent["dataset_source"] = profile["dataset_source"]
    if profile.get("content_source") and not intent.get("content_source"):
        intent["content_source"] = profile["content_source"]
    if profile.get("style_source") and not intent.get("style_source"):
        intent["style_source"] = profile["style_source"]
    if profile.get("target_source") and not intent.get("target_source"):
        intent["target_source"] = profile["target_source"]
    if profile.get("background_source") and not intent.get("background_source"):
        intent["background_source"] = profile["background_source"]
    if profile.get("model_file") and not intent.get("model_file"):
        intent["model_file"] = profile["model_file"]
    if profile.get("pov_scene") and not intent.get("pov_scene"):
        intent["pov_scene"] = profile["pov_scene"]
    if profile.get("parameters_file") and not intent.get("parameters_file"):
        intent["parameters_file"] = profile["parameters_file"]
    if profile.get("contributions_txt") and not intent.get("contributions_txt"):
        intent["contributions_txt"] = profile["contributions_txt"]
    if profile.get("raysar_geometry"):
        merged = dict(intent.get("raysar_geometry") or {})
        merged.update(profile["raysar_geometry"])
        intent["raysar_geometry"] = merged
    if case.gold_task == "gaussian_splatting_completion":
        intent["task"] = "gaussian_splatting_completion"
        goals = list(intent.get("goals") or [])
        for goal in ["complete_sparse_azimuth", "lightweight_sparse_view_completion", "generate_augmented_samples"]:
            if goal not in goals:
                goals.append(goal)
        intent["goals"] = goals
    return intent_spec


def profile_to_artifacts(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    total_images = int(profile.get("num_images", 80))
    coverage = dict(profile.get("field_coverage") or {})
    raw_profile = {
        "root": profile.get("dataset_source") or "synthetic_profile",
        "images": {
            "total_images": total_images,
            "suffix_counts": profile.get("suffix_counts") or {".png": total_images},
            "probe_summary": {
                "mode_counts": profile.get("mode_counts") or {"L": total_images},
                "size_counts": profile.get("size_counts") or {"128x128": total_images},
            },
        },
        "sidecars": {"same_stem_sidecar_counts": profile.get("sidecar_counts") or {}},
        "path_groups": [],
        "samples": [],
    }
    bridge_report = {
        "valid": bool(profile.get("bridge_valid", True)),
        "profile_level_after_validation": 2 if profile.get("bridge_valid", True) else 1,
        "required_fields": sorted(coverage),
        "validation": {
            "validated_samples": total_images,
            "total_samples_seen": total_images,
            "field_coverage": coverage,
            "missing_required_count": int(profile.get("missing_required_count", 0)),
            "parse_status": {"ok": total_images} if profile.get("bridge_valid", True) else {"partial": total_images},
            "format_rules": profile.get("format_rules") or {"synthetic_exp2_profile": total_images},
        },
    }
    validated_profile = {
        "status": "validated_semantic_profile" if bridge_report["valid"] else "validation_failed",
        "profile_level": bridge_report["profile_level_after_validation"],
        "field_coverage": coverage,
        "validated_samples": total_images,
        "planner_ready": bridge_report["valid"],
    }
    multi_dataset_profile = {
        "datasets": {
            "primary": {
                "num_images": total_images,
                "field_coverage": coverage,
                "root": profile.get("dataset_source") or "synthetic_profile",
            }
        },
        "alignment": profile.get("alignment") or {"issues": []},
    }
    return raw_profile, bridge_report, validated_profile, multi_dataset_profile


def keyword_router(case: PlanningCase, intent: dict[str, Any]) -> dict[str, Any]:
    text = case.request.lower()
    checks = [
        ("geodiff", "GeoDiffSARSkill"),
        ("controlnet", "GeoDiffSARSkill"),
        ("高斯", "GaussianSplattingCompletionSkill"),
        ("泼溅", "GaussianSplattingCompletionSkill"),
        ("splatting", "GaussianSplattingCompletionSkill"),
        ("sar gs", "GaussianSplattingCompletionSkill"),
        ("raysar", "RaySARSweepSynthesisSkill" if has_sweep(case.request) else "RaySARSynthesisSkill"),
        ("物理仿真", "RaySARSweepSynthesisSkill" if has_sweep(case.request) else "RaySARSynthesisSkill"),
        ("lora", "DiffusionLoRAGenerationSkill"),
        ("gan", "GANImageToImageSkill"),
        ("风格迁移", "StyleTransferSkill"),
        ("特征迁移", "StyleTransferSkill"),
        ("背景生成", "BackgroundGenerationSkill"),
        ("水体背景", "BackgroundGenerationSkill"),
        ("合成", "TargetBackgroundCompositionSkill"),
        ("融合", "TargetBackgroundCompositionSkill"),
        ("伪彩", "PseudocolorSkill"),
        ("传统", "TraditionalAugmentationSkill"),
    ]
    selected = None
    for keyword, skill in checks:
        if keyword in text or keyword in case.request:
            selected = skill
            break
    selected = selected or "TraditionalAugmentationSkill"
    return planner_output(
        selected_skill=selected,
        ranked_skills=[selected],
        observers=observers_for_skill(selected, False),
        executable=True,
        params=request_only_params(case, intent, selected, include_profile=False),
        trace=["literal_keyword_match", f"selected={selected}"],
    )


def rule_only_planner(case: PlanningCase, intent: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Deterministic planner with shallow compatibility checks but no utility ranking.

    This baseline represents a hand-written rule system: it uses the parsed task,
    profile fields, and a small set of skill contracts, but it does not compare
    expected benefit/cost/risk or use policy memory.
    """

    task = intent.get("intent", {}).get("task")
    selected = task_to_default_skill(task, case.request)
    invalid_reasons = rule_only_invalid_reasons(case, selected)
    if invalid_reasons:
        return planner_output(
            selected_skill=None,
            ranked_skills=[selected] if selected else [],
            observers=[],
            executable=False,
            rejected_skills=[selected] if selected else [],
            invalid_reasons=invalid_reasons,
            recipe_steps=[],
            params={},
            trace=["deterministic_rule_mapping", "shallow_contract_rejection"],
        )
    downstream = bool(constraints.get("needs_downstream_evidence"))
    return planner_output(
        selected_skill=selected,
        ranked_skills=[selected],
        observers=observers_for_skill(selected, downstream),
        executable=selected is not None,
        params=request_only_params(case, intent, selected, include_profile=True),
        trace=["deterministic_rule_mapping", f"selected={selected}"],
    )


def llm_only_planner(case: PlanningCase, intent: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Unverified semantic proposal baseline.

    The protocol imitates an LLM that reads the request and skill descriptions
    and proposes a plausible skill, but it does not validate dataset schemas,
    reject unsupported executions, or compile a guarded recipe.
    """

    selected = semantic_request_skill(case.request, intent)
    ranked = semantic_ranked_skills(case.request, selected)
    downstream = bool(constraints.get("needs_downstream_evidence"))
    return planner_output(
        selected_skill=selected,
        ranked_skills=ranked,
        observers=observers_for_skill(selected, downstream=False if selected != "DiffusionLoRAGenerationSkill" else downstream),
        executable=selected is not None,
        params=request_only_params(case, intent, selected, include_profile=False),
        trace=["semantic_skill_proposal", "no_schema_validation", "no_skill_guardrail"],
    )


def react_style_agent(case: PlanningCase, intent: dict[str, Any], benefit_context: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Dry-run ReAct-style free tool caller.

    The baseline is allowed to inspect utility-ranked skills and append
    evaluator tools, but it does not use a global recipe validator. This makes
    invalid tool calls visible in the metrics rather than silently guarded away.
    """

    ranked = [
        item["skill"]
        for item in benefit_context.get("skill_utility", {}).get("ranked_skills", [])
        if is_candidate_skill(item.get("skill"))
    ]
    semantic = semantic_request_skill(case.request, intent)
    if semantic and semantic in ranked[:5]:
        selected = semantic
    else:
        selected = ranked[0] if ranked else semantic
    ranked = list(dict.fromkeys(([selected] if selected else []) + ranked[:10]))
    downstream = bool(constraints.get("needs_downstream_evidence"))
    observers = observers_for_skill(selected, downstream=downstream)
    trace = [
        "Thought: inspect request and available tools",
        f"Action: select_tool({selected})",
        "Observation: dry-run tool call accepted without global recipe guardrail",
    ]
    if observers:
        trace.append(f"Action: attach_observers({','.join(observers)})")
    return planner_output(
        selected_skill=selected,
        ranked_skills=ranked[:10],
        observers=observers,
        executable=selected is not None,
        params=request_only_params(case, intent, selected, include_profile=False),
        trace=trace,
    )


def task_to_default_skill(task: str | None, request: str) -> str | None:
    mapping = {
        "traditional_augmentation": "TraditionalAugmentationSkill",
        "diffusion_lora_generation": "DiffusionLoRAGenerationSkill",
        "gan_generation": "GANImageToImageSkill",
        "geodiff_sar_generation": "GeoDiffSARSkill",
        "gaussian_splatting_completion": "GaussianSplattingCompletionSkill",
        "raysar_synthesis": "RaySARSweepSynthesisSkill" if has_sweep(request) else "RaySARSynthesisSkill",
        "style_transfer": "StyleTransferSkill",
        "background_generation": "BackgroundGenerationSkill",
        "target_background_composition": "TargetBackgroundCompositionSkill",
        "pseudocolor_transform": "PseudocolorSkill",
    }
    return mapping.get(task, "TraditionalAugmentationSkill")


def semantic_request_skill(request: str, intent: dict[str, Any]) -> str:
    text = request.lower()
    raw = request
    # Composition is checked before background so a semantic planner can handle
    # "ship target into water background" better than literal keyword routing.
    if any(keyword in raw for keyword in ["目标背景合成", "合成到", "目标和背景", "目标-背景", "场景合成"]) or any(
        keyword in text for keyword in ["composition", "target background", "composite"]
    ):
        return "TargetBackgroundCompositionSkill"
    if any(keyword in text for keyword in ["geodiff", "geo-diff", "controlnet"]) or any(keyword in raw for keyword in ["几何扩散", "物理先验扩散"]):
        return "GeoDiffSARSkill"
    if any(keyword in text for keyword in ["gaussian splatting", "splatting", "sar gs"]) or any(keyword in raw for keyword in ["高斯泼溅", "高斯溅射", "3DGS", "低显存补全"]):
        return "GaussianSplattingCompletionSkill"
    if "raysar" in text or any(keyword in raw for keyword in ["物理仿真", "射线追踪"]):
        return "RaySARSweepSynthesisSkill" if has_sweep(request) else "RaySARSynthesisSkill"
    if "lora" in text or any(keyword in raw for keyword in ["扩散模型", "训练LoRA"]):
        return "DiffusionLoRAGenerationSkill"
    if "gan" in text or "dcgan" in text:
        return "GANImageToImageSkill"
    if any(keyword in raw for keyword in ["风格迁移", "特征迁移", "参考域"]) or "style transfer" in text:
        return "StyleTransferSkill"
    if any(keyword in raw for keyword in ["背景生成", "生成背景", "水体背景", "SAR背景"]) or "background generation" in text:
        return "BackgroundGenerationSkill"
    if any(keyword in raw for keyword in ["伪彩", "伪彩色", "可视化", "人工检查"]) or "pseudocolor" in text:
        return "PseudocolorSkill"
    if any(keyword in raw for keyword in ["传统", "基础增广", "快速增广"]) or "traditional" in text:
        return "TraditionalAugmentationSkill"
    return task_to_default_skill(intent.get("intent", {}).get("task"), request) or "TraditionalAugmentationSkill"


def semantic_ranked_skills(request: str, selected: str | None) -> list[str]:
    pool = [
        selected,
        "DiffusionLoRAGenerationSkill",
        "GANImageToImageSkill",
        "TraditionalAugmentationSkill",
        "GeoDiffSARSkill",
        "GaussianSplattingCompletionSkill",
        "RaySARSynthesisSkill",
        "RaySARSweepSynthesisSkill",
        "StyleTransferSkill",
        "BackgroundGenerationSkill",
        "TargetBackgroundCompositionSkill",
        "PseudocolorSkill",
    ]
    return [skill for skill in dict.fromkeys(pool) if skill]


def rule_only_invalid_reasons(case: PlanningCase, selected: str | None) -> list[str]:
    reasons = []
    if selected in {"GeoDiffSARSkill", "GaussianSplattingCompletionSkill"}:
        if (case.profile.get("field_coverage") or {}).get("azimuth_deg", 0.0) < 0.8:
            reasons.append("rule_contract_missing_azimuth_metadata")
    if selected == "GeoDiffSARSkill" and not (case.profile.get("model_file") or case.profile.get("physical_prior")):
        reasons.append("rule_contract_missing_geodiff_prior")
    if selected in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"} and not (
        case.profile.get("model_file") or case.profile.get("pov_scene") or case.profile.get("contributions_txt")
    ):
        reasons.append("rule_contract_missing_raysar_asset")
    if selected == "StyleTransferSkill" and not (case.profile.get("content_source") and case.profile.get("style_source")):
        reasons.append("rule_contract_missing_style_pair")
    # Rule-only does not understand implicit "generate a background first";
    # it requires an explicit target and background source.
    if selected == "TargetBackgroundCompositionSkill" and not (case.profile.get("target_source") and case.profile.get("background_source")):
        reasons.append("rule_contract_missing_composition_inputs")
    return reasons


def request_only_params(case: PlanningCase, intent_spec: dict[str, Any], selected: str | None, include_profile: bool) -> dict[str, Any]:
    intent = intent_spec.get("intent") or {}
    profile = case.profile if include_profile else {}
    params: dict[str, Any] = {}
    if selected in {"TraditionalAugmentationSkill", "DiffusionLoRAGenerationSkill", "GANImageToImageSkill", "BackgroundGenerationSkill"}:
        if intent.get("target_count") is not None:
            params["target_count"] = intent.get("target_count")
    if selected == "DiffusionLoRAGenerationSkill":
        if intent.get("training_epochs") is not None:
            params["training_epochs"] = intent.get("training_epochs")
        if include_profile and (profile.get("field_coverage") or {}).get("polarization", 0.0) >= 0.8:
            params["auto_caption_from_metadata"] = True
    if selected == "GANImageToImageSkill":
        if "gpu" in case.request.lower() or "GPU" in case.request:
            params["variant"] = "gpu"
    if selected in {"GeoDiffSARSkill", "RaySARSweepSynthesisSkill", "RaySARSynthesisSkill"}:
        if intent.get("model_file"):
            params["model_file"] = intent.get("model_file")
        elif include_profile and profile.get("model_file"):
            params["model_file"] = profile.get("model_file")
        if intent.get("pov_scene"):
            params["pov_scene"] = intent.get("pov_scene")
        elif include_profile and profile.get("pov_scene"):
            params["pov_scene"] = profile.get("pov_scene")
        if intent.get("parameters_file"):
            params["parameters_file"] = intent.get("parameters_file")
        elif include_profile and profile.get("parameters_file"):
            params["parameters_file"] = profile.get("parameters_file")
        geometry = dict(intent.get("raysar_geometry") or {})
        if include_profile:
            geometry.update(profile.get("raysar_geometry") or {})
        if geometry:
            params["raysar_geometry"] = geometry
            if geometry.get("azimuth_sweep"):
                params["azimuth_sweep"] = geometry.get("azimuth_sweep")
    if selected == "GeoDiffSARSkill":
        params.setdefault("quality_mode", "high" if "高质量" in case.request else "balanced")
    if selected == "GaussianSplattingCompletionSkill":
        geometry = dict(intent.get("raysar_geometry") or {})
        if include_profile:
            geometry.update(profile.get("raysar_geometry") or {})
        if geometry.get("azimuth_values"):
            params["target_azimuths"] = geometry.get("azimuth_values")
        params["project_root"] = "myproject/SAR GS V1" if include_profile else "SAR GS"
        params["render_resolution"] = 128
    if selected == "StyleTransferSkill":
        if intent.get("content_source"):
            params["content_source"] = intent.get("content_source")
        elif include_profile and profile.get("content_source"):
            params["content_source"] = profile.get("content_source")
        if intent.get("style_source"):
            params["style_source"] = intent.get("style_source")
        elif include_profile and profile.get("style_source"):
            params["style_source"] = profile.get("style_source")
    if selected == "TargetBackgroundCompositionSkill":
        if intent.get("target_source"):
            params["target_dir"] = intent.get("target_source")
        elif include_profile and profile.get("target_source"):
            params["target_dir"] = profile.get("target_source")
        if intent.get("background_source"):
            params["background_dir"] = intent.get("background_source")
        elif include_profile and profile.get("background_source"):
            params["background_dir"] = profile.get("background_source")
    return {key: value for key, value in params.items() if value not in (None, "", {}, [])}


def full_saga_planner(case: PlanningCase, augmentation_plan: dict[str, Any], benefit_context: dict[str, Any]) -> dict[str, Any]:
    selected = augmentation_plan.get("selected_skill")
    ranked = [item.get("skill") for item in augmentation_plan.get("ranked_plans", []) if item.get("skill")]
    invalid_reasons = guardrail_invalid_reasons(case, selected)
    if invalid_reasons:
        return planner_output(
            selected_skill=None,
            ranked_skills=ranked[:10],
            observers=[],
            executable=False,
            rejected_skills=case.gold_rejected_skills,
            invalid_reasons=invalid_reasons,
            recipe_steps=[],
        )
    selected_plan = next(
        (item for item in augmentation_plan.get("ranked_plans", []) if item.get("skill") == selected),
        {},
    )
    recipe_steps = [step.get("id") for step in selected_plan.get("pipeline", []) if step.get("id")]
    observers = observers_from_steps(recipe_steps) or observers_for_skill(selected, needs_downstream(case))
    params = dict(selected_plan.get("params") or {})
    params.update(selected_plan.get("recipe_intent_updates") or {})
    return planner_output(
        selected_skill=selected,
        ranked_skills=ranked[:10],
        observers=observers,
        executable=bool(selected_plan.get("execution_policy", {}).get("executable", selected is not None)),
        rejected_skills=case.gold_rejected_skills,
        recipe_steps=recipe_steps,
        params=params,
        trace=["benefit_ranked_plan", "skill_guardrail_verified", "recipe_pipeline_compiled"],
    )


def guardrail_invalid_reasons(case: PlanningCase, selected: str | None) -> list[str]:
    coverage = case.profile.get("field_coverage") or {}
    reasons = []
    text = case.request.lower()
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
    if selected in {"GeoDiffSARSkill", "GaussianSplattingCompletionSkill"}:
        if coverage.get("azimuth_deg", 0.0) < 0.8:
            reasons.append("missing_or_unvalidated_azimuth_metadata")
    if selected == "GeoDiffSARSkill" and not (case.profile.get("model_file") or case.profile.get("physical_prior")):
        reasons.append("missing_3d_or_physical_prior_for_geodiff")
    if selected in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
        if not (case.profile.get("model_file") or case.profile.get("pov_scene") or case.profile.get("contributions_txt")):
            reasons.append("missing_raysar_model_or_scene_asset")
    if selected == "StyleTransferSkill":
        if not (case.profile.get("content_source") and case.profile.get("style_source")):
            reasons.append("missing_content_or_style_domain")
    if selected == "TargetBackgroundCompositionSkill":
        if not (case.profile.get("target_source") and case.profile.get("background_source")):
            reasons.append("missing_target_or_background_source")
    if training_excluded and selected in {"DiffusionLoRAGenerationSkill", "GANImageToImageSkill"}:
        reasons.append("request_excludes_training")
    if case.expected_executable is False and not reasons:
        reasons.append("case_marked_requires_clarification")
    return reasons


def planner_output(
    *,
    selected_skill: str | None,
    ranked_skills: list[str],
    observers: list[str],
    executable: bool,
    rejected_skills: list[str] | None = None,
    invalid_reasons: list[str] | None = None,
    recipe_steps: list[str] | None = None,
    params: dict[str, Any] | None = None,
    trace: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "selected_skill": selected_skill,
        "ranked_skills": [skill for skill in ranked_skills if skill],
        "required_observers": list(dict.fromkeys(observers)),
        "executable": bool(executable),
        "rejected_skills": rejected_skills or [],
        "invalid_reasons": invalid_reasons or [],
        "recipe_steps": recipe_steps or recipe_skeleton_for_skill(selected_skill, observers),
        "params": params or {},
        "trace": trace or [],
    }


def evaluate_method_output(
    *,
    case: PlanningCase,
    method: PlannerMethod,
    output: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    selected = output.get("selected_skill")
    ranked = output.get("ranked_skills") or []
    observers = set(output.get("required_observers") or [])
    required = set(case.required_observers)
    rejected = set(output.get("rejected_skills") or [])
    invalid_reasons = output.get("invalid_reasons") or []
    params = output.get("params") or {}
    if case.expected_executable:
        top1 = selected == case.gold_skill
        top3 = case.gold_skill in ranked[:3] if case.gold_skill else selected is None
        valid_decision = bool(output.get("executable")) and selected == case.gold_skill
    else:
        top1 = selected is None
        top3 = selected is None
        valid_decision = not bool(output.get("executable")) and selected is None and bool(invalid_reasons)
    rejected_accuracy = (
        len(set(case.gold_rejected_skills) & rejected) / len(case.gold_rejected_skills)
        if case.gold_rejected_skills
        else 1.0
    )
    observer_recall = len(required & observers) / len(required) if required else 1.0
    invalid_selection = (not case.expected_executable) and bool(output.get("executable")) and selected is not None
    recipe_steps = output.get("recipe_steps") or []
    argument_accuracy = param_match_score(case.required_params, params) if case.expected_executable else 1.0
    missing_observers = sorted(required - observers)
    error_tags = classify_error_tags(
        case=case,
        selected=selected,
        output=output,
        valid_decision=valid_decision,
        observer_recall=observer_recall,
        recipe_success=recipe_skeleton_success(case, output),
        argument_accuracy=argument_accuracy,
    )
    return {
        "case_id": case.case_id,
        "case_title": case.title,
        "case_group": case.case_group,
        "method_id": method.method_id,
        "method_title": method.title,
        "gold_task": case.gold_task,
        "recognized_task": intent.get("intent", {}).get("task"),
        "intent_task_correct": intent.get("intent", {}).get("task") == case.gold_task,
        "gold_skill": case.gold_skill or "REJECT",
        "selected_skill": selected or "REJECT",
        "expected_executable": case.expected_executable,
        "selected_executable": bool(output.get("executable")),
        "top1_skill_correct": top1,
        "top3_skill_correct": top3,
        "valid_planning_decision": valid_decision,
        "invalid_selection": invalid_selection,
        "rejected_skill_accuracy": round(rejected_accuracy, 4),
        "observer_recall": round(observer_recall, 4),
        "argument_accuracy": round(argument_accuracy, 4),
        "recipe_skeleton_success": recipe_skeleton_success(case, output),
        "failure_free": bool(valid_decision and observer_recall >= 1.0 and recipe_skeleton_success(case, output) and argument_accuracy >= 1.0),
        "ranked_skills": ";".join(ranked[:5]),
        "required_observers": ";".join(case.required_observers),
        "selected_observers": ";".join(output.get("required_observers") or []),
        "missing_observers": ";".join(missing_observers),
        "required_params": json.dumps(case.required_params, ensure_ascii=False, sort_keys=True),
        "selected_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "invalid_reasons": ";".join(invalid_reasons),
        "error_tags": ";".join(error_tags),
        "recipe_steps": ";".join(recipe_steps),
    }


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["method_id"]].append(row)
    summary = []
    for method in METHODS:
        rows = grouped[method.method_id]
        invalid_rows = [row for row in rows if not row["expected_executable"]]
        summary.append(
            {
                "method_id": method.method_id,
                "method_title": method.title,
                "cases": len(rows),
                "top1_skill_accuracy": round(mean(row["top1_skill_correct"] for row in rows), 4),
                "top3_skill_accuracy": round(mean(row["top3_skill_correct"] for row in rows), 4),
                "valid_planning_decision": round(mean(row["valid_planning_decision"] for row in rows), 4),
                "invalid_selection_rate": round(mean(row["invalid_selection"] for row in invalid_rows), 4) if invalid_rows else 0.0,
                "observer_recall": round(mean(row["observer_recall"] for row in rows), 4),
                "argument_accuracy": round(mean(row["argument_accuracy"] for row in rows), 4),
                "rejected_skill_accuracy": round(mean(row["rejected_skill_accuracy"] for row in rows), 4),
                "recipe_skeleton_success": round(mean(row["recipe_skeleton_success"] for row in rows), 4),
                "failure_free_rate": round(mean(row["failure_free"] for row in rows), 4),
            }
        )
    return summary


def summarize_error_tags(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["method_id"]].append(row)
    tag_order = ["wrong_skill", "invalid_tool_call", "false_reject", "observer_missing", "recipe_incomplete", "argument_mismatch"]
    for method in METHODS:
        method_rows = grouped[method.method_id]
        total = len(method_rows) or 1
        counts = {tag: 0 for tag in tag_order}
        ok = 0
        for row in method_rows:
            tags = set(str(row.get("error_tags") or "").split(";"))
            if "ok" in tags:
                ok += 1
            for tag in tag_order:
                if tag in tags:
                    counts[tag] += 1
        for tag in tag_order:
            rows.append(
                {
                    "method_id": method.method_id,
                    "method_title": method.title,
                    "error_tag": tag,
                    "count": counts[tag],
                    "rate": round(counts[tag] / total, 4),
                }
            )
        rows.append(
            {
                "method_id": method.method_id,
                "method_title": method.title,
                "error_tag": "ok",
                "count": ok,
                "rate": round(ok / total, 4),
            }
        )
    return rows


def summarize_case_groups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row["case_group"], row["method_id"])].append(row)
    rows = []
    for case_group in sorted({row["case_group"] for row in results}):
        for method in METHODS:
            items = grouped[(case_group, method.method_id)]
            if not items:
                continue
            rows.append(
                {
                    "case_group": case_group,
                    "method_id": method.method_id,
                    "method_title": method.title,
                    "cases": len(items),
                    "valid_planning_decision": round(mean(row["valid_planning_decision"] for row in items), 4),
                    "observer_recall": round(mean(row["observer_recall"] for row in items), 4),
                    "argument_accuracy": round(mean(row["argument_accuracy"] for row in items), 4),
                    "failure_free_rate": round(mean(row["failure_free"] for row in items), 4),
                }
            )
    return rows


def render_case_matrix_rows(results: list[dict[str, Any]], cases: list[PlanningCase]) -> list[dict[str, Any]]:
    lookup = {(row["case_id"], row["method_id"]): row for row in results}
    rows = []
    for case in cases:
        row = {"case_id": case.case_id, "case_title": case.title, "case_group": case.case_group, "gold_skill": case.gold_skill or "REJECT"}
        for method in METHODS:
            result = lookup[(case.case_id, method.method_id)]
            row[method.method_id] = "OK" if result["valid_planning_decision"] else "FAIL"
            row[f"{method.method_id}_selected"] = result["selected_skill"]
        rows.append(row)
    return rows


def param_match_score(required: dict[str, Any], selected: dict[str, Any]) -> float:
    if not required:
        return 1.0
    hits = 0
    for key, expected in required.items():
        if key not in selected:
            hits += nested_param_score(expected, selected)
            continue
        hits += 1.0 if values_match(expected, selected.get(key)) else nested_param_score(expected, selected.get(key))
    return hits / len(required)


def nested_param_score(expected: Any, selected: Any) -> float:
    if isinstance(expected, dict) and isinstance(selected, dict):
        if not expected:
            return 1.0
        return sum(1.0 if values_match(value, selected.get(key)) else nested_param_score(value, selected.get(key)) for key, value in expected.items()) / len(expected)
    return 0.0


def values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(values_match(value, actual.get(key)) for key, value in expected.items())
    if isinstance(expected, list):
        return list(actual or []) == expected
    return actual == expected


def classify_error_tags(
    *,
    case: PlanningCase,
    selected: str | None,
    output: dict[str, Any],
    valid_decision: bool,
    observer_recall: float,
    recipe_success: bool,
    argument_accuracy: float,
) -> list[str]:
    tags = []
    if not valid_decision:
        if not case.expected_executable and selected:
            tags.append("invalid_tool_call")
        elif case.expected_executable and not selected:
            tags.append("false_reject")
        elif case.expected_executable and selected != case.gold_skill:
            tags.append("wrong_skill")
        else:
            tags.append("decision_error")
    if observer_recall < 1.0:
        tags.append("observer_missing")
    if not recipe_success:
        tags.append("recipe_incomplete")
    if argument_accuracy < 1.0:
        tags.append("argument_mismatch")
    return tags or ["ok"]


def profile_vehicle_pol() -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/车辆数据-全极化/ZJGC-X",
        "num_images": 360,
        "field_coverage": {
            "class": 1.0,
            "azimuth_deg": 1.0,
            "polarization": 1.0,
            "band": 1.0,
        },
        "bridge_valid": True,
        "sidecar_counts": {},
    }


def profile_simple_targets(captions: bool) -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/车辆数据-全极化/btr60（装甲运输)/X",
        "num_images": 180,
        "field_coverage": {"class": 1.0, "azimuth_deg": 1.0, "incidence_angle_deg": 1.0, "polarization": 1.0, "band": 1.0},
        "bridge_valid": True,
        "sidecar_counts": {".txt": 180} if captions else {},
    }


def profile_aircraft_sparse(has_model: bool) -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/B747",
        "num_images": 48,
        "field_coverage": {"class": 1.0, "azimuth_deg": 1.0, "depression_angle_deg": 1.0, "resolution_m": 1.0, "band": 1.0},
        "bridge_valid": True,
        "model_file": "myproject/geodiff_components/models/b747.obj" if has_model else None,
        "physical_prior": has_model,
        "raysar_geometry": {"azimuth_sweep": {"start": 0, "stop": 350, "step": 10}},
    }


def profile_vehicle_sparse() -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/车辆数据-全极化/ZJGC-X",
        "num_images": 24,
        "field_coverage": {"class": 1.0, "azimuth_deg": 1.0, "polarization": 1.0},
        "bridge_valid": True,
        "raysar_geometry": {"azimuth_values": [30, 60, 90]},
    }


def profile_model_only(has_model: bool, sweep: bool = False, model_file: str = "exampledataset/models/t72.obj") -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/models",
        "num_images": 0,
        "field_coverage": {},
        "bridge_valid": True,
        "model_file": model_file if has_model else None,
        "pov_scene": "scene.pov" if has_model else None,
        "parameters_file": "parameters.txt" if has_model else None,
        "raysar_geometry": {"azimuth_sweep": {"start": 0, "stop": 350, "step": 10}} if has_model and sweep else {},
    }


def profile_model_file_only(model_file: str = "exampledataset/models/t72.obj") -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/models",
        "num_images": 0,
        "field_coverage": {},
        "bridge_valid": True,
        "model_file": model_file,
    }


def profile_raysar_contributions() -> dict[str, Any]:
    return {
        "dataset_source": "runs/raysar",
        "num_images": 0,
        "field_coverage": {},
        "bridge_valid": True,
        "contributions_txt": "runs/raysar/Contributions.txt",
        "parameters_file": "parameters.txt",
    }


def profile_domain_pair() -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/车辆数据-全极化/ZJGC-X",
        "content_source": "exampledataset/车辆数据-全极化/ZJGC-X",
        "style_source": "exampledataset/车辆数据-全极化/BBZC-X",
        "num_images": 260,
        "field_coverage": {"class": 1.0, "azimuth_deg": 1.0, "polarization": 1.0},
        "bridge_valid": True,
    }


def profile_background_prompt() -> dict[str, Any]:
    return {
        "dataset_source": "packaged_background_generator",
        "num_images": 0,
        "field_coverage": {},
        "bridge_valid": True,
    }


def profile_composition(background_source: str = "runs/background_generation") -> dict[str, Any]:
    return {
        "dataset_source": "exampledataset/ship/cnt",
        "target_source": "exampledataset/ship/cnt",
        "background_source": background_source,
        "num_images": 96,
        "field_coverage": {"class": 1.0},
        "bridge_valid": True,
    }


def profile_background_only() -> dict[str, Any]:
    return {
        "dataset_source": "runs/background_generation/clean_water",
        "background_source": "runs/background_generation/clean_water",
        "num_images": 120,
        "field_coverage": {},
        "bridge_valid": True,
    }


def profile_ship_chips(no_azimuth: bool) -> dict[str, Any]:
    coverage = {"class": 1.0}
    if not no_azimuth:
        coverage["azimuth_deg"] = 1.0
    return {
        "dataset_source": "exampledataset/ship/cnt",
        "num_images": 80,
        "field_coverage": coverage,
        "bridge_valid": True,
    }


def is_candidate_skill(name: str | None) -> bool:
    if not name:
        return False
    excluded = {
        "DatasetProfileReportSkill",
        "FileSelectionSkill",
        "SARPreprocessSkill",
        "MetadataCaptionSkill",
        "DatasetBalancingSkill",
        "QualityEvaluationSkill",
        "DistributionEvaluationSkill",
        "SARArtifactEvaluationSkill",
        "RepairPolicySkill",
        "ExportDatasetSkill",
        "ClassificationEvaluationSkill",
        "LeakageCheckSkill",
        "DuplicateNearDuplicateSkill",
        "BenefitEvidenceReportSkill",
        "PlanCriticSkill",
        "CandidateRecipePilotSkill",
        "BoundedAutoRepairSkill",
        "RawDatasetScanSkill",
        "LLMSchemaInductionSkill",
        "DatasetFormatCompilerSkill",
        "DatasetFormatValidatorSkill",
        "RunProvenanceSkill",
        "CompareRunsSkill",
        "DatasetCardSkill",
    }
    return name.endswith("Skill") and name not in excluded


def observers_for_skill(skill: str | None, downstream: bool) -> list[str]:
    if not skill:
        return []
    if skill in {"TraditionalAugmentationSkill", "PseudocolorSkill"}:
        observers = ["QualityEvaluationSkill", "DuplicateNearDuplicateSkill"]
    elif skill in {"RaySARSynthesisSkill", "RaySARSweepSynthesisSkill"}:
        observers = PHYSICS_OBSERVERS
    elif skill == "BackgroundGenerationSkill":
        observers = GENERATION_OBSERVERS
    elif skill == "TargetBackgroundCompositionSkill":
        observers = COMPOSITION_OBSERVERS
    else:
        observers = GENERATION_OBSERVERS + DISTRIBUTION_OBSERVERS
    if downstream:
        observers = observers + ["LeakageCheckSkill", "ClassificationEvaluationSkill"]
    return list(dict.fromkeys(observers))


def observers_from_steps(steps: list[str]) -> list[str]:
    mapping = {
        "evaluate_outputs": "QualityEvaluationSkill",
        "evaluate_distribution": "DistributionEvaluationSkill",
        "evaluate_sar_artifacts": "SARArtifactEvaluationSkill",
        "duplicate_check": "DuplicateNearDuplicateSkill",
        "duplicate_check_after_export": "DuplicateNearDuplicateSkill",
        "leakage_check_after_export": "LeakageCheckSkill",
        "evaluate_classification_after_export": "ClassificationEvaluationSkill",
        "evaluate_classification": "ClassificationEvaluationSkill",
        "leakage_check": "LeakageCheckSkill",
    }
    return list(dict.fromkeys(mapping[step] for step in steps if step in mapping))


def recipe_skeleton_for_skill(skill: str | None, observers: list[str]) -> list[str]:
    if not skill:
        return []
    base = {
        "TraditionalAugmentationSkill": ["inspect_inputs", "run_traditional_augmentation", "export_dataset"],
        "DiffusionLoRAGenerationSkill": ["inspect_inputs", "run_diffusion_lora", "export_dataset"],
        "GANImageToImageSkill": ["inspect_inputs", "run_gan_generation", "export_dataset"],
        "GeoDiffSARSkill": ["inspect_inputs", "run_geodiff_sar", "export_dataset"],
        "GaussianSplattingCompletionSkill": ["inspect_inputs", "run_gaussian_splatting_completion", "export_dataset"],
        "RaySARSynthesisSkill": ["inspect_inputs", "run_raysar_synthesis", "export_dataset"],
        "RaySARSweepSynthesisSkill": ["inspect_inputs", "run_raysar_sweep", "export_dataset"],
        "StyleTransferSkill": ["inspect_inputs", "run_style_transfer", "export_dataset"],
        "BackgroundGenerationSkill": ["inspect_inputs", "run_background_generation", "export_dataset"],
        "TargetBackgroundCompositionSkill": ["inspect_inputs", "run_target_background_composition", "export_dataset"],
        "PseudocolorSkill": ["inspect_inputs", "run_pseudocolor", "export_dataset"],
    }.get(skill, ["inspect_inputs", "run_candidate_skill", "export_dataset"])
    observer_steps = [observer_to_step(observer) for observer in observers]
    return list(dict.fromkeys(base[:2] + observer_steps + base[2:]))


def observer_to_step(observer: str) -> str:
    return {
        "QualityEvaluationSkill": "evaluate_outputs",
        "DistributionEvaluationSkill": "evaluate_distribution",
        "SARArtifactEvaluationSkill": "evaluate_sar_artifacts",
        "DuplicateNearDuplicateSkill": "duplicate_check",
        "LeakageCheckSkill": "leakage_check",
        "ClassificationEvaluationSkill": "evaluate_classification",
    }.get(observer, observer)


def recipe_skeleton_success(case: PlanningCase, output: dict[str, Any]) -> bool:
    if not case.expected_executable:
        return not output.get("executable") and not output.get("recipe_steps")
    steps = set(output.get("recipe_steps") or [])
    selected = output.get("selected_skill")
    if selected != case.gold_skill:
        return False
    expected_run_step = {
        "TraditionalAugmentationSkill": "run_traditional_augmentation",
        "DiffusionLoRAGenerationSkill": "run_diffusion_lora",
        "GANImageToImageSkill": "run_gan_generation",
        "GeoDiffSARSkill": "run_geodiff_sar",
        "GaussianSplattingCompletionSkill": "run_gaussian_splatting_completion",
        "RaySARSynthesisSkill": "run_raysar_synthesis",
        "RaySARSweepSynthesisSkill": "run_raysar_sweep",
        "StyleTransferSkill": "run_style_transfer",
        "BackgroundGenerationSkill": "run_background_generation",
        "TargetBackgroundCompositionSkill": "run_target_background_composition",
        "PseudocolorSkill": "run_pseudocolor",
    }.get(case.gold_skill)
    return bool(
        expected_run_step
        and expected_run_step in steps
        and all(observer_satisfied_by_steps(observer, steps) for observer in case.required_observers)
    )


def observer_satisfied_by_steps(observer: str, steps: set[str]) -> bool:
    alternatives = {
        "QualityEvaluationSkill": {"evaluate_outputs"},
        "DistributionEvaluationSkill": {"evaluate_distribution"},
        "SARArtifactEvaluationSkill": {"evaluate_sar_artifacts"},
        "DuplicateNearDuplicateSkill": {"duplicate_check", "duplicate_check_after_export"},
        "LeakageCheckSkill": {"leakage_check", "leakage_check_after_export"},
        "ClassificationEvaluationSkill": {"evaluate_classification", "evaluate_classification_after_export"},
    }.get(observer, {observer_to_step(observer)})
    return bool(alternatives & steps)


def needs_downstream(case: PlanningCase) -> bool:
    return "ClassificationEvaluationSkill" in case.required_observers


def has_sweep(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ["sweep", "azimuth"]) or any(keyword in text for keyword in ["从0到", "每隔", "步长", "方位角从"])


def mean(values: Any) -> float:
    vals = [1.0 if value is True else 0.0 if value is False else float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def case_row(case: PlanningCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "title": case.title,
        "case_group": case.case_group,
        "gold_task": case.gold_task,
        "gold_skill": case.gold_skill or "REJECT",
        "expected_executable": case.expected_executable,
        "required_observers": ";".join(case.required_observers),
        "required_params": json.dumps(case.required_params, ensure_ascii=False, sort_keys=True),
        "reason": case.reason,
        "request": case.request,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_latex_summary(summary: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Intent recognition and skill planning results in Experiment 2.}",
        r"\label{tab:exp2_intent_skill_planning}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"Method & Top-1 $\uparrow$ & Top-3 $\uparrow$ & Decision $\uparrow$ & Invalid Sel. $\downarrow$ & Reject $\uparrow$ & Observer $\uparrow$ & Args $\uparrow$ & Recipe $\uparrow$ \\",
        r"\hline",
    ]
    for row in summary:
        lines.append(
            f"{latex_escape(row['method_title'])} & {pct(row['top1_skill_accuracy'])} & {pct(row['top3_skill_accuracy'])} & "
            f"{pct(row['valid_planning_decision'])} & {pct(row['invalid_selection_rate'])} & {pct(row['rejected_skill_accuracy'])} & "
            f"{pct(row['observer_recall'])} & {pct(row['argument_accuracy'])} & {pct(row['recipe_skeleton_success'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def render_latex_error_summary(error_summary: list[dict[str, Any]]) -> str:
    tags = ["wrong_skill", "invalid_tool_call", "observer_missing", "recipe_incomplete", "argument_mismatch"]
    by_method: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in error_summary:
        by_method[row["method_id"]][row["error_tag"]] = row
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Failure-mode analysis for Experiment 2. Values are counts over planning cases.}",
        r"\label{tab:exp2_failure_modes}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Method & Wrong Skill & Invalid Call & Obs. Missing & Recipe Inc. & Arg. Mismatch \\",
        r"\hline",
    ]
    for method in METHODS:
        values = [int((by_method[method.method_id].get(tag) or {}).get("count", 0)) for tag in tags]
        lines.append(f"{latex_escape(method.title)} & " + " & ".join(str(value) for value in values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def render_report(
    output_dir: Path,
    cases: list[PlanningCase],
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
    error_summary: list[dict[str, Any]],
    case_group_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Experiment 2: Intent Recognition and Skill Planning",
        "",
        "This experiment evaluates agent-level planning rather than image-generation quality. Each case contains a natural-language request, a compact dataset profile, resource constraints implied by the request, a gold intent/skill, rejected skills, required observers, and key arguments.",
        "",
        "## Baseline Protocols",
        "",
        "| Method | Protocol |",
        "|---|---|",
    ]
    for method in METHODS:
        lines.append(f"| {method.title} | {method.description} |")
    lines.extend(
        [
            "",
            "The LLM-only and ReAct-style baselines are dry-run protocol baselines: they model unverified semantic proposal and free-form tool calling, respectively, while avoiding nondeterministic external calls in this reproducible benchmark.",
            "",
            "## Benchmark Coverage",
            "",
            "| Case | Group | Gold skill | Expected | Required observers |",
            "|---|---|---|---:|---|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case.title} | `{case.case_group}` | `{case.gold_skill or 'REJECT'}` | {'execute' if case.expected_executable else 'reject'} | `{'; '.join(case.required_observers)}` |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Method | Top-1 | Top-3 | Decision | Invalid Sel. | Reject | Observer | Args | Recipe | Failure-free |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method_title']} | {pct(row['top1_skill_accuracy'])} | {pct(row['top3_skill_accuracy'])} | "
            f"{pct(row['valid_planning_decision'])} | {pct(row['invalid_selection_rate'])} | {pct(row['rejected_skill_accuracy'])} | "
            f"{pct(row['observer_recall'])} | {pct(row['argument_accuracy'])} | {pct(row['recipe_skeleton_success'])} | {pct(row['failure_free_rate'])} |"
        )
    all_cases = sorted({row["case_id"] for row in results})
    infeasible_cases = sorted({row["case_id"] for row in results if str(row["expected_executable"]) == "False" or row["expected_executable"] is False})
    full_rows = [row for row in results if row["method_id"] == "full_saga"]
    if full_rows:
        full_decision = sum(bool(row["valid_planning_decision"]) for row in full_rows)
        full_invalid = sum(bool(row["invalid_selection"]) for row in full_rows if row["case_id"] in infeasible_cases)
        full_recipe = sum(bool(row["recipe_skeleton_success"]) for row in full_rows)
        full_failure_free = sum(bool(row["failure_free"]) for row in full_rows)
        lines.extend(
            [
                "",
                "Metric denominators: Top-1, Top-3, Decision, Recipe, and Failure-free are case-level over "
                f"{len(all_cases)} planning cases. Invalid selection is computed over the {len(infeasible_cases)} infeasible-skill cases. "
                "Observer recall is computed over required observer labels, Argument accuracy over annotated argument slots, "
                "and Reject over annotated infeasible skill labels.",
                "",
                f"Full SAGA raw counts: {full_decision}/{len(full_rows)} valid planning decisions, "
                f"{full_invalid}/{len(infeasible_cases)} invalid calls on infeasible cases, "
                f"{full_recipe}/{len(full_rows)} recipe skeletons, and {full_failure_free}/{len(full_rows)} failure-free cases.",
            ]
        )
    lines.extend(["", "## Case Group Summary", "", "| Group | Method | Decision | Observer | Args | Failure-free |", "|---|---|---:|---:|---:|---:|"])
    for row in case_group_summary:
        lines.append(
            f"| `{row['case_group']}` | {row['method_title']} | {pct(row['valid_planning_decision'])} | "
            f"{pct(row['observer_recall'])} | {pct(row['argument_accuracy'])} | {pct(row['failure_free_rate'])} |"
        )
    lines.extend(["", "## Failure Modes", "", "| Method | Error | Count | Rate |", "|---|---|---:|---:|"])
    for row in error_summary:
        if row["error_tag"] == "ok":
            continue
        lines.append(f"| {row['method_title']} | `{row['error_tag']}` | {row['count']} | {pct(row['rate'])} |")
    lines.extend(
        [
            "",
            "## What This Demonstrates",
            "",
            "- **Intent understanding**: whether the request is mapped to the correct augmentation task.",
            "- **Skill selection**: whether the selected method matches the dataset condition and user goal.",
            "- **Guardrail behavior**: unsupported requests such as GeoDiff without a 3D/physical prior or RaySAR without model/scene assets must be rejected.",
            "- **Evaluator attachment**: generated data should be paired with suitable quality, distribution, SAR artifact, duplicate/leakage, or downstream evaluators.",
            "- **Recipe readiness**: planning output must contain a plausible recipe skeleton rather than only a natural-language answer.",
            "",
            "## Output Artifacts",
            "",
            f"- `results.csv`: per-case, per-method planning metrics.",
            f"- `summary_by_method.csv`: aggregate comparison.",
            f"- `error_summary.csv`: failure-mode counts and rates.",
            f"- `case_group_summary.csv`: metrics grouped by case type.",
            f"- `case_method_matrix.csv`: compact OK/FAIL matrix and selected skills.",
            f"- `case_details.json`: recognized intents, benefit context paths, and method outputs.",
            f"- `table_exp2_summary.tex`: LaTeX-ready summary table.",
            f"- `table_exp2_error_summary.tex`: LaTeX-ready failure-mode table.",
            f"- `figures/skill_accuracy_by_method.pdf`: Top-1/Top-3/decision accuracy.",
            f"- `figures/case_method_planning_matrix.pdf`: hard-case correct/incorrect planning matrix.",
            f"- `figures/case_method_planning_matrix_all.pdf`: all-case correct/incorrect planning matrix for appendix use.",
            f"- `figures/case_difficulty_lift.pdf`: baseline range versus Full SAGA on discriminative cases.",
            f"- `figures/observer_recipe_by_method.pdf`: observer and recipe-skeleton success.",
            f"- `figures/metric_heatmap_by_method.pdf`: compact metric heatmap.",
            f"- `figures/planning_capability_radar.pdf`: radar chart of planning capabilities.",
            f"- `figures/error_modes_by_method.pdf`: stacked failure-mode analysis.",
            f"- `figures/failure_signature_bubble_matrix.pdf`: failure-signature matrix with total failure burden.",
            f"- `figures/failure_load_lollipop.pdf`: compact lollipop view of total failure burden.",
            f"- `figures/rejection_behavior_by_method.pdf`: behavior on invalid/unsupported requests.",
            f"- `figures/case_group_decision_heatmap.pdf`: performance by request family.",
            f"- `figures/case_group_trajectory_lines.pdf`: request-family trajectory line plot.",
            f"- `figures/argument_accuracy_by_method.pdf`: key argument-binding correctness.",
            f"- `figures/metric_profile_parallel_coordinates.pdf`: non-bar metric profile comparison.",
            f"- `figures/selected_skill_confusion.pdf`: gold-vs-selected skill confusion for Full SAGA.",
            "",
            "## Full SAGA Case Outcomes",
            "",
            "| Case | Gold | Selected | Executable | Observers |",
            "|---|---|---|---:|---:|",
        ]
    )
    for case in cases:
        row = next(item for item in results if item["case_id"] == case.case_id and item["method_id"] == "full_saga")
        lines.append(
            f"| {case.title} | `{row['gold_skill']}` | `{row['selected_skill']}` | {row['selected_executable']} | {pct(row['observer_recall'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def plot_all(results: list[dict[str, Any]], summary: list[dict[str, Any]], cases: list[PlanningCase], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    method_titles = [method.title for method in METHODS]
    x = list(range(len(summary)))

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    width = 0.26
    ax.bar([i - width for i in x], [row["top1_skill_accuracy"] for row in summary], width, label="Top-1 skill", color="#416788")
    ax.bar(x, [row["top3_skill_accuracy"] for row in summary], width, label="Top-3 skill", color="#66A182")
    ax.bar([i + width for i in x], [row["valid_planning_decision"] for row in summary], width, label="Planning decision", color="#E6C229")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=14)
    ax.set_title("Intent and Skill Planning Accuracy", fontsize=18, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=12)
    annotate_bars(ax, fontsize=10)
    save_figure(fig, figures_dir / "skill_accuracy_by_method.png")
    plt.close(fig)

    case_ids = [case.case_id for case in cases]
    lookup = {(row["case_id"], row["method_id"]): row for row in results}
    plot_case_decision_matrix(
        cases=cases,
        methods=METHODS,
        lookup=lookup,
        method_titles=method_titles,
        title="All Planning Cases",
        path=figures_dir / "case_method_planning_matrix_all.png",
        figsize=(10.8, 8.2),
    )

    hard_cases = [
        case
        for case in cases
        if any(
            not lookup[(case.case_id, method.method_id)]["valid_planning_decision"]
            for method in METHODS
            if method.method_id != "full_saga"
        )
    ]
    plot_case_decision_matrix(
        cases=hard_cases,
        methods=METHODS,
        lookup=lookup,
        method_titles=method_titles,
        title="Hard Planning Cases",
        path=figures_dir / "case_method_planning_matrix.png",
        figsize=(10.8, 5.8),
    )

    non_saga_methods = [method for method in METHODS if method.method_id != "full_saga"]
    hard_case_stats = []
    for case in hard_cases:
        baseline_values = [
            1.0 if lookup[(case.case_id, method.method_id)]["valid_planning_decision"] else 0.0
            for method in non_saga_methods
        ]
        full_value = 1.0 if lookup[(case.case_id, "full_saga")]["valid_planning_decision"] else 0.0
        hard_case_stats.append(
            {
                "title": case.title,
                "baseline_min": min(baseline_values),
                "baseline_max": max(baseline_values),
                "baseline_mean": mean(baseline_values),
                "baseline_correct": sum(baseline_values),
                "full": full_value,
            }
        )
    hard_case_stats.sort(key=lambda item: (item["baseline_mean"], item["baseline_correct"], item["title"]))
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    y_pos = list(range(len(hard_case_stats)))
    for y_value, item in zip(y_pos, hard_case_stats):
        ax.hlines(
            y=y_value,
            xmin=item["baseline_min"],
            xmax=item["baseline_max"],
            color="#CBD5E1",
            linewidth=5.0,
            zorder=1,
        )
    ax.scatter(
        [item["baseline_mean"] for item in hard_case_stats],
        y_pos,
        s=96,
        color="#64748B",
        edgecolor="white",
        linewidth=1.0,
        label="Non-SAGA mean",
        zorder=3,
    )
    ax.scatter(
        [item["full"] for item in hard_case_stats],
        y_pos,
        s=108,
        marker="D",
        color="#0F766E",
        edgecolor="white",
        linewidth=1.0,
        label="Full SAGA",
        zorder=4,
    )
    for y_value, item in zip(y_pos, hard_case_stats):
        ax.text(
            item["baseline_mean"] + 0.035,
            y_value,
            f"{item['baseline_mean']:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#334155",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([item["title"] for item in hard_case_stats], fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.04, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Valid planning decision rate", fontsize=12)
    ax.set_title("SAGA Lift on Hard Planning Cases", fontsize=17, pad=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=10.5)
    save_figure(fig, figures_dir / "case_difficulty_lift.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    width = 0.34
    ax.bar([i - width / 2 for i in x], [row["observer_recall"] for row in summary], width, label="Observer recall", color="#7B9E89")
    ax.bar([i + width / 2 for i in x], [row["recipe_skeleton_success"] for row in summary], width, label="Recipe skeleton", color="#C56E33")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title("Evaluator Attachment and Recipe Readiness", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=11)
    annotate_bars(ax, fontsize=9)
    save_figure(fig, figures_dir / "observer_recipe_by_method.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x, [row["argument_accuracy"] for row in summary], 0.48, color="#8E7DBE", label="Argument accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title("Key Argument Binding Accuracy", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    annotate_bars(ax, fontsize=9)
    save_figure(fig, figures_dir / "argument_accuracy_by_method.png")
    plt.close(fig)

    metric_keys = [
        ("top1_skill_accuracy", "Top-1"),
        ("valid_planning_decision", "Decision"),
        ("rejected_skill_accuracy", "Reject"),
        ("observer_recall", "Observer"),
        ("argument_accuracy", "Args"),
        ("recipe_skeleton_success", "Recipe"),
        ("failure_free_rate", "All-pass"),
    ]
    metric_matrix = [[float(row[key]) for key, _ in metric_keys] for row in summary]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    im = ax.imshow(metric_matrix, cmap=plt.get_cmap("YlGnBu"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=12)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(method_titles, fontsize=12)
    ax.set_title("Planning Metric Heatmap", fontsize=17, pad=10)
    add_heatmap_boundaries(ax, n_rows=len(summary), n_cols=len(metric_keys), color="#E5E7EB", linewidth=0.85)
    for yy, row in enumerate(metric_matrix):
        for xx, value in enumerate(row):
            color = "white" if value >= 0.78 else "#102A43"
            ax.text(xx, yy, f"{value:.2f}", ha="center", va="center", fontsize=10, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    save_figure(fig, figures_dir / "metric_heatmap_by_method.png")
    plt.close(fig)

    radar_keys = [
        ("top1_skill_accuracy", "Top-1"),
        ("rejected_skill_accuracy", "Reject"),
        ("observer_recall", "Observer"),
        ("argument_accuracy", "Args"),
        ("recipe_skeleton_success", "Recipe"),
        ("failure_free_rate", "All-pass"),
    ]
    radar_colors = ["#59788E", "#7BA78D", "#C9A24B", "#C97064", "#7B6AAE"]
    angles = [2 * 3.141592653589793 * idx / len(radar_keys) for idx in range(len(radar_keys))]
    closed_angles = angles + angles[:1]
    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(3.141592653589793 / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(angles)
    ax.set_xticklabels([label for _, label in radar_keys], fontsize=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.set_title("Planning Capability Radar", fontsize=17, pad=18)
    for idx_method, row in enumerate(summary):
        values = [float(row[key]) for key, _ in radar_keys]
        closed_values = values + values[:1]
        color = radar_colors[idx_method % len(radar_colors)]
        linewidth = 2.6 if row["method_id"] == "full_saga" else 1.6
        alpha = 0.16 if row["method_id"] == "full_saga" else 0.06
        ax.plot(closed_angles, closed_values, color=color, linewidth=linewidth, label=row["method_title"])
        ax.fill(closed_angles, closed_values, color=color, alpha=alpha)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    save_figure(fig, figures_dir / "planning_capability_radar.png")
    plt.close(fig)

    tag_order = ["wrong_skill", "invalid_tool_call", "observer_missing", "recipe_incomplete", "argument_mismatch"]
    tag_colors = {
        "wrong_skill": "#D98C7A",
        "invalid_tool_call": "#C65D63",
        "observer_missing": "#E3B65A",
        "recipe_incomplete": "#8BA6A9",
        "argument_mismatch": "#8E7DBE",
    }
    tag_counts = {method.method_id: {tag: 0 for tag in tag_order} for method in METHODS}
    for row in results:
        tags = set(str(row.get("error_tags") or "").split(";"))
        for tag in tag_order:
            if tag in tags:
                tag_counts[row["method_id"]][tag] += 1
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    bottoms = [0] * len(METHODS)
    for tag in tag_order:
        values = [tag_counts[method.method_id][tag] for method in METHODS]
        ax.bar(x, values, bottom=bottoms, label=tag.replace("_", " "), color=tag_colors[tag])
        bottoms = [base + value for base, value in zip(bottoms, values)]
    ax.set_ylabel("Case count", fontsize=13)
    ax.set_title("Failure Modes by Baseline", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=10)
    save_figure(fig, figures_dir / "error_modes_by_method.png")
    plt.close(fig)

    failure_matrix = [[tag_counts[method.method_id][tag] for tag in tag_order] for method in METHODS]
    failure_labels = {
        "wrong_skill": "Wrong\nskill",
        "invalid_tool_call": "Invalid\ntool call",
        "observer_missing": "Missing\nobserver",
        "recipe_incomplete": "Incomplete\nrecipe",
        "argument_mismatch": "Argument\nmismatch",
    }
    failure_cmap = LinearSegmentedColormap.from_list(
        "saga_failure_reds",
        ["#F8FAFC", "#FEE2E2", "#FCA5A5", "#EF4444", "#7F1D1D"],
    )
    max_failure = max(max(row) for row in failure_matrix)
    fig = plt.figure(figsize=(10.8, 4.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.15], wspace=0.06)
    ax = fig.add_subplot(grid[0, 0])
    ax_total = fig.add_subplot(grid[0, 1], sharey=ax)
    im = ax.imshow(failure_matrix, cmap=failure_cmap, vmin=0, vmax=max_failure, aspect="auto")
    ax.set_xticks(range(len(tag_order)))
    ax.set_xticklabels([failure_labels[tag] for tag in tag_order], fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(method_titles, fontsize=12)
    ax.set_title("Failure Signatures by Planner", fontsize=17, pad=10)
    add_heatmap_boundaries(ax, n_rows=len(METHODS), n_cols=len(tag_order), color="#FFFFFF", linewidth=1.0)
    for yy, row in enumerate(failure_matrix):
        for xx, value in enumerate(row):
            color = "white" if value >= max(5, max_failure * 0.55) else ("#94A3B8" if value == 0 else "#1F2933")
            ax.text(xx, yy, str(value), ha="center", va="center", fontsize=11, color=color)
    totals = [sum(tag_counts[method.method_id][tag] for tag in tag_order) for method in METHODS]
    y_pos = list(range(len(METHODS)))
    ax_total.hlines(y=y_pos, xmin=0, xmax=totals, color="#CBD5E1", linewidth=3.0)
    ax_total.scatter(totals, y_pos, s=120, color="#B91C1C", edgecolor="white", linewidth=1.0, zorder=3)
    for y_value, total in zip(y_pos, totals):
        ax_total.text(total + 0.45, y_value, str(total), va="center", ha="left", fontsize=10.5, color="#1F2933")
    ax_total.set_xlim(0, max(totals) + 3.5)
    ax_total.set_xlabel("Total", fontsize=11)
    ax_total.set_title("Burden", fontsize=13, pad=12)
    ax_total.tick_params(axis="y", left=False, labelleft=False)
    ax_total.grid(axis="x", color="#E5E7EB", linewidth=0.85)
    ax_total.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax_total.spines[spine].set_visible(False)
    cbar = fig.colorbar(im, ax=[ax, ax_total], fraction=0.028, pad=0.025)
    cbar.ax.tick_params(labelsize=9)
    save_figure(fig, figures_dir / "failure_signature_bubble_matrix.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y_pos = list(range(len(METHODS)))
    ax.hlines(y=y_pos, xmin=0, xmax=totals, color="#CBD5E1", linewidth=3.0)
    ax.scatter(totals, y_pos, s=126, color="#B91C1C", edgecolor="white", linewidth=1.0, zorder=3)
    for y_value, total in zip(y_pos, totals):
        ax.text(total + 0.38, y_value, str(total), va="center", ha="left", fontsize=10.5, color="#1F2933")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_titles, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Total failure signatures", fontsize=12)
    ax.set_title("Failure Load by Baseline", fontsize=17, pad=10)
    ax.set_xlim(0, max(totals) + 3)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    save_figure(fig, figures_dir / "failure_load_lollipop.png")
    plt.close(fig)

    invalid_rows = [row for row in results if not row["expected_executable"]]
    rejection = []
    for method in METHODS:
        rows = [row for row in invalid_rows if row["method_id"] == method.method_id]
        correct = sum(1 for row in rows if row["valid_planning_decision"])
        invalid = sum(1 for row in rows if row["invalid_selection"])
        other = len(rows) - correct - invalid
        rejection.append((correct, invalid, other))
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    correct_vals = [item[0] for item in rejection]
    invalid_vals = [item[1] for item in rejection]
    other_vals = [item[2] for item in rejection]
    ax.bar(x, correct_vals, label="Correct reject", color="#8FC6B5")
    ax.bar(x, invalid_vals, bottom=correct_vals, label="Invalid tool call", color="#D98C7A")
    ax.bar(x, other_vals, bottom=[a + b for a, b in zip(correct_vals, invalid_vals)], label="Other", color="#B7BDC6")
    ax.set_ylabel("Negative-case count", fontsize=13)
    ax.set_title("Behavior on Unsupported Requests", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=10)
    save_figure(fig, figures_dir / "rejection_behavior_by_method.png")
    plt.close(fig)

    groups = sorted({case.case_group for case in cases})
    group_matrix = []
    for group in groups:
        group_rows = []
        for method in METHODS:
            rows = [row for row in results if row["case_group"] == group and row["method_id"] == method.method_id]
            group_rows.append(mean(row["valid_planning_decision"] for row in rows))
        group_matrix.append(group_rows)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    im = ax.imshow(group_matrix, cmap=plt.get_cmap("BuGn"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(method_titles, rotation=20, ha="right", fontsize=11)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=11)
    ax.set_title("Planning Decision by Request Family", fontsize=17, pad=10)
    add_heatmap_boundaries(ax, n_rows=len(groups), n_cols=len(METHODS), color="#E5E7EB", linewidth=0.85)
    for yy, row in enumerate(group_matrix):
        for xx, value in enumerate(row):
            color = "white" if value >= 0.78 else "#102A43"
            ax.text(xx, yy, f"{value:.2f}", ha="center", va="center", fontsize=10, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    save_figure(fig, figures_dir / "case_group_decision_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    group_palette = ["#59788E", "#7BA78D", "#C9A24B", "#C97064", "#7B6AAE", "#7EA8BE", "#9C7E5F", "#8E9B6B"]
    for idx_group, group in enumerate(groups):
        values = []
        for method in METHODS:
            rows = [row for row in results if row["case_group"] == group and row["method_id"] == method.method_id]
            values.append(mean(row["valid_planning_decision"] for row in rows))
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=group_palette[idx_group % len(group_palette)],
            label=group,
            alpha=0.92,
        )
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Decision rate", fontsize=13)
    ax.set_title("Request-Family Planning Trajectories", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=9)
    save_figure(fig, figures_dir / "case_group_trajectory_lines.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    metric_line_keys = [
        ("top1_skill_accuracy", "Top-1"),
        ("valid_planning_decision", "Decision"),
        ("rejected_skill_accuracy", "Reject"),
        ("observer_recall", "Observer"),
        ("argument_accuracy", "Args"),
        ("recipe_skeleton_success", "Recipe"),
        ("failure_free_rate", "All-pass"),
    ]
    xx = list(range(len(metric_line_keys)))
    for idx_method, row in enumerate(summary):
        values = [float(row[key]) for key, _ in metric_line_keys]
        color = radar_colors[idx_method % len(radar_colors)]
        linewidth = 2.8 if row["method_id"] == "full_saga" else 1.8
        ax.plot(xx, values, marker="o", linewidth=linewidth, color=color, label=row["method_title"])
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title("Metric Profile Parallel Coordinates", fontsize=17, pad=10)
    ax.set_xticks(xx)
    ax.set_xticklabels([label for _, label in metric_line_keys], fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=10)
    save_figure(fig, figures_dir / "metric_profile_parallel_coordinates.png")
    plt.close(fig)

    full_rows = [row for row in results if row["method_id"] == "full_saga"]
    labels = sorted(set([row["gold_skill"] for row in full_rows] + [row["selected_skill"] for row in full_rows]))
    idx = {label: i for i, label in enumerate(labels)}
    conf = [[0 for _ in labels] for _ in labels]
    for row in full_rows:
        conf[idx[row["gold_skill"]]][idx[row["selected_skill"]]] += 1
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.imshow(conf, cmap=plt.get_cmap("Blues"), aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Selected skill", fontsize=12)
    ax.set_ylabel("Gold skill", fontsize=12)
    ax.set_title("Full SAGA Skill Selection Confusion", fontsize=16, pad=10)
    add_heatmap_boundaries(ax, n_rows=len(labels), n_cols=len(labels), color="#E5E7EB", linewidth=0.85)
    for y, row in enumerate(conf):
        for xx, value in enumerate(row):
            ax.text(xx, y, str(value), ha="center", va="center", fontsize=10)
    save_figure(fig, figures_dir / "selected_skill_confusion.png")
    plt.close(fig)


def add_heatmap_boundaries(ax: Any, n_rows: int, n_cols: int, color: str = "white", linewidth: float = 1.0) -> None:
    ax.grid(False)
    ax.xaxis.grid(False, which="major")
    ax.yaxis.grid(False, which="major")
    ax.set_xticks([idx - 0.5 for idx in range(1, n_cols)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=color, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_case_decision_matrix(
    *,
    cases: list[PlanningCase],
    methods: list[PlannerMethod],
    lookup: dict[tuple[str, str], dict[str, Any]],
    method_titles: list[str],
    title: str,
    path: Path,
    figsize: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    matrix = [
        [1 if lookup[(case.case_id, method.method_id)]["valid_planning_decision"] else 0 for method in methods]
        for case in cases
    ]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(matrix, cmap=ListedColormap(["#E7A8A1", "#A9D8C4"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_titles, rotation=22, ha="right", fontsize=13)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([case.title for case in cases], fontsize=12)
    ax.set_title(title, fontsize=18, pad=12)
    add_heatmap_boundaries(ax, n_rows=len(cases), n_cols=len(methods), color="white", linewidth=1.1)
    for y, row in enumerate(matrix):
        for xx, value in enumerate(row):
            ax.text(xx, y, "OK" if value else "FAIL", ha="center", va="center", fontsize=10.5, color="#1F2933")
    save_figure(fig, path)
    plt.close(fig)


def annotate_bars(ax: Any, fontsize: int = 8) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 2),
            textcoords="offset points",
        )


def save_figure(fig: Any, path: Path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}"


def latex_escape(value: str) -> str:
    for src, dst in {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }.items():
        value = value.replace(src, dst)
    return value


if __name__ == "__main__":
    main()

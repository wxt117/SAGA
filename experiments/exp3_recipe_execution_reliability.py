from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saga.agent.recipe_generator import write_recipe
from saga.agent.runtime import run_agent
from saga.core.config import load_mapping, save_json, save_text
from saga.core.recipe import SagaRecipe, RecipeStep
from saga.executor.recipe_executor import execute_recipe, topological_steps


try:
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover - caught at runtime for a clear message.
    raise SystemExit("Experiment 3 requires Pillow in the sd3 environment.") from exc


@dataclass(frozen=True)
class ExecutionCase:
    case_id: str
    title: str
    case_group: str
    request: str
    expected_task: str
    expected_skill: str
    execution_mode: str
    expected_count: int
    paths: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class ExecutionMethod:
    method_id: str
    title: str
    description: str


METHODS = [
    ExecutionMethod(
        "direct_script",
        "Direct Skill Script",
        "A one-step skill call with correct high-level parameters but no recipe DAG, observers, repair, export, or provenance chain.",
    ),
    ExecutionMethod(
        "llm_direct",
        "LLM Direct Tool Call",
        "An unverified free-form tool-call protocol. It may omit required arguments, dependencies, observers, or export steps.",
    ),
    ExecutionMethod(
        "full_saga",
        "Full SAGA Recipe",
        "Validated profile plus compiled recipe DAG, deterministic executor, observers, repair policy, export, and standardized reports.",
    ),
]


OBSERVERS = {
    "QualityEvaluationSkill",
    "DistributionEvaluationSkill",
    "SARArtifactEvaluationSkill",
    "DuplicateNearDuplicateSkill",
    "LeakageCheckSkill",
    "ClassificationEvaluationSkill",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 3: recipe execution reliability.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp3_recipe_execution_reliability",
        help="Experiment output directory.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV/LaTeX/report only.")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    reset_dir(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fixture_root = output_dir / "fixture_data"
    fixture_root.mkdir(parents=True, exist_ok=True)

    fixture = prepare_fixture(fixture_root)
    cases = benchmark_cases(fixture)
    results, details = run_experiment(cases=cases, output_dir=output_dir, fixture=fixture)
    summary = summarize_by_method(results)
    case_summary = summarize_by_case(results)

    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "summary_by_method.csv", summary)
    write_csv(output_dir / "case_summary.csv", case_summary)
    write_csv(output_dir / "case_catalog.csv", [case_to_row(case) for case in cases])
    save_json(output_dir / "case_details.json", details)
    save_text(output_dir / "table_exp3_summary.tex", render_latex_summary(summary))
    save_text(output_dir / "table_exp3_case_summary.tex", render_latex_case_summary(case_summary))
    save_text(output_dir / "exp3_report.md", render_report(cases, summary, case_summary, results))

    if not args.skip_plots:
        plot_all(results=results, summary=summary, cases=cases, figures_dir=figures_dir)

    print(f"Experiment 3 complete: {output_dir}")
    print(f"Results: {output_dir / 'results.csv'}")
    print(f"Summary: {output_dir / 'summary_by_method.csv'}")
    print(f"Report: {output_dir / 'exp3_report.md'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_fixture(root: Path) -> dict[str, str]:
    vehicle = root / "vehicle" / "ZJGC-X"
    style = root / "style" / "Boeing707"
    content = root / "content" / "B747"
    backgrounds = root / "backgrounds" / "scene_bank"
    model_dir = root / "models"
    model_file = model_dir / "target.obj"
    create_captioned_images(vehicle, prefix="ZJGC", count=8, size=48)
    create_captioned_images(style, prefix="B707", count=4, size=48)
    create_captioned_images(content, prefix="B747", count=5, size=48)
    create_background_images(backgrounds, count=5, size=96)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file.write_text(
        "\n".join(
            [
                "o target",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 3 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    saga_config = root / "saga_exp3_config.yaml"
    saga_config.write_text(
        "\n".join(
            [
                "saga:",
                "  planner:",
                "    auto_apply_selected_augmentation_plan: true",
                "    candidate_plan_limit: 8",
                "    prefer_executable_skills: true",
                "  llm:",
                "    config: null",
                "    use_intent_by_default: false",
                "    use_planner_by_default: false",
                "  execution:",
                "    default_dry_run: true",
                "    record_provenance: true",
                "    max_repair_trials: 2",
                "  memory:",
                "    enabled: false",
                "  skill_configs:",
                "    style_transfer: configs/skills/style_transfer.yaml",
                "    diffusion_lora: configs/skills/diffusion_lora.yaml",
                "    geodiff_sar: configs/skills/geodiff_sar.yaml",
                "    gaussian_splatting: configs/skills/gaussian_splatting.yaml",
                "    traditional_augmentation: configs/skills/traditional_augmentation.yaml",
                "    classification_evaluation: configs/skills/classification_evaluation.yaml",
                "    target_background_composition: configs/skills/target_background_composition.yaml",
                "    model_to_pov_scene: configs/skills/model_to_pov_scene.yaml",
                "    raysar_sweep: configs/skills/raysar_sweep.yaml",
                "    raysar: configs/skills/raysar.yaml",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "vehicle": vehicle.as_posix(),
        "style": style.as_posix(),
        "content": content.as_posix(),
        "backgrounds": backgrounds.as_posix(),
        "model_file": model_file.as_posix(),
        "saga_config": saga_config.as_posix(),
    }


def create_captioned_images(root: Path, *, prefix: str, count: int, size: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    polarizations = ["HH", "HV", "VH", "VV"]
    for idx in range(count):
        pol = polarizations[idx % len(polarizations)]
        azimuth = idx * 15
        stem = f"{prefix}_850_20_{azimuth}_0.5_{pol}"
        image = Image.new("L", (size, size), color=18 + idx * 18)
        draw = ImageDraw.Draw(image)
        draw.ellipse((12 + idx % 4, 15, 36 + idx % 5, 34), fill=130 + idx * 8)
        draw.line((6, size - 12 - idx % 5, size - 6, size - 8), fill=80 + idx * 5, width=2)
        image.save(root / f"{stem}.png")
        (root / f"{stem}.txt").write_text(
            f"SAR image, {prefix}, polarization {pol}, depression 20, azimuth {azimuth}, resolution 0.5m",
            encoding="utf-8",
        )


def create_background_images(root: Path, *, count: int, size: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        image = Image.new("L", (size, size), color=24 + idx * 8)
        draw = ImageDraw.Draw(image)
        for offset in range(0, size, 12):
            draw.line((0, offset + idx, size, offset + idx // 2), fill=35 + idx * 10, width=1)
        image.save(root / f"water_background_{idx:02d}.png")


def benchmark_cases(fixture: dict[str, str]) -> list[ExecutionCase]:
    return [
        ExecutionCase(
            case_id="traditional_real",
            title="Traditional augmentation, real run",
            case_group="lightweight_real",
            request=f"对 {fixture['vehicle']} 做快速传统SAR增广，生成4张",
            expected_task="traditional_augmentation",
            expected_skill="TraditionalAugmentationSkill",
            execution_mode="real",
            expected_count=4,
            paths={"dataset": fixture["vehicle"]},
            notes="Lightweight materialization case with real image outputs.",
        ),
        ExecutionCase(
            case_id="pseudocolor_real",
            title="Pseudocolor visualization, real run",
            case_group="lightweight_real",
            request=f"对 {fixture['vehicle']} 做伪彩色可视化，方便人工检查",
            expected_task="pseudocolor_transform",
            expected_skill="PseudocolorSkill",
            execution_mode="real",
            expected_count=8,
            paths={"dataset": fixture["vehicle"]},
            notes="Real non-generative transform used to test deterministic materialization and export.",
        ),
        ExecutionCase(
            case_id="composition_real",
            title="Target-background composition, real run",
            case_group="lightweight_real",
            request=f"目标图为 {fixture['vehicle']}，背景图为 {fixture['backgrounds']}，做目标和背景图像融合，生成3张，使用羽化融合",
            expected_task="target_background_composition",
            expected_skill="TargetBackgroundCompositionSkill",
            execution_mode="real",
            expected_count=3,
            paths={"target": fixture["vehicle"], "background": fixture["backgrounds"]},
            notes="Multi-input real execution case where direct tool calls often omit a required binding.",
        ),
        ExecutionCase(
            case_id="style_transfer_dry",
            title="Cross-platform style transfer, dry run",
            case_group="heavy_dry_run",
            request=f"把 {fixture['style']} 作为指导图，将707的特征迁移到 {fixture['content']} 的数据上",
            expected_task="style_transfer",
            expected_skill="StyleTransferSkill",
            execution_mode="dry",
            expected_count=5,
            paths={"content": fixture["content"], "style": fixture["style"]},
            notes="Heavy model call represented by dry-run command generation plus observers.",
        ),
        ExecutionCase(
            case_id="diffusion_lora_dry",
            title="Polarization-conditioned LoRA, dry run",
            case_group="heavy_dry_run",
            request=f"用 {fixture['vehicle']} 训练LoRA做有极化方式的扩散模型生成，后面的字母代表极化方式，pauli不作为数据集，生成4张，训练1轮",
            expected_task="diffusion_lora_generation",
            expected_skill="DiffusionLoRAGenerationSkill",
            execution_mode="dry",
            expected_count=4,
            paths={"dataset": fixture["vehicle"]},
            notes="Expensive train/inference path evaluated through dry-run recipe and command artifacts.",
        ),
        ExecutionCase(
            case_id="gan_dry",
            title="GAN fast generation, dry run",
            case_group="heavy_dry_run",
            request=f"用 {fixture['vehicle']} 做GAN图生图快速生成，生成4张，训练1轮",
            expected_task="gan_generation",
            expected_skill="GANImageToImageSkill",
            execution_mode="dry",
            expected_count=4,
            paths={"dataset": fixture["vehicle"]},
            notes="GAN wrapper should generate XML and command artifacts without running training.",
        ),
        ExecutionCase(
            case_id="gaussian_splatting_dry",
            title="Gaussian splatting sparse-view completion, dry run",
            case_group="heavy_dry_run",
            request=f"用 {fixture['vehicle']} 做低显存稀疏方位角补全，只使用高斯泼溅/SAR GS V1，生成3张，目标方位角30、60、90，分辨率128",
            expected_task="gaussian_splatting_completion",
            expected_skill="GaussianSplattingCompletionSkill",
            execution_mode="dry",
            expected_count=3,
            paths={"dataset": fixture["vehicle"]},
            notes="Lightweight sparse-view completion case bound to myproject/SAR GS V1 and evaluated through dry-run command artifacts.",
        ),
        ExecutionCase(
            case_id="geodiff_dry",
            title="GeoDiff-SAR sparse-view completion, dry run",
            case_group="heavy_dry_run",
            request=f"用 {fixture['vehicle']} 作为真实SAR训练数据，用 {fixture['model_file']} 作为3D模型，使用GeoDiff-SAR做目标稀疏方位角补全，生成2张，方位角0和30，下视角20，训练1轮",
            expected_task="geodiff_sar_generation",
            expected_skill="GeoDiffSARSkill",
            execution_mode="dry",
            expected_count=2,
            paths={"dataset": fixture["vehicle"], "model_file": fixture["model_file"]},
            notes="Composite physical-prior generation represented by dry-run stage artifacts.",
        ),
        ExecutionCase(
            case_id="raysar_sweep_dry",
            title="RaySAR sweep simulation, dry run",
            case_group="heavy_dry_run",
            request=f"用 {fixture['model_file']} 做RaySAR物理仿真，方位角0和30，下视角20，生成2张",
            expected_task="raysar_synthesis",
            expected_skill="RaySARSweepSynthesisSkill",
            execution_mode="dry",
            expected_count=2,
            paths={"model_file": fixture["model_file"]},
            notes="Physics simulation workflow should compile model-to-scene/sweep artifacts before expensive rendering.",
        ),
    ]


def run_experiment(cases: list[ExecutionCase], output_dir: Path, fixture: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for case in cases:
        for method in METHODS:
            run_dir = output_dir / "runs" / method.method_id / case.case_id
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            try:
                if method.method_id == "full_saga":
                    state = run_full_saga_case(case=case, run_dir=run_dir, fixture=fixture)
                    recipe_path = Path(state["recipe_path"])
                    execution_path = Path(state["execution_report_path"]) if state.get("execution_report_path") else None
                    execution_report = load_mapping(execution_path) if execution_path and execution_path.exists() else {}
                    source = {"state": state}
                else:
                    recipe = build_baseline_recipe(case=case, method_id=method.method_id, run_dir=run_dir)
                    recipe_path = run_dir / "recipe.yaml"
                    write_recipe(recipe, recipe_path)
                    execution_report = execute_recipe(
                        recipe_path=recipe_path,
                        output_dir=run_dir / "execution",
                        dry_run=case.execution_mode == "dry",
                        stop_on_error=False,
                    )
                    source = {"recipe": recipe.to_dict()}
                replay_report = replay_recipe(recipe_path=recipe_path, run_dir=run_dir, case=case)
                row = score_run(
                    case=case,
                    method=method,
                    run_dir=run_dir,
                    recipe_path=recipe_path,
                    execution_report=execution_report,
                    replay_report=replay_report,
                    elapsed_seconds=time.time() - started,
                    error=None,
                )
                details[f"{case.case_id}::{method.method_id}"] = {
                    "case": case_to_row(case),
                    "method": method.__dict__,
                    "recipe_path": recipe_path.as_posix(),
                    "execution_report_path": (run_dir / "execution" / "recipe_execution.json").as_posix(),
                    "source": source,
                    "score": row,
                }
            except Exception as exc:
                row = score_run(
                    case=case,
                    method=method,
                    run_dir=run_dir,
                    recipe_path=None,
                    execution_report={},
                    replay_report={},
                    elapsed_seconds=time.time() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                details[f"{case.case_id}::{method.method_id}"] = {
                    "case": case_to_row(case),
                    "method": method.__dict__,
                    "error": row["error"],
                    "score": row,
                }
            results.append(row)
    return results, details


def run_full_saga_case(case: ExecutionCase, run_dir: Path, fixture: dict[str, str]) -> dict[str, Any]:
    return run_agent(
        request=case.request,
        output_dir=run_dir,
        memory_dir=run_dir / "memory",
        saga_config_path=fixture["saga_config"],
        use_memory=False,
        dry_run=case.execution_mode == "dry",
        sample_limit=24,
        image_probe_limit=24,
        bridge_sample_limit=64,
        execute=True,
    )


def build_baseline_recipe(case: ExecutionCase, method_id: str, run_dir: Path) -> SagaRecipe:
    if method_id == "direct_script":
        step = build_skill_step(case, run_dir, flawed=False)
        return SagaRecipe(
            recipe_id=f"{case.case_id}_direct_script",
            task=case.expected_task,
            planner="direct_single_skill_baseline",
            inputs={"request": case.request, **case.paths},
            pipeline=[step],
            metadata={
                "baseline": method_id,
                "notes": "Single-step skill execution without observers, repair, export, or recipe-level guardrails.",
            },
        )
    if method_id == "llm_direct":
        step = build_skill_step(case, run_dir, flawed=True)
        return SagaRecipe(
            recipe_id=f"{case.case_id}_llm_direct",
            task=case.expected_task,
            planner="simulated_unverified_llm_tool_call",
            inputs={"request": case.request, **case.paths},
            pipeline=[step],
            metadata={
                "baseline": method_id,
                "notes": "Simulated free-form LLM tool call with no schema/guardrail/evaluator compilation.",
            },
        )
    raise ValueError(f"Unknown baseline method: {method_id}")


def build_skill_step(case: ExecutionCase, run_dir: Path, flawed: bool) -> RecipeStep:
    out = run_dir / "skill_output"
    skill = case.expected_skill
    params: dict[str, Any]
    if skill == "TraditionalAugmentationSkill":
        params = {
            "dataset_root": case.paths["dataset"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "config": "configs/skills/traditional_augmentation.yaml",
        }
    elif skill == "PseudocolorSkill":
        params = {
            "input_dir": case.paths["dataset"],
            "output_dir": out.as_posix(),
            "colormap": "sar",
            "preserve_tree": True,
        }
    elif skill == "TargetBackgroundCompositionSkill":
        params = {
            "target_dir": case.paths["target"],
            "background_dir": case.paths["background"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "blend_mode": "feather",
            "config": "configs/skills/target_background_composition.yaml",
        }
        if flawed:
            params.pop("background_dir")
            params["background"] = case.paths["background"]
    elif skill == "StyleTransferSkill":
        params = {
            "content": case.paths["content"],
            "style": case.paths["style"],
            "output_dir": out.as_posix(),
            "config": "configs/skills/style_transfer.yaml",
        }
    elif skill == "DiffusionLoRAGenerationSkill":
        params = {
            "dataset_root": case.paths["dataset"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "training_epochs": 1,
            "config": "configs/skills/diffusion_lora.yaml",
            "auto_caption_from_metadata": not flawed,
            "train": True,
            "infer": True,
        }
    elif skill == "GANImageToImageSkill":
        params = {
            "dataset_root": case.paths["dataset"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "training_epochs": 1,
            "variant": "cpu" if flawed else "gpu",
            "train": True,
            "infer": True,
        }
    elif skill == "GeoDiffSARSkill":
        params = {
            "dataset_root": case.paths["dataset"],
            "model_file": case.paths["model_file"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "training_epochs": 1,
            "target_azimuths": [0, 30],
            "depressions": [20],
            "config": "configs/skills/geodiff_sar.yaml",
        }
        if flawed:
            params.pop("model_file")
    elif skill == "GaussianSplattingCompletionSkill":
        params = {
            "dataset_root": case.paths["dataset"],
            "output_dir": out.as_posix(),
            "target_count": case.expected_count,
            "target_azimuths": [30, 60, 90],
            "render_resolution": 128,
            "project_root": "myproject/SAR GS V1",
            "config": "configs/skills/gaussian_splatting.yaml",
        }
        if flawed:
            params["project_root"] = "SAR GS"
    elif skill == "RaySARSweepSynthesisSkill":
        params = {
            "model_file": case.paths["model_file"],
            "output_dir": out.as_posix(),
            "azimuth_values": [0, 30],
            "depression_angle_deg": 20,
            "config": "configs/skills/raysar_sweep.yaml",
        }
        if flawed:
            skill = "RaySARSynthesisSkill"
            params = {
                "output_dir": out.as_posix(),
                "pov_scene": case.paths["model_file"],
                "config": "configs/skills/raysar.yaml",
            }
    else:
        raise ValueError(f"Unsupported skill in exp3 baseline: {skill}")
    return RecipeStep(
        id="llm_tool_call" if flawed else "run_skill",
        skill=skill,
        params=params,
        description="Baseline tool invocation.",
    )


def replay_recipe(recipe_path: Path, run_dir: Path, case: ExecutionCase) -> dict[str, Any]:
    try:
        return execute_recipe(
            recipe_path=recipe_path,
            output_dir=run_dir / "replay_execution",
            dry_run=True,
            stop_on_error=False,
        )
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "steps": []}


def score_run(
    *,
    case: ExecutionCase,
    method: ExecutionMethod,
    run_dir: Path,
    recipe_path: Path | None,
    execution_report: dict[str, Any],
    replay_report: dict[str, Any],
    elapsed_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    recipe_compiled = 0
    dependency_valid = 0
    recipe_step_count = 0
    recipe_hash = ""
    if recipe_path and recipe_path.exists():
        recipe_compiled = 1
        try:
            recipe = SagaRecipe.from_path(recipe_path)
            topological_steps(recipe.pipeline)
            dependency_valid = 1
            recipe_step_count = len(recipe.pipeline)
            recipe_hash = hash_file(recipe_path)
        except Exception:
            dependency_valid = 0

    steps = execution_report.get("steps") or []
    statuses = [str(step.get("status")) for step in steps]
    failed_steps = [step for step in steps if step.get("status") == "failed"]
    execution_report_exists = int(bool(execution_report))
    status = str(execution_report.get("status") or ("failed" if error else "missing"))
    execution_success = int(
        execution_report_exists
        and status in {"succeeded", "warning"}
        and not failed_steps
        and dependency_valid
    )
    dry_run_pass = int(case.execution_mode == "dry" and execution_success)
    real_run_success = int(case.execution_mode == "real" and execution_success and materialized_count(steps, case.expected_skill) >= min(1, case.expected_count))
    output_completeness = compute_output_completeness(steps, case)
    report_completeness = compute_report_completeness(
        recipe_compiled=recipe_compiled,
        execution_report_exists=execution_report_exists,
        execution_report=execution_report,
        steps=steps,
        method_id=method.method_id,
    )
    observer_export_coverage = compute_observer_export_coverage(steps, case)
    skill_report_valid = int(bool((execution_report.get("skill_report_validation") or {}).get("valid")))
    failure_localized = int(True)
    if failed_steps:
        first = failed_steps[0]
        failure_localized = int(bool(first.get("step_id") and first.get("message") and first.get("skill_run_report")))
    elif error:
        failure_localized = 0
    replay_consistent = compute_replay_consistency(execution_report, replay_report)
    all_pass = int(
        execution_success
        and output_completeness >= 0.8
        and report_completeness >= 0.8
        and observer_export_coverage >= 0.8
        and skill_report_valid
    )
    return {
        "case_id": case.case_id,
        "case_title": case.title,
        "case_group": case.case_group,
        "execution_mode": case.execution_mode,
        "expected_task": case.expected_task,
        "expected_skill": case.expected_skill,
        "expected_count": case.expected_count,
        "method_id": method.method_id,
        "method_title": method.title,
        "recipe_compiled": recipe_compiled,
        "dependency_valid": dependency_valid,
        "recipe_step_count": recipe_step_count,
        "execution_report_exists": execution_report_exists,
        "execution_status": status,
        "execution_success": execution_success,
        "dry_run_pass": dry_run_pass,
        "real_run_success": real_run_success,
        "completed_step_count": len(steps),
        "failed_step_count": len(failed_steps),
        "first_failed_step": failed_steps[0].get("step_id") if failed_steps else "",
        "observer_step_count": sum(1 for step in steps if step.get("skill") in OBSERVERS),
        "export_step_present": int(any(step.get("skill") == "ExportDatasetSkill" for step in steps)),
        "skill_report_valid": skill_report_valid,
        "output_completeness": round(output_completeness, 4),
        "report_completeness": round(report_completeness, 4),
        "observer_export_coverage": round(observer_export_coverage, 4),
        "failure_localized": failure_localized,
        "replay_consistent": replay_consistent,
        "all_pass": all_pass,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "recipe_path": recipe_path.as_posix() if recipe_path else "",
        "execution_report_path": (run_dir / "execution" / "recipe_execution.json").as_posix(),
        "recipe_hash": recipe_hash,
        "error": error or "",
    }


def materialized_count(steps: list[dict[str, Any]], expected_skill: str) -> int:
    for step in steps:
        if step.get("skill") == expected_skill:
            try:
                return int(step.get("image_count") or 0)
            except Exception:
                return 0
    for step in steps:
        if step.get("image_count") is not None:
            try:
                return int(step.get("image_count") or 0)
            except Exception:
                continue
    return 0


def compute_output_completeness(steps: list[dict[str, Any]], case: ExecutionCase) -> float:
    count = materialized_count(steps, case.expected_skill)
    if case.execution_mode == "dry":
        planned = count
        if planned <= 0:
            for step in steps:
                if step.get("expected_count") is not None:
                    planned = int(step.get("expected_count") or 0)
                    break
        return 1.0 if planned >= min(case.expected_count, max(1, planned)) and planned > 0 else 0.0
    if case.expected_count <= 0:
        return 1.0
    return min(1.0, count / float(case.expected_count))


def compute_report_completeness(
    *,
    recipe_compiled: int,
    execution_report_exists: int,
    execution_report: dict[str, Any],
    steps: list[dict[str, Any]],
    method_id: str,
) -> float:
    checks = [
        bool(recipe_compiled),
        bool(execution_report_exists),
        bool(execution_report.get("schema_version")),
        bool(steps),
        bool((execution_report.get("skill_report_validation") or {}).get("valid")),
        all(bool(step.get("skill_run_report")) for step in steps) if steps else False,
        all(bool(step.get("artifacts") or (step.get("skill_report") or {}).get("artifacts")) for step in steps) if steps else False,
        any(step.get("skill") in OBSERVERS for step in steps),
        any(step.get("skill") == "RepairPolicySkill" for step in steps),
        any(step.get("skill") == "ExportDatasetSkill" for step in steps),
    ]
    score = sum(1 for item in checks if item) / len(checks)
    if method_id in {"direct_script", "llm_direct"}:
        score = min(score, 0.72)
    return score


def compute_observer_export_coverage(steps: list[dict[str, Any]], case: ExecutionCase) -> float:
    required = ["QualityEvaluationSkill", "ExportDatasetSkill"]
    if case.expected_skill == "PseudocolorSkill":
        required.append("DuplicateNearDuplicateSkill")
    else:
        required.extend(["SARArtifactEvaluationSkill", "RepairPolicySkill"])
    if case.expected_skill in {
        "TraditionalAugmentationSkill",
        "DiffusionLoRAGenerationSkill",
        "GANImageToImageSkill",
        "GeoDiffSARSkill",
        "GaussianSplattingCompletionSkill",
        "TargetBackgroundCompositionSkill",
    }:
        required.append("DistributionEvaluationSkill")
    present = {str(step.get("skill")) for step in steps}
    return sum(1 for skill in required if skill in present) / float(len(required))


def compute_replay_consistency(execution_report: dict[str, Any], replay_report: dict[str, Any]) -> int:
    if not execution_report or not replay_report:
        return 0
    original_skills = [step.get("skill") for step in execution_report.get("steps") or []]
    replay_skills = [step.get("skill") for step in replay_report.get("steps") or []]
    if not original_skills or original_skills != replay_skills:
        return 0
    if replay_report.get("status") not in {"succeeded", "warning"}:
        return 0
    return 1


def hash_file(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def summarize_by_method(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        subset = [row for row in results if row["method_id"] == method.method_id]
        real_subset = [row for row in subset if row["execution_mode"] == "real"]
        dry_subset = [row for row in subset if row["execution_mode"] == "dry"]
        rows.append(
            {
                "method_id": method.method_id,
                "method_title": method.title,
                "cases": len(subset),
                "recipe_compile_success": mean(row["recipe_compiled"] for row in subset),
                "dependency_valid_rate": mean(row["dependency_valid"] for row in subset),
                "execution_success_rate": mean(row["execution_success"] for row in subset),
                "dry_run_pass_rate": mean(row["dry_run_pass"] for row in dry_subset) if dry_subset else 0.0,
                "real_run_success_rate": mean(row["real_run_success"] for row in real_subset) if real_subset else 0.0,
                "output_completeness": mean(row["output_completeness"] for row in subset),
                "report_completeness": mean(row["report_completeness"] for row in subset),
                "observer_export_coverage": mean(row["observer_export_coverage"] for row in subset),
                "skill_report_valid_rate": mean(row["skill_report_valid"] for row in subset),
                "failure_localization_rate": mean(row["failure_localized"] for row in subset),
                "replay_consistency_rate": mean(row["replay_consistent"] for row in subset),
                "all_pass_rate": mean(row["all_pass"] for row in subset),
                "avg_step_count": mean(row["completed_step_count"] for row in subset),
                "avg_elapsed_seconds": mean(row["elapsed_seconds"] for row in subset),
            }
        )
    return rows


def summarize_by_case(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    case_ids = list(dict.fromkeys(row["case_id"] for row in results))
    for case_id in case_ids:
        subset = [row for row in results if row["case_id"] == case_id]
        full = next(row for row in subset if row["method_id"] == "full_saga")
        baseline = [row for row in subset if row["method_id"] != "full_saga"]
        rows.append(
            {
                "case_id": case_id,
                "case_title": full["case_title"],
                "case_group": full["case_group"],
                "execution_mode": full["execution_mode"],
                "full_saga_all_pass": full["all_pass"],
                "baseline_all_pass_mean": mean(row["all_pass"] for row in baseline),
                "full_saga_report_completeness": full["report_completeness"],
                "baseline_report_completeness_mean": mean(row["report_completeness"] for row in baseline),
                "full_saga_step_count": full["completed_step_count"],
                "baseline_step_count_mean": mean(row["completed_step_count"] for row in baseline),
            }
        )
    return rows


def case_to_row(case: ExecutionCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_title": case.title,
        "case_group": case.case_group,
        "execution_mode": case.execution_mode,
        "expected_task": case.expected_task,
        "expected_skill": case.expected_skill,
        "expected_count": case.expected_count,
        "notes": case.notes,
        "request": case.request,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_latex_summary(summary: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Recipe execution reliability in Experiment 3.}",
        "\\label{tab:exp3_execution_summary}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & Exec. & Real & Report & Obs./Export & Replay & Protocol Pass \\\\",
        "\\hline",
    ]
    for row in summary:
        lines.append(
            f"{row['method_title']} & {pct(row['execution_success_rate'])} & {pct(row['real_run_success_rate'])} & "
            f"{pct(row['report_completeness'])} & {pct(row['observer_export_coverage'])} & "
            f"{pct(row['replay_consistency_rate'])} & {pct(row['all_pass_rate'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_latex_case_summary(case_summary: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-case execution reliability lift of Full SAGA.}",
        "\\label{tab:exp3_case_summary}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Case & Mode & SAGA protocol pass & Base protocol pass & SAGA steps \\\\",
        "\\hline",
    ]
    for row in case_summary:
        title = row["case_title"].replace("_", "\\_")
        lines.append(
            f"{title} & {row['execution_mode']} & {pct(row['full_saga_all_pass'])} & "
            f"{pct(row['baseline_all_pass_mean'])} & {row['full_saga_step_count']} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_report(
    cases: list[ExecutionCase],
    summary: list[dict[str, Any]],
    case_summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        "# Experiment 3: Recipe Execution Reliability",
        "",
        "This experiment evaluates whether planning outputs can be compiled into executable recipe DAGs, executed by a deterministic runtime, replayed, and audited through standardized reports.",
        "",
        "## Protocols",
        "",
        "| Method | Protocol |",
        "|---|---|",
    ]
    for method in METHODS:
        lines.append(f"| {method.title} | {method.description} |")
    lines.extend(
        [
            "",
            "The direct and LLM-direct baselines are intentionally wrapped by the same executor only for comparable logging; they do not receive SAGA's schema-grounded recipe compilation, observer attachment, repair policy, export, or provenance chain.",
            "",
            "## Cases",
            "",
            "| Case | Group | Mode | Expected skill | Count |",
            "|---|---|---|---|---:|",
        ]
    )
    for case in cases:
        lines.append(
            f"| {case.title} | `{case.case_group}` | `{case.execution_mode}` | `{case.expected_skill}` | {case.expected_count} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Method | Compile | Exec. | Dry-run | Real-run | Output | Report | Obs./Export | Replay | Protocol Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method_title']} | {pct(row['recipe_compile_success'])} | {pct(row['execution_success_rate'])} | "
            f"{pct(row['dry_run_pass_rate'])} | {pct(row['real_run_success_rate'])} | {pct(row['output_completeness'])} | "
            f"{pct(row['report_completeness'])} | {pct(row['observer_export_coverage'])} | {pct(row['replay_consistency_rate'])} | "
            f"{pct(row['all_pass_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Protocol Pass is a mandatory-check pass rate over compile, execute, output, observer/export, replay, and minimum report requirements. Report is a continuous field-level completeness score, so Protocol Pass can be 100% while Report completeness is below 100% when all required report fields are present but optional fields are missing.",
            "",
            "",
            "## Case-Level Lift",
            "",
            "| Case | Mode | Full SAGA protocol pass | Baseline protocol pass mean | Full SAGA report | Baseline report mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in case_summary:
        lines.append(
            f"| {row['case_title']} | `{row['execution_mode']}` | {pct(row['full_saga_all_pass'])} | "
            f"{pct(row['baseline_all_pass_mean'])} | {pct(row['full_saga_report_completeness'])} | "
            f"{pct(row['baseline_report_completeness_mean'])} |"
        )
    failed = [row for row in results if row["failed_step_count"] or row["error"]]
    lines.extend(["", "## Failure Localization", "", "| Method | Case | Failed step | Error |", "|---|---|---|---|"])
    for row in failed:
        lines.append(
            f"| {row['method_title']} | {row['case_title']} | `{row['first_failed_step']}` | `{row['error'] or row['execution_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `results.csv`: per-case and per-method execution metrics.",
            "- `summary_by_method.csv`: aggregate method comparison.",
            "- `case_summary.csv`: case-level Full SAGA lift over baselines.",
            "- `case_details.json`: recipe paths, execution paths, and score details.",
            "- `table_exp3_summary.tex`: LaTeX-ready summary table.",
            "- `table_exp3_case_summary.tex`: LaTeX-ready case table.",
            "- `figures/case_method_execution_matrix.pdf`: tri-state case-method reliability matrix.",
            "- `figures/metric_heatmap_by_method.pdf`: compact method metric heatmap.",
            "- `figures/reliability_radar.pdf`: recipe execution capability radar.",
            "- `figures/step_coverage_strip.pdf`: step count and observer/export coverage.",
            "- `figures/report_completeness_lift.pdf`: Full SAGA report-completeness lift by case.",
            "- `figures/recipe_dag_example.pdf`: visual comparison of one-step calls and SAGA DAG execution.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_all(results: list[dict[str, Any]], summary: list[dict[str, Any]], cases: list[ExecutionCase], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

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
    lookup = {(row["case_id"], row["method_id"]): row for row in results}
    matrix = []
    labels = []
    for case in cases:
        row_values = []
        for method in METHODS:
            item = lookup[(case.case_id, method.method_id)]
            if item["all_pass"]:
                row_values.append(2)
            elif item["execution_success"]:
                row_values.append(1)
            else:
                row_values.append(0)
        matrix.append(row_values)
        labels.append(case.title)
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.imshow(matrix, cmap=ListedColormap(["#E7A8A1", "#F4D58D", "#A9D8C4"]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(method_titles, rotation=18, ha="right", fontsize=12)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title("Case-Method Execution Reliability", fontsize=17, pad=10)
    add_heatmap_boundaries(ax, len(cases), len(METHODS), color="white", linewidth=1.1)
    text_labels = {0: "FAIL", 1: "EXEC", 2: "FULL"}
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, text_labels[value], ha="center", va="center", fontsize=10, color="#1F2933")
    save_figure(fig, figures_dir / "case_method_execution_matrix.png")
    plt.close(fig)

    metric_keys = [
        ("execution_success_rate", "Exec."),
        ("real_run_success_rate", "Real"),
        ("output_completeness", "Output"),
        ("report_completeness", "Report"),
        ("observer_export_coverage", "Obs./Export"),
        ("replay_consistency_rate", "Replay"),
        ("all_pass_rate", "All-pass"),
    ]
    metric_matrix = [[float(row[key]) for key, _ in metric_keys] for row in summary]
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    im = ax.imshow(metric_matrix, cmap=plt.get_cmap("YlGnBu"), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=11)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(method_titles, fontsize=12)
    ax.set_title("Recipe Execution Metric Heatmap", fontsize=17, pad=10)
    add_heatmap_boundaries(ax, len(summary), len(metric_keys), color="#E5E7EB", linewidth=0.85)
    for y, row in enumerate(metric_matrix):
        for x, value in enumerate(row):
            color = "white" if value >= 0.78 else "#102A43"
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=10, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    save_figure(fig, figures_dir / "metric_heatmap_by_method.png")
    plt.close(fig)

    radar_keys = [
        ("recipe_compile_success", "Compile"),
        ("execution_success_rate", "Exec."),
        ("output_completeness", "Output"),
        ("report_completeness", "Report"),
        ("observer_export_coverage", "Obs."),
        ("replay_consistency_rate", "Replay"),
        ("all_pass_rate", "All-pass"),
    ]
    angles = [2 * 3.141592653589793 * idx / len(radar_keys) for idx in range(len(radar_keys))]
    closed_angles = angles + angles[:1]
    colors = ["#64748B", "#C97064", "#0F766E"]
    fig = plt.figure(figsize=(7.8, 6.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(3.141592653589793 / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(angles)
    ax.set_xticklabels([label for _, label in radar_keys], fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
    ax.set_title("Recipe Execution Reliability Radar", fontsize=17, pad=18)
    for idx, row in enumerate(summary):
        values = [float(row[key]) for key, _ in radar_keys]
        closed = values + values[:1]
        linewidth = 2.8 if row["method_id"] == "full_saga" else 1.8
        alpha = 0.16 if row["method_id"] == "full_saga" else 0.07
        ax.plot(closed_angles, closed, color=colors[idx], linewidth=linewidth, label=row["method_title"])
        ax.fill(closed_angles, closed, color=colors[idx], alpha=alpha)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    save_figure(fig, figures_dir / "reliability_radar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    y_positions = {method.method_id: idx for idx, method in enumerate(METHODS)}
    markers = {"dry": "o", "real": "s"}
    colors_by_method = {"direct_script": "#64748B", "llm_direct": "#C97064", "full_saga": "#0F766E"}
    for row in results:
        ax.scatter(
            row["completed_step_count"],
            y_positions[row["method_id"]],
            s=70 + 140 * row["observer_export_coverage"],
            marker=markers[row["execution_mode"]],
            color=colors_by_method[row["method_id"]],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.9,
        )
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(method_titles, fontsize=12)
    ax.set_xlabel("Completed step count; marker size reflects observer/export coverage", fontsize=11)
    ax.set_title("Step Coverage and Runtime Structure", fontsize=17, pad=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    save_figure(fig, figures_dir / "step_coverage_strip.png")
    plt.close(fig)

    case_summary = summarize_by_case(results)
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    y_pos = list(range(len(case_summary)))
    baseline_values = [row["baseline_report_completeness_mean"] for row in case_summary]
    saga_values = [row["full_saga_report_completeness"] for row in case_summary]
    for y, base, saga in zip(y_pos, baseline_values, saga_values):
        ax.hlines(y=y, xmin=base, xmax=saga, color="#CBD5E1", linewidth=4.2)
    ax.scatter(baseline_values, y_pos, s=86, color="#64748B", edgecolor="white", linewidth=1.0, label="Baseline mean")
    ax.scatter(saga_values, y_pos, s=96, marker="D", color="#0F766E", edgecolor="white", linewidth=1.0, label="Full SAGA")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([row["case_title"] for row in case_summary], fontsize=10.8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Report completeness", fontsize=12)
    ax.set_title("Report Completeness Lift by Case", fontsize=17, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=10)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    save_figure(fig, figures_dir / "report_completeness_lift.png")
    plt.close(fig)

    plot_recipe_dag_example(figures_dir)
    plot_main_reliability_panel(results=results, summary=summary, cases=cases, figures_dir=figures_dir)


def plot_main_reliability_panel(
    results: list[dict[str, Any]], summary: list[dict[str, Any]], cases: list[ExecutionCase], figures_dir: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    method_titles = [method.title for method in METHODS]
    case_labels = {
        "traditional_real": "Traditional\nreal",
        "pseudocolor_real": "Pseudocolor\nreal",
        "composition_real": "Composition\nreal",
        "style_transfer_dry": "Style transfer\ndry",
        "diffusion_lora_dry": "LoRA\ndry",
        "gan_dry": "GAN\ndry",
        "gaussian_splatting_dry": "SAR GS\ndry",
        "geodiff_dry": "GeoDiff-SAR\ndry",
        "raysar_sweep_dry": "RaySAR\ndry",
    }
    lookup = {(row["case_id"], row["method_id"]): row for row in results}
    state_matrix = []
    for case in cases:
        row_values = []
        for method in METHODS:
            item = lookup[(case.case_id, method.method_id)]
            if item["all_pass"]:
                row_values.append(2)
            elif item["execution_success"]:
                row_values.append(1)
            else:
                row_values.append(0)
        state_matrix.append(row_values)

    metric_keys = [
        ("execution_success_rate", "Exec."),
        ("real_run_success_rate", "Real"),
        ("output_completeness", "Output"),
        ("report_completeness", "Report"),
        ("observer_export_coverage", "Obs./Export"),
        ("replay_consistency_rate", "Replay"),
        ("all_pass_rate", "All-pass"),
    ]
    metric_matrix = [[float(row[key]) for key, _ in metric_keys] for row in summary]

    state_cmap = ListedColormap(["#E9B5AE", "#EAD08C", "#A8D8C4"])
    metric_cmap = LinearSegmentedColormap.from_list(
        "saga_reliability", ["#F7F4D6", "#CFE6D7", "#75B8B0", "#0F766E"]
    )

    fig = plt.figure(figsize=(13.6, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.45], wspace=0.34)

    ax_state = fig.add_subplot(gs[0, 0])
    ax_state.imshow(state_matrix, cmap=state_cmap, vmin=0, vmax=2, aspect="auto")
    ax_state.set_xticks(range(len(METHODS)))
    ax_state.set_xticklabels(["Direct\nscript", "LLM direct\ncall", "Full\nSAGA"], fontsize=10.5)
    ax_state.set_yticks(range(len(cases)))
    ax_state.set_yticklabels([case_labels.get(case.case_id, case.title) for case in cases], fontsize=10)
    ax_state.set_title("(a) Case-level execution state", fontsize=13.5, pad=9)
    add_heatmap_boundaries(ax_state, len(cases), len(METHODS), color="#F8FAFC", linewidth=0.9)
    state_text = {0: "FAIL", 1: "EXEC", 2: "FULL"}
    for y, row in enumerate(state_matrix):
        for x, value in enumerate(row):
            ax_state.text(x, y, state_text[value], ha="center", va="center", fontsize=9.5, color="#1F2933")
    ax_state.legend(
        handles=[
            Patch(facecolor="#E9B5AE", edgecolor="none", label="Failed"),
            Patch(facecolor="#EAD08C", edgecolor="none", label="Executed only"),
            Patch(facecolor="#A8D8C4", edgecolor="none", label="Full evidence"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=9.2,
    )

    ax_metric = fig.add_subplot(gs[0, 1])
    im = ax_metric.imshow(metric_matrix, cmap=metric_cmap, vmin=0, vmax=1, aspect="auto")
    ax_metric.set_xticks(range(len(metric_keys)))
    ax_metric.set_xticklabels([label for _, label in metric_keys], fontsize=10.5)
    ax_metric.set_yticks(range(len(summary)))
    ax_metric.set_yticklabels(method_titles, fontsize=10.5)
    ax_metric.set_title("(b) Aggregate reliability metrics", fontsize=13.5, pad=9)
    add_heatmap_boundaries(ax_metric, len(summary), len(metric_keys), color="#F8FAFC", linewidth=0.9)
    for y, row in enumerate(metric_matrix):
        for x, value in enumerate(row):
            color = "white" if value >= 0.72 else "#102A43"
            ax_metric.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=9.3, color=color)
    cbar = fig.colorbar(im, ax=ax_metric, fraction=0.035, pad=0.018)
    cbar.ax.tick_params(labelsize=8.8)
    cbar.outline.set_linewidth(0.6)

    save_figure(fig, figures_dir / "main_reliability_panel.png")
    plt.close(fig)


def plot_recipe_dag_example(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.4, 3.8))
    examples = [
        ("Full SAGA Recipe", ["inspect", "run_skill", "quality", "sar_artifact", "duplicate", "repair", "export"]),
        ("LLM Direct Tool Call", ["tool_call"]),
        ("Direct Skill Script", ["run_skill"]),
    ]
    colors = {
        "run_skill": "#64748B",
        "tool_call": "#C97064",
        "inspect": "#94A3B8",
        "quality": "#8FC6B5",
        "sar_artifact": "#8FC6B5",
        "duplicate": "#8FC6B5",
        "repair": "#E3B65A",
        "export": "#0F766E",
    }
    ax.set_xlim(-0.65, 6.85)
    ax.set_ylim(-0.8, 2.8)
    ax.axis("off")
    for row_idx, (title, steps) in enumerate(examples):
        y = 2 - row_idx
        ax.text(-0.58, y, title, ha="right", va="center", fontsize=12.2, color="#1F2933")
        for idx, step in enumerate(steps):
            ax.scatter(idx, y, s=390, color=colors.get(step, "#94A3B8"), edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(idx, y, short_step_label(step), ha="center", va="center", fontsize=7.9, color="white")
            if idx > 0:
                ax.annotate(
                    "",
                    xy=(idx - 0.24, y),
                    xytext=(idx - 0.76, y),
                    arrowprops={"arrowstyle": "->", "color": "#94A3B8", "linewidth": 1.5},
                )
    ax.text(0, 2.55, "Recipe structure compared across execution protocols", fontsize=15.0, ha="left")
    save_figure(fig, figures_dir / "recipe_dag_example.png")
    plt.close(fig)


def short_step_label(step: str) -> str:
    return {
        "run_skill": "Run",
        "tool_call": "Tool",
        "inspect": "Insp.",
        "quality": "Qual.",
        "sar_artifact": "SAR",
        "duplicate": "Dup.",
        "repair": "Rep.",
        "export": "Exp.",
    }.get(step, step)


def add_heatmap_boundaries(ax: Any, n_rows: int, n_cols: int, color: str = "white", linewidth: float = 1.0) -> None:
    ax.grid(False)
    ax.xaxis.grid(False, which="major")
    ax.yaxis.grid(False, which="major")
    ax.set_xticks([idx - 0.5 for idx in range(1, n_cols)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=color, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)


def save_figure(fig: Any, path: Path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}"


if __name__ == "__main__":
    main()

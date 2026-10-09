from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from saga.core.config import load_dataset_config, load_mapping, save_json, save_text
from saga.core.protocol import UNKNOWN
from saga.core.recipe import EXECUTION_REPORT_VERSION, SagaRecipe, RecipeStep
from saga.core.skill_report import attach_skill_run_report, summarize_skill_report_validations, validate_step_result_skill_report
from saga.data.discovery import is_image, iter_images
from saga.data.loader import load_samples
from saga.exporter import export_augmented_dataset
from saga.observer.distribution import evaluate_distribution
from saga.observer.quality import evaluate_output_directory
from saga.observer.repair_policy import build_repair_policy_report
from saga.observer.sar_artifacts import evaluate_sar_artifacts
from saga.skills.background_generation.skill import run_background_generation_skill
from saga.skills.diffusion_lora.skill import run_diffusion_lora_generation_skill
from saga.skills.style_transfer.skill import run_style_transfer_skill
from saga.skills.target_background_composition.skill import run_target_background_composition_skill
from saga.skills.classification_evaluation.skill import run_classification_evaluation_skill
from saga.skills.evaluation_utils.skill import run_duplicate_near_duplicate_skill, run_leakage_check_skill
from saga.skills.gan_generation.skill import run_gan_generation_skill
from saga.skills.geodiff_sar.skill import run_geodiff_sar_skill
from saga.skills.model_to_pov_scene.skill import run_model_to_pov_scene_compiler_skill
from saga.skills.pseudocolor.skill import run_pseudocolor_skill
from saga.skills.raysar.skill import run_raysar_synthesis_skill
from saga.skills.raysar_sweep.skill import run_raysar_sweep_synthesis_skill
from saga.skills.traditional_augmentation.skill import run_traditional_augmentation_skill


def execute_recipe(
    recipe_path: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    stop_on_error: bool = True,
) -> dict[str, Any]:
    recipe = SagaRecipe.from_path(recipe_path)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    step_results: dict[str, dict[str, Any]] = {}
    ordered_steps = topological_steps(recipe.pipeline)
    status = "succeeded"
    status_path = output_path / "recipe_execution_status.json"
    save_json(
        status_path,
        {
            "schema_version": EXECUTION_REPORT_VERSION,
            "recipe_path": Path(recipe_path).expanduser().resolve().as_posix(),
            "recipe_id": recipe.recipe_id,
            "task": recipe.task,
            "status": "running",
            "dry_run": dry_run,
            "started_at": wall_time(started),
            "updated_at": wall_time(),
            "current_step": None,
            "completed_step_count": 0,
            "total_step_count": len(ordered_steps),
            "steps": [],
        },
    )
    for step in ordered_steps:
        step_started = time.time()
        save_json(
            output_path / "steps" / f"{step.id}.started.json",
            {
                "step_id": step.id,
                "skill": step.skill,
                "status": "running",
                "dry_run": dry_run,
                "started_at": wall_time(step_started),
                "depends_on": step.depends_on,
                "params": step.params,
            },
        )
        save_json(
            status_path,
            {
                "schema_version": EXECUTION_REPORT_VERSION,
                "recipe_path": Path(recipe_path).expanduser().resolve().as_posix(),
                "recipe_id": recipe.recipe_id,
                "task": recipe.task,
                "status": "running",
                "dry_run": dry_run,
                "started_at": wall_time(started),
                "updated_at": wall_time(),
                "current_step": {"id": step.id, "skill": step.skill, "started_at": wall_time(step_started)},
                "completed_step_count": len(step_results),
                "total_step_count": len(ordered_steps),
                "steps": list(step_results.values()),
            },
        )
        missing = [dep for dep in step.depends_on if dep not in step_results]
        if missing:
            result = step_error(step, f"Missing dependencies: {missing}", dry_run=dry_run)
        else:
            try:
                result = execute_step(
                    step=step,
                    recipe=recipe,
                    output_dir=output_path,
                    dry_run=dry_run,
                    step_results=step_results,
                )
            except Exception as exc:
                result = step_error(step, f"{type(exc).__name__}: {exc}", dry_run=dry_run)
        step_results[step.id] = result
        save_json(output_path / "steps" / f"{step.id}.json", result)
        save_json(
            status_path,
            {
                "schema_version": EXECUTION_REPORT_VERSION,
                "recipe_path": Path(recipe_path).expanduser().resolve().as_posix(),
                "recipe_id": recipe.recipe_id,
                "task": recipe.task,
                "status": "running",
                "dry_run": dry_run,
                "started_at": wall_time(started),
                "updated_at": wall_time(),
                "current_step": None,
                "completed_step_count": len(step_results),
                "total_step_count": len(ordered_steps),
                "steps": list(step_results.values()),
            },
        )
        if result.get("status") == "failed":
            status = "failed"
            if stop_on_error:
                break
        if status == "succeeded" and result.get("status") in {"warning", "triggered", "blocked"}:
            status = "warning"

    skill_report_validation = summarize_skill_report_validations(list(step_results.values()))
    if status == "succeeded" and not skill_report_validation.get("valid"):
        status = "warning"
    report = {
        "schema_version": EXECUTION_REPORT_VERSION,
        "recipe_path": Path(recipe_path).expanduser().resolve().as_posix(),
        "recipe_id": recipe.recipe_id,
        "task": recipe.task,
        "status": status,
        "dry_run": dry_run,
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": output_path.as_posix(),
        "skill_report_validation": skill_report_validation,
        "steps": list(step_results.values()),
    }
    save_json(output_path / "recipe_execution.json", report)
    save_text(output_path / "recipe_execution.md", render_execution_markdown(report))
    save_json(
        status_path,
        {
            **report,
            "updated_at": wall_time(),
            "completed_at": wall_time(),
        },
    )
    return report


def execute_step(
    step: RecipeStep,
    recipe: SagaRecipe,
    output_dir: Path,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.time()
    previous_results = step_results or {}
    if step.skill == "DatasetProfileReportSkill":
        result = run_dataset_profile_report_step(step, dry_run=dry_run)
    elif step.skill == "FileSelectionSkill":
        result = run_file_selection_step(step, dry_run=dry_run)
    elif step.skill == "StyleTransferSkill":
        result = run_style_transfer_step(step, dry_run=dry_run)
    elif step.skill in {"DiffusionLoRAGenerationSkill", "SARLoRAGenerationSkill"}:
        result = run_diffusion_lora_generation_step(step, dry_run=dry_run)
    elif step.skill in {"TraditionalAugmentationSkill", "TraditionalAugSkill"}:
        result = run_traditional_augmentation_step(step, dry_run=dry_run)
    elif step.skill == "GANImageToImageSkill":
        result = run_gan_generation_step(step, dry_run=dry_run)
    elif step.skill == "GeoDiffSARSkill":
        result = run_geodiff_sar_step(step, dry_run=dry_run)
    elif step.skill == "GaussianSplattingCompletionSkill":
        result = run_gaussian_splatting_completion_step(step, dry_run=dry_run)
    elif step.skill == "PseudocolorSkill":
        result = run_pseudocolor_step(step, dry_run=dry_run)
    elif step.skill == "BackgroundGenerationSkill":
        result = run_background_generation_step(step, dry_run=dry_run)
    elif step.skill == "TargetBackgroundCompositionSkill":
        result = run_target_background_composition_step(step, dry_run=dry_run)
    elif step.skill == "ModelToPOVSceneCompilerSkill":
        result = run_model_to_pov_scene_compiler_step(step, dry_run=dry_run)
    elif step.skill == "RaySARSynthesisSkill":
        result = run_raysar_synthesis_step(step, dry_run=dry_run, step_results=previous_results)
    elif step.skill == "RaySARSweepSynthesisSkill":
        result = run_raysar_sweep_synthesis_step(step, dry_run=dry_run)
    elif step.skill in {"BasicOutputObserver", "QualityEvaluationSkill", "DataQualityEvaluationSkill"}:
        result = run_quality_evaluation_step(
            step=step,
            output_dir=output_dir,
            dry_run=dry_run,
            step_results=previous_results,
            legacy_observer=step.skill == "BasicOutputObserver",
        )
    elif step.skill in {"DistributionEvaluationSkill", "FIDDistributionEvaluationSkill"}:
        result = run_distribution_evaluation_step(
            step=step,
            output_dir=output_dir,
            dry_run=dry_run,
            step_results=previous_results,
        )
    elif step.skill in {"SARArtifactEvaluationSkill", "SARArtifactObserver"}:
        result = run_sar_artifact_evaluation_step(
            step=step,
            output_dir=output_dir,
            dry_run=dry_run,
            step_results=previous_results,
        )
    elif step.skill == "ClassificationEvaluationSkill":
        result = run_classification_evaluation_step(step, dry_run=dry_run, step_results=previous_results)
    elif step.skill == "LeakageCheckSkill":
        result = run_leakage_check_step(step=step, dry_run=dry_run, step_results=previous_results)
    elif step.skill == "DuplicateNearDuplicateSkill":
        result = run_duplicate_near_duplicate_step(step=step, dry_run=dry_run, step_results=previous_results)
    elif step.skill == "RepairPolicySkill":
        result = run_repair_policy_step(
            step=step,
            recipe=recipe,
            output_dir=output_dir,
            dry_run=dry_run,
            step_results=previous_results,
        )
    elif step.skill == "ExportDatasetSkill":
        result = run_export_dataset_step(
            step=step,
            recipe=recipe,
            dry_run=dry_run,
            step_results=previous_results,
        )
    else:
        result = {
            "status": "skipped" if dry_run else "failed",
            "message": f"Unsupported recipe skill: {step.skill}",
        }
    elapsed_seconds = round(time.time() - started, 3)
    result.update(
        {
            "step_id": step.id,
            "skill": step.skill,
            "dry_run": dry_run,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    attach_skill_run_report(
        result,
        step_id=step.id,
        skill=step.skill,
        dry_run=dry_run,
        elapsed_seconds=elapsed_seconds,
        params=step.params,
    )
    validation = validate_step_result_skill_report(result)
    result["skill_run_report_validation"] = validation
    if not validation.get("valid") and result.get("status") not in {"failed", "blocked"}:
        result.setdefault("warnings", [])
        if isinstance(result["warnings"], list):
            result["warnings"].append(
                {
                    "type": "skill_run_report_validation",
                    "message": "SkillRunReport failed structural validation.",
                    "issue_count": validation.get("issue_count", 0),
                }
            )
        result["status"] = "warning"
    return result


def run_dataset_profile_report_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    paths = {
        "dataset_profile_path": step.params.get("dataset_profile_path"),
        "validated_profile_path": step.params.get("validated_profile_path"),
    }
    existing = {key: value for key, value in paths.items() if value and Path(value).exists()}
    return {
        "status": "dry_run" if dry_run else "succeeded",
        "message": "Profile artifacts recorded.",
        "artifacts": existing,
        "params": step.params,
    }


def run_file_selection_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    content_root = Path(str(step.params["content_root"])).expanduser().resolve()
    output_dir = Path(str(step.params["output_dir"])).expanduser().resolve()
    filters = step.params.get("filters") or {}
    exclude_filters = step.params.get("exclude_filters") or {}
    dataset_config = step.params.get("dataset_config")
    fail_if_empty = bool(step.params.get("fail_if_empty", True))

    candidates = select_images_with_filters(
        content_root=content_root,
        dataset_config=Path(dataset_config).expanduser().resolve() if dataset_config else None,
        filters=filters,
        exclude_filters=exclude_filters,
    )
    if not candidates and fail_if_empty:
        return {
            "status": "failed",
            "message": "No content images matched filters.",
            "content_root": content_root.as_posix(),
            "filters": filters,
            "exclude_filters": exclude_filters,
            "selected_count": 0,
        }
    if dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dry_run": True,
            "content_root": content_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "filters": filters,
            "exclude_filters": exclude_filters,
            "selected_count": len(candidates),
            "sources": [path.as_posix() for path in candidates],
        }
        save_json(output_dir / "selection_manifest.json", manifest)
        return {
            "status": "dry_run",
            "message": "Would stage selected content images.",
            "content_root": content_root.as_posix(),
            "output_dir": output_dir.as_posix(),
            "filters": filters,
            "exclude_filters": exclude_filters,
            "selected_count": len(candidates),
            "examples": [path.as_posix() for path in candidates[:20]],
            "manifest": (output_dir / "selection_manifest.json").as_posix(),
        }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    used_names: set[str] = set()
    for path in candidates:
        name = unique_stage_name(path.name, used_names)
        target = output_dir / name
        target.symlink_to(path.resolve())
        staged.append(target)
    manifest = {
        "content_root": content_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "filters": filters,
        "exclude_filters": exclude_filters,
        "selected_count": len(candidates),
        "staged": [path.as_posix() for path in staged],
        "sources": [path.as_posix() for path in candidates],
    }
    save_json(output_dir / "selection_manifest.json", manifest)
    return {
        "status": "succeeded",
        "message": "Staged selected content images.",
        "content_root": content_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "filters": filters,
        "exclude_filters": exclude_filters,
        "selected_count": len(candidates),
        "examples": [path.as_posix() for path in candidates[:20]],
        "manifest": (output_dir / "selection_manifest.json").as_posix(),
    }


def run_style_transfer_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    content = step.params["content"]
    style = step.params["style"]
    output_dir = step.params["output_dir"]
    config = step.params.get("config")
    output_path = Path(output_dir).expanduser().resolve()
    if not dry_run and output_path.exists():
        shutil.rmtree(output_path)
    report = run_style_transfer_skill(
        content=content,
        style=style,
        output_dir=output_dir,
        config_path=config,
        skill_overrides=step.params.get("skill_overrides") or {},
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": "Style transfer skill completed." if not dry_run else "Style transfer dry-run command generated.",
        "skill_report": report,
        "output_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "image_count": report.get("output_count") or report.get("generated_count"),
        "expected_count": report.get("selected_count") or report.get("planned_count"),
        "artifacts": report.get("artifacts", {}),
    }


def run_diffusion_lora_generation_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    dataset_root = step.params["dataset_root"]
    output_dir = step.params["output_dir"]
    config = step.params.get("config")
    report = run_diffusion_lora_generation_skill(
        dataset_root=dataset_root,
        output_dir=output_dir,
        config_path=config,
        model_family=step.params.get("model_family"),
        target_count=step.params.get("target_count"),
        output_name=step.params.get("output_name"),
        prompt=step.params.get("prompt"),
        lora_weights=step.params.get("lora_weights"),
        dataset_config=step.params.get("dataset_config"),
        filters=step.params.get("filters") or {},
        exclude_filters=step.params.get("exclude_filters") or {},
        auto_caption_from_metadata=bool(step.params.get("auto_caption_from_metadata", False)),
        training_epochs=step.params.get("training_epochs"),
        skill_overrides=step.params.get("skill_overrides") or {},
        train=bool(step.params.get("train", True)),
        infer=bool(step.params.get("infer", True)),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Diffusion LoRA generation skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("target_count"),
        "expected_count": report.get("target_count"),
        "caption_profile": report.get("caption_profile"),
        "artifacts": report.get("artifacts", {}),
    }


def run_traditional_augmentation_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    dataset_root = step.params["dataset_root"]
    output_dir = step.params["output_dir"]
    report = run_traditional_augmentation_skill(
        dataset_root=dataset_root,
        output_dir=output_dir,
        config_path=step.params.get("config"),
        dataset_config=step.params.get("dataset_config"),
        filters=step.params.get("filters") or {},
        exclude_filters=step.params.get("exclude_filters") or {},
        target_count=step.params.get("target_count"),
        multiplier=step.params.get("multiplier"),
        skill_overrides=step.params.get("skill_overrides") or {},
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Traditional augmentation skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("generated_count") if not dry_run else report.get("planned_count"),
        "expected_count": report.get("planned_count"),
        "artifacts": report.get("artifacts", {}),
    }


def run_gan_generation_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    dataset_root = step.params["dataset_root"]
    output_dir = step.params["output_dir"]
    report = run_gan_generation_skill(
        dataset_root=dataset_root,
        output_dir=output_dir,
        project_dir=step.params.get("project_dir"),
        variant=str(step.params.get("variant") or "gpu"),
        target_count=int(step.params.get("target_count") or 100),
        epochs=step.params.get("epochs") or step.params.get("training_epochs"),
        model_path=step.params.get("model_path"),
        train=bool(step.params.get("train", True)),
        infer=bool(step.params.get("infer", True)),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "GAN generation skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "reference_dir": Path(str(dataset_root)).expanduser().resolve().as_posix(),
        "image_count": report.get("target_count"),
        "expected_count": report.get("target_count"),
        "artifacts": report.get("artifacts", {}),
    }


def run_geodiff_sar_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    report = run_geodiff_sar_skill(
        output_dir=output_dir,
        config_path=step.params.get("config"),
        dataset_root=step.params.get("dataset_root"),
        model_file=step.params.get("model_file"),
        target_azimuths=step.params.get("target_azimuths"),
        azimuth_sweep=step.params.get("azimuth_sweep"),
        depressions=step.params.get("depressions"),
        target_count=step.params.get("target_count"),
        training_epochs=step.params.get("training_epochs"),
        controlnet_weights=step.params.get("controlnet_weights"),
        controlnet_init_weights=step.params.get("controlnet_init_weights"),
        prompt=step.params.get("prompt"),
        model_name=step.params.get("model_name"),
        train=bool(step.params.get("train", True)),
        infer=bool(step.params.get("infer", True)),
        extract_real_gefm=bool(step.params.get("extract_real_gefm", True)),
        render_gefm=bool(step.params.get("render_gefm", True)),
        cuda=step.params.get("cuda"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "GeoDiff-SAR skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": report.get("output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("image_count") if not dry_run else report.get("expected_count"),
        "expected_count": report.get("expected_count"),
        "controlnet_weights": report.get("controlnet_weights") or report.get("expected_controlnet_weights"),
        "metrics": report.get("metrics", {}),
        "warnings": report.get("warnings", []),
        "issues": report.get("issues", []),
        "artifacts": report.get("artifacts", {}),
    }


def run_gaussian_splatting_completion_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = Path(str(step.params["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered_views"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(str(step.params.get("dataset_root") or ".")).expanduser().resolve()
    project_root = Path(str(step.params.get("project_root") or "myproject/SAR GS V1")).expanduser().resolve()
    target_azimuths = step.params.get("target_azimuths") or []
    target_count = int(step.params.get("target_count") or (len(target_azimuths) if isinstance(target_azimuths, list) else 0) or 20)
    render_resolution = int(step.params.get("render_resolution") or 128)
    iterations = int(step.params.get("iterations") or 1000)
    train_script = project_root / "train_sar.py"
    view_script = project_root / "view_result.py"
    train_command = (
        f"python {train_script.as_posix()} --data {dataset_root.as_posix()} "
        f"--output {output_dir.as_posix()} --iterations {iterations}"
    )
    render_command = (
        f"python {view_script.as_posix()} --model {(output_dir / 'gaussians_final.npz').as_posix()} "
        f"--output {rendered_dir.as_posix()} --resolution {render_resolution} --count {target_count}"
    )
    status = "dry_run" if dry_run else "blocked"
    message = (
        "Gaussian splatting completion dry-run command artifacts generated."
        if dry_run
        else "Gaussian splatting real execution requires the external SAR GS runtime and is intentionally blocked here."
    )
    report = {
        "schema_version": "saga_skill_run_report_v1",
        "skill": "GaussianSplattingCompletionSkill",
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "dataset_root": dataset_root.as_posix(),
        "project_root": project_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "generated_output_dir": rendered_dir.as_posix(),
        "target_count": target_count,
        "expected_count": target_count,
        "target_azimuths": target_azimuths,
        "render_resolution": render_resolution,
        "iterations": iterations,
        "commands": {
            "train": train_command,
            "render": render_command,
        },
        "artifacts": {
            "run_report": (output_dir / "gaussian_splatting_run.json").as_posix(),
            "train_command": (output_dir / "train_command.sh").as_posix(),
            "render_command": (output_dir / "render_command.sh").as_posix(),
            "rendered_views": rendered_dir.as_posix(),
        },
        "warnings": [] if project_root.exists() else [{"type": "missing_project_root", "path": project_root.as_posix()}],
    }
    save_json(output_dir / "gaussian_splatting_run.json", report)
    save_text(output_dir / "gaussian_splatting_run.md", render_gaussian_splatting_markdown(report))
    save_text(output_dir / "train_command.sh", train_command + "\n")
    save_text(output_dir / "render_command.sh", render_command + "\n")
    return {
        "status": status,
        "message": message,
        "skill_report": report,
        "output_dir": output_dir.as_posix(),
        "run_dir": output_dir.as_posix(),
        "generated_output_dir": rendered_dir.as_posix(),
        "image_count": target_count if dry_run else 0,
        "expected_count": target_count,
        "artifacts": report["artifacts"],
        "warnings": report["warnings"],
    }


def run_pseudocolor_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    input_dir = step.params["input_dir"]
    output_dir = step.params["output_dir"]
    report = run_pseudocolor_skill(
        input_dir=input_dir,
        output_dir=output_dir,
        colormap=str(step.params.get("colormap") or "sar"),
        preserve_tree=bool(step.params.get("preserve_tree", True)),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Pseudocolor skill completed."),
        "skill_report": report,
        "output_dir": report.get("pseudocolor_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("pseudocolor_output_dir"),
        "image_count": report.get("materialized_count") if not dry_run else report.get("planned_count"),
        "expected_count": report.get("planned_count"),
        "artifacts": report.get("artifacts", {}),
    }


def run_background_generation_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    report = run_background_generation_skill(
        scene_prompt=str(step.params.get("scene_prompt") or "SAR background scene"),
        output_dir=output_dir,
        config_path=step.params.get("config"),
        project_dir=step.params.get("project_dir"),
        num_images=step.params.get("num_images") or step.params.get("target_count"),
        split=step.params.get("split"),
        top_k=step.params.get("top_k"),
        selection_seed=step.params.get("selection_seed"),
        seed=step.params.get("seed"),
        cns=step.params.get("cns"),
        steps=step.params.get("steps"),
        scale=step.params.get("scale"),
        width=step.params.get("width"),
        height=step.params.get("height"),
        cuda=step.params.get("cuda"),
        cpu_threads=step.params.get("cpu_threads"),
        extra_prompt=step.params.get("extra_prompt"),
        clear_cache_each_image=step.params.get("clear_cache_each_image"),
        show_top=step.params.get("show_top"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Background generation skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": report.get("run_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("generated_count") if not dry_run else report.get("target_count"),
        "expected_count": report.get("target_count"),
        "scene_prompt": report.get("scene_prompt"),
        "artifacts": report.get("artifacts", {}),
    }


def run_target_background_composition_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    report = run_target_background_composition_skill(
        target_dir=step.params["target_dir"],
        background_dir=step.params["background_dir"],
        output_dir=output_dir,
        config_path=step.params.get("config"),
        mask_dir=step.params.get("mask_dir"),
        target_count=step.params.get("target_count"),
        blend_mode=step.params.get("blend_mode"),
        mask_mode=step.params.get("mask_mode"),
        placement_policy=step.params.get("placement_policy"),
        skill_overrides=step.params.get("skill_overrides") or {},
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Target/background composition skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("generated_count") if not dry_run else report.get("target_count"),
        "expected_count": report.get("target_count"),
        "reference_dir": Path(str(step.params["background_dir"])).expanduser().resolve().as_posix(),
        "artifacts": report.get("artifacts", {}),
    }


def run_model_to_pov_scene_compiler_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    report = run_model_to_pov_scene_compiler_skill(
        model_file=step.params["model_file"],
        output_dir=output_dir,
        config_path=step.params.get("config"),
        incidence_angle_deg=step.params.get("incidence_angle_deg"),
        depression_angle_deg=step.params.get("depression_angle_deg"),
        azimuth_deg=step.params.get("azimuth_deg"),
        pitch_deg=step.params.get("pitch_deg"),
        roll_deg=step.params.get("roll_deg"),
        target_extent_m=step.params.get("target_extent_m"),
        scale_factor=step.params.get("scale_factor"),
        input_up_axis=step.params.get("input_up_axis"),
        pov_height_axis=step.params.get("pov_height_axis"),
        range_distance_m=step.params.get("range_distance_m"),
        sensor_plane_m=step.params.get("sensor_plane_m"),
        scene_center=step.params.get("scene_center"),
        add_ground=step.params.get("add_ground"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "Model-to-POV scene compiler completed."),
        "skill_report": report,
        "model_file": report.get("model_file"),
        "compiled_scene": report.get("compiled_scene"),
        "pov_scene": report.get("compiled_scene"),
        "output_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "metrics": report.get("metrics", {}),
        "warnings": report.get("warnings", []),
        "artifacts": report.get("artifacts", {}),
    }


def run_raysar_synthesis_step(
    step: RecipeStep,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    pov_scene = step.params.get("pov_scene")
    if step.params.get("pov_scene_from_step"):
        previous = (step_results or {}).get(str(step.params["pov_scene_from_step"])) or {}
        pov_scene = previous.get("compiled_scene") or previous.get("pov_scene") or pov_scene
        if not pov_scene and isinstance(previous.get("skill_report"), dict):
            pov_scene = previous["skill_report"].get("compiled_scene")
    report = run_raysar_synthesis_skill(
        output_dir=output_dir,
        pov_scene=pov_scene,
        parameters_file=step.params.get("parameters_file"),
        simulation_parameters=step.params.get("simulation_parameters") or {},
        contributions_txt=step.params.get("contributions_txt"),
        adapted_povray=step.params.get("adapted_povray"),
        width=step.params.get("width"),
        height=step.params.get("height"),
        postprocess=step.params.get("postprocess"),
        render=bool(step.params.get("render", True)),
        auto_fix_scene=step.params.get("auto_fix_scene"),
        sar_intersection=step.params.get("sar_intersection"),
        build_if_missing=step.params.get("build_if_missing"),
        config_path=step.params.get("config"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "RaySAR synthesis skill completed."),
        "skill_report": report,
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": report.get("run_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("map_count") or report.get("image_count"),
        "expected_count": 3 if report.get("postprocess") else None,
        "metrics": report.get("metrics", {}),
        "warnings": report.get("warnings", []),
        "artifacts": report.get("artifacts", {}),
    }


def run_raysar_sweep_synthesis_step(step: RecipeStep, dry_run: bool) -> dict[str, Any]:
    output_dir = step.params["output_dir"]
    report = run_raysar_sweep_synthesis_skill(
        model_file=step.params["model_file"],
        output_dir=output_dir,
        config_path=step.params.get("config"),
        azimuth_values=step.params.get("azimuth_values"),
        azimuth_sweep=step.params.get("azimuth_sweep"),
        incidence_angle_deg=step.params.get("incidence_angle_deg"),
        depression_angle_deg=step.params.get("depression_angle_deg"),
        target_extent_m=step.params.get("target_extent_m"),
        scale_factor=step.params.get("scale_factor"),
        input_up_axis=step.params.get("input_up_axis"),
        pov_height_axis=step.params.get("pov_height_axis"),
        range_distance_m=step.params.get("range_distance_m"),
        sensor_plane_m=step.params.get("sensor_plane_m"),
        parameters_file=step.params.get("parameters_file"),
        adapted_povray=step.params.get("adapted_povray"),
        width=step.params.get("width"),
        height=step.params.get("height"),
        render=step.params.get("render"),
        postprocess=step.params.get("postprocess"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status", "unknown"),
        "message": report.get("message", "RaySAR sweep synthesis skill completed."),
        "skill_report": report,
        "model_file": report.get("model_file"),
        "output_dir": report.get("generated_output_dir") or Path(output_dir).expanduser().resolve().as_posix(),
        "run_dir": Path(output_dir).expanduser().resolve().as_posix(),
        "generated_output_dir": report.get("generated_output_dir"),
        "image_count": report.get("map_count") or report.get("image_count"),
        "expected_count": report.get("expected_count"),
        "view_count": report.get("view_count"),
        "metrics": report.get("metrics", {}),
        "warnings": report.get("warnings", []),
        "artifacts": report.get("artifacts", {}),
    }


def run_quality_evaluation_step(
    step: RecipeStep,
    output_dir: Path,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
    legacy_observer: bool = False,
) -> dict[str, Any]:
    input_dir = Path(str(step.params["input_dir"])).expanduser().resolve()
    expected_count = resolve_expected_count(step.params, step_results)
    report_dir = Path(str(step.params.get("report_dir") or output_dir / "observer" / step.id)).expanduser().resolve()
    report = evaluate_output_directory(
        input_dir=input_dir,
        output_dir=report_dir,
        expected_count=expected_count,
        dry_run=dry_run,
        sample_limit=int(step.params.get("sample_limit", 100)),
        thresholds=step.params.get("thresholds") or {},
    )
    if legacy_observer:
        report["message"] = "Observed output directory." if not dry_run else "Would observe output directory after execution."
    return {
        "status": normalize_evaluator_status(report.get("status")),
        "message": report.get("message", "Quality evaluation completed."),
        "input_dir": input_dir.as_posix(),
        "report_dir": report_dir.as_posix(),
        "expected_count": expected_count,
        "image_count": report.get("image_count", 0),
        "existing_image_count": report.get("existing_image_count"),
        "triggers": report.get("triggers", []),
        "quality_report": report,
        "artifacts": {
            "quality_evaluation_json": (report_dir / "quality_evaluation.json").as_posix(),
            "quality_evaluation_md": (report_dir / "quality_evaluation.md").as_posix(),
        },
    }


def run_distribution_evaluation_step(
    step: RecipeStep,
    output_dir: Path,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    generated_dir = resolve_path_from_step_result(
        params=step.params,
        step_results=step_results,
        from_key="generated_from_step",
        result_key="generated_output_dir",
        fallback=step.params.get("generated_dir") or step.params.get("input_dir"),
    )
    reference_dir = resolve_reference_dir(step.params, step_results)
    report_dir = Path(str(step.params.get("report_dir") or output_dir / "observer" / step.id)).expanduser().resolve()
    report = evaluate_distribution(
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        output_dir=report_dir,
        dry_run=dry_run,
        reference_sample_limit=int(step.params.get("reference_sample_limit", 200)),
        generated_sample_limit=int(step.params.get("generated_sample_limit", 50)),
        image_size=int(step.params.get("image_size", 64)),
        standard_fid=bool(step.params.get("standard_fid", True)),
        standard_fid_dims=int(step.params.get("standard_fid_dims", 2048)),
        standard_fid_batch_size=int(step.params.get("standard_fid_batch_size", 16)),
        thresholds=step.params.get("thresholds") or {},
    )
    return {
        "status": normalize_evaluator_status(report.get("status")),
        "message": report.get("message", "Distribution evaluation completed."),
        "reference_dir": Path(str(reference_dir)).expanduser().resolve().as_posix(),
        "generated_dir": Path(str(generated_dir)).expanduser().resolve().as_posix(),
        "report_dir": report_dir.as_posix(),
        "metrics": report.get("metrics", {}),
        "triggers": report.get("triggers", []),
        "distribution_report": report,
        "artifacts": {
            "distribution_evaluation_json": (report_dir / "distribution_evaluation.json").as_posix(),
            "distribution_evaluation_md": (report_dir / "distribution_evaluation.md").as_posix(),
        },
    }


def run_sar_artifact_evaluation_step(
    step: RecipeStep,
    output_dir: Path,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_dir = resolve_path_from_step_result(
        params=step.params,
        step_results=step_results,
        from_key="input_from_step",
        result_key="generated_output_dir",
        fallback=step.params.get("input_dir"),
    )
    report_dir = Path(str(step.params.get("report_dir") or output_dir / "observer" / step.id)).expanduser().resolve()
    report = evaluate_sar_artifacts(
        input_dir=input_dir,
        output_dir=report_dir,
        dry_run=dry_run,
        sample_limit=int(step.params.get("sample_limit", 100)),
        image_size=int(step.params.get("image_size", 128)),
        thresholds=step.params.get("thresholds") or {},
    )
    return {
        "status": normalize_evaluator_status(report.get("status")),
        "message": report.get("message", "SAR artifact evaluation completed."),
        "input_dir": Path(str(input_dir)).expanduser().resolve().as_posix(),
        "report_dir": report_dir.as_posix(),
        "triggers": report.get("triggers", []),
        "sar_artifact_report": report,
        "artifacts": {
            "sar_artifact_evaluation_json": (report_dir / "sar_artifact_evaluation.json").as_posix(),
            "sar_artifact_evaluation_md": (report_dir / "sar_artifact_evaluation.md").as_posix(),
        },
    }


def run_classification_evaluation_step(
    step: RecipeStep,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    augmented_dataset = step.params.get("augmented_dataset")
    if step.params.get("augmented_from_step"):
        previous = step_results.get(str(step.params["augmented_from_step"])) or {}
        augmented_dataset = previous.get("output_dir") or previous.get("source_dir") or augmented_dataset
    report = run_classification_evaluation_skill(
        baseline_dataset=step.params["baseline_dataset"],
        augmented_dataset=augmented_dataset,
        val_dataset=step.params.get("val_dataset"),
        baseline_dataset_config=step.params.get("baseline_dataset_config"),
        augmented_dataset_config=step.params.get("augmented_dataset_config"),
        val_dataset_config=step.params.get("val_dataset_config"),
        output_dir=step.params["output_dir"],
        config_path=step.params.get("config"),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status"),
        "message": report.get("message"),
        "skill_report": report,
        "evaluator_report": report.get("evaluator_report"),
        "output_dir": Path(str(step.params["output_dir"])).expanduser().resolve().as_posix(),
        "metrics": report.get("metrics", {}),
        "artifacts": report.get("artifacts", {}),
    }


def run_leakage_check_step(
    step: RecipeStep,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    augmented_dataset = step.params.get("augmented_dataset")
    if step.params.get("augmented_from_step"):
        previous = step_results.get(str(step.params["augmented_from_step"])) or {}
        augmented_dataset = previous.get("output_dir") or previous.get("source_dir") or augmented_dataset
    report = run_leakage_check_skill(
        baseline_dataset=step.params["baseline_dataset"],
        augmented_dataset=augmented_dataset,
        val_dataset=step.params.get("val_dataset"),
        output_dir=step.params["output_dir"],
        sample_limit=int(step.params.get("sample_limit", 20000)),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status"),
        "message": report.get("message"),
        "baseline_dataset": Path(str(step.params["baseline_dataset"])).expanduser().resolve().as_posix(),
        "augmented_dataset": Path(str(augmented_dataset)).expanduser().resolve().as_posix() if augmented_dataset else None,
        "val_dataset": Path(str(step.params["val_dataset"])).expanduser().resolve().as_posix()
        if step.params.get("val_dataset")
        else None,
        "output_dir": Path(str(step.params["output_dir"])).expanduser().resolve().as_posix(),
        "trigger_count": report.get("trigger_count"),
        "overlaps": report.get("overlaps", []),
        "leakage_report": report,
        "artifacts": report.get("artifacts", {}),
    }


def run_duplicate_near_duplicate_step(
    step: RecipeStep,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    input_dir = step.params.get("input_dir")
    if step.params.get("input_from_step"):
        previous = step_results.get(str(step.params["input_from_step"])) or {}
        input_dir = previous.get("output_dir") or previous.get("generated_output_dir") or previous.get("source_dir") or input_dir
    report = run_duplicate_near_duplicate_skill(
        input_dir=input_dir,
        output_dir=step.params["output_dir"],
        hash_size=int(step.params.get("hash_size", 16)),
        hamming_threshold=int(step.params.get("hamming_threshold", 6)),
        sample_limit=int(step.params.get("sample_limit", 5000)),
        dry_run=dry_run,
    )
    return {
        "status": report.get("status"),
        "message": report.get("message"),
        "input_dir": Path(str(input_dir)).expanduser().resolve().as_posix(),
        "output_dir": Path(str(step.params["output_dir"])).expanduser().resolve().as_posix(),
        "pair_count": report.get("pair_count"),
        "duplicate_report": report,
        "artifacts": report.get("artifacts", {}),
    }


def wall_time(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or time.time()))


def resolve_reference_dir(params: dict[str, Any], step_results: dict[str, dict[str, Any]]) -> str:
    if params.get("reference_dir"):
        return str(params["reference_dir"])
    step_id = params.get("reference_from_step")
    if step_id:
        previous = step_results.get(str(step_id)) or {}
        skill_report = previous.get("skill_report") or {}
        inputs = skill_report.get("inputs") or {}
        for key in ("prepared_dataset", "dataset_root"):
            if inputs.get(key):
                return str(inputs[key])
        for key in ("reference_dir", "dataset_root"):
            if previous.get(key):
                return str(previous[key])
            if skill_report.get(key):
                return str(skill_report[key])
        caption_profile = previous.get("caption_profile") or skill_report.get("caption_profile") or {}
        if caption_profile.get("dataset_root"):
            return str(caption_profile["dataset_root"])
    raise ValueError("Distribution evaluation needs reference_dir or reference_from_step.")


def run_repair_policy_step(
    step: RecipeStep,
    recipe: SagaRecipe,
    output_dir: Path,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evaluator_step = str(step.params.get("evaluator_step") or step.params.get("evaluation_step") or "")
    evaluation_result = step_results.get(evaluator_step, {}) if evaluator_step else {}
    evaluation = evaluation_result.get("quality_report") or evaluation_result
    distribution_step = str(step.params.get("distribution_step") or "")
    if distribution_step:
        distribution_result = step_results.get(distribution_step, {})
        distribution = distribution_result.get("distribution_report") or distribution_result
        evaluation = merge_evaluations(evaluation, distribution)
    sar_artifact_step = str(step.params.get("sar_artifact_step") or "")
    if sar_artifact_step:
        sar_artifact_result = step_results.get(sar_artifact_step, {})
        sar_artifact = sar_artifact_result.get("sar_artifact_report") or sar_artifact_result
        evaluation = merge_evaluations(evaluation, sar_artifact)
    report_dir = Path(str(step.params.get("report_dir") or output_dir / "observer" / step.id)).expanduser().resolve()
    report = build_repair_policy_report(
        evaluation=evaluation,
        output_dir=report_dir,
        max_trials=int(step.params.get("max_trials", 3)),
        auto_rerun=bool(step.params.get("auto_rerun", False)),
        dry_run=dry_run,
        recipe=recipe,
        step_results=step_results,
        trial_index=int(step.params.get("trial_index", 0)),
    )
    return {
        "status": report.get("status"),
        "message": report.get("reason", "Repair policy completed."),
        "report_dir": report_dir.as_posix(),
        "trigger_count": len(report.get("triggers") or []),
        "suggested_action_count": len(report.get("suggested_actions") or []),
        "repair_policy": report,
        "artifacts": {
            "repair_policy_json": (report_dir / "repair_policy.json").as_posix(),
            "repair_policy_md": (report_dir / "repair_policy.md").as_posix(),
            "parameter_repair_plan_json": (
                report.get("parameter_repair", {}).get("artifacts", {}) or {}
            ).get("parameter_repair_plan_json"),
            "revised_recipe_yaml": (
                report.get("parameter_repair", {}).get("artifacts", {}) or {}
            ).get("revised_recipe_yaml"),
        },
    }


def merge_evaluations(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    if not secondary:
        return primary
    merged = dict(primary or {})
    triggers = list((primary or {}).get("triggers") or [])
    triggers.extend((secondary or {}).get("triggers") or [])
    merged["triggers"] = triggers
    merged["secondary_evaluation"] = secondary
    return merged


def run_export_dataset_step(
    step: RecipeStep,
    recipe: SagaRecipe,
    dry_run: bool,
    step_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_dir = resolve_path_from_step_result(
        params=step.params,
        step_results=step_results,
        from_key="source_from_step",
        result_key="output_dir",
        fallback=step.params.get("source_dir"),
    )
    output_dir = Path(str(step.params["output_dir"])).expanduser().resolve()
    selection_manifest = None
    if step.params.get("selection_from_step") or step.params.get("selection_manifest"):
        selection_manifest = resolve_path_from_step_result(
            params=step.params,
            step_results=step_results,
            from_key="selection_from_step",
            result_key="manifest",
            fallback=step.params.get("selection_manifest"),
        )
    quality_report = resolve_report_from_step_result(
        params=step.params,
        step_results=step_results,
        from_key="quality_from_step",
        result_key="quality_report",
    )
    repair_policy = resolve_report_from_step_result(
        params=step.params,
        step_results=step_results,
        from_key="repair_from_step",
        result_key="repair_policy",
    )
    report = export_augmented_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        recipe_id=recipe.recipe_id,
        task=recipe.task,
        dry_run=dry_run,
        mode=str(step.params.get("mode", "copy")),
        selection_manifest=selection_manifest,
        quality_report=quality_report,
        repair_policy=repair_policy,
        include_originals=bool(step.params.get("include_originals", False)),
        require_quality_pass=bool(step.params.get("require_quality_pass", False)),
        max_items=step.params.get("max_items"),
        write_caption_sidecars=bool(step.params.get("write_caption_sidecars", False)),
    )
    return {
        "status": report.get("status"),
        "message": report.get("message"),
        "output_dir": output_dir.as_posix(),
        "source_dir": Path(str(source_dir)).expanduser().resolve().as_posix(),
        "generated_image_count": report.get("generated_image_count"),
        "manifest_count": report.get("manifest_count"),
        "caption_sidecar_count": report.get("caption_sidecar_count"),
        "export_report": report,
        "artifacts": report.get("artifacts", {}),
    }


def resolve_path_from_step_result(
    params: dict[str, Any],
    step_results: dict[str, dict[str, Any]],
    from_key: str,
    result_key: str,
    fallback: Any = None,
) -> str:
    step_id = params.get(from_key)
    if step_id:
        value = (step_results.get(str(step_id)) or {}).get(result_key)
        if value:
            return str(value)
    if fallback:
        return str(fallback)
    raise ValueError(f"Cannot resolve path for {from_key}/{result_key}")


def resolve_report_from_step_result(
    params: dict[str, Any],
    step_results: dict[str, dict[str, Any]],
    from_key: str,
    result_key: str,
) -> dict[str, Any] | None:
    step_id = params.get(from_key)
    if not step_id:
        return None
    value = (step_results.get(str(step_id)) or {}).get(result_key)
    return value if isinstance(value, dict) else None


def resolve_expected_count(params: dict[str, Any], step_results: dict[str, dict[str, Any]]) -> int | None:
    if params.get("expected_count") is not None:
        return int(params["expected_count"])
    expected_from_step = params.get("expected_from_step")
    if not expected_from_step:
        return None
    previous = step_results.get(str(expected_from_step)) or {}
    value = previous.get("selected_count") or previous.get("expected_count") or previous.get("image_count")
    return int(value) if value is not None else None


def normalize_evaluator_status(status: Any) -> str:
    if status == "passed":
        return "succeeded"
    return str(status or "unknown")


def select_images_with_filters(
    content_root: Path,
    dataset_config: Path | None,
    filters: dict[str, Any],
    exclude_filters: dict[str, Any] | None = None,
) -> list[Path]:
    if not content_root.exists():
        raise FileNotFoundError(f"Content root does not exist: {content_root}")
    if dataset_config and dataset_config.exists():
        config = load_dataset_config(dataset_config)
        samples = load_samples(config, read_image_size=False)
        selected = []
        for sample in samples:
            path = Path(sample.image.path).expanduser().resolve()
            if not is_under(path, content_root):
                continue
            if sample_matches_filters(sample.metadata, filters) and not sample_matches_any_filter(sample.metadata, exclude_filters or {}):
                selected.append(path)
        if selected or filters or exclude_filters:
            return selected
        # Style/content workflows may profile the style dataset first while the
        # content root is a separate user path. With no metadata filters, an
        # empty config-based selection should fall back to raw content images.
        return list(iter_images(content_root))
    return [
        path
        for path in iter_images(content_root)
        if sample_matches_filters({}, filters) and not sample_matches_any_filter({}, exclude_filters or {})
    ]


def sample_matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = metadata.get(field, UNKNOWN)
        if not values_equal(actual, expected):
            return False
    return True


def sample_matches_any_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = metadata.get(field, UNKNOWN)
        if values_equal(actual, expected):
            return True
    return False


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(values_equal(actual, item) for item in expected)
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except (TypeError, ValueError):
        return str(actual).lower() == str(expected).lower()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def unique_stage_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    idx = 2
    while f"{stem}_{idx}{suffix}" in used:
        idx += 1
    value = f"{stem}_{idx}{suffix}"
    used.add(value)
    return value


def topological_steps(steps: list[RecipeStep]) -> list[RecipeStep]:
    ordered: list[RecipeStep] = []
    remaining = list(steps)
    done: set[str] = set()
    while remaining:
        progressed = False
        for step in list(remaining):
            if all(dep in done for dep in step.depends_on):
                ordered.append(step)
                remaining.remove(step)
                done.add(step.id)
                progressed = True
        if not progressed:
            cycle = [step.id for step in remaining]
            raise ValueError(f"Recipe dependency cycle or missing dependency: {cycle}")
    return ordered


def step_error(step: RecipeStep, message: str, dry_run: bool) -> dict[str, Any]:
    result = {
        "step_id": step.id,
        "skill": step.skill,
        "status": "failed",
        "dry_run": dry_run,
        "message": message,
    }
    attach_skill_run_report(
        result,
        step_id=step.id,
        skill=step.skill,
        dry_run=dry_run,
        elapsed_seconds=None,
        params=step.params,
    )
    result["skill_run_report_validation"] = validate_step_result_skill_report(result)
    return result


def render_gaussian_splatting_markdown(report: dict[str, Any]) -> str:
    commands = report.get("commands") or {}
    lines = [
        "# Gaussian Splatting Completion",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Dataset: `{report.get('dataset_root')}`",
        f"- Project root: `{report.get('project_root')}`",
        f"- Target count: {report.get('target_count')}",
        f"- Render resolution: {report.get('render_resolution')}",
        "",
        "## Commands",
        "",
        f"- Train: `{commands.get('train', '')}`",
        f"- Render: `{commands.get('render', '')}`",
        "",
    ]
    return "\n".join(lines)


def render_execution_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Recipe Execution",
        "",
        f"- Recipe: `{report['recipe_id']}`",
        f"- Task: `{report['task']}`",
        f"- Status: {report['status']}",
        f"- Dry run: {report['dry_run']}",
        f"- Output dir: `{report['output_dir']}`",
        f"- Elapsed seconds: {report['elapsed_seconds']}",
        "",
        "## Steps",
        "",
    ]
    for step in report.get("steps", []):
        lines.append(
            f"- `{step.get('step_id')}` {step.get('skill')} -> {step.get('status')}: {step.get('message', '')}"
        )
        if "selected_count" in step:
            lines.append(f"  selected_count: {step['selected_count']}")
        if "image_count" in step:
            lines.append(f"  image_count: {step['image_count']}")
        if "trigger_count" in step:
            lines.append(f"  trigger_count: {step['trigger_count']}")
        if "pair_count" in step:
            lines.append(f"  pair_count: {step['pair_count']}")
    validation = report.get("skill_report_validation") or {}
    if validation:
        lines.extend(
            [
                "",
                "## SkillRunReport Validation",
                "",
                f"- Valid: {validation.get('valid')}",
                f"- Invalid steps: {validation.get('invalid_step_count')}",
                f"- Severity counts: `{validation.get('severity_counts')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)

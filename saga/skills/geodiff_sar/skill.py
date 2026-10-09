from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text


REPO_ROOT = Path(__file__).resolve().parents[3]
GEODIFF_SAR_RUN_VERSION = "saga_geodiff_sar_run_v1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class GeoDiffSARSkillConfig:
    name: str = "GeoDiffSARSkill"
    description: str = (
        "GeoDiff-SAR composite skill: real-image GEFM extraction, FLUX ControlNet training, "
        "3D-model GEFM rendering, and ControlNet inference."
    )
    conda_env: str = "sd3"
    qinglong_project_dir: str = "myproject/qinglong_trainer_29b"
    preprocess_dir: str = "myproject/geodiff_components/preprocess3"
    raytracing_dir: str = "myproject/geodiff_components/raytracing"
    real_gefm_script: str = "generate_training_data_for_folder.py"
    raytracing_batch_script: str = "batch_generate_gecm.py"
    controlnet_train_script: str = "sd-scripts/flux_train_control_net.py"
    controlnet_inference_script: str = "inference/flux_controlnet_inference.py"
    output_name_prefix: str = "saga_geodiff_controlnet"
    generated_output_name: str = "saga_geodiff"
    default_model_name: str = "target"
    max_scattering_points: int = 10
    pose_method: str = "azimuth"
    raytracing_jobs: int = 1
    raytracing_pipeline_mode: str = "pure3d"
    raytracing_pol: str = "all"
    raytracing_image_size: int = 900
    raytracing_only_skeleton_png: bool = True
    raytracing_strong_method: str = "raytrace"
    raytracing_strong_top_k: int = 10
    raytracing_strong_hard_max: int = 10
    raytracing_strong_threshold_db: float = -9.5
    raytracing_strong_nms_bins: int = 44
    raytracing_extra_args: list[str] = field(default_factory=list)
    target_count: int = 20
    training_epochs: int = 25
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    save_every_n_epochs: int = 5
    learning_rate: str = "1e-4"
    seed: int = 1026
    mixed_precision: str = "bf16"
    save_precision: str = "bf16"
    caption_extension: str = ".txt"
    vae_batch_size: int = 4
    resolution: str = "512,512"
    max_data_loader_n_workers: int = 8
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    persistent_data_loader_workers: bool = True
    gradient_checkpointing: bool = False
    full_bf16: bool = False
    fp8_base: bool = False
    fp8_base_unet: bool = False
    blocks_to_swap: int = 8
    timestep_sampling: str = "shift"
    sigmoid_scale: float = 1.0
    discrete_flow_shift: float = 3.185
    model_prediction_type: str = "raw"
    guidance_scale: float = 1.0
    apply_t5_attn_mask: bool = True
    disable_mmap_load_safetensors: bool = False
    pretrained_model: str = "Stable-diffusion/flux/flux1-dev2pro.safetensors"
    ae: str = "VAE/ae.sft"
    clip_l: str = "clip/clip_l.safetensors"
    t5xxl: str = "clip/t5xxl_fp16.safetensors"
    controlnet_init_weights: str = ""
    inference_width: int = 512
    inference_height: int = 512
    inference_steps: int = 20
    inference_scale: float = 3.5
    inference_seed: int = 1026
    inference_cpu_threads: int = 8
    train_cpu_threads: int = 8
    cuda: str | None = None
    prompt_template: str = (
        "SAR image, {model_name}, azimuth {azimuth:g} deg, depression {depression:g} deg, "
        "GeoDiff-SAR physical prior ControlNet"
    )
    prompt_fallback: str = "SAR image, target, GeoDiff-SAR physical prior ControlNet"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "GeoDiffSARSkillConfig":
        raw = raw or {}
        body = raw.get("geodiff_sar", raw)
        default = cls()
        values: dict[str, Any] = {}
        for field_name in default.__dataclass_fields__:
            if field_name in body:
                values[field_name] = body[field_name]
        values.setdefault("name", str(body.get("name", default.name)))
        values.setdefault("description", str(body.get("description", default.description)))
        config = cls(**values)
        config.conda_env = str(config.conda_env)
        config.raytracing_extra_args = [str(item) for item in (config.raytracing_extra_args or [])]
        return config

    @classmethod
    def from_path(cls, path: str | Path | None) -> "GeoDiffSARSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class GeoDiffSARSkill:
    def __init__(self, config: GeoDiffSARSkillConfig | None = None) -> None:
        self.config = config or GeoDiffSARSkillConfig()

    @property
    def qinglong_project_dir(self) -> Path:
        return resolve_repo_path(self.config.qinglong_project_dir)

    @property
    def preprocess_dir(self) -> Path:
        return resolve_repo_path(self.config.preprocess_dir)

    @property
    def raytracing_dir(self) -> Path:
        return resolve_repo_path(self.config.raytracing_dir)

    def run(
        self,
        output_dir: str | Path,
        dataset_root: str | Path | None = None,
        model_file: str | Path | None = None,
        target_azimuths: list[float] | None = None,
        azimuth_sweep: dict[str, Any] | None = None,
        depressions: list[float] | None = None,
        target_count: int | None = None,
        training_epochs: int | None = None,
        controlnet_weights: str | Path | None = None,
        controlnet_init_weights: str | Path | None = None,
        prompt: str | None = None,
        model_name: str | None = None,
        train: bool = True,
        infer: bool = True,
        extract_real_gefm: bool = True,
        render_gefm: bool = True,
        cuda: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        cfg = self.config
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_path = resolve_optional_path(dataset_root)
        model_path = resolve_optional_path(model_file)
        requested_target_count = int(target_count or cfg.target_count)
        requested_epochs = int(training_epochs or cfg.training_epochs)
        run_model_name = sanitize_model_name(model_name or infer_model_name(model_path) or cfg.default_model_name)
        azimuth_values = resolve_azimuths(target_azimuths, azimuth_sweep)
        depression_values = [float(v) for v in (depressions or [80.0])]
        cuda_value = cuda if cuda is not None else cfg.cuda

        paths = self.resolve_component_paths()
        validation = validate_inputs(
            dataset_path=dataset_path,
            model_path=model_path,
            paths=paths,
            train=train,
            infer=infer,
            extract_real_gefm=extract_real_gefm,
            render_gefm=render_gefm,
        )

        train_condition_raw_dir = output_path / "gefm_train_raw"
        train_condition_dir = output_path / "gefm_train_conditions"
        train_visual_dir = output_path / "gefm_train_visual"
        ray_gefm_dir = output_path / "gefm_inference"
        prompt_dir = output_path / "controlnet_prompts"
        generated_dir = output_path / "generated_images"
        qinglong_weights_dir = self.qinglong_project_dir / "output"
        output_name = build_output_name(cfg.output_name_prefix, run_model_name)
        expected_controlnet = qinglong_weights_dir / f"{output_name}-{requested_epochs:06d}.safetensors"

        commands: dict[str, Any] = {}
        stage_results: dict[str, Any] = {}
        status = "dry_run" if dry_run else "succeeded"
        issues: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = [
            {
                "type": "domain_gap",
                "message": "GeoDiff-SAR uses physical/geometric conditions plus learned diffusion priors; outputs still require observer/evaluator checks before downstream benefit claims.",
            }
        ]

        if validation["blocking"]:
            status = "blocked"
            issues.extend(validation["issues"])

        if train and extract_real_gefm and dataset_path and status != "blocked":
            extract_cmd = self.build_extract_real_gefm_command(
                dataset_path=dataset_path,
                output_original=train_visual_dir,
                output_training=train_condition_raw_dir,
            )
            commands["extract_real_gefm"] = extract_cmd
            stage_results["extract_real_gefm"] = run_command_stage(
                command=extract_cmd,
                cwd=self.preprocess_dir,
                dry_run=dry_run,
                env_cuda=None,
            )
            if not dry_run and stage_results["extract_real_gefm"]["status"] == "succeeded":
                paired = stage_training_conditions(
                    raw_condition_dir=train_condition_raw_dir,
                    staged_condition_dir=train_condition_dir,
                )
                stage_results["stage_training_conditions"] = paired
            elif dry_run:
                stage_results["stage_training_conditions"] = {
                    "status": "dry_run",
                    "message": "Would rename *_pose_scattering.png outputs to same-stem ControlNet condition images.",
                    "output_dir": train_condition_dir.as_posix(),
                }

        if train and dataset_path and status != "blocked":
            train_condition_source = train_condition_dir if extract_real_gefm else Path(str(dataset_path)).expanduser().resolve()
            train_cmd = self.build_controlnet_train_command(
                dataset_path=dataset_path,
                condition_dir=train_condition_source,
                output_name=output_name,
                training_epochs=requested_epochs,
                controlnet_init_weights=controlnet_init_weights,
            )
            commands["train_controlnet"] = train_cmd
            stage_results["train_controlnet"] = run_command_stage(
                command=train_cmd,
                cwd=self.qinglong_project_dir,
                dry_run=dry_run,
                env_cuda=cuda_value,
            )
            if not dry_run and stage_results["train_controlnet"]["status"] != "succeeded":
                status = "failed"

        resolved_controlnet = resolve_controlnet_weights(
            explicit_weights=controlnet_weights,
            expected_weights=expected_controlnet,
            output_name=output_name,
            weights_dir=qinglong_weights_dir,
            dry_run=dry_run,
        )

        if infer and render_gefm and model_path and status != "blocked":
            ray_cmd = self.build_raytracing_command(
                model_path=model_path,
                model_name=run_model_name,
                azimuth_values=azimuth_values,
                depressions=depression_values,
                output_dir=ray_gefm_dir,
                dry_run=dry_run,
            )
            commands["render_inference_gefm"] = ray_cmd
            stage_results["render_inference_gefm"] = run_command_stage(
                command=ray_cmd,
                cwd=self.raytracing_dir,
                dry_run=dry_run,
                env_cuda=None,
            )
            if not dry_run and stage_results["render_inference_gefm"]["status"] != "succeeded":
                status = "failed"

        prompt_plan = build_controlnet_prompt_plan(
            prompt_dir=prompt_dir,
            ray_gefm_dir=ray_gefm_dir,
            model_name=run_model_name,
            azimuth_values=azimuth_values,
            depressions=depression_values,
            target_count=requested_target_count,
            prompt=prompt,
            prompt_template=cfg.prompt_template,
            prompt_fallback=cfg.prompt_fallback,
            seed=cfg.inference_seed,
            dry_run=dry_run,
        )
        stage_results["prompt_plan"] = prompt_plan

        if infer and status != "blocked":
            if not resolved_controlnet and not dry_run:
                status = "blocked"
                issues.append(
                    {
                        "type": "missing_controlnet_weights",
                        "message": "ControlNet weights were not found. Provide controlnet_weights or run training successfully first.",
                    }
                )
            else:
                infer_cmd = self.build_controlnet_inference_command(
                    prompt_dir=prompt_dir,
                    output_dir=generated_dir,
                    output_name=cfg.generated_output_name,
                    controlnet_weights=resolved_controlnet or expected_controlnet,
                    target_count=requested_target_count,
                )
                commands["controlnet_inference"] = infer_cmd
                stage_results["controlnet_inference"] = run_command_stage(
                    command=infer_cmd,
                    cwd=self.qinglong_project_dir,
                    dry_run=dry_run,
                    env_cuda=cuda_value,
                )
                if not dry_run and stage_results["controlnet_inference"]["status"] != "succeeded":
                    status = "failed"

        generated_count = count_images(generated_dir) if generated_dir.exists() and not dry_run else 0
        if status not in {"blocked", "failed"}:
            status = "dry_run" if dry_run else "succeeded"
        report = {
            "schema_version": GEODIFF_SAR_RUN_VERSION,
            "skill": "GeoDiffSARSkill",
            "status": status,
            "message": render_status_message(status, dry_run=dry_run),
            "dry_run": dry_run,
            "inputs": {
                "dataset_root": dataset_path.as_posix() if dataset_path else None,
                "model_file": model_path.as_posix() if model_path else None,
                "target_azimuths": azimuth_values,
                "depressions": depression_values,
                "target_count": requested_target_count,
                "training_epochs": requested_epochs,
                "train": train,
                "infer": infer,
                "extract_real_gefm": extract_real_gefm,
                "render_gefm": render_gefm,
            },
            "component_paths": {key: path.as_posix() for key, path in paths.items()},
            "commands": commands,
            "stage_results": stage_results,
            "controlnet_weights": resolved_controlnet.as_posix() if resolved_controlnet else None,
            "expected_controlnet_weights": expected_controlnet.as_posix(),
            "output_dir": output_path.as_posix(),
            "generated_output_dir": generated_dir.as_posix(),
            "image_count": generated_count,
            "expected_count": requested_target_count,
            "metrics": {
                "planned_prompt_count": prompt_plan.get("planned_count", 0),
                "generated_count": generated_count,
                "azimuth_count": len(azimuth_values),
                "depression_count": len(depression_values),
            },
            "issues": issues,
            "warnings": warnings,
            "artifacts": {
                "train_gefm_raw_dir": train_condition_raw_dir.as_posix(),
                "train_gefm_condition_dir": train_condition_dir.as_posix(),
                "train_gefm_visual_dir": train_visual_dir.as_posix(),
                "inference_gefm_dir": ray_gefm_dir.as_posix(),
                "prompt_dir": prompt_dir.as_posix(),
                "prompt_plan": (prompt_dir / "prompt_plan.json").as_posix(),
                "generated_images": generated_dir.as_posix(),
                "report_json": (output_path / "geodiff_sar_report.json").as_posix(),
                "report_md": (output_path / "geodiff_sar_report.md").as_posix(),
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(output_path / "geodiff_sar_report.json", report)
        save_text(output_path / "geodiff_sar_report.md", render_report_markdown(report))
        return report

    def resolve_component_paths(self) -> dict[str, Path]:
        return {
            "qinglong_project_dir": self.qinglong_project_dir,
            "preprocess_dir": self.preprocess_dir,
            "raytracing_dir": self.raytracing_dir,
            "real_gefm_script": self.preprocess_dir / self.config.real_gefm_script,
            "raytracing_batch_script": self.raytracing_dir / self.config.raytracing_batch_script,
            "controlnet_train_script": self.qinglong_project_dir / self.config.controlnet_train_script,
            "controlnet_inference_script": self.qinglong_project_dir / self.config.controlnet_inference_script,
        }

    def build_extract_real_gefm_command(self, dataset_path: Path, output_original: Path, output_training: Path) -> list[str]:
        cfg = self.config
        return [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            (self.preprocess_dir / cfg.real_gefm_script).as_posix(),
            "--input-dir",
            dataset_path.as_posix(),
            "--output-original",
            output_original.as_posix(),
            "--output-training",
            output_training.as_posix(),
            "--max-scattering-points",
            str(int(cfg.max_scattering_points)),
            "--pose-method",
            cfg.pose_method,
        ]

    def build_controlnet_train_command(
        self,
        dataset_path: Path,
        condition_dir: Path,
        output_name: str,
        training_epochs: int,
        controlnet_init_weights: str | Path | None = None,
    ) -> list[str]:
        cfg = self.config
        command = [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            "-m",
            "accelerate.commands.launch",
            "--num_processes=1",
            "--num_cpu_threads_per_process",
            str(int(cfg.train_cpu_threads)),
            "--mixed_precision",
            cfg.mixed_precision,
            (self.qinglong_project_dir / cfg.controlnet_train_script).as_posix(),
            "--pretrained_model_name_or_path",
            self.resolve_qinglong_path(cfg.pretrained_model).as_posix(),
            "--train_data_dir",
            dataset_path.as_posix(),
            "--conditioning_data_dir",
            condition_dir.as_posix(),
            "--output_dir",
            (self.qinglong_project_dir / "output").as_posix(),
            "--logging_dir",
            (self.qinglong_project_dir / "logs").as_posix(),
            "--max_train_epochs",
            str(int(training_epochs)),
            "--learning_rate",
            str(cfg.learning_rate),
            "--output_name",
            output_name,
            "--save_every_n_epochs",
            str(int(min(cfg.save_every_n_epochs, training_epochs))),
            "--save_precision",
            cfg.save_precision,
            "--seed",
            str(int(cfg.seed)),
            "--caption_extension",
            cfg.caption_extension,
            "--vae_batch_size",
            str(int(cfg.vae_batch_size)),
            "--resolution",
            cfg.resolution,
            "--train_batch_size",
            str(int(cfg.train_batch_size)),
            "--gradient_accumulation_steps",
            str(int(cfg.gradient_accumulation_steps)),
            "--max_data_loader_n_workers",
            str(int(cfg.max_data_loader_n_workers)),
            "--mixed_precision",
            cfg.mixed_precision,
            "--ae",
            self.resolve_qinglong_path(cfg.ae).as_posix(),
            "--clip_l",
            self.resolve_qinglong_path(cfg.clip_l).as_posix(),
            "--t5xxl",
            self.resolve_qinglong_path(cfg.t5xxl).as_posix(),
            "--timestep_sampling",
            cfg.timestep_sampling,
            "--sigmoid_scale",
            str(float(cfg.sigmoid_scale)),
            "--discrete_flow_shift",
            str(float(cfg.discrete_flow_shift)),
            "--model_prediction_type",
            cfg.model_prediction_type,
            "--guidance_scale",
            str(float(cfg.guidance_scale)),
            "--blocks_to_swap",
            str(int(cfg.blocks_to_swap)),
        ]
        init_weights = controlnet_init_weights or cfg.controlnet_init_weights
        if init_weights:
            command.extend(["--controlnet_model_name_or_path", resolve_path(init_weights).as_posix()])
        append_bool(command, cfg.apply_t5_attn_mask, "--apply_t5_attn_mask")
        append_bool(command, cfg.cache_latents, "--cache_latents")
        append_bool(command, cfg.cache_latents_to_disk, "--cache_latents_to_disk")
        append_bool(command, cfg.cache_text_encoder_outputs, "--cache_text_encoder_outputs")
        append_bool(command, cfg.cache_text_encoder_outputs_to_disk, "--cache_text_encoder_outputs_to_disk")
        append_bool(command, cfg.persistent_data_loader_workers, "--persistent_data_loader_workers")
        append_bool(command, cfg.gradient_checkpointing, "--gradient_checkpointing")
        append_bool(command, cfg.full_bf16, "--full_bf16")
        append_bool(command, cfg.fp8_base, "--fp8_base")
        append_bool(command, cfg.fp8_base_unet, "--fp8_base_unet")
        append_bool(command, cfg.disable_mmap_load_safetensors, "--disable_mmap_load_safetensors")
        return command

    def build_raytracing_command(
        self,
        model_path: Path,
        model_name: str,
        azimuth_values: list[float],
        depressions: list[float],
        output_dir: Path,
        dry_run: bool,
    ) -> list[str]:
        cfg = self.config
        command = [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            (self.raytracing_dir / cfg.raytracing_batch_script).as_posix(),
            "--models",
            f"{model_name}={model_path.as_posix()}",
            "--azimuths",
            ",".join(format_float(v) for v in azimuth_values),
            "--depressions",
            ",".join(format_float(v) for v in depressions),
            "--out-root",
            output_dir.as_posix(),
            "--jobs",
            str(int(cfg.raytracing_jobs)),
            "--python",
            "python",
            "--pipeline-mode",
            cfg.raytracing_pipeline_mode,
            "--reference-enable",
            "0",
            "--pol",
            cfg.raytracing_pol,
            "--image-size",
            str(int(cfg.raytracing_image_size)),
            "--strong-method",
            cfg.raytracing_strong_method,
            "--strong-top-k",
            str(int(cfg.raytracing_strong_top_k)),
            "--strong-hard-max",
            str(int(cfg.raytracing_strong_hard_max)),
            "--strong-threshold-db",
            str(float(cfg.raytracing_strong_threshold_db)),
            "--strong-nms-bins",
            str(int(cfg.raytracing_strong_nms_bins)),
            "--only-skeleton-png",
            "1" if cfg.raytracing_only_skeleton_png else "0",
            "--dry-run",
            "1" if dry_run else "0",
        ]
        for blob in cfg.raytracing_extra_args:
            command.extend(shlex.split(blob))
        return command

    def build_controlnet_inference_command(
        self,
        prompt_dir: Path,
        output_dir: Path,
        output_name: str,
        controlnet_weights: Path,
        target_count: int,
    ) -> list[str]:
        cfg = self.config
        return [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            "-m",
            "accelerate.commands.launch",
            "--num_processes=1",
            "--num_cpu_threads_per_process",
            str(int(cfg.inference_cpu_threads)),
            "--mixed_precision",
            cfg.mixed_precision,
            (self.qinglong_project_dir / cfg.controlnet_inference_script).as_posix(),
            "--pretrained_model_name_or_path",
            self.resolve_qinglong_path(cfg.pretrained_model).as_posix(),
            "--controlnet_model_name_or_path",
            controlnet_weights.as_posix(),
            "--ae",
            self.resolve_qinglong_path(cfg.ae).as_posix(),
            "--clip_l",
            self.resolve_qinglong_path(cfg.clip_l).as_posix(),
            "--t5xxl",
            self.resolve_qinglong_path(cfg.t5xxl).as_posix(),
            "--sample_prompts",
            prompt_dir.as_posix(),
            "--output_dir",
            output_dir.as_posix(),
            "--output_name",
            output_name,
            "--apply_t5_attn_mask",
            "--save_caption_txt",
            "--mixed_precision",
            cfg.mixed_precision,
            "--disable_mmap_load_safetensors",
            "1" if cfg.disable_mmap_load_safetensors else "0",
            "--default_width",
            str(int(cfg.inference_width)),
            "--default_height",
            str(int(cfg.inference_height)),
            "--default_steps",
            str(int(cfg.inference_steps)),
            "--default_scale",
            str(float(cfg.inference_scale)),
            "--default_seed",
            str(int(cfg.inference_seed)),
        ]

    def resolve_qinglong_path(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        if value.is_absolute():
            return value
        return (self.qinglong_project_dir / value).resolve()


def run_geodiff_sar_skill(
    output_dir: str | Path,
    config_path: str | Path | None = "configs/skills/geodiff_sar.yaml",
    dataset_root: str | Path | None = None,
    model_file: str | Path | None = None,
    target_azimuths: list[float] | None = None,
    azimuth_sweep: dict[str, Any] | None = None,
    depressions: list[float] | None = None,
    target_count: int | None = None,
    training_epochs: int | None = None,
    controlnet_weights: str | Path | None = None,
    controlnet_init_weights: str | Path | None = None,
    prompt: str | None = None,
    model_name: str | None = None,
    train: bool = True,
    infer: bool = True,
    extract_real_gefm: bool = True,
    render_gefm: bool = True,
    cuda: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    skill = GeoDiffSARSkill(GeoDiffSARSkillConfig.from_path(config_path))
    return skill.run(
        dataset_root=dataset_root,
        model_file=model_file,
        output_dir=output_dir,
        target_azimuths=target_azimuths,
        azimuth_sweep=azimuth_sweep,
        depressions=depressions,
        target_count=target_count,
        training_epochs=training_epochs,
        controlnet_weights=controlnet_weights,
        controlnet_init_weights=controlnet_init_weights,
        prompt=prompt,
        model_name=model_name,
        train=train,
        infer=infer,
        extract_real_gefm=extract_real_gefm,
        render_gefm=render_gefm,
        cuda=cuda,
        dry_run=dry_run,
    )


def validate_inputs(
    *,
    dataset_path: Path | None,
    model_path: Path | None,
    paths: dict[str, Path],
    train: bool,
    infer: bool,
    extract_real_gefm: bool,
    render_gefm: bool,
) -> dict[str, Any]:
    issues = []
    for key, path in paths.items():
        if not path.exists():
            issues.append({"type": "missing_component", "component": key, "path": path.as_posix()})
    if train and dataset_path is None:
        issues.append({"type": "missing_dataset_root", "message": "GeoDiff-SAR training needs dataset_root."})
    if train and dataset_path is not None and not dataset_path.exists():
        issues.append({"type": "dataset_root_not_found", "path": dataset_path.as_posix()})
    if infer and render_gefm and model_path is None:
        issues.append({"type": "missing_model_file", "message": "GeoDiff-SAR inference GEFM rendering needs a 3D model_file."})
    if infer and render_gefm and model_path is not None and not model_path.exists():
        issues.append({"type": "model_file_not_found", "path": model_path.as_posix()})
    return {"blocking": bool(issues), "issues": issues}


def run_command_stage(command: list[str], cwd: Path, dry_run: bool, env_cuda: str | None) -> dict[str, Any]:
    if dry_run:
        return {"status": "dry_run", "command": command, "cwd": cwd.as_posix()}
    env = os.environ.copy()
    if env_cuda not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(env_cuda)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "cwd": cwd.as_posix(),
        "stdout_tail": tail_text(completed.stdout),
        "stderr_tail": tail_text(completed.stderr),
    }


def stage_training_conditions(raw_condition_dir: Path, staged_condition_dir: Path) -> dict[str, Any]:
    if staged_condition_dir.exists():
        shutil.rmtree(staged_condition_dir)
    staged_condition_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    skipped = []
    for path in sorted(raw_condition_dir.glob("*.png")):
        stem = path.stem
        if stem.endswith("_pose_scattering"):
            stem = stem[: -len("_pose_scattering")]
        target = staged_condition_dir / f"{stem}.png"
        shutil.copy2(path, target)
        staged.append(target.as_posix())
    if not staged:
        skipped.append("No *_pose_scattering.png condition files were found.")
    return {
        "status": "succeeded" if staged else "warning",
        "message": f"Staged {len(staged)} ControlNet condition images.",
        "output_dir": staged_condition_dir.as_posix(),
        "staged_count": len(staged),
        "staged_examples": staged[:20],
        "warnings": skipped,
    }


def build_controlnet_prompt_plan(
    *,
    prompt_dir: Path,
    ray_gefm_dir: Path,
    model_name: str,
    azimuth_values: list[float],
    depressions: list[float],
    target_count: int,
    prompt: str | None,
    prompt_template: str,
    prompt_fallback: str,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    control_paths = discover_control_paths(
        ray_gefm_dir=ray_gefm_dir,
        model_name=model_name,
        azimuth_values=azimuth_values,
        depressions=depressions,
        dry_run=dry_run,
    )
    planned = []
    if not control_paths:
        save_json(prompt_dir / "prompt_plan.json", {"planned_count": 0, "prompts": [], "warning": "No control images found or planned."})
        return {"status": "warning", "planned_count": 0, "prompt_dir": prompt_dir.as_posix()}
    for index in range(max(0, int(target_count))):
        control = control_paths[index % len(control_paths)]
        az = control.get("azimuth")
        dep = control.get("depression")
        text = prompt or build_prompt(prompt_template, prompt_fallback, model_name, az, dep)
        item_seed = int(seed) + index * 7919
        line = f"{text} --d {item_seed} --cn {control['path']}"
        prompt_file = prompt_dir / f"{index + 1:06d}.txt"
        if not dry_run:
            prompt_file.write_text(line + "\n", encoding="utf-8")
        else:
            prompt_file.write_text(line + "\n", encoding="utf-8")
        planned.append(
            {
                "index": index + 1,
                "prompt": text,
                "seed": item_seed,
                "controlnet_image": control["path"],
                "azimuth": az,
                "depression": dep,
                "prompt_file": prompt_file.as_posix(),
            }
        )
    save_json(prompt_dir / "prompt_plan.json", {"planned_count": len(planned), "prompts": planned})
    return {
        "status": "dry_run" if dry_run else "succeeded",
        "planned_count": len(planned),
        "prompt_dir": prompt_dir.as_posix(),
        "examples": planned[:10],
    }


def discover_control_paths(
    *,
    ray_gefm_dir: Path,
    model_name: str,
    azimuth_values: list[float],
    depressions: list[float],
    dry_run: bool,
) -> list[dict[str, Any]]:
    if ray_gefm_dir.exists() and not dry_run:
        paths = sorted(ray_gefm_dir.glob(f"{model_name}/*_skeleton_only.png"))
        if not paths:
            paths = sorted(ray_gefm_dir.glob(f"{model_name}/*.png"))
        return [parse_condition_path(path) for path in paths]
    planned = []
    for dep in depressions:
        for az in azimuth_values:
            name = f"{model_name}_az{format_label(az)}_dep{format_label(dep)}_skeleton_only.png"
            planned.append(
                {
                    "path": (ray_gefm_dir / model_name / name).as_posix(),
                    "azimuth": float(az),
                    "depression": float(dep),
                }
            )
    return planned


def parse_condition_path(path: Path) -> dict[str, Any]:
    text = path.stem
    az = parse_float_after(text, "az")
    dep = parse_float_after(text, "dep")
    return {"path": path.as_posix(), "azimuth": az, "depression": dep}


def resolve_controlnet_weights(
    *,
    explicit_weights: str | Path | None,
    expected_weights: Path,
    output_name: str,
    weights_dir: Path,
    dry_run: bool,
) -> Path | None:
    if explicit_weights:
        return resolve_path(explicit_weights)
    matches = sorted(weights_dir.glob(f"{output_name}-*.safetensors"))
    if matches:
        return matches[-1].resolve()
    return expected_weights.resolve() if dry_run else None


def resolve_azimuths(values: list[float] | None, sweep: dict[str, Any] | None) -> list[float]:
    if values:
        return [float(v) % 360.0 for v in values]
    if sweep:
        start = float(sweep.get("start", 0))
        stop = float(sweep.get("stop", 360))
        step = float(sweep.get("step", 30))
        if step <= 0:
            step = 30
        out = []
        cur = start
        guard = 0
        while cur < stop and guard < 10000:
            out.append(cur % 360.0)
            cur += step
            guard += 1
        if out:
            return out
    return [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0]


def build_prompt(template: str, fallback: str, model_name: str, azimuth: float | None, depression: float | None) -> str:
    if azimuth is None or depression is None:
        return fallback
    try:
        return template.format(model_name=model_name, azimuth=float(azimuth), depression=float(depression))
    except Exception:
        return fallback


def infer_model_name(model_path: Path | None) -> str | None:
    if model_path is None:
        return None
    return model_path.stem


def sanitize_model_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_\-]+", "_", value.strip())
    return out.strip("_") or "target"


def build_output_name(prefix: str, model_name: str) -> str:
    return f"{sanitize_model_name(prefix)}_{sanitize_model_name(model_name)}"


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (REPO_ROOT / value).resolve()


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (REPO_ROOT / value).resolve()


def resolve_optional_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    return resolve_path(path)


def append_bool(command: list[str], enabled: bool, flag: str) -> None:
    if bool(enabled):
        command.append(flag)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def parse_float_after(text: str, prefix: str) -> float | None:
    match = re.search(rf"{re.escape(prefix)}([0-9mp.\-]+)", text)
    if not match:
        return None
    token = match.group(1).replace("p", ".").replace("m", "-")
    try:
        return float(token)
    except ValueError:
        return None


def format_float(value: float) -> str:
    as_float = float(value)
    if as_float.is_integer():
        return str(int(as_float))
    return f"{as_float:.4f}".rstrip("0").rstrip(".")


def format_label(value: float) -> str:
    text = format_float(value)
    return text.replace("-", "m").replace(".", "p")


def tail_text(text: str, line_count: int = 60) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= line_count:
        return "\n".join(lines)
    return "\n".join(lines[-line_count:])


def render_status_message(status: str, dry_run: bool) -> str:
    if status == "blocked":
        return "GeoDiff-SAR run is blocked by missing inputs or component paths."
    if status == "failed":
        return "GeoDiff-SAR run failed in at least one execution stage."
    if dry_run:
        return "GeoDiff-SAR dry-run generated a four-stage command plan."
    return "GeoDiff-SAR pipeline completed."


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GeoDiff-SAR Skill Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Dry run: {report.get('dry_run')}",
        f"- Output: `{report.get('output_dir')}`",
        f"- Generated output: `{report.get('generated_output_dir')}`",
        f"- Expected count: {report.get('expected_count')}",
        f"- Image count: {report.get('image_count')}",
        "",
        "## Stages",
        "",
    ]
    for key, value in (report.get("stage_results") or {}).items():
        lines.append(f"- `{key}`: {value.get('status') if isinstance(value, dict) else value}")
    if report.get("issues"):
        lines.extend(["", "## Issues", ""])
        for item in report.get("issues") or []:
            lines.append(f"- `{item.get('type')}`: {item.get('message') or item.get('path') or item}")
    lines.extend(["", "## Commands", ""])
    for key, command in (report.get("commands") or {}).items():
        lines.extend([f"### {key}", "", "```bash", shlex.join(command), "```", ""])
    return "\n".join(lines)

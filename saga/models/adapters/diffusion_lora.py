from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


MODEL_FAMILY_ALIASES = {
    "flux": "flux",
    "flux.1": "flux",
    "sd3": "sd3",
    "sd3.5": "sd3",
    "sd35": "sd3",
    "sd3.5m": "sd3",
    "sdxl": "sdxl",
}


@dataclass
class DiffusionLoRAAdapterConfig:
    project_root: str = "myproject/qinglong_trainer_29b"
    conda_env: str = "sd3"
    default_model_family: str = "flux"
    output_root: str = "output"
    logging_dir: str = "logs"
    train_scripts: dict[str, str] = field(
        default_factory=lambda: {
            "flux": "sd-scripts/flux_train_network.py",
            "sd3": "sd-scripts/sd3_train_network.py",
            "sdxl": "sd-scripts/sdxl_train_network.py",
        }
    )
    inference_scripts: dict[str, str] = field(
        default_factory=lambda: {
            "flux": "sd-scripts/flux_minimal_inference.py",
            "sd3": "sd-scripts/sd3_minimal_inference.py",
            "sdxl": "sd-scripts/sdxl_gen_img.py",
        }
    )
    network_modules: dict[str, str] = field(
        default_factory=lambda: {
            "flux": "networks.lora_flux",
            "sd3": "networks.lora_sd3",
            "sdxl": "networks.lora",
        }
    )
    pretrained_models: dict[str, str] = field(
        default_factory=lambda: {
            "flux": "Stable-diffusion/flux/flux1-dev2pro.safetensors",
            "sd3": "",
            "sdxl": "",
        }
    )
    vae: str = "VAE/ae.sft"
    ae: str = "VAE/ae.sft"
    clip_l: str = "clip/clip_l.safetensors"
    clip_g: str = "clip/clip_g.safetensors"
    t5xxl: str = "clip/t5xxl_fp16.safetensors"
    resolution: str = "512,512"
    train_batch_size: int = 2
    max_train_epochs: int = 100
    save_every_n_epochs: int = 5
    learning_rate: str = "1e-4"
    unet_lr: str = "5e-4"
    text_encoder_lr: str = "2e-5"
    optimizer_type: str = "AdamW8bit"
    optimizer_args: list[str] = field(default_factory=list)
    lr_scheduler: str = "constant"
    gradient_accumulation_steps: int = 8
    network_dim: int = 32
    network_alpha: float = 16.0
    seed: int = 1026
    mixed_precision: str = "bf16"
    save_precision: str = "bf16"
    save_model_as: str = "safetensors"
    caption_extension: str = ".txt"
    max_token_length: int = 225
    vae_batch_size: int = 4
    max_data_loader_n_workers: int = 8
    persistent_data_loader_workers: bool = True
    enable_bucket: bool = True
    min_bucket_reso: int = 512
    max_bucket_reso: int = 1024
    bucket_no_upscale: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    gradient_checkpointing: bool = False
    sdpa: bool = True
    xformers: bool = False
    disable_mmap_load_safetensors: bool = False
    fp8_base_unet: bool = True
    blocks_to_swap: int = 8
    timestep_sampling: str = "flux_shift"
    sigmoid_scale: float = 1.0
    model_prediction_type: str = "raw"
    training_guidance_scale: float = 1.0
    discrete_flow_shift: float = 3.185
    inference_width: int = 512
    inference_height: int = 512
    inference_steps: int = 20
    inference_guidance: float = 3.5
    inference_cfg_scale: float = 1.0
    inference_seed: int = 1026
    lora_multiplier: float = 1.0
    offload: bool = False
    extra_train_args: list[str] = field(default_factory=list)
    extra_inference_args: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DiffusionLoRAAdapterConfig":
        raw = raw or {}
        if "diffusion_lora" in raw and isinstance(raw["diffusion_lora"], dict):
            raw = raw["diffusion_lora"]
        if "adapter" in raw and isinstance(raw["adapter"], dict):
            raw = raw["adapter"]
        default = cls()
        return cls(
            project_root=str(raw.get("project_root", default.project_root)),
            conda_env=str(raw.get("conda_env", default.conda_env)),
            default_model_family=normalize_model_family(raw.get("default_model_family", default.default_model_family)),
            output_root=str(raw.get("output_root", default.output_root)),
            logging_dir=str(raw.get("logging_dir", default.logging_dir)),
            train_scripts=merge_dict(default.train_scripts, raw.get("train_scripts")),
            inference_scripts=merge_dict(default.inference_scripts, raw.get("inference_scripts")),
            network_modules=merge_dict(default.network_modules, raw.get("network_modules")),
            pretrained_models=merge_dict(default.pretrained_models, raw.get("pretrained_models")),
            vae=str(raw.get("vae", default.vae)),
            ae=str(raw.get("ae", raw.get("vae", default.ae))),
            clip_l=str(raw.get("clip_l", default.clip_l)),
            clip_g=str(raw.get("clip_g", default.clip_g)),
            t5xxl=str(raw.get("t5xxl", default.t5xxl)),
            resolution=str(raw.get("resolution", default.resolution)),
            train_batch_size=int(raw.get("train_batch_size", raw.get("batch_size", default.train_batch_size))),
            max_train_epochs=int(raw.get("max_train_epochs", default.max_train_epochs)),
            save_every_n_epochs=int(raw.get("save_every_n_epochs", default.save_every_n_epochs)),
            learning_rate=str(raw.get("learning_rate", raw.get("lr", default.learning_rate))),
            unet_lr=str(raw.get("unet_lr", default.unet_lr)),
            text_encoder_lr=str(raw.get("text_encoder_lr", default.text_encoder_lr)),
            optimizer_type=str(raw.get("optimizer_type", default.optimizer_type)),
            optimizer_args=list(raw.get("optimizer_args", default.optimizer_args) or []),
            lr_scheduler=str(raw.get("lr_scheduler", default.lr_scheduler)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", default.gradient_accumulation_steps)),
            network_dim=int(raw.get("network_dim", default.network_dim)),
            network_alpha=float(raw.get("network_alpha", default.network_alpha)),
            seed=int(raw.get("seed", default.seed)),
            mixed_precision=str(raw.get("mixed_precision", default.mixed_precision)),
            save_precision=str(raw.get("save_precision", default.save_precision)),
            save_model_as=str(raw.get("save_model_as", default.save_model_as)),
            caption_extension=str(raw.get("caption_extension", default.caption_extension)),
            max_token_length=int(raw.get("max_token_length", default.max_token_length)),
            vae_batch_size=int(raw.get("vae_batch_size", default.vae_batch_size)),
            max_data_loader_n_workers=int(raw.get("max_data_loader_n_workers", default.max_data_loader_n_workers)),
            persistent_data_loader_workers=parse_bool(
                raw.get("persistent_data_loader_workers", default.persistent_data_loader_workers)
            ),
            enable_bucket=parse_bool(raw.get("enable_bucket", default.enable_bucket)),
            min_bucket_reso=int(raw.get("min_bucket_reso", default.min_bucket_reso)),
            max_bucket_reso=int(raw.get("max_bucket_reso", default.max_bucket_reso)),
            bucket_no_upscale=parse_bool(raw.get("bucket_no_upscale", default.bucket_no_upscale)),
            cache_latents=parse_bool(raw.get("cache_latents", default.cache_latents)),
            cache_latents_to_disk=parse_bool(raw.get("cache_latents_to_disk", default.cache_latents_to_disk)),
            cache_text_encoder_outputs=parse_bool(
                raw.get("cache_text_encoder_outputs", default.cache_text_encoder_outputs)
            ),
            cache_text_encoder_outputs_to_disk=parse_bool(
                raw.get("cache_text_encoder_outputs_to_disk", default.cache_text_encoder_outputs_to_disk)
            ),
            gradient_checkpointing=parse_bool(raw.get("gradient_checkpointing", default.gradient_checkpointing)),
            sdpa=parse_bool(raw.get("sdpa", default.sdpa)),
            xformers=parse_bool(raw.get("xformers", default.xformers)),
            disable_mmap_load_safetensors=parse_bool(
                raw.get("disable_mmap_load_safetensors", default.disable_mmap_load_safetensors)
            ),
            fp8_base_unet=parse_bool(raw.get("fp8_base_unet", default.fp8_base_unet)),
            blocks_to_swap=int(raw.get("blocks_to_swap", default.blocks_to_swap)),
            timestep_sampling=str(raw.get("timestep_sampling", default.timestep_sampling)),
            sigmoid_scale=float(raw.get("sigmoid_scale", default.sigmoid_scale)),
            model_prediction_type=str(raw.get("model_prediction_type", default.model_prediction_type)),
            training_guidance_scale=float(raw.get("training_guidance_scale", default.training_guidance_scale)),
            discrete_flow_shift=float(raw.get("discrete_flow_shift", default.discrete_flow_shift)),
            inference_width=int(raw.get("inference_width", raw.get("width", default.inference_width))),
            inference_height=int(raw.get("inference_height", raw.get("height", default.inference_height))),
            inference_steps=int(raw.get("inference_steps", default.inference_steps)),
            inference_guidance=float(raw.get("inference_guidance", default.inference_guidance)),
            inference_cfg_scale=float(raw.get("inference_cfg_scale", default.inference_cfg_scale)),
            inference_seed=int(raw.get("inference_seed", default.inference_seed)),
            lora_multiplier=float(raw.get("lora_multiplier", default.lora_multiplier)),
            offload=parse_bool(raw.get("offload", default.offload)),
            extra_train_args=[str(item) for item in raw.get("extra_train_args", [])],
            extra_inference_args=[str(item) for item in raw.get("extra_inference_args", [])],
        )


class DiffusionLoRAAdapter:
    def __init__(self, config: DiffusionLoRAAdapterConfig | None = None) -> None:
        self.config = config or DiffusionLoRAAdapterConfig()

    @property
    def project_root(self) -> Path:
        return resolve_repo_path(self.config.project_root)

    def build_train_command(
        self,
        dataset_config: str | Path,
        output_name: str,
        model_family: str | None = None,
    ) -> list[str]:
        cfg = self.config
        family = normalize_model_family(model_family or cfg.default_model_family)
        script = self.resolve_project_path(required_mapping_value(cfg.train_scripts, family, "train script"))
        command = base_accelerate_command(cfg.conda_env, script)
        command.extend(
            [
                "--dataset_config",
                str(dataset_config),
                "--output_dir",
                self.resolve_project_path(cfg.output_root).as_posix(),
                "--logging_dir",
                self.resolve_project_path(cfg.logging_dir).as_posix(),
                "--max_train_epochs",
                str(cfg.max_train_epochs),
                "--learning_rate",
                cfg.learning_rate,
                "--output_name",
                output_name,
                "--save_every_n_epochs",
                str(cfg.save_every_n_epochs),
                "--save_precision",
                cfg.save_precision,
                "--seed",
                str(cfg.seed),
                "--caption_extension",
                cfg.caption_extension,
                "--vae_batch_size",
                str(cfg.vae_batch_size),
                "--mixed_precision",
                cfg.mixed_precision,
                "--network_module",
                required_mapping_value(cfg.network_modules, family, "network module"),
                "--network_dim",
                str(cfg.network_dim),
                "--network_alpha",
                numeric_to_cli(cfg.network_alpha),
                "--gradient_accumulation_steps",
                str(cfg.gradient_accumulation_steps),
                "--lr_scheduler",
                cfg.lr_scheduler,
                "--save_model_as",
                cfg.save_model_as,
                "--max_data_loader_n_workers",
                str(cfg.max_data_loader_n_workers),
            ]
        )
        pretrained_model = required_mapping_value(cfg.pretrained_models, family, "pretrained model")
        if pretrained_model:
            command.extend(["--pretrained_model_name_or_path", self.resolve_project_path(pretrained_model).as_posix()])
        if cfg.unet_lr:
            command.extend(["--unet_lr", cfg.unet_lr])
        if cfg.text_encoder_lr:
            command.extend(["--text_encoder_lr", cfg.text_encoder_lr])
        append_optimizer_args(command, cfg)
        append_common_boolean_args(command, cfg)
        append_family_train_args(command, cfg, family, self.project_root)
        command.extend(cfg.extra_train_args)
        return command

    def build_inference_command(
        self,
        prompt: str,
        output_dir: str | Path,
        lora_weights: str | Path | None,
        model_family: str | None = None,
        seed: int | None = None,
    ) -> list[str]:
        cfg = self.config
        family = normalize_model_family(model_family or cfg.default_model_family)
        script = self.resolve_project_path(required_mapping_value(cfg.inference_scripts, family, "inference script"))
        command = ["conda", "run", "-n", cfg.conda_env, "python", script.as_posix()]
        pretrained_model = required_mapping_value(cfg.pretrained_models, family, "pretrained model")
        if family == "sdxl":
            command.extend(["--ckpt", self.resolve_project_path(pretrained_model).as_posix()])
            command.extend(["--prompt", prompt, "--outdir", str(output_dir)])
            command.extend(["--W", str(cfg.inference_width), "--H", str(cfg.inference_height)])
            command.extend(["--steps", str(cfg.inference_steps), "--scale", str(cfg.inference_guidance)])
            command.extend(["--seed", str(seed if seed is not None else cfg.inference_seed)])
            if lora_weights:
                command.extend(["--network_module", required_mapping_value(cfg.network_modules, family, "network module")])
                command.extend(["--network_weights", str(lora_weights)])
                command.extend(["--network_mul", numeric_to_cli(cfg.lora_multiplier)])
            if cfg.sdpa:
                command.append("--sdpa")
            if cfg.xformers:
                command.append("--xformers")
            if cfg.mixed_precision == "bf16":
                command.append("--bf16")
            elif cfg.mixed_precision == "fp16":
                command.append("--fp16")
            command.extend(cfg.extra_inference_args)
            return command

        command.extend(["--ckpt_path", self.resolve_project_path(pretrained_model).as_posix()])
        command.extend(["--prompt", prompt, "--output_dir", str(output_dir)])
        command.extend(["--seed", str(seed if seed is not None else cfg.inference_seed)])
        command.extend(["--steps", str(cfg.inference_steps)])
        command.extend(["--width", str(cfg.inference_width), "--height", str(cfg.inference_height)])
        if family == "flux":
            command.extend(["--clip_l", self.resolve_project_path(cfg.clip_l).as_posix()])
            command.extend(["--t5xxl", self.resolve_project_path(cfg.t5xxl).as_posix()])
            command.extend(["--ae", self.resolve_project_path(cfg.ae).as_posix()])
            command.extend(["--guidance", str(cfg.inference_guidance)])
            command.extend(["--cfg_scale", str(cfg.inference_cfg_scale)])
        elif family == "sd3":
            command.extend(["--clip_l", self.resolve_project_path(cfg.clip_l).as_posix()])
            command.extend(["--clip_g", self.resolve_project_path(cfg.clip_g).as_posix()])
            command.extend(["--t5xxl", self.resolve_project_path(cfg.t5xxl).as_posix()])
            command.extend(["--cfg_scale", str(cfg.inference_guidance)])
            if cfg.mixed_precision == "bf16":
                command.append("--bf16")
            elif cfg.mixed_precision == "fp16":
                command.append("--fp16")
        if lora_weights:
            command.extend(["--lora_weights", f"{lora_weights};{numeric_to_cli(cfg.lora_multiplier)}"])
        if cfg.offload:
            command.append("--offload")
        command.extend(cfg.extra_inference_args)
        return command

    def run_train(
        self,
        dataset_config: str | Path,
        output_name: str,
        model_family: str | None = None,
        dry_run: bool = True,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        command = self.build_train_command(
            dataset_config=dataset_config,
            output_name=output_name,
            model_family=model_family,
        )
        if dry_run:
            return {"status": "dry_run", "command": command}
        if log_dir:
            return run_logged_process(command, cwd=self.project_root, log_dir=log_dir, prefix="train")
        completed = subprocess.run(command, check=False, text=True, capture_output=True, cwd=self.project_root)
        return completed_to_result(completed, command)

    def run_inference_commands(
        self,
        commands: list[list[str]],
        dry_run: bool = True,
        log_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if dry_run:
            return {"status": "dry_run", "commands": commands, "completed": 0}
        results = []
        status = "succeeded"
        for index, command in enumerate(commands, start=1):
            if log_dir:
                result = run_logged_process(
                    command,
                    cwd=self.project_root,
                    log_dir=log_dir,
                    prefix=f"inference_{index:06d}",
                )
            else:
                completed = subprocess.run(command, check=False, text=True, capture_output=True, cwd=self.project_root)
                result = completed_to_result(completed, command)
            results.append(result)
            if result["status"] != "succeeded":
                status = "failed"
                break
        return {"status": status, "commands": commands, "completed": len(results), "results": results}

    def resolve_project_path(self, path: str | Path) -> Path:
        value = Path(path).expanduser()
        if value.is_absolute():
            return value
        return (self.project_root / value).resolve()


def base_accelerate_command(conda_env: str, script: Path) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        "8",
        script.as_posix(),
    ]


def append_optimizer_args(command: list[str], cfg: DiffusionLoRAAdapterConfig) -> None:
    if cfg.optimizer_type:
        command.extend(["--optimizer_type", cfg.optimizer_type])
    if cfg.optimizer_args:
        command.append("--optimizer_args")
        command.extend(cfg.optimizer_args)


def append_common_boolean_args(command: list[str], cfg: DiffusionLoRAAdapterConfig) -> None:
    if cfg.persistent_data_loader_workers:
        command.append("--persistent_data_loader_workers")
    if cfg.cache_latents:
        command.append("--cache_latents")
    if cfg.cache_latents_to_disk:
        command.append("--cache_latents_to_disk")
    if cfg.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if cfg.disable_mmap_load_safetensors:
        command.append("--disable_mmap_load_safetensors")
    if cfg.sdpa:
        command.append("--sdpa")
    elif cfg.xformers:
        command.append("--xformers")


def append_family_train_args(
    command: list[str],
    cfg: DiffusionLoRAAdapterConfig,
    family: str,
    project_root: Path,
) -> None:
    if family == "sdxl":
        if cfg.vae:
            command.extend(["--vae", resolve_project_path(project_root, cfg.vae).as_posix()])
        if cfg.cache_text_encoder_outputs:
            command.append("--cache_text_encoder_outputs")
        if cfg.cache_text_encoder_outputs_to_disk:
            command.append("--cache_text_encoder_outputs_to_disk")
        return

    command.extend(["--clip_l", resolve_project_path(project_root, cfg.clip_l).as_posix()])
    command.extend(["--t5xxl", resolve_project_path(project_root, cfg.t5xxl).as_posix()])
    command.extend(["--discrete_flow_shift", str(cfg.discrete_flow_shift)])
    command.extend(["--blocks_to_swap", str(cfg.blocks_to_swap)])
    if cfg.cache_text_encoder_outputs:
        command.append("--cache_text_encoder_outputs")
    if cfg.cache_text_encoder_outputs_to_disk:
        command.append("--cache_text_encoder_outputs_to_disk")
    if cfg.fp8_base_unet:
        command.append("--fp8_base_unet")
    if family == "flux":
        command.extend(["--ae", resolve_project_path(project_root, cfg.ae).as_posix()])
        command.extend(["--timestep_sampling", cfg.timestep_sampling])
        command.extend(["--sigmoid_scale", str(cfg.sigmoid_scale)])
        command.extend(["--model_prediction_type", cfg.model_prediction_type])
        command.extend(["--guidance_scale", str(cfg.training_guidance_scale)])
    elif family == "sd3":
        command.extend(["--clip_g", resolve_project_path(project_root, cfg.clip_g).as_posix()])


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (REPO_ROOT / value).resolve()


def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


def normalize_model_family(value: Any) -> str:
    key = str(value or "flux").strip().lower()
    return MODEL_FAMILY_ALIASES.get(key, key)


def required_mapping_value(mapping: dict[str, str], key: str, label: str) -> str:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"No {label} configured for model_family={key}")
    return str(value)


def merge_dict(default: dict[str, str], override: Any) -> dict[str, str]:
    merged = dict(default)
    if isinstance(override, dict):
        merged.update({str(key): str(value) for key, value in override.items()})
    return merged


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def numeric_to_cli(value: int | float) -> str:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if as_float.is_integer():
        return str(int(as_float))
    return str(as_float)


def completed_to_result(completed: subprocess.CompletedProcess[str], command: list[str]) -> dict[str, Any]:
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_logged_process(command: list[str], cwd: str | Path, log_dir: str | Path, prefix: str) -> dict[str, Any]:
    started = time.time()
    log_path = Path(log_dir).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path / f"{prefix}.stdout.log"
    stderr_path = log_path / f"{prefix}.stderr.log"
    command_path = log_path / f"{prefix}.command.txt"
    command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            cwd=cwd,
        )
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "command_path": command_path.as_posix(),
        "stdout_tail": read_tail(stdout_path),
        "stderr_tail": read_tail(stderr_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def read_tail(path: Path, max_chars: int = 12000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_chars))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)

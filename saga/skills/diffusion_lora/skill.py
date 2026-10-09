from __future__ import annotations

import re
import shlex
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from saga.core.config import load_dataset_config, load_mapping, save_json, save_text
from saga.core.protocol import UNKNOWN
from saga.data.discovery import is_image, iter_images, natural_key
from saga.data.loader import load_samples
from saga.models.adapters.diffusion_lora import (
    DiffusionLoRAAdapter,
    DiffusionLoRAAdapterConfig,
    normalize_model_family,
)


CAPTION_PROFILE_VERSION = "saga_caption_dataset_profile_v1"
DIFFUSION_LORA_RUN_VERSION = "saga_diffusion_lora_run_v1"

CONTROL_ARG_RE = re.compile(r"\s+--(?:cn|d|w|h|s|seed|steps|cfg|scale)\b(?:\s+\"[^\"]*\"|\s+'[^']*'|\s+\S+)?")
IMAGE_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class DiffusionLoRAGenerationSkillConfig:
    name: str = "DiffusionLoRAGenerationSkill"
    description: str = "Train a SAR text-caption LoRA and generate augmented images through qinglong_trainer scripts."
    adapter: DiffusionLoRAAdapterConfig = field(default_factory=DiffusionLoRAAdapterConfig)
    num_repeats: int = 1
    target_count: int = 20
    prompt_count: int = 20
    output_name_prefix: str = "saga_lora"
    stage_dataset: bool = False
    max_train_samples: int | None = None
    stage_balance_fields: list[str] = field(default_factory=lambda: ["polarization"])
    strip_generation_args_from_caption: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "DiffusionLoRAGenerationSkillConfig":
        raw = raw or {}
        body = raw.get("diffusion_lora", raw)
        adapter_raw = body.get("adapter", body) if isinstance(body, dict) else {}
        default = cls()
        return cls(
            name=str(body.get("name", default.name)) if isinstance(body, dict) else default.name,
            description=str(body.get("description", default.description)) if isinstance(body, dict) else default.description,
            adapter=DiffusionLoRAAdapterConfig.from_mapping(adapter_raw),
            num_repeats=int(body.get("num_repeats", default.num_repeats)) if isinstance(body, dict) else default.num_repeats,
            target_count=int(body.get("target_count", default.target_count)) if isinstance(body, dict) else default.target_count,
            prompt_count=int(body.get("prompt_count", default.prompt_count)) if isinstance(body, dict) else default.prompt_count,
            output_name_prefix=str(body.get("output_name_prefix", default.output_name_prefix))
            if isinstance(body, dict)
            else default.output_name_prefix,
            stage_dataset=parse_bool(body.get("stage_dataset", default.stage_dataset)) if isinstance(body, dict) else False,
            max_train_samples=parse_optional_int(body.get("max_train_samples")) if isinstance(body, dict) else None,
            stage_balance_fields=[str(item) for item in body.get("stage_balance_fields", default.stage_balance_fields)]
            if isinstance(body, dict)
            else default.stage_balance_fields,
            strip_generation_args_from_caption=parse_bool(
                body.get("strip_generation_args_from_caption", default.strip_generation_args_from_caption)
            )
            if isinstance(body, dict)
            else True,
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "DiffusionLoRAGenerationSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class DiffusionLoRAGenerationSkill:
    def __init__(self, config: DiffusionLoRAGenerationSkillConfig | None = None) -> None:
        self.config = config or DiffusionLoRAGenerationSkillConfig()
        self.adapter = DiffusionLoRAAdapter(self.config.adapter)

    def run(
        self,
        dataset_root: str | Path,
        output_dir: str | Path,
        model_family: str | None = None,
        target_count: int | None = None,
        output_name: str | None = None,
        prompt: str | None = None,
        lora_weights: str | Path | None = None,
        dataset_config: str | Path | None = None,
        filters: dict[str, Any] | None = None,
        exclude_filters: dict[str, Any] | None = None,
        auto_caption_from_metadata: bool = False,
        training_epochs: int | None = None,
        skill_overrides: dict[str, Any] | None = None,
        train: bool = True,
        infer: bool = True,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        dataset_path = Path(dataset_root).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        validate_dataset_root(dataset_path)
        override_report = apply_skill_overrides(self.config, skill_overrides or {})
        if training_epochs is not None:
            training_epochs = int(training_epochs)
            if training_epochs <= 0:
                raise ValueError("training_epochs must be a positive integer.")
            self.config.adapter.max_train_epochs = training_epochs
            if self.config.adapter.save_every_n_epochs > training_epochs:
                self.config.adapter.save_every_n_epochs = training_epochs

        family = normalize_model_family(model_family or self.config.adapter.default_model_family)
        requested_count = int(target_count or self.config.target_count)
        run_output_name = output_name or build_output_name(self.config.output_name_prefix, dataset_path, family)
        profile = inspect_caption_dataset(
            dataset_root=dataset_path,
            strip_generation_args=self.config.strip_generation_args_from_caption,
        )
        prepared_dataset = dataset_path
        preparation_report: dict[str, Any] = {
            "enabled": False,
            "reason": "input_caption_dataset_valid",
            "filters": filters or {},
            "exclude_filters": exclude_filters or {},
        }
        needs_metadata_stage = bool(dataset_config) and (
            bool(filters)
            or bool(exclude_filters)
            or (auto_caption_from_metadata and not profile["valid"])
        )
        if needs_metadata_stage:
            prepared_dataset = stage_caption_dataset_from_metadata(
                dataset_config=Path(dataset_config).expanduser().resolve(),
                output_dir=output_path / "metadata_caption_dataset",
                filters=filters or {},
                exclude_filters=exclude_filters or {},
                sample_limit=self.config.max_train_samples,
                balance_fields=self.config.stage_balance_fields,
            )
            profile = inspect_caption_dataset(
                dataset_root=prepared_dataset,
                strip_generation_args=self.config.strip_generation_args_from_caption,
            )
            preparation_report = {
                "enabled": True,
                "reason": "metadata_caption_staging",
                "dataset_config": Path(dataset_config).expanduser().resolve().as_posix(),
                "output_dir": prepared_dataset.as_posix(),
                "filters": filters or {},
                "exclude_filters": exclude_filters or {},
                "staged_image_count": profile.get("image_count"),
                "staged_caption_count": profile.get("txt_count"),
                "max_train_samples": self.config.max_train_samples,
                "stage_balance_fields": self.config.stage_balance_fields,
            }
        if not profile["valid"]:
            status = "blocked"
            message = "Caption dataset did not pass validation; no train/inference command was executed."
        else:
            status = "dry_run" if dry_run else "succeeded"
            message = (
                "Diffusion LoRA plan generated."
                if dry_run
                else "Diffusion LoRA train/inference run completed."
            )

        should_stage = self.config.stage_dataset or bool(profile.get("sanitized_caption_count"))
        if should_stage:
            prepared_dataset = stage_caption_dataset(
                dataset_root=dataset_path,
                output_dir=output_path / "prepared_dataset",
                profile=profile,
            )

        dataset_config_path = output_path / "lora_dataset_config.toml"
        write_dataset_config(
            output_path=dataset_config_path,
            image_dir=prepared_dataset,
            resolution=self.config.adapter.resolution,
            batch_size=self.config.adapter.train_batch_size,
            enable_bucket=self.config.adapter.enable_bucket,
            min_bucket_reso=self.config.adapter.min_bucket_reso,
            max_bucket_reso=self.config.adapter.max_bucket_reso,
            bucket_no_upscale=self.config.adapter.bucket_no_upscale,
            caption_extension=self.config.adapter.caption_extension,
            num_repeats=self.config.num_repeats,
        )

        prompt_plan = build_prompt_plan(
            profile=profile,
            explicit_prompt=prompt,
            target_count=requested_count,
            prompt_count=self.config.prompt_count,
            base_seed=self.config.adapter.inference_seed,
        )
        prompt_plan_path = output_path / "prompt_plan.txt"
        save_text(prompt_plan_path, "\n".join(item["prompt"] for item in prompt_plan) + "\n")

        train_result = {"status": "skipped", "command": []}
        train_command: list[str] = []
        process_log_dir = output_path / "logs"
        if train and profile["valid"]:
            train_command = self.adapter.build_train_command(
                dataset_config=dataset_config_path,
                output_name=run_output_name,
                model_family=family,
            )
            if dry_run:
                train_result = {"status": "dry_run", "command": train_command}
            else:
                train_result = self.adapter.run_train(
                    dataset_config=dataset_config_path,
                    output_name=run_output_name,
                    model_family=family,
                    dry_run=False,
                    log_dir=process_log_dir,
                )
                status = "failed" if train_result.get("status") == "failed" else status

        inferred_lora_weights = (
            Path(lora_weights).expanduser().resolve().as_posix()
            if lora_weights
            else self.adapter.resolve_project_path(
                Path(self.config.adapter.output_root) / f"{run_output_name}.safetensors"
            ).as_posix()
        )
        generated_output_dir = output_path / "generated_images"
        inference_commands: list[list[str]] = []
        if infer and profile["valid"]:
            for item in prompt_plan:
                item_output_dir = generated_output_dir / f"{int(item['index']):06d}"
                item["output_dir"] = item_output_dir.as_posix()
                command = self.adapter.build_inference_command(
                    prompt=item["prompt"],
                    output_dir=item_output_dir,
                    lora_weights=inferred_lora_weights,
                    model_family=family,
                    seed=int(item["seed"]),
                )
                inference_commands.append(command)

        inference_result = {"status": "skipped", "commands": []}
        if infer and profile["valid"]:
            if dry_run:
                inference_result = {"status": "dry_run", "commands": inference_commands, "completed": 0}
            else:
                inference_block_reason = None
                if train and train_result.get("status") != "succeeded" and not lora_weights:
                    inference_block_reason = "Training did not succeed, so SAGA did not run LoRA inference."
                elif inferred_lora_weights and not Path(inferred_lora_weights).exists():
                    inference_block_reason = f"LoRA weights were not found: {inferred_lora_weights}"

                if inference_block_reason:
                    inference_result = {
                        "status": "blocked",
                        "commands": inference_commands,
                        "completed": 0,
                        "message": inference_block_reason,
                    }
                    status = "failed"
                else:
                    generated_output_dir.mkdir(parents=True, exist_ok=True)
                    before_images = set(iter_generation_images(generated_output_dir))
                    inference_result = self.adapter.run_inference_commands(
                        inference_commands,
                        dry_run=False,
                        log_dir=process_log_dir,
                    )
                    write_generated_sidecars(
                        output_dir=generated_output_dir,
                        prompt_plan=prompt_plan,
                        before_images=before_images,
                    )
                    status = "failed" if inference_result.get("status") == "failed" else status

        artifacts = {
            "caption_dataset_profile_json": (output_path / "caption_dataset_profile.json").as_posix(),
            "caption_dataset_profile_md": (output_path / "caption_dataset_profile.md").as_posix(),
            "dataset_config": dataset_config_path.as_posix(),
            "prompt_plan": prompt_plan_path.as_posix(),
            "train_command": (output_path / "train_command.sh").as_posix(),
            "inference_commands": (output_path / "inference_commands.sh").as_posix(),
            "run_json": (output_path / "diffusion_lora_run.json").as_posix(),
            "run_md": (output_path / "diffusion_lora_run.md").as_posix(),
            "process_log_dir": process_log_dir.as_posix(),
        }
        report = {
            "schema_version": DIFFUSION_LORA_RUN_VERSION,
            "skill": self.config.name,
            "status": status,
            "message": message,
            "dry_run": dry_run,
            "model_family": family,
            "inputs": {
                "dataset_root": dataset_path.as_posix(),
                "prepared_dataset": prepared_dataset.as_posix(),
                "dataset_config": Path(dataset_config).expanduser().resolve().as_posix() if dataset_config else None,
                "filters": filters or {},
                "exclude_filters": exclude_filters or {},
            },
            "output_dir": output_path.as_posix(),
            "generated_output_dir": generated_output_dir.as_posix(),
            "output_name": run_output_name,
            "expected_lora_weights": inferred_lora_weights,
            "target_count": requested_count,
            "training_epochs": self.config.adapter.max_train_epochs,
            "skill_overrides": override_report,
            "caption_profile": profile,
            "preparation": preparation_report,
            "train": {
                "enabled": train,
                "status": train_result.get("status"),
                "command": train_command or train_result.get("command", []),
                "result": prune_process_result(train_result),
            },
            "inference": {
                "enabled": infer,
                "status": inference_result.get("status"),
                "command_count": len(inference_commands),
                "commands": inference_commands,
                "result": prune_process_result(inference_result),
            },
            "prompt_plan": prompt_plan,
            "adapter_config": asdict(self.config.adapter),
            "artifacts": artifacts,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_run_artifacts(output_path, report)
        return report


def run_diffusion_lora_generation_skill(
    dataset_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    model_family: str | None = None,
    target_count: int | None = None,
    output_name: str | None = None,
    prompt: str | None = None,
    lora_weights: str | Path | None = None,
    dataset_config: str | Path | None = None,
    filters: dict[str, Any] | None = None,
    exclude_filters: dict[str, Any] | None = None,
    auto_caption_from_metadata: bool = False,
    training_epochs: int | None = None,
    skill_overrides: dict[str, Any] | None = None,
    train: bool = True,
    infer: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = DiffusionLoRAGenerationSkillConfig.from_path(config_path)
    return DiffusionLoRAGenerationSkill(config).run(
        dataset_root=dataset_root,
        output_dir=output_dir,
        model_family=model_family,
        target_count=target_count,
        output_name=output_name,
        prompt=prompt,
        lora_weights=lora_weights,
        dataset_config=dataset_config,
        filters=filters,
        exclude_filters=exclude_filters,
        auto_caption_from_metadata=auto_caption_from_metadata,
        training_epochs=training_epochs,
        skill_overrides=skill_overrides,
        train=train,
        infer=infer,
        dry_run=dry_run,
    )


ALLOWED_SKILL_OVERRIDES = {
    "target_count",
    "prompt_count",
    "num_repeats",
    "max_train_samples",
    "stage_balance_fields",
    "adapter",
}

ALLOWED_ADAPTER_OVERRIDES = {
    "train_batch_size",
    "max_train_epochs",
    "save_every_n_epochs",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "gradient_accumulation_steps",
    "network_dim",
    "network_alpha",
    "vae_batch_size",
    "max_data_loader_n_workers",
    "inference_steps",
    "inference_guidance",
    "inference_cfg_scale",
    "lora_multiplier",
    "blocks_to_swap",
}


def apply_skill_overrides(
    config: DiffusionLoRAGenerationSkillConfig,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    if not overrides:
        return {"applied": [], "ignored": [], "allowed_keys": sorted(ALLOWED_SKILL_OVERRIDES)}
    applied = []
    ignored = []
    for key, value in overrides.items():
        if key not in ALLOWED_SKILL_OVERRIDES:
            ignored.append({"key": key, "reason": "not_in_skill_override_whitelist"})
            continue
        if key == "adapter":
            if not isinstance(value, dict):
                ignored.append({"key": key, "reason": "adapter_override_must_be_mapping"})
                continue
            for adapter_key, adapter_value in value.items():
                if adapter_key not in ALLOWED_ADAPTER_OVERRIDES:
                    ignored.append({"key": f"adapter.{adapter_key}", "reason": "not_in_adapter_override_whitelist"})
                    continue
                old_value = getattr(config.adapter, adapter_key)
                new_value = coerce_like(adapter_value, old_value)
                setattr(config.adapter, adapter_key, new_value)
                applied.append({"key": f"adapter.{adapter_key}", "old_value": old_value, "new_value": new_value})
            continue
        old_value = getattr(config, key)
        new_value = coerce_skill_override(key, value, old_value)
        setattr(config, key, new_value)
        applied.append({"key": key, "old_value": old_value, "new_value": new_value})
    if config.adapter.save_every_n_epochs > config.adapter.max_train_epochs:
        applied.append(
            {
                "key": "adapter.save_every_n_epochs",
                "old_value": config.adapter.save_every_n_epochs,
                "new_value": config.adapter.max_train_epochs,
                "reason": "clamped_to_max_train_epochs",
            }
        )
        config.adapter.save_every_n_epochs = config.adapter.max_train_epochs
    return {
        "applied": applied,
        "ignored": ignored,
        "allowed_keys": sorted(ALLOWED_SKILL_OVERRIDES),
        "allowed_adapter_keys": sorted(ALLOWED_ADAPTER_OVERRIDES),
    }


def coerce_like(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        return parse_bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list):
        return list(value) if isinstance(value, (list, tuple)) else [str(value)]
    return str(value)


def coerce_skill_override(key: str, value: Any, current: Any) -> Any:
    if key in {"target_count", "prompt_count", "num_repeats"}:
        return int(float(value))
    if key == "max_train_samples":
        return parse_optional_int(value)
    if key == "stage_balance_fields":
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]
    return coerce_like(value, current)


def inspect_caption_dataset(dataset_root: Path, strip_generation_args: bool = True) -> dict[str, Any]:
    images = iter_images(dataset_root)
    sidecars = sorted(dataset_root.rglob("*.txt"), key=lambda p: natural_key(p.as_posix()))
    image_by_stem = {path.with_suffix("").as_posix(): path for path in images}
    txt_by_stem = {path.with_suffix("").as_posix(): path for path in sidecars}
    paired_stems = sorted(set(image_by_stem) & set(txt_by_stem), key=natural_key)
    missing_txt = [path for stem, path in image_by_stem.items() if stem not in txt_by_stem]
    orphan_txt = [path for stem, path in txt_by_stem.items() if stem not in image_by_stem]
    caption_rows = []
    empty_captions = []
    control_arg_captions = []
    token_counter: Counter[str] = Counter()
    for stem in paired_stems:
        txt_path = txt_by_stem[stem]
        raw_caption = read_first_line(txt_path)
        caption = sanitize_caption(raw_caption) if strip_generation_args else raw_caption.strip()
        if not caption:
            empty_captions.append(txt_path)
        if raw_caption != caption:
            control_arg_captions.append(txt_path)
        tokens = [token.strip() for token in caption.split(",") if token.strip()]
        token_counter.update(tokens)
        caption_rows.append(
            {
                "image": image_by_stem[stem].as_posix(),
                "caption_file": txt_path.as_posix(),
                "caption": caption,
                "raw_caption": raw_caption,
                "tokens": tokens,
            }
        )

    image_probe = probe_images(images[: min(len(images), 200)])
    valid = bool(images) and len(missing_txt) == 0 and len(empty_captions) == 0
    issues = []
    if not images:
        issues.append({"severity": "high", "name": "no_images", "message": "No train images were found."})
    if missing_txt:
        issues.append(
            {
                "severity": "high",
                "name": "missing_caption_sidecars",
                "count": len(missing_txt),
                "examples": [path.as_posix() for path in missing_txt[:20]],
            }
        )
    if empty_captions:
        issues.append(
            {
                "severity": "high",
                "name": "empty_captions",
                "count": len(empty_captions),
                "examples": [path.as_posix() for path in empty_captions[:20]],
            }
        )
    if orphan_txt:
        issues.append(
            {
                "severity": "low",
                "name": "orphan_caption_sidecars",
                "count": len(orphan_txt),
                "examples": [path.as_posix() for path in orphan_txt[:20]],
            }
        )
    if control_arg_captions:
        issues.append(
            {
                "severity": "medium",
                "name": "caption_generation_args_detected",
                "count": len(control_arg_captions),
                "message": "Some captions contain generation-only flags such as --cn/--d; sanitized prompts were used.",
                "examples": [path.as_posix() for path in control_arg_captions[:20]],
            }
        )

    return {
        "schema_version": CAPTION_PROFILE_VERSION,
        "valid": valid,
        "dataset_root": dataset_root.as_posix(),
        "image_count": len(images),
        "txt_count": len(sidecars),
        "paired_count": len(paired_stems),
        "missing_txt_count": len(missing_txt),
        "orphan_txt_count": len(orphan_txt),
        "empty_caption_count": len(empty_captions),
        "sanitized_caption_count": len(control_arg_captions),
        "image_suffix_counts": dict(Counter(path.suffix.lower() for path in images).most_common()),
        "image_probe": image_probe,
        "top_caption_tokens": [{"token": token, "count": count} for token, count in token_counter.most_common(30)],
        "sample_captions": caption_rows[:20],
        "issues": issues,
    }


def probe_images(images: list[Path]) -> dict[str, Any]:
    size_counter: Counter[str] = Counter()
    mode_counter: Counter[str] = Counter()
    unreadable = []
    for path in images:
        try:
            with Image.open(path) as image:
                size_counter[f"{image.size[0]}x{image.size[1]}"] += 1
                mode_counter[str(image.mode)] += 1
        except Exception as exc:
            unreadable.append({"path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "sampled_count": len(images),
        "size_counts": dict(size_counter.most_common()),
        "mode_counts": dict(mode_counter.most_common()),
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:10],
    }


def stage_caption_dataset(dataset_root: Path, output_dir: Path, profile: dict[str, Any]) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in profile.get("sample_captions", []):
        image = Path(row["image"])
        txt = Path(row["caption_file"])
        shutil.copy2(image, output_dir / image.name)
        save_text(output_dir / txt.name, row["caption"].strip() + "\n")
    sampled_paths = {row["image"] for row in profile.get("sample_captions", [])}
    for image in iter_images(dataset_root):
        if image.as_posix() in sampled_paths:
            continue
        txt = image.with_suffix(".txt")
        if not txt.exists():
            continue
        shutil.copy2(image, output_dir / image.name)
        save_text(output_dir / txt.name, sanitize_caption(read_first_line(txt)).strip() + "\n")
    return output_dir


def stage_caption_dataset_from_metadata(
    dataset_config: Path,
    output_dir: Path,
    filters: dict[str, Any] | None = None,
    exclude_filters: dict[str, Any] | None = None,
    sample_limit: int | None = None,
    balance_fields: list[str] | None = None,
) -> Path:
    config = load_dataset_config(dataset_config)
    samples = load_samples(config, read_image_size=False)
    selected = [
        sample
        for sample in samples
        if sample_matches_filters(sample.metadata, filters or {})
        and not sample_matches_any_filter(sample.metadata, exclude_filters or {})
    ]
    original_selected_count = len(selected)
    selected = limit_samples(selected, sample_limit=sample_limit, balance_fields=balance_fields or [])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    manifest = []
    for sample in selected:
        image = Path(sample.image.path).expanduser().resolve()
        staged_name = unique_stage_name(image.name, used_names)
        staged_image = output_dir / staged_name
        shutil.copy2(image, staged_image)
        caption = str(sample.label.caption or build_caption_from_metadata(sample.label.class_name, sample.metadata)).strip()
        if not caption:
            caption = "SAR image"
        staged_txt = staged_image.with_suffix(".txt")
        save_text(staged_txt, caption + "\n")
        manifest.append(
            {
                "source_image": image.as_posix(),
                "staged_image": staged_image.as_posix(),
                "staged_caption": staged_txt.as_posix(),
                "caption": caption,
                "metadata": sample.metadata,
            }
        )
    save_json(
        output_dir / "metadata_caption_manifest.json",
        {
            "dataset_config": dataset_config.as_posix(),
            "filters": filters or {},
            "exclude_filters": exclude_filters or {},
            "candidate_count": original_selected_count,
            "selected_count": len(selected),
            "sample_limit": sample_limit,
            "balance_fields": balance_fields or [],
            "items": manifest,
        },
    )
    return output_dir


def limit_samples(samples: list[Any], sample_limit: int | None, balance_fields: list[str]) -> list[Any]:
    if sample_limit is None or sample_limit <= 0 or len(samples) <= sample_limit:
        return samples
    if not balance_fields:
        return deterministic_sample_values(samples, sample_limit)
    buckets: dict[tuple[str, ...], list[Any]] = {}
    for sample in samples:
        key = tuple(str(sample.metadata.get(field, UNKNOWN)).lower() for field in balance_fields)
        buckets.setdefault(key, []).append(sample)
    for key in list(buckets):
        buckets[key] = deterministic_sample_values(buckets[key], len(buckets[key]))
    selected = []
    keys = sorted(buckets)
    while len(selected) < sample_limit and any(buckets.values()):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) >= sample_limit:
                    break
    return selected


def deterministic_sample_values(values: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[0]]
    last = len(values) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [values[index] for index in indexes]


def sample_matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = metadata.get(field, UNKNOWN)
        if not value_matches(actual, expected):
            return False
    return True


def sample_matches_any_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = metadata.get(field, UNKNOWN)
        if value_matches(actual, expected):
            return True
    return False


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(value_matches(actual, item) for item in expected)
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except (TypeError, ValueError):
        return str(actual).lower() == str(expected).lower()


def unique_stage_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    idx = 2
    while f"{stem}_{idx}{suffix}" in used_names:
        idx += 1
    value = f"{stem}_{idx}{suffix}"
    used_names.add(value)
    return value


def build_caption_from_metadata(class_name: str, metadata: dict[str, Any]) -> str:
    pieces = ["SAR image"]
    band = metadata.get("band")
    if band and band != UNKNOWN:
        pieces.append(f"{band}-band")
    if class_name and class_name != UNKNOWN:
        pieces.append(str(class_name))
    azimuth = metadata.get("azimuth_deg")
    if azimuth not in {None, UNKNOWN}:
        pieces.append(f"azimuth {azimuth} deg")
    pol = metadata.get("polarization")
    if pol and pol != UNKNOWN:
        pieces.append(f"{pol} polarization")
    return ", ".join(pieces)


def write_dataset_config(
    output_path: Path,
    image_dir: Path,
    resolution: str,
    batch_size: int,
    enable_bucket: bool,
    min_bucket_reso: int,
    max_bucket_reso: int,
    bucket_no_upscale: bool,
    caption_extension: str,
    num_repeats: int,
) -> None:
    width, height = parse_resolution(resolution)
    text = "\n".join(
        [
            "[[datasets]]",
            f"batch_size = {int(batch_size)}",
            f"enable_bucket = {toml_bool(enable_bucket)}",
            f"resolution = [{width}, {height}]",
            f"min_bucket_reso = {int(min_bucket_reso)}",
            f"max_bucket_reso = {int(max_bucket_reso)}",
            f"bucket_no_upscale = {toml_bool(bucket_no_upscale)}",
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = {toml_string(image_dir.as_posix())}",
            f"  caption_extension = {toml_string(caption_extension)}",
            f"  num_repeats = {int(num_repeats)}",
            "",
        ]
    )
    save_text(output_path, text)


def build_prompt_plan(
    profile: dict[str, Any],
    explicit_prompt: str | None,
    target_count: int,
    prompt_count: int,
    base_seed: int = 1026,
) -> list[dict[str, Any]]:
    count = max(0, int(target_count))
    if count == 0:
        return []
    captions = []
    if explicit_prompt:
        captions.append(sanitize_caption(explicit_prompt))
    for row in profile.get("sample_captions", []):
        caption = sanitize_caption(row.get("caption") or row.get("raw_caption") or "")
        if caption:
            captions.append(caption)
    if not captions:
        captions = ["SAR image"]
    plan = []
    for index in range(count):
        prompt = captions[index % len(captions)]
        plan.append(
            {
                "index": index + 1,
                "prompt": prompt,
                "seed": int(base_seed) + index * 7919,
                "source_caption_index": index % len(captions),
            }
        )
    return plan


def write_generated_sidecars(output_dir: Path, prompt_plan: list[dict[str, Any]], before_images: set[Path]) -> None:
    after_images = [path for path in iter_generation_images(output_dir) if path not in before_images]
    after_images = sorted(after_images, key=lambda p: natural_key(p.as_posix()))
    for image, plan in zip(after_images, prompt_plan):
        sidecar = image.with_suffix(".txt")
        if not sidecar.exists():
            save_text(sidecar, str(plan["prompt"]).strip() + "\n")


def iter_generation_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_OUTPUT_EXTS],
        key=lambda p: natural_key(p.as_posix()),
    )


def write_run_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    save_json(output_dir / "caption_dataset_profile.json", report["caption_profile"])
    save_text(output_dir / "caption_dataset_profile.md", render_caption_profile_markdown(report["caption_profile"]))
    save_json(output_dir / "diffusion_lora_run.json", report)
    save_text(output_dir / "diffusion_lora_run.md", render_run_markdown(report))
    train_command = report.get("train", {}).get("command") or []
    if train_command:
        save_text(output_dir / "train_command.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\n" + shell_join(train_command) + "\n")
    inference_commands = report.get("inference", {}).get("commands") or []
    if inference_commands:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        lines.extend(shell_join(command) for command in inference_commands)
        lines.append("")
        save_text(output_dir / "inference_commands.sh", "\n".join(lines))


def render_caption_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Caption Dataset Profile",
        "",
        f"- Dataset root: `{profile.get('dataset_root')}`",
        f"- Valid: {profile.get('valid')}",
        f"- Images: {profile.get('image_count')}",
        f"- TXT sidecars: {profile.get('txt_count')}",
        f"- Paired samples: {profile.get('paired_count')}",
        f"- Missing txt: {profile.get('missing_txt_count')}",
        f"- Empty captions: {profile.get('empty_caption_count')}",
        "",
        "## Image Probe",
        "",
        f"- Size counts: `{profile.get('image_probe', {}).get('size_counts')}`",
        f"- Mode counts: `{profile.get('image_probe', {}).get('mode_counts')}`",
        "",
        "## Top Caption Tokens",
        "",
    ]
    for item in profile.get("top_caption_tokens", [])[:20]:
        lines.append(f"- `{item['token']}`: {item['count']}")
    if profile.get("issues"):
        lines.extend(["", "## Issues", ""])
        for issue in profile["issues"]:
            lines.append(f"- {issue.get('severity')}: {issue.get('name')} ({issue.get('count', '')})")
    lines.append("")
    return "\n".join(lines)


def render_run_markdown(report: dict[str, Any]) -> str:
    train_command = report.get("train", {}).get("command") or []
    inference_commands = report.get("inference", {}).get("commands") or []
    lines = [
        "# Diffusion LoRA Generation Run",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Model family: `{report.get('model_family')}`",
        f"- Dataset: `{report.get('inputs', {}).get('dataset_root')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Generated output dir: `{report.get('generated_output_dir')}`",
        f"- Output LoRA name: `{report.get('output_name')}`",
        f"- Expected LoRA weights: `{report.get('expected_lora_weights')}`",
        f"- Target count: {report.get('target_count')}",
        f"- Training epochs: {report.get('training_epochs')}",
        "",
        "## Caption Dataset",
        "",
        f"- Valid: {report.get('caption_profile', {}).get('valid')}",
        f"- Paired samples: {report.get('caption_profile', {}).get('paired_count')}",
        f"- Issues: {len(report.get('caption_profile', {}).get('issues') or [])}",
        f"- Preparation: `{report.get('preparation')}`",
        "",
        "## Train Command",
        "",
        "```bash",
        shell_join(train_command) if train_command else "",
        "```",
        "",
        "## Inference Commands",
        "",
        f"- Command count: {len(inference_commands)}",
        "",
        "```bash",
    ]
    preview = inference_commands[:3]
    lines.extend(shell_join(command) for command in preview)
    if len(inference_commands) > len(preview):
        lines.append(f"# ... {len(inference_commands) - len(preview)} more commands")
    lines.extend(["```", ""])
    return "\n".join(lines)


def sanitize_caption(caption: str) -> str:
    value = CONTROL_ARG_RE.sub("", caption.strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,")


def read_first_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def parse_resolution(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    return 512, 512


def build_output_name(prefix: str, dataset_root: Path, model_family: str) -> str:
    raw = f"{prefix}_{dataset_root.name}_{model_family}"
    return re.sub(r"\W+", "_", raw, flags=re.UNICODE).strip("_")[:120]


def validate_dataset_root(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Dataset root must be a directory: {path}")


def toml_bool(value: bool) -> str:
    return "true" if value else "false"


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def prune_process_result(result: dict[str, Any]) -> dict[str, Any]:
    return prune_process_value(result)


def prune_process_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: prune_process_text(item) if key in {"stdout", "stderr"} else prune_process_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [prune_process_value(item) for item in value]
    return value


def prune_process_text(text: Any, head_chars: int = 2000, tail_chars: int = 12000) -> Any:
    if not isinstance(text, str):
        return text
    budget = head_chars + tail_chars
    if len(text) <= budget:
        return text
    omitted = len(text) - budget
    return text[:head_chars] + f"\n...[truncated {omitted} chars]...\n" + text[-tail_chars:]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    number = int(value)
    return number if number > 0 else None

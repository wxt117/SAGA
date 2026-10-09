from __future__ import annotations

import hashlib
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from saga.core.config import load_dataset_config, load_mapping, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.data.discovery import iter_images, natural_key
from saga.data.loader import load_samples


TRADITIONAL_AUGMENTATION_VERSION = "saga_traditional_augmentation_run_v1"


@dataclass
class TraditionalAugmentationSkillConfig:
    name: str = "TraditionalAugmentationSkill"
    description: str = "Fast deterministic SAR-aware image augmentation for classification-style target images."
    target_count: int | None = None
    multiplier: int = 2
    seed: int = 1026
    output_image_format: str = "png"
    preserve_sidecars: bool = True
    copy_originals: bool = False
    operations: list[str] = field(
        default_factory=lambda: [
            "hflip",
            "vflip",
            "rotate",
            "crop_resize",
            "intensity_scale",
            "speckle_noise",
            "gaussian_noise",
        ]
    )
    rotate_degrees: list[float] = field(default_factory=lambda: [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0, 90.0, 180.0, 270.0])
    crop_scale_range: tuple[float, float] = (0.86, 0.98)
    intensity_scale_range: tuple[float, float] = (0.85, 1.18)
    contrast_range: tuple[float, float] = (0.9, 1.12)
    speckle_std_range: tuple[float, float] = (0.03, 0.12)
    gaussian_std_range: tuple[float, float] = (2.0, 10.0)
    blur_probability: float = 0.08

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "TraditionalAugmentationSkillConfig":
        raw = raw or {}
        body = raw.get("traditional_augmentation", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            target_count=parse_optional_int(body.get("target_count", default.target_count)),
            multiplier=int(body.get("multiplier", default.multiplier)),
            seed=int(body.get("seed", default.seed)),
            output_image_format=str(body.get("output_image_format", default.output_image_format)).lower().lstrip("."),
            preserve_sidecars=parse_bool(body.get("preserve_sidecars", default.preserve_sidecars)),
            copy_originals=parse_bool(body.get("copy_originals", default.copy_originals)),
            operations=[str(item) for item in body.get("operations", default.operations)],
            rotate_degrees=[float(item) for item in body.get("rotate_degrees", default.rotate_degrees)],
            crop_scale_range=parse_range(body.get("crop_scale_range"), default.crop_scale_range),
            intensity_scale_range=parse_range(body.get("intensity_scale_range"), default.intensity_scale_range),
            contrast_range=parse_range(body.get("contrast_range"), default.contrast_range),
            speckle_std_range=parse_range(body.get("speckle_std_range"), default.speckle_std_range),
            gaussian_std_range=parse_range(body.get("gaussian_std_range"), default.gaussian_std_range),
            blur_probability=float(body.get("blur_probability", default.blur_probability)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "TraditionalAugmentationSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class TraditionalAugmentationSkill:
    def __init__(self, config: TraditionalAugmentationSkillConfig | None = None) -> None:
        self.config = config or TraditionalAugmentationSkillConfig()

    def run(
        self,
        dataset_root: str | Path,
        output_dir: str | Path,
        dataset_config: str | Path | None = None,
        filters: dict[str, Any] | None = None,
        exclude_filters: dict[str, Any] | None = None,
        target_count: int | None = None,
        multiplier: int | None = None,
        skill_overrides: dict[str, Any] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        override_report = apply_skill_overrides(self.config, skill_overrides or {})
        dataset_path = Path(dataset_root).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")

        selected = collect_input_items(
            dataset_root=dataset_path,
            dataset_config=Path(dataset_config).expanduser().resolve() if dataset_config else None,
            filters=filters or {},
            exclude_filters=exclude_filters or {},
        )
        requested_count = int(target_count or self.config.target_count or max(0, len(selected) * int(multiplier or self.config.multiplier)))
        plan = build_augmentation_plan(
            items=selected,
            config=self.config,
            target_count=requested_count,
            multiplier=int(multiplier or self.config.multiplier),
        )

        generated_dir = output_path / "generated_images"
        originals_dir = output_path / "originals"
        if dry_run:
            report = build_run_report(
                status="dry_run",
                message="Traditional SAR-aware augmentation plan generated.",
                dry_run=True,
                dataset_root=dataset_path,
                output_dir=output_path,
                generated_dir=generated_dir,
                items=selected,
                plan=plan,
                rows=[],
                config=self.config,
                started=started,
                dataset_config=dataset_config,
                filters=filters or {},
                exclude_filters=exclude_filters or {},
                override_report=override_report,
            )
            write_run_artifacts(output_path, report, plan=plan, rows=[])
            return report

        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        generated_dir.mkdir(parents=True, exist_ok=True)
        if self.config.copy_originals:
            if originals_dir.exists():
                shutil.rmtree(originals_dir)
            originals_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        errors = []
        for idx, plan_item in enumerate(plan):
            try:
                row = apply_plan_item(
                    plan_item=plan_item,
                    index=idx,
                    output_dir=generated_dir,
                    config=self.config,
                    copy_originals_dir=originals_dir if self.config.copy_originals else None,
                )
                rows.append(row)
            except Exception as exc:
                errors.append(
                    {
                        "source": plan_item["source_path"],
                        "operation": plan_item["operation"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        status = "succeeded" if not errors else ("warning" if rows else "failed")
        report = build_run_report(
            status=status,
            message="Traditional SAR-aware augmentation completed." if rows else "No augmented images were produced.",
            dry_run=False,
            dataset_root=dataset_path,
            output_dir=output_path,
            generated_dir=generated_dir,
            items=selected,
            plan=plan,
            rows=rows,
            config=self.config,
            started=started,
            dataset_config=dataset_config,
            filters=filters or {},
            exclude_filters=exclude_filters or {},
            errors=errors,
            override_report=override_report,
        )
        write_run_artifacts(output_path, report, plan=plan, rows=rows)
        return report


def run_traditional_augmentation_skill(
    dataset_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    dataset_config: str | Path | None = None,
    filters: dict[str, Any] | None = None,
    exclude_filters: dict[str, Any] | None = None,
    target_count: int | None = None,
    multiplier: int | None = None,
    skill_overrides: dict[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = TraditionalAugmentationSkillConfig.from_path(config_path)
    return TraditionalAugmentationSkill(config).run(
        dataset_root=dataset_root,
        output_dir=output_dir,
        dataset_config=dataset_config,
        filters=filters,
        exclude_filters=exclude_filters,
        target_count=target_count,
        multiplier=multiplier,
        skill_overrides=skill_overrides,
        dry_run=dry_run,
    )


def apply_skill_overrides(config: TraditionalAugmentationSkillConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    if not overrides:
        return {"applied": [], "ignored": []}
    allowed = {
        "operations",
        "rotate_degrees",
        "crop_scale_range",
        "intensity_scale_range",
        "contrast_range",
        "speckle_std_range",
        "gaussian_std_range",
        "blur_probability",
    }
    applied = []
    ignored = []
    for key, value in overrides.items():
        if key not in allowed:
            ignored.append({"field": key, "reason": "not_whitelisted"})
            continue
        old_value = getattr(config, key)
        if key == "operations":
            new_value = [str(item) for item in value] if isinstance(value, list) else old_value
        elif key in {"crop_scale_range", "intensity_scale_range", "contrast_range", "speckle_std_range", "gaussian_std_range"}:
            new_value = parse_range(value, old_value)
        elif key == "rotate_degrees":
            new_value = [float(item) for item in value] if isinstance(value, list) else old_value
        elif key == "blur_probability":
            new_value = max(0.0, min(1.0, float(value)))
        else:
            new_value = value
        setattr(config, key, new_value)
        applied.append({"field": key, "old_value": old_value, "new_value": new_value})
    return {"applied": applied, "ignored": ignored}


def collect_input_items(
    dataset_root: Path,
    dataset_config: Path | None,
    filters: dict[str, Any],
    exclude_filters: dict[str, Any],
) -> list[dict[str, Any]]:
    if dataset_config and dataset_config.exists():
        samples = load_samples(load_dataset_config(dataset_config), read_image_size=False)
        items = []
        for sample in samples:
            image = Path(sample.image.path).expanduser().resolve()
            if not is_under(image, dataset_root):
                continue
            if not metadata_matches_filters(sample.metadata, filters):
                continue
            if metadata_matches_any_filter(sample.metadata, exclude_filters):
                continue
            items.append(
                {
                    "source_path": image.as_posix(),
                    "class_name": sample.label.class_name,
                    "caption": sample.label.caption,
                    "metadata": sample.metadata,
                    "split": sample.split,
                }
            )
        return items

    items = []
    for image in iter_images(dataset_root):
        items.append(
            {
                "source_path": image.as_posix(),
                "class_name": image.parent.name,
                "caption": read_sidecar(image) or f"SAR image, {image.parent.name}",
                "metadata": {"class": image.parent.name, "original_relpath": safe_relative(image, dataset_root)},
                "split": "train",
            }
        )
    return items


def build_augmentation_plan(
    items: list[dict[str, Any]],
    config: TraditionalAugmentationSkillConfig,
    target_count: int,
    multiplier: int,
) -> list[dict[str, Any]]:
    if not items or target_count <= 0:
        return []
    plan = []
    sorted_items = sorted(items, key=lambda item: natural_key(item["source_path"]))
    operation_cycle = config.operations or ["hflip"]
    total = int(target_count)
    if total <= 0:
        total = len(sorted_items) * max(1, multiplier)
    for index in range(total):
        source = sorted_items[index % len(sorted_items)]
        op = operation_cycle[index % len(operation_cycle)]
        seed = stable_seed(config.seed, source["source_path"], index, op)
        rng = random.Random(seed)
        params = sample_operation_params(op, config, rng)
        plan.append(
            {
                "index": index + 1,
                "source_path": source["source_path"],
                "operation": op,
                "params": params,
                "seed": seed,
                "class_name": source.get("class_name"),
                "caption": source.get("caption"),
                "metadata": source.get("metadata") or {},
                "split": source.get("split") or "train",
            }
        )
    return plan


def sample_operation_params(op: str, config: TraditionalAugmentationSkillConfig, rng: random.Random) -> dict[str, Any]:
    if op == "rotate":
        return {"degrees": rng.choice(config.rotate_degrees)}
    if op == "crop_resize":
        return {"scale": round(rng.uniform(*config.crop_scale_range), 4)}
    if op == "intensity_scale":
        return {
            "scale": round(rng.uniform(*config.intensity_scale_range), 4),
            "contrast": round(rng.uniform(*config.contrast_range), 4),
        }
    if op == "speckle_noise":
        return {"std": round(rng.uniform(*config.speckle_std_range), 5)}
    if op == "gaussian_noise":
        return {"std": round(rng.uniform(*config.gaussian_std_range), 4)}
    if op == "blur":
        return {"radius": round(rng.uniform(0.3, 0.9), 3)}
    return {}


def apply_plan_item(
    plan_item: dict[str, Any],
    index: int,
    output_dir: Path,
    config: TraditionalAugmentationSkillConfig,
    copy_originals_dir: Path | None = None,
) -> dict[str, Any]:
    source = Path(plan_item["source_path"]).expanduser().resolve()
    with Image.open(source) as image:
        original_mode = image.mode
        transformed = apply_operation(image, plan_item["operation"], plan_item["params"], seed=int(plan_item["seed"]), config=config)
        target_name = build_augmented_name(source, plan_item["operation"], index, config.output_image_format)
        target = output_dir / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        save_image(transformed, target, original_mode=original_mode)
    caption = augment_caption(plan_item.get("caption") or read_sidecar(source), plan_item["operation"], plan_item["params"])
    if config.preserve_sidecars:
        save_text(target.with_suffix(".txt"), caption.strip() + "\n")
    original_copy = None
    if copy_originals_dir is not None:
        original_copy = copy_original_image(source, copy_originals_dir)
    return {
        "sample_id": build_sample_id(source, index, plan_item["operation"]),
        "source_path": source.as_posix(),
        "image_path": target.as_posix(),
        "sidecar_path": target.with_suffix(".txt").as_posix() if config.preserve_sidecars else None,
        "original_copy": original_copy,
        "operation": plan_item["operation"],
        "params": plan_item["params"],
        "seed": plan_item["seed"],
        "split": plan_item.get("split") or "train",
        "class_name": plan_item.get("class_name"),
        "caption": caption,
        "metadata": {
            **(plan_item.get("metadata") or {}),
            "augmentation": "traditional_sar",
            "augmentation_operation": plan_item["operation"],
            "generated": True,
        },
    }


def apply_operation(
    image: Image.Image,
    operation: str,
    params: dict[str, Any],
    seed: int,
    config: TraditionalAugmentationSkillConfig,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    if operation == "hflip":
        return ImageOps.mirror(image)
    if operation == "vflip":
        return ImageOps.flip(image)
    if operation == "rotate":
        return rotate_preserve_size(image, float(params.get("degrees", 0.0)))
    if operation == "crop_resize":
        return crop_resize(image, float(params.get("scale", 0.92)))
    if operation == "intensity_scale":
        return intensity_scale(image, scale=float(params.get("scale", 1.0)), contrast=float(params.get("contrast", 1.0)))
    if operation == "speckle_noise":
        return speckle_noise(image, std=float(params.get("std", 0.06)), rng=rng)
    if operation == "gaussian_noise":
        return additive_noise(image, std=float(params.get("std", 5.0)), rng=rng)
    if operation == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(params.get("radius", 0.5))))
    return image.copy()


def rotate_preserve_size(image: Image.Image, degrees: float) -> Image.Image:
    if abs(degrees) in {90.0, 180.0, 270.0}:
        return image.rotate(degrees, expand=False)
    return image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)


def crop_resize(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    scale = min(max(scale, 0.5), 1.0)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def intensity_scale(image: Image.Image, scale: float, contrast: float) -> Image.Image:
    working = ImageEnhance.Brightness(image).enhance(scale)
    return ImageEnhance.Contrast(working).enhance(contrast)


def speckle_noise(image: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    arr, mode = image_to_float_array(image)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape)
    out = arr + arr * noise
    return float_array_to_image(out, mode=mode)


def additive_noise(image: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    arr, mode = image_to_float_array(image)
    out = arr + rng.normal(loc=0.0, scale=std, size=arr.shape)
    return float_array_to_image(out, mode=mode)


def image_to_float_array(image: Image.Image) -> tuple[np.ndarray, str]:
    mode = image.mode
    if mode in {"I;16", "I;16B", "I;16L"}:
        arr = np.asarray(image, dtype=np.float32)
        return arr, mode
    if mode == "I":
        arr = np.asarray(image, dtype=np.float32)
        return arr, mode
    arr = np.asarray(image.convert("RGB") if mode not in {"L", "RGB", "RGBA"} else image, dtype=np.float32)
    return arr, mode


def float_array_to_image(arr: np.ndarray, mode: str) -> Image.Image:
    if mode in {"I;16", "I;16B", "I;16L"}:
        clipped = np.clip(arr, 0, 65535).astype(np.uint16)
        return Image.fromarray(clipped)
    if mode == "I":
        clipped = np.clip(arr, 0, np.iinfo(np.int32).max).astype(np.int32)
        return Image.fromarray(clipped)
    clipped = np.clip(arr, 0, 255).astype(np.uint8)
    if clipped.ndim == 2:
        return Image.fromarray(clipped, mode="L")
    if clipped.shape[-1] == 4 and mode == "RGBA":
        return Image.fromarray(clipped, mode="RGBA")
    return Image.fromarray(clipped[..., :3], mode="RGB")


def save_image(image: Image.Image, path: Path, original_mode: str) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    image.save(path)


def build_run_report(
    status: str,
    message: str,
    dry_run: bool,
    dataset_root: Path,
    output_dir: Path,
    generated_dir: Path,
    items: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: TraditionalAugmentationSkillConfig,
    started: float,
    dataset_config: str | Path | None,
    filters: dict[str, Any],
    exclude_filters: dict[str, Any],
    errors: list[dict[str, Any]] | None = None,
    override_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operation_counts = Counter(item["operation"] for item in plan)
    return {
        "schema_version": TRADITIONAL_AUGMENTATION_VERSION,
        "skill": config.name,
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "dataset_root": dataset_root.as_posix(),
        "dataset_config": Path(dataset_config).expanduser().resolve().as_posix() if dataset_config else None,
        "output_dir": output_dir.as_posix(),
        "generated_output_dir": generated_dir.as_posix(),
        "input_count": len(items),
        "planned_count": len(plan),
        "generated_count": len(rows),
        "filters": filters,
        "exclude_filters": exclude_filters,
        "operation_counts": dict(operation_counts.most_common()),
        "override_report": override_report or {"applied": [], "ignored": []},
        "plan_count": len(plan),
        "plan_digest": digest_jsonable(plan),
        "config": asdict(config),
        "artifacts": {
            "generated_images": generated_dir.as_posix(),
            "manifest_jsonl": (output_dir / "manifest.jsonl").as_posix(),
            "run_json": (output_dir / "traditional_aug_run.json").as_posix(),
            "run_md": (output_dir / "traditional_aug_run.md").as_posix(),
            "plan_json": (output_dir / "augmentation_plan.json").as_posix(),
        },
        "plan_preview": plan[:20],
        "errors": errors or [],
        "elapsed_seconds": round(time.time() - started, 3),
    }


def write_run_artifacts(
    output_dir: Path,
    report: dict[str, Any],
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    save_json(output_dir / "traditional_aug_run.json", report)
    save_json(output_dir / "augmentation_plan.json", plan)
    save_text(output_dir / "traditional_aug_run.md", render_run_markdown(report))
    write_jsonl(output_dir / "manifest.jsonl", rows)


def render_run_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Traditional SAR-aware Augmentation Run",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Dataset root: `{report.get('dataset_root')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Generated output dir: `{report.get('generated_output_dir')}`",
        f"- Input images: {report.get('input_count')}",
        f"- Planned outputs: {report.get('planned_count')}",
        f"- Generated outputs: {report.get('generated_count')}",
        f"- Operation counts: `{report.get('operation_counts')}`",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in (report.get("artifacts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    errors = report.get("errors") or []
    if errors:
        lines.extend(["", "## Errors", ""])
        for item in errors[:20]:
            lines.append(f"- `{item.get('operation')}` on `{item.get('source')}`: {item.get('error')}")
    lines.append("")
    return "\n".join(lines)


def metadata_matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in (filters or {}).items():
        if not value_matches(metadata.get(key), expected):
            return False
    return True


def metadata_matches_any_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in (filters or {}).items():
        if value_matches(metadata.get(key), expected):
            return True
    return False


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(value_matches(actual, item) for item in expected)
    try:
        return abs(float(actual) - float(expected)) < 1e-6
    except (TypeError, ValueError):
        return str(actual).lower() == str(expected).lower()


def stable_seed(base_seed: int, source_path: str, index: int, operation: str) -> int:
    digest = hashlib.sha1(f"{base_seed}|{source_path}|{index}|{operation}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def digest_jsonable(value: Any) -> str:
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_augmented_name(source: Path, operation: str, index: int, image_format: str) -> str:
    suffix = image_format.lower().lstrip(".") or source.suffix.lower().lstrip(".") or "png"
    stem = source.stem
    return f"{stem}__trad_{index + 1:06d}_{operation}.{suffix}"


def build_sample_id(source: Path, index: int, operation: str) -> str:
    digest = hashlib.sha1(f"{source.as_posix()}|{index}|{operation}".encode("utf-8")).hexdigest()[:14]
    return f"trad_{digest}"


def augment_caption(caption: str, operation: str, params: dict[str, Any]) -> str:
    base = caption.strip() or "SAR image"
    return f"{base}, traditional augmentation {operation}"


def read_sidecar(image: Path) -> str:
    sidecar = image.with_suffix(".txt")
    if not sidecar.exists():
        return ""
    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def copy_original_image(source: Path, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    if target.exists():
        stem = source.stem
        suffix = source.suffix
        idx = 2
        while (output_dir / f"{stem}_{idx}{suffix}").exists():
            idx += 1
        target = output_dir / f"{stem}_{idx}{suffix}"
    shutil.copy2(source, target)
    sidecar = source.with_suffix(".txt")
    if sidecar.exists():
        shutil.copy2(sidecar, target.with_suffix(".txt"))
    return target.as_posix()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


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


def parse_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.replace(",", " ").split() if piece.strip()]
    else:
        pieces = list(value)
    if len(pieces) < 2:
        return default
    low = float(pieces[0])
    high = float(pieces[1])
    return (min(low, high), max(low, high))

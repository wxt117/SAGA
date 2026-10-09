from __future__ import annotations

import hashlib
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from saga.core.config import load_mapping, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.data.discovery import iter_images, natural_key


TARGET_BACKGROUND_COMPOSITION_VERSION = "saga_target_background_composition_run_v1"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class TargetBackgroundCompositionSkillConfig:
    name: str = "TargetBackgroundCompositionSkill"
    description: str = "SAR-aware target/background image fusion for synthetic scene composition."
    target_count: int = 20
    blend_mode: str = "feather"
    mask_mode: str = "auto"
    placement_policy: str = "random"
    seed: int = 1026
    output_image_format: str = "png"
    target_scale_range: tuple[float, float] = (0.65, 1.15)
    target_fraction_range: tuple[float, float] = (0.08, 0.32)
    feather_radius: int = 9
    mask_threshold_percentile: float = 82.0
    mask_min_area_ratio: float = 0.002
    mask_dilate: int = 2
    mask_erode: int = 0
    intensity_match: bool = True
    preserve_target_contrast: float = 0.85
    poisson_clone_mode: str = "normal"
    allow_partial_target: bool = False
    copy_sidecars: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "TargetBackgroundCompositionSkillConfig":
        raw = raw or {}
        body = raw.get("target_background_composition", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            target_count=int(body.get("target_count", default.target_count)),
            blend_mode=str(body.get("blend_mode", default.blend_mode)),
            mask_mode=str(body.get("mask_mode", default.mask_mode)),
            placement_policy=str(body.get("placement_policy", default.placement_policy)),
            seed=int(body.get("seed", default.seed)),
            output_image_format=str(body.get("output_image_format", default.output_image_format)).lower().lstrip("."),
            target_scale_range=parse_range(body.get("target_scale_range"), default.target_scale_range),
            target_fraction_range=parse_range(body.get("target_fraction_range"), default.target_fraction_range),
            feather_radius=int(body.get("feather_radius", default.feather_radius)),
            mask_threshold_percentile=float(body.get("mask_threshold_percentile", default.mask_threshold_percentile)),
            mask_min_area_ratio=float(body.get("mask_min_area_ratio", default.mask_min_area_ratio)),
            mask_dilate=int(body.get("mask_dilate", default.mask_dilate)),
            mask_erode=int(body.get("mask_erode", default.mask_erode)),
            intensity_match=parse_bool(body.get("intensity_match", default.intensity_match)),
            preserve_target_contrast=float(body.get("preserve_target_contrast", default.preserve_target_contrast)),
            poisson_clone_mode=str(body.get("poisson_clone_mode", default.poisson_clone_mode)),
            allow_partial_target=parse_bool(body.get("allow_partial_target", default.allow_partial_target)),
            copy_sidecars=parse_bool(body.get("copy_sidecars", default.copy_sidecars)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "TargetBackgroundCompositionSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class TargetBackgroundCompositionSkill:
    def __init__(self, config: TargetBackgroundCompositionSkillConfig | None = None) -> None:
        self.config = config or TargetBackgroundCompositionSkillConfig()

    def run(
        self,
        target_dir: str | Path,
        background_dir: str | Path,
        output_dir: str | Path,
        mask_dir: str | Path | None = None,
        target_count: int | None = None,
        blend_mode: str | None = None,
        mask_mode: str | None = None,
        placement_policy: str | None = None,
        skill_overrides: dict[str, Any] | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        cfg = apply_skill_overrides(self.config, skill_overrides or {})
        target_path = Path(target_dir).expanduser().resolve()
        background_path = Path(background_dir).expanduser().resolve()
        mask_path = Path(mask_dir).expanduser().resolve() if mask_dir else None
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        validate_dir(target_path, "target_dir")
        validate_dir(background_path, "background_dir")
        if mask_path and not mask_path.is_dir():
            raise FileNotFoundError(f"mask_dir does not exist or is not a directory: {mask_path}")

        targets = iter_images(target_path)
        backgrounds = iter_images(background_path)
        masks = index_masks(mask_path) if mask_path else {}
        if not targets:
            raise ValueError(f"No target images found in {target_path}")
        if not backgrounds:
            raise ValueError(f"No background images found in {background_path}")

        requested_count = int(target_count or cfg.target_count)
        mode = normalize_blend_mode(blend_mode or cfg.blend_mode)
        current_mask_mode = str(mask_mode or cfg.mask_mode).lower()
        current_placement = str(placement_policy or cfg.placement_policy).lower()
        plan = build_composition_plan(
            targets=targets,
            backgrounds=backgrounds,
            masks=masks,
            count=requested_count,
            seed=cfg.seed,
            placement_policy=current_placement,
            blend_mode=mode,
            mask_mode=current_mask_mode,
        )

        composed_dir = output_path / "composed_images"
        mask_out_dir = output_path / "masks"
        preview_dir = output_path / "previews"
        if dry_run:
            report = build_report(
                status="dry_run",
                message="Target/background composition plan generated.",
                dry_run=True,
                target_dir=target_path,
                background_dir=background_path,
                mask_dir=mask_path,
                output_dir=output_path,
                composed_dir=composed_dir,
                plan=plan,
                rows=[],
                errors=[],
                config=cfg,
                override_report=skill_overrides or {},
                started=started,
            )
            write_artifacts(output_path, report, plan=plan, rows=[])
            return report

        reset_dir(composed_dir)
        reset_dir(mask_out_dir)
        reset_dir(preview_dir)
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(plan, start=1):
            try:
                row = compose_plan_item(
                    item=item,
                    index=index,
                    config=cfg,
                    composed_dir=composed_dir,
                    mask_out_dir=mask_out_dir,
                    preview_dir=preview_dir,
                )
                rows.append(row)
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "target": item.get("target"),
                        "background": item.get("background"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        status = "succeeded" if rows and not errors else "warning" if rows else "failed"
        report = build_report(
            status=status,
            message="Target/background composition completed." if rows else "No composited images were produced.",
            dry_run=False,
            target_dir=target_path,
            background_dir=background_path,
            mask_dir=mask_path,
            output_dir=output_path,
            composed_dir=composed_dir,
            plan=plan,
            rows=rows,
            errors=errors,
            config=cfg,
            override_report=skill_overrides or {},
            started=started,
        )
        write_artifacts(output_path, report, plan=plan, rows=rows)
        return report


def run_target_background_composition_skill(
    target_dir: str | Path,
    background_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    mask_dir: str | Path | None = None,
    target_count: int | None = None,
    blend_mode: str | None = None,
    mask_mode: str | None = None,
    placement_policy: str | None = None,
    skill_overrides: dict[str, Any] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = TargetBackgroundCompositionSkillConfig.from_path(config_path)
    return TargetBackgroundCompositionSkill(config).run(
        target_dir=target_dir,
        background_dir=background_dir,
        output_dir=output_dir,
        mask_dir=mask_dir,
        target_count=target_count,
        blend_mode=blend_mode,
        mask_mode=mask_mode,
        placement_policy=placement_policy,
        skill_overrides=skill_overrides,
        dry_run=dry_run,
    )


def build_composition_plan(
    targets: list[Path],
    backgrounds: list[Path],
    masks: dict[str, Path],
    count: int,
    seed: int,
    placement_policy: str,
    blend_mode: str,
    mask_mode: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    targets = sorted(targets, key=lambda p: natural_key(p.as_posix()))
    backgrounds = sorted(backgrounds, key=lambda p: natural_key(p.as_posix()))
    plan = []
    for index in range(max(0, count)):
        target = targets[index % len(targets)]
        background = backgrounds[(index // max(1, len(targets))) % len(backgrounds)]
        if placement_policy == "random":
            background = rng.choice(backgrounds)
            target = rng.choice(targets)
        mask = masks.get(target.stem)
        plan.append(
            {
                "index": index + 1,
                "target": target.as_posix(),
                "background": background.as_posix(),
                "mask": mask.as_posix() if mask else None,
                "blend_mode": blend_mode,
                "mask_mode": mask_mode,
                "seed": stable_seed(seed, target, background, index),
                "placement_policy": placement_policy,
            }
        )
    return plan


def compose_plan_item(
    item: dict[str, Any],
    index: int,
    config: TargetBackgroundCompositionSkillConfig,
    composed_dir: Path,
    mask_out_dir: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    rng = random.Random(int(item["seed"]))
    target_image = load_grayscale(Path(item["target"]))
    background_image = load_grayscale(Path(item["background"]))
    mask = load_or_infer_mask(
        target=target_image,
        mask_path=Path(item["mask"]) if item.get("mask") else None,
        config=config,
    )
    target_crop, mask_crop = crop_to_mask(target_image, mask)
    target_resized, mask_resized, scale = resize_target_for_background(target_crop, mask_crop, background_image, config, rng)
    x, y = choose_position(
        background_image.size,
        target_resized.size,
        placement_policy=str(item.get("placement_policy") or config.placement_policy),
        rng=rng,
    )
    composed, alpha = blend_target(
        background=background_image,
        target=target_resized,
        mask=mask_resized,
        x=x,
        y=y,
        blend_mode=str(item["blend_mode"]),
        config=config,
    )
    out_name = f"composition_{index:06d}.{config.output_image_format}"
    mask_name = f"composition_{index:06d}_mask.png"
    preview_name = f"composition_{index:06d}_preview.png"
    out_path = composed_dir / out_name
    mask_path = mask_out_dir / mask_name
    preview_path = preview_dir / preview_name
    save_image(composed, out_path)
    save_image(alpha, mask_path)
    save_preview(background_image, composed, alpha, preview_path)
    sidecar = out_path.with_suffix(".txt")
    caption = build_caption(Path(item["target"]), Path(item["background"]), item)
    save_text(sidecar, caption + "\n")
    bbox = mask_bbox(alpha)
    return {
        "index": index,
        "output_image": out_path.as_posix(),
        "output_mask": mask_path.as_posix(),
        "preview": preview_path.as_posix(),
        "caption_file": sidecar.as_posix(),
        "target": item["target"],
        "background": item["background"],
        "mask_source": item.get("mask") or "auto",
        "blend_mode": item["blend_mode"],
        "placement": {"x": x, "y": y},
        "scale": round(float(scale), 6),
        "bbox_xyxy": bbox,
        "bbox_xywh": [bbox[0], bbox[1], max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])] if bbox else None,
        "caption": caption,
    }


def load_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


def load_or_infer_mask(target: Image.Image, mask_path: Path | None, config: TargetBackgroundCompositionSkillConfig) -> Image.Image:
    if mask_path and mask_path.exists() and str(config.mask_mode).lower() in {"auto", "provided", "explicit"}:
        with Image.open(mask_path) as image:
            return image.convert("L").resize(target.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 0 else 0)
    arr = np.asarray(target, dtype=np.float32)
    if arr.size == 0:
        return Image.new("L", target.size, 0)
    threshold = np.percentile(arr, float(config.mask_threshold_percentile))
    mask = (arr >= threshold).astype(np.uint8) * 255
    if mask.mean() / 255.0 < float(config.mask_min_area_ratio):
        threshold = np.percentile(arr, 100.0 * (1.0 - float(config.mask_min_area_ratio)))
        mask = (arr >= threshold).astype(np.uint8) * 255
    image = Image.fromarray(mask, mode="L")
    if config.mask_erode > 0:
        for _ in range(config.mask_erode):
            image = image.filter(ImageFilter.MinFilter(3))
    if config.mask_dilate > 0:
        for _ in range(config.mask_dilate):
            image = image.filter(ImageFilter.MaxFilter(3))
    return image.point(lambda v: 255 if v > 0 else 0)


def crop_to_mask(target: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    bbox = mask.getbbox()
    if not bbox:
        return target, Image.new("L", target.size, 255)
    pad_x = max(4, int((bbox[2] - bbox[0]) * 0.18))
    pad_y = max(4, int((bbox[3] - bbox[1]) * 0.18))
    crop_box = (
        max(0, bbox[0] - pad_x),
        max(0, bbox[1] - pad_y),
        min(target.size[0], bbox[2] + pad_x),
        min(target.size[1], bbox[3] + pad_y),
    )
    return target.crop(crop_box), mask.crop(crop_box)


def resize_target_for_background(
    target: Image.Image,
    mask: Image.Image,
    background: Image.Image,
    config: TargetBackgroundCompositionSkillConfig,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, float]:
    bg_w, bg_h = background.size
    target_w, target_h = target.size
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target crop has invalid size.")
    fraction = rng.uniform(*config.target_fraction_range)
    max_side = max(8, int(min(bg_w, bg_h) * fraction))
    scale_by_fraction = max_side / max(target_w, target_h)
    jitter = rng.uniform(*config.target_scale_range)
    scale = max(0.02, scale_by_fraction * jitter)
    new_w = max(4, min(bg_w, int(round(target_w * scale))))
    new_h = max(4, min(bg_h, int(round(target_h * scale))))
    if not config.allow_partial_target:
        new_w = min(new_w, bg_w)
        new_h = min(new_h, bg_h)
    resized_target = target.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_mask = mask.resize((new_w, new_h), Image.Resampling.BICUBIC).point(lambda v: 255 if v > 20 else 0)
    return resized_target, resized_mask, scale


def choose_position(
    background_size: tuple[int, int],
    target_size: tuple[int, int],
    placement_policy: str,
    rng: random.Random,
) -> tuple[int, int]:
    bg_w, bg_h = background_size
    target_w, target_h = target_size
    if str(placement_policy).lower() == "center":
        return max(0, (bg_w - target_w) // 2), max(0, (bg_h - target_h) // 2)
    max_x = max(0, bg_w - target_w)
    max_y = max(0, bg_h - target_h)
    margin_x = int(bg_w * 0.05) if max_x > 4 else 0
    margin_y = int(bg_h * 0.05) if max_y > 4 else 0
    low_x, high_x = min(margin_x, max_x), max(0, max_x - margin_x)
    low_y, high_y = min(margin_y, max_y), max(0, max_y - margin_y)
    return rng.randint(low_x, max(low_x, high_x)), rng.randint(low_y, max(low_y, high_y))


def blend_target(
    background: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    x: int,
    y: int,
    blend_mode: str,
    config: TargetBackgroundCompositionSkillConfig,
) -> tuple[Image.Image, Image.Image]:
    bg_arr = np.asarray(background, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    mask_arr = (np.asarray(mask, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    h, w = target_arr.shape[:2]
    roi = bg_arr[y : y + h, x : x + w]
    matched = match_intensity(target_arr, roi, mask_arr, config) if config.intensity_match else target_arr
    if blend_mode == "poisson":
        composed = poisson_blend(background, matched, mask_arr, x, y, config)
    elif blend_mode == "laplacian":
        composed = laplacian_blend(bg_arr, matched, mask_arr, x, y, config)
    elif blend_mode == "hard":
        composed = hard_blend(bg_arr, matched, mask_arr, x, y)
    else:
        composed = feather_blend(bg_arr, matched, mask_arr, x, y, config)
    alpha = np.zeros_like(bg_arr, dtype=np.float32)
    alpha[y : y + h, x : x + w] = (mask_arr > 0.02).astype(np.float32) * 255.0
    return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), mode="L"), Image.fromarray(alpha.astype(np.uint8), mode="L")


def match_intensity(
    target: np.ndarray,
    roi: np.ndarray,
    mask: np.ndarray,
    config: TargetBackgroundCompositionSkillConfig,
) -> np.ndarray:
    inside = mask > 0.1
    if not np.any(inside):
        return target
    edge_alpha = feather_alpha(mask, max(3, int(config.feather_radius) * 2))
    ring = (edge_alpha > 0.02) & (edge_alpha < 0.45)
    background_values = roi[ring] if np.any(ring) else roi.reshape(-1)
    target_values = target[inside]
    bg_mean = float(np.mean(background_values))
    bg_std = float(np.std(background_values) + 1e-6)
    target_mean = float(np.mean(target_values))
    target_std = float(np.std(target_values) + 1e-6)
    normalized = (target - target_mean) / target_std
    contrast = float(config.preserve_target_contrast)
    matched = normalized * (bg_std * contrast + target_std * (1.0 - contrast)) + bg_mean
    return np.clip(matched, 0, 255)


def feather_blend(
    background: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    config: TargetBackgroundCompositionSkillConfig,
) -> np.ndarray:
    output = background.copy()
    h, w = target.shape[:2]
    alpha = feather_alpha(mask, int(config.feather_radius))
    roi = output[y : y + h, x : x + w]
    output[y : y + h, x : x + w] = target * alpha + roi * (1.0 - alpha)
    return output


def hard_blend(
    background: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    output = background.copy()
    h, w = target.shape[:2]
    alpha = (mask > 0.05).astype(np.float32)
    roi = output[y : y + h, x : x + w]
    output[y : y + h, x : x + w] = target * alpha + roi * (1.0 - alpha)
    return output


def laplacian_blend(
    background: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    config: TargetBackgroundCompositionSkillConfig,
) -> np.ndarray:
    output = background.copy()
    h, w = target.shape[:2]
    roi = output[y : y + h, x : x + w]
    alpha = feather_alpha(mask, max(3, int(config.feather_radius)))
    levels = max(2, min(5, int(math.log2(max(8, min(h, w)))) - 2))
    blended = pyramid_blend(roi, target, alpha, levels=levels)
    output[y : y + h, x : x + w] = blended
    return output


def poisson_blend(
    background: Image.Image,
    target: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    config: TargetBackgroundCompositionSkillConfig,
) -> np.ndarray:
    try:
        import cv2
    except Exception:
        bg_arr = np.asarray(background, dtype=np.float32)
        return feather_blend(bg_arr, target, mask, x, y, config)
    target_u8 = np.clip(target, 0, 255).astype(np.uint8)
    mask_u8 = ((mask > 0.05).astype(np.uint8) * 255)
    src = cv2.cvtColor(target_u8, cv2.COLOR_GRAY2BGR)
    dst = cv2.cvtColor(np.asarray(background, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    center = (int(x + target.shape[1] / 2), int(y + target.shape[0] / 2))
    clone_mode = cv2.MIXED_CLONE if str(config.poisson_clone_mode).lower() == "mixed" else cv2.NORMAL_CLONE
    try:
        blended = cv2.seamlessClone(src, dst, mask_u8, center, clone_mode)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32)
    except Exception:
        bg_arr = np.asarray(background, dtype=np.float32)
        return feather_blend(bg_arr, target, mask, x, y, config)


def feather_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def pyramid_blend(background: np.ndarray, target: np.ndarray, alpha: np.ndarray, levels: int) -> np.ndarray:
    bg_levels = gaussian_pyramid(background, levels)
    target_levels = gaussian_pyramid(target, levels)
    alpha_levels = gaussian_pyramid(alpha, levels)
    bg_lap = laplacian_pyramid(bg_levels)
    target_lap = laplacian_pyramid(target_levels)
    blended = []
    for bg_l, target_l, alpha_l in zip(bg_lap, target_lap, alpha_levels):
        blended.append(target_l * alpha_l + bg_l * (1.0 - alpha_l))
    current = blended[-1]
    for level in reversed(blended[:-1]):
        current = resize_array(current, level.shape) + level
    return np.clip(current, 0, 255)


def gaussian_pyramid(arr: np.ndarray, levels: int) -> list[np.ndarray]:
    values = [arr.astype(np.float32)]
    current = arr.astype(np.float32)
    for _ in range(max(1, levels - 1)):
        if min(current.shape[:2]) <= 8:
            break
        current = blur_and_downsample(current)
        values.append(current)
    return values


def laplacian_pyramid(levels: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for idx in range(len(levels) - 1):
        expanded = resize_array(levels[idx + 1], levels[idx].shape)
        out.append(levels[idx] - expanded)
    out.append(levels[-1])
    return out


def blur_and_downsample(arr: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L" if arr.ndim == 2 else None)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.asarray(image.resize((max(1, arr.shape[1] // 2), max(1, arr.shape[0] // 2)), Image.Resampling.BICUBIC), dtype=np.float32)


def resize_array(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L" if arr.ndim == 2 else None)
    resized = image.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32)


def save_preview(background: Image.Image, composed: Image.Image, alpha: Image.Image, path: Path) -> None:
    bg_rgb = background.convert("RGB")
    comp_rgb = composed.convert("RGB")
    overlay = comp_rgb.copy()
    red = Image.new("RGB", alpha.size, (255, 0, 0))
    overlay = Image.composite(red, overlay, alpha.point(lambda v: min(120, v)))
    w, h = background.size
    preview = Image.new("RGB", (w * 3, h), (0, 0, 0))
    preview.paste(bg_rgb, (0, 0))
    preview.paste(comp_rgb, (w, 0))
    preview.paste(overlay, (w * 2, 0))
    preview.save(path)


def mask_bbox(mask: Image.Image) -> list[int] | None:
    bbox = mask.getbbox()
    if not bbox:
        return None
    return [int(value) for value in bbox]


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def build_caption(target: Path, background: Path, item: dict[str, Any]) -> str:
    return (
        "SAR composed image, "
        f"target {target.stem}, background {background.stem}, "
        f"blend {item.get('blend_mode')}"
    )


def build_report(
    status: str,
    message: str,
    dry_run: bool,
    target_dir: Path,
    background_dir: Path,
    mask_dir: Path | None,
    output_dir: Path,
    composed_dir: Path,
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None,
    config: TargetBackgroundCompositionSkillConfig,
    override_report: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": TARGET_BACKGROUND_COMPOSITION_VERSION,
        "skill": config.name,
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "target_dir": target_dir.as_posix(),
        "background_dir": background_dir.as_posix(),
        "mask_dir": mask_dir.as_posix() if mask_dir else None,
        "output_dir": output_dir.as_posix(),
        "generated_output_dir": composed_dir.as_posix(),
        "target_count": len(plan),
        "generated_count": len(rows),
        "plan_count": len(plan),
        "errors": errors or [],
        "config": asdict(config),
        "skill_overrides": override_report,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output_dir / "target_background_composition_report.json").as_posix(),
            "report_md": (output_dir / "target_background_composition_report.md").as_posix(),
            "plan_json": (output_dir / "composition_plan.json").as_posix(),
            "manifest_jsonl": (output_dir / "composition_manifest.jsonl").as_posix(),
            "generated_dir": composed_dir.as_posix(),
            "mask_dir": (output_dir / "masks").as_posix(),
            "preview_dir": (output_dir / "previews").as_posix(),
        },
        "notes": [
            "This is deterministic SAR-aware image fusion for composition, not a downstream benefit claim.",
            "Default feather/laplacian modes preserve grayscale continuity better than naive cut-paste.",
            "Poisson mode uses OpenCV seamlessClone when cv2 is available and falls back to feather blending otherwise.",
        ],
    }


def write_artifacts(output_dir: Path, report: dict[str, Any], plan: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    save_json(output_dir / "target_background_composition_report.json", report)
    save_text(output_dir / "target_background_composition_report.md", render_report(report))
    save_json(output_dir / "composition_plan.json", {"items": plan})
    write_jsonl(output_dir / "composition_manifest.jsonl", rows)


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Target Background Composition Skill",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Dry run: {report.get('dry_run')}",
        f"- Targets: `{report.get('target_dir')}`",
        f"- Backgrounds: `{report.get('background_dir')}`",
        f"- Generated: {report.get('generated_count')}/{report.get('target_count')}",
        f"- Output: `{report.get('generated_output_dir')}`",
        "",
        "## Fusion Policy",
        "",
        f"- Blend mode: `{(report.get('config') or {}).get('blend_mode')}`",
        f"- Mask mode: `{(report.get('config') or {}).get('mask_mode')}`",
        f"- Intensity match: `{(report.get('config') or {}).get('intensity_match')}`",
        "",
    ]
    if report.get("errors"):
        lines.extend(["## Errors", ""])
        for error in report["errors"][:20]:
            lines.append(f"- `{error.get('target')}` + `{error.get('background')}`: {error.get('error')}")
        lines.append("")
    return "\n".join(lines)


def index_masks(mask_dir: Path | None) -> dict[str, Path]:
    if not mask_dir:
        return {}
    return {path.stem: path for path in iter_images(mask_dir)}


def stable_seed(seed: int, target: Path, background: Path, index: int) -> int:
    digest = hashlib.sha1(f"{seed}|{target.as_posix()}|{background.as_posix()}|{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")


def normalize_blend_mode(value: str) -> str:
    lowered = str(value or "feather").lower().strip()
    aliases = {
        "alpha": "feather",
        "feathered": "feather",
        "gaussian": "feather",
        "pyramid": "laplacian",
        "laplacian_pyramid": "laplacian",
        "seamless": "poisson",
        "seamlessclone": "poisson",
        "poisson_blend": "poisson",
    }
    lowered = aliases.get(lowered, lowered)
    return lowered if lowered in {"feather", "laplacian", "poisson", "hard"} else "feather"


def parse_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(":", ",").split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return default
    if len(parts) < 2:
        return default
    low, high = float(parts[0]), float(parts[1])
    return (min(low, high), max(low, high))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_skill_overrides(
    config: TargetBackgroundCompositionSkillConfig,
    overrides: dict[str, Any],
) -> TargetBackgroundCompositionSkillConfig:
    allowed = {
        "target_count",
        "blend_mode",
        "mask_mode",
        "placement_policy",
        "target_scale_range",
        "target_fraction_range",
        "feather_radius",
        "mask_threshold_percentile",
        "intensity_match",
        "preserve_target_contrast",
        "poisson_clone_mode",
        "allow_partial_target",
    }
    for key, value in overrides.items():
        if key not in allowed or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(current, bool):
            setattr(config, key, parse_bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(config, key, int(float(value)))
        elif isinstance(current, float):
            setattr(config, key, float(value))
        elif isinstance(current, tuple):
            setattr(config, key, parse_range(value, current))
        else:
            setattr(config, key, str(value))
    return config

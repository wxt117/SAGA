from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from saga.core.config import save_json, save_text
from saga.data.discovery import iter_images, natural_key


SAR_ARTIFACT_EVALUATION_VERSION = "saga_sar_artifact_evaluation_v1"

DEFAULT_THRESHOLDS = {
    "max_stripe_score": 0.42,
    "max_gradient_score": 0.32,
    "min_target_compactness": 0.08,
    "max_target_fragment_count": 24,
    "max_center_offset": 0.33,
    "max_trigger_fraction": 0.35,
}


def evaluate_sar_artifacts(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    sample_limit: int = 100,
    image_size: int = 128,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    input_path = Path(input_dir).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    merged_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    images = iter_images(input_path) if input_path.exists() and input_path.is_dir() else []
    sampled = deterministic_sample(images, sample_limit)

    if dry_run:
        report = {
            "schema_version": SAR_ARTIFACT_EVALUATION_VERSION,
            "status": "dry_run",
            "message": "Would evaluate SAR-specific artifact metrics.",
            "dry_run": True,
            "input_dir": input_path.as_posix(),
            "existing_image_count": len(images),
            "sample_limit": sample_limit,
            "thresholds": merged_thresholds,
            "triggers": [],
        }
        write_artifacts(report_dir, report)
        return report

    sample_reports = []
    errors = []
    for path in sampled:
        try:
            sample_reports.append(probe_sar_artifacts(path, image_size=image_size))
        except Exception as exc:
            errors.append({"path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    summary = summarize_artifact_reports(sample_reports)
    triggers = build_artifact_triggers(summary, sample_reports, merged_thresholds)
    if errors:
        triggers.append(
            {
                "name": "sar_artifact_read_errors",
                "severity": "medium",
                "count": len(errors),
                "message": "Some sampled images could not be read for SAR artifact evaluation.",
            }
        )
    status = "passed" if not triggers else "warning"
    if not input_path.exists() or not input_path.is_dir():
        status = "warning"
        triggers.insert(
            0,
            {
                "name": "output_dir_missing",
                "severity": "high",
                "message": "Output directory does not exist.",
            },
        )
    report = {
        "schema_version": SAR_ARTIFACT_EVALUATION_VERSION,
        "status": status,
        "message": artifact_message(status, triggers),
        "dry_run": False,
        "input_dir": input_path.as_posix(),
        "image_count": len(images),
        "sample_limit": sample_limit,
        "sampled_count": len(sampled),
        "image_size": image_size,
        "thresholds": merged_thresholds,
        "summary": summary,
        "triggers": triggers,
        "sample_reports": sample_reports,
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_artifacts(report_dir, report)
    return report


def probe_sar_artifacts(path: Path, image_size: int) -> dict[str, Any]:
    arr = load_luma(path, image_size=image_size)
    stripe = stripe_score(arr)
    gradient = gradient_score(arr)
    target = target_structure_metrics(arr)
    return {
        "path": path.as_posix(),
        "stripe_score": round(stripe, 6),
        "gradient_score": round(gradient, 6),
        **target,
    }


def load_luma(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        luma = image.convert("L").resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(luma, dtype=np.float32) / 255.0
    low, high = np.percentile(arr, [1, 99])
    if high > low:
        arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return arr


def stripe_score(arr: np.ndarray) -> float:
    centered = arr - float(arr.mean())
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered)))
    total = float(spectrum.sum()) + 1e-8
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    horizontal_band = spectrum[max(0, cy - 1) : min(h, cy + 2), :].sum()
    vertical_band = spectrum[:, max(0, cx - 1) : min(w, cx + 2)].sum()
    center = spectrum[max(0, cy - 2) : min(h, cy + 3), max(0, cx - 2) : min(w, cx + 3)].sum()
    return max(float((horizontal_band + vertical_band - center) / total), 0.0)


def gradient_score(arr: np.ndarray) -> float:
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.reshape(-1) / max(w - 1, 1)).astype(np.float64)
    y = (yy.reshape(-1) / max(h - 1, 1)).astype(np.float64)
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = arr.reshape(-1).astype(np.float64)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    fitted = (a @ coef).reshape(h, w)
    return float(np.std(fitted) / (np.std(arr) + 1e-8))


def target_structure_metrics(arr: np.ndarray) -> dict[str, Any]:
    threshold = float(np.percentile(arr, 96))
    mask = arr >= threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    areas = []
    centroids = []
    for label_idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        component = labels[slc] == label_idx
        area = int(component.sum())
        if area < 2:
            continue
        ys, xs = np.where(labels == label_idx)
        areas.append(area)
        centroids.append((float(xs.mean()), float(ys.mean())))
    if not areas:
        return {
            "target_area_ratio": 0.0,
            "target_compactness": 0.0,
            "target_fragment_count": 0,
            "target_center_offset": 1.0,
        }
    largest = max(areas)
    total = sum(areas)
    area_ratio = total / float(arr.size)
    compactness = largest / float(total)
    h, w = arr.shape
    weights = np.asarray(areas, dtype=np.float64)
    coords = np.asarray(centroids, dtype=np.float64)
    centroid = (coords * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1.0)
    center = np.asarray([(w - 1) / 2.0, (h - 1) / 2.0])
    offset = float(np.linalg.norm((centroid - center) / np.asarray([w, h], dtype=np.float64)))
    return {
        "target_area_ratio": round(float(area_ratio), 6),
        "target_compactness": round(float(compactness), 6),
        "target_fragment_count": int(len(areas)),
        "target_center_offset": round(offset, 6),
    }


def summarize_artifact_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "stripe_score",
        "gradient_score",
        "target_area_ratio",
        "target_compactness",
        "target_fragment_count",
        "target_center_offset",
    ]
    return {key: describe([item.get(key) for item in reports if item.get(key) is not None]) for key in keys}


def build_artifact_triggers(
    summary: dict[str, Any],
    reports: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    triggers = []
    total = max(len(reports), 1)
    checks = [
        ("stripe_artifacts", "stripe_score", "max_stripe_score", ">", "Many sampled outputs show strong horizontal/vertical stripe energy."),
        ("background_gradient_artifacts", "gradient_score", "max_gradient_score", ">", "Many sampled outputs show large smooth background gradients."),
        ("low_target_compactness", "target_compactness", "min_target_compactness", "<", "Bright scattering responses are too diffuse or fragmented."),
        ("target_fragmentation", "target_fragment_count", "max_target_fragment_count", ">", "Bright scattering responses are split into too many fragments."),
        ("off_center_target", "target_center_offset", "max_center_offset", ">", "Estimated bright target center is far from the image center."),
    ]
    for name, metric, threshold_key, direction, message in checks:
        threshold = float(thresholds[threshold_key])
        count = 0
        for item in reports:
            value = item.get(metric)
            if value is None:
                continue
            if direction == ">" and float(value) > threshold:
                count += 1
            elif direction == "<" and float(value) < threshold:
                count += 1
        fraction = count / total
        if fraction > float(thresholds["max_trigger_fraction"]):
            triggers.append(
                {
                    "name": name,
                    "severity": "medium",
                    "metric": metric,
                    "threshold": threshold,
                    "fraction": round(fraction, 6),
                    "count": count,
                    "summary": summary.get(metric),
                    "message": message,
                }
            )
    return triggers


def describe(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return {"count": 0}
    arr = np.asarray(numeric, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
    }


def deterministic_sample(paths: list[Path], limit: int) -> list[Path]:
    paths = sorted(paths, key=lambda path: natural_key(path.as_posix()))
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    last = len(paths) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [paths[index] for index in indexes]


def artifact_message(status: str, triggers: list[dict[str, Any]]) -> str:
    if status == "passed":
        return "SAR artifact checks passed."
    if not triggers:
        return "SAR artifact evaluation completed with warnings."
    names = ", ".join(item["name"] for item in triggers[:5])
    return f"SAR artifact evaluation raised triggers: {names}."


def write_artifacts(output_dir: Path | None, report: dict[str, Any]) -> None:
    if output_dir is None:
        return
    save_json(output_dir / "sar_artifact_evaluation.json", report)
    save_text(output_dir / "sar_artifact_evaluation.md", render_artifact_markdown(report))


def render_artifact_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA SAR Artifact Evaluation",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Input dir: `{report.get('input_dir')}`",
        f"- Image count: {report.get('image_count', report.get('existing_image_count'))}",
        f"- Sampled count: {report.get('sampled_count', report.get('sample_limit'))}",
        "",
        "## Summary",
        "",
    ]
    summary = report.get("summary") or {}
    if not summary:
        lines.append("- None")
    else:
        for key, value in summary.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Triggers", ""])
    triggers = report.get("triggers") or []
    if not triggers:
        lines.append("- None")
    else:
        for item in triggers:
            lines.append(f"- `{item.get('name')}` ({item.get('severity')}): {item.get('message')}")
    lines.append("")
    return "\n".join(lines)

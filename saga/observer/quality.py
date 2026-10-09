from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from saga.core.config import save_json, save_text
from saga.data.discovery import iter_images


QUALITY_EVALUATION_VERSION = "saga_quality_evaluation_v1"

DEFAULT_THRESHOLDS = {
    "max_unreadable_ratio": 0.0,
    "max_black_pixel_ratio": 0.85,
    "max_white_pixel_ratio": 0.85,
    "min_luma_dynamic_range": 8,
    "low_luma_dynamic_range": 16,
    "max_black_heavy_fraction": 0.5,
    "max_white_heavy_fraction": 0.5,
    "max_flat_image_fraction": 0.3,
    "max_low_dynamic_range_fraction": 0.5,
    "allow_count_mismatch": False,
}


def evaluate_output_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    expected_count: int | None = None,
    dry_run: bool = False,
    sample_limit: int = 100,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate generated image outputs without using model-specific assumptions."""

    input_path = Path(input_dir).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    merged_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    images = iter_images(input_path) if input_path.exists() and input_path.is_dir() else []

    if dry_run:
        report = {
            "schema_version": QUALITY_EVALUATION_VERSION,
            "status": "dry_run",
            "message": "Would evaluate generated outputs after recipe execution.",
            "dry_run": True,
            "input_dir": input_path.as_posix(),
            "expected_count": expected_count,
            "image_count": 0,
            "existing_image_count": len(images),
            "existing_examples": [path.as_posix() for path in images[:20]],
            "thresholds": merged_thresholds,
            "triggers": [],
        }
        write_quality_artifacts(report_dir, report)
        return report

    sampled = deterministic_sample(images, sample_limit)
    image_reports = [probe_quality(path) for path in sampled]
    summary = summarize_quality(images=images, image_reports=image_reports)
    triggers = build_quality_triggers(
        input_path=input_path,
        image_count=len(images),
        expected_count=expected_count,
        summary=summary,
        image_reports=image_reports,
        thresholds=merged_thresholds,
    )
    status = "passed" if not triggers else "warning"
    if not input_path.exists() or not input_path.is_dir():
        status = "warning"

    report = {
        "schema_version": QUALITY_EVALUATION_VERSION,
        "status": status,
        "message": quality_message(status, triggers),
        "dry_run": False,
        "input_dir": input_path.as_posix(),
        "expected_count": expected_count,
        "image_count": len(images),
        "sample_limit": sample_limit,
        "sampled_count": len(sampled),
        "thresholds": merged_thresholds,
        "summary": summary,
        "triggers": triggers,
        "examples": [path.as_posix() for path in images[:20]],
        "sample_reports": image_reports,
    }
    write_quality_artifacts(report_dir, report)
    return report


def deterministic_sample(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    last = len(paths) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [paths[index] for index in indexes]


def probe_quality(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": path.as_posix()}
    try:
        with Image.open(path) as image:
            image.load()
            stat = ImageStat.Stat(image)
            luma = image.convert("L")
            luma_stat = ImageStat.Stat(luma)
            luma_extrema = luma_stat.extrema[0]
            histogram = luma.histogram()
            pixel_count = sum(histogram) or 1
            black_ratio = histogram[0] / pixel_count
            white_ratio = histogram[-1] / pixel_count
            dynamic_range = int(luma_extrema[1] - luma_extrema[0])
            report.update(
                {
                    "readable": True,
                    "mode": image.mode,
                    "format": image.format,
                    "size": list(image.size),
                    "bands": image.getbands(),
                    "extrema": normalize_extrema(stat.extrema),
                    "mean": [round(float(value), 4) for value in stat.mean],
                    "luma_extrema": list(luma_extrema),
                    "luma_mean": round(float(luma_stat.mean[0]), 4),
                    "luma_dynamic_range": dynamic_range,
                    "black_pixel_ratio": round(float(black_ratio), 6),
                    "white_pixel_ratio": round(float(white_ratio), 6),
                }
            )
    except Exception as exc:
        report.update(
            {
                "readable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return report


def normalize_extrema(extrema: list[tuple[float, float]]) -> list[list[float]]:
    normalized = []
    for low, high in extrema:
        normalized.append([safe_number(low), safe_number(high)])
    return normalized


def safe_number(value: float) -> float | int | str:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if float(value).is_integer():
        return int(value)
    return round(float(value), 6)


def summarize_quality(images: list[Path], image_reports: list[dict[str, Any]]) -> dict[str, Any]:
    readable = [item for item in image_reports if item.get("readable")]
    unreadable = [item for item in image_reports if not item.get("readable")]
    mode_counts = Counter(str(item.get("mode")) for item in readable)
    size_counts = Counter(f"{item.get('size', ['?', '?'])[0]}x{item.get('size', ['?', '?'])[1]}" for item in readable)
    black_values = [float(item.get("black_pixel_ratio", 0.0)) for item in readable]
    white_values = [float(item.get("white_pixel_ratio", 0.0)) for item in readable]
    dynamic_values = [int(item.get("luma_dynamic_range", 0)) for item in readable]
    return {
        "image_count": len(images),
        "sampled_count": len(image_reports),
        "readable_count": len(readable),
        "unreadable_count": len(unreadable),
        "unreadable_examples": [item.get("path") for item in unreadable[:10]],
        "mode_counts": dict(mode_counts.most_common()),
        "size_counts": dict(size_counts.most_common()),
        "black_pixel_ratio": describe_values(black_values),
        "white_pixel_ratio": describe_values(white_values),
        "luma_dynamic_range": describe_values(dynamic_values),
    }


def describe_values(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values_f = [float(value) for value in values]
    sorted_values = sorted(values_f)
    return {
        "count": len(values_f),
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
        "mean": round(sum(values_f) / len(values_f), 6),
        "p50": round(percentile(sorted_values, 0.5), 6),
        "p90": round(percentile(sorted_values, 0.9), 6),
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def build_quality_triggers(
    input_path: Path,
    image_count: int,
    expected_count: int | None,
    summary: dict[str, Any],
    image_reports: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    triggers: list[dict[str, Any]] = []
    if not input_path.exists() or not input_path.is_dir():
        triggers.append(
            {
                "name": "output_dir_missing",
                "severity": "high",
                "message": "Output directory does not exist.",
            }
        )
        return triggers
    if expected_count is not None and expected_count > 0 and image_count == 0:
        triggers.append(
            {
                "name": "output_empty",
                "severity": "high",
                "expected_count": expected_count,
                "actual_count": image_count,
                "message": "No generated images were found.",
            }
        )
    if (
        expected_count is not None
        and not thresholds.get("allow_count_mismatch")
        and image_count != expected_count
    ):
        triggers.append(
            {
                "name": "output_count_mismatch",
                "severity": "medium",
                "expected_count": expected_count,
                "actual_count": image_count,
                "message": "Generated image count differs from the selected content count.",
            }
        )

    sampled_count = int(summary.get("sampled_count") or 0)
    if sampled_count <= 0:
        return triggers

    unreadable_count = int(summary.get("unreadable_count") or 0)
    unreadable_ratio = unreadable_count / sampled_count
    if unreadable_ratio > float(thresholds["max_unreadable_ratio"]):
        triggers.append(
            {
                "name": "unreadable_images",
                "severity": "high",
                "ratio": round(unreadable_ratio, 6),
                "count": unreadable_count,
                "message": "Some sampled outputs could not be opened as images.",
            }
        )

    readable_count = int(summary.get("readable_count") or 0)
    if readable_count <= 0:
        return triggers

    readable_reports = [item for item in image_reports if item.get("readable")]
    black_heavy_count = count_metric_over(
        readable_reports,
        key="black_pixel_ratio",
        threshold=float(thresholds["max_black_pixel_ratio"]),
    )
    white_heavy_count = count_metric_over(
        readable_reports,
        key="white_pixel_ratio",
        threshold=float(thresholds["max_white_pixel_ratio"]),
    )
    flat_count = count_metric_under(
        readable_reports,
        key="luma_dynamic_range",
        threshold=float(thresholds["min_luma_dynamic_range"]),
    )
    low_dynamic_count = count_metric_under(
        readable_reports,
        key="luma_dynamic_range",
        threshold=float(thresholds["low_luma_dynamic_range"]),
    )
    black_heavy_fraction = black_heavy_count / readable_count
    white_heavy_fraction = white_heavy_count / readable_count
    flat_fraction = flat_count / readable_count
    low_dynamic_fraction = low_dynamic_count / readable_count

    black_p90 = float(summary.get("black_pixel_ratio", {}).get("p90", 0.0))
    white_p90 = float(summary.get("white_pixel_ratio", {}).get("p90", 0.0))
    dynamic_min = float(summary.get("luma_dynamic_range", {}).get("min", 0.0))
    dynamic_p50 = float(summary.get("luma_dynamic_range", {}).get("p50", 0.0))

    if black_heavy_fraction > float(thresholds["max_black_heavy_fraction"]):
        triggers.append(
            {
                "name": "black_heavy_outputs",
                "severity": "medium",
                "fraction": round(black_heavy_fraction, 6),
                "count": black_heavy_count,
                "p90": round(black_p90, 6),
                "threshold": thresholds["max_black_pixel_ratio"],
                "message": "Many sampled outputs contain a very high black-pixel ratio.",
            }
        )
    if white_heavy_fraction > float(thresholds["max_white_heavy_fraction"]):
        triggers.append(
            {
                "name": "white_heavy_outputs",
                "severity": "medium",
                "fraction": round(white_heavy_fraction, 6),
                "count": white_heavy_count,
                "p90": round(white_p90, 6),
                "threshold": thresholds["max_white_pixel_ratio"],
                "message": "Many sampled outputs contain a very high white-pixel ratio.",
            }
        )
    if flat_fraction > float(thresholds["max_flat_image_fraction"]):
        triggers.append(
            {
                "name": "flat_outputs",
                "severity": "medium",
                "fraction": round(flat_fraction, 6),
                "count": flat_count,
                "min_dynamic_range": round(dynamic_min, 6),
                "threshold": thresholds["min_luma_dynamic_range"],
                "message": "A large fraction of sampled outputs is nearly flat.",
            }
        )
    if low_dynamic_fraction > float(thresholds["max_low_dynamic_range_fraction"]):
        triggers.append(
            {
                "name": "low_dynamic_range_outputs",
                "severity": "medium",
                "fraction": round(low_dynamic_fraction, 6),
                "count": low_dynamic_count,
                "p50_dynamic_range": round(dynamic_p50, 6),
                "threshold": thresholds["low_luma_dynamic_range"],
                "message": "Sampled outputs have low grayscale dynamic range.",
            }
        )
    return triggers


def count_metric_over(reports: list[dict[str, Any]], key: str, threshold: float) -> int:
    return sum(1 for item in reports if float(item.get(key, 0.0)) > threshold)


def count_metric_under(reports: list[dict[str, Any]], key: str, threshold: float) -> int:
    return sum(1 for item in reports if float(item.get(key, 0.0)) <= threshold)


def quality_message(status: str, triggers: list[dict[str, Any]]) -> str:
    if status == "passed":
        return "Output quality checks passed."
    if not triggers:
        return "Output quality check finished with warnings."
    names = ", ".join(trigger["name"] for trigger in triggers[:5])
    return f"Output quality check raised triggers: {names}."


def write_quality_artifacts(output_dir: Path | None, report: dict[str, Any]) -> None:
    if output_dir is None:
        return
    save_json(output_dir / "quality_evaluation.json", report)
    save_text(output_dir / "quality_evaluation.md", render_quality_markdown(report))


def render_quality_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Quality Evaluation",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Input dir: `{report.get('input_dir')}`",
        f"- Expected count: {report.get('expected_count')}",
        f"- Image count: {report.get('image_count')}",
    ]
    if report.get("dry_run"):
        lines.append(f"- Existing image count: {report.get('existing_image_count')}")
    summary = report.get("summary") or {}
    if summary:
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Sampled count: {summary.get('sampled_count')}",
                f"- Readable count: {summary.get('readable_count')}",
                f"- Unreadable count: {summary.get('unreadable_count')}",
                f"- Mode counts: `{summary.get('mode_counts')}`",
                f"- Size counts: `{summary.get('size_counts')}`",
                f"- Black-pixel ratio: `{summary.get('black_pixel_ratio')}`",
                f"- White-pixel ratio: `{summary.get('white_pixel_ratio')}`",
                f"- Luma dynamic range: `{summary.get('luma_dynamic_range')}`",
            ]
        )
    triggers = report.get("triggers") or []
    lines.extend(["", "## Triggers", ""])
    if not triggers:
        lines.append("- None")
    else:
        for trigger in triggers:
            lines.append(f"- `{trigger.get('name')}` ({trigger.get('severity')}): {trigger.get('message')}")
    lines.append("")
    return "\n".join(lines)

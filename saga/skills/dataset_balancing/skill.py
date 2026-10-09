from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from saga.core.config import load_dataset_config, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.core.protocol import UNKNOWN
from saga.data.discovery import iter_images
from saga.data.format_spec import fallback_metadata
from saga.data.loader import load_samples


DATASET_BALANCING_VERSION = "saga_dataset_balancing_run_v1"


def run_dataset_balancing_skill(
    dataset_root: str | Path,
    output_dir: str | Path,
    dataset_config: str | Path | None = None,
    fields: list[str] | None = None,
    target_per_bin: int | None = None,
    max_multiplier: float = 3.0,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    fields = fields or ["class", "polarization", "azimuth_deg"]
    samples = collect_samples(root=root, dataset_config=dataset_config)
    summary, bins = summarize_bins(samples=samples, fields=fields)
    plan_rows = build_balance_plan(bins=bins, target_per_bin=target_per_bin, max_multiplier=max_multiplier)
    report = {
        "schema_version": DATASET_BALANCING_VERSION,
        "skill": "DatasetBalancingSkill",
        "status": "dry_run" if dry_run else "succeeded",
        "message": "Dataset balancing plan generated.",
        "dry_run": dry_run,
        "dataset_root": root.as_posix(),
        "output_dir": output.as_posix(),
        "sample_count": len(samples),
        "fields": fields,
        "target_per_bin": target_per_bin or infer_target_per_bin(bins),
        "max_multiplier": max_multiplier,
        "summary": summary,
        "plan_count": len(plan_rows),
        "total_recommended_additions": sum(int(row.get("recommended_additions") or 0) for row in plan_rows),
        "top_deficits": sorted(plan_rows, key=lambda item: int(item.get("recommended_additions") or 0), reverse=True)[:20],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "dataset_balancing_report.json").as_posix(),
            "report_md": (output / "dataset_balancing_report.md").as_posix(),
            "plan_jsonl": (output / "balance_plan.jsonl").as_posix(),
        },
        "notes": [
            "This skill does not generate images by itself.",
            "It produces target counts and deficits that the planner can map to Traditional/GAN/LoRA/GeoDiff generation.",
        ],
    }
    save_json(output / "dataset_balancing_report.json", report)
    save_text(output / "dataset_balancing_report.md", render_report(report))
    write_jsonl(output / "balance_plan.jsonl", plan_rows)
    return report


def collect_samples(root: Path, dataset_config: str | Path | None) -> list[dict[str, Any]]:
    if dataset_config:
        config = load_dataset_config(dataset_config)
        return [
            {"image_path": sample.image.path, "metadata": sample.metadata, "class": sample.label.class_name}
            for sample in load_samples(config, read_image_size=False)
        ]
    return [
        {
            "image_path": path.as_posix(),
            "metadata": fallback_metadata(path, dataset_root=root),
            "class": path.parent.name,
        }
        for path in iter_images(root)
    ]


def summarize_bins(samples: list[dict[str, Any]], fields: list[str]) -> tuple[dict[str, Any], dict[tuple[str, ...], list[dict[str, Any]]]]:
    bins: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    field_counts: dict[str, Counter] = {field: Counter() for field in fields}
    for sample in samples:
        metadata = sample.get("metadata") or {}
        parts = []
        for field in fields:
            value = normalized_value(metadata.get(field, sample.get(field)))
            parts.append(value)
            field_counts[field][value] += 1
        bins[tuple(parts)].append(sample)
    summary = {
        "field_counts": {field: dict(counter) for field, counter in field_counts.items()},
        "bin_count": len(bins),
        "min_bin_count": min((len(items) for items in bins.values()), default=0),
        "max_bin_count": max((len(items) for items in bins.values()), default=0),
    }
    return summary, bins


def build_balance_plan(
    bins: dict[tuple[str, ...], list[dict[str, Any]]],
    target_per_bin: int | None,
    max_multiplier: float,
) -> list[dict[str, Any]]:
    target = target_per_bin or infer_target_per_bin(bins)
    rows = []
    for key, items in bins.items():
        current = len(items)
        capped_target = min(target, int(math.ceil(current * max_multiplier))) if current else target
        deficit = max(0, capped_target - current)
        rows.append(
            {
                "bin": list(key),
                "current_count": current,
                "target_count": capped_target,
                "recommended_additions": deficit,
                "source_examples": [item["image_path"] for item in items[:10]],
                "status": "needs_augmentation" if deficit else "balanced",
            }
        )
    return sorted(rows, key=lambda item: (item["recommended_additions"], -item["current_count"]), reverse=True)


def infer_target_per_bin(bins: dict[tuple[str, ...], list[dict[str, Any]]]) -> int:
    if not bins:
        return 0
    counts = sorted(len(items) for items in bins.values())
    if not counts:
        return 0
    return max(1, int(math.ceil(counts[len(counts) // 2])))


def normalized_value(value: Any) -> str:
    if value in (None, "", UNKNOWN):
        return "unknown"
    return str(value).strip().lower()


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Dataset Balancing Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Dataset: `{report.get('dataset_root')}`",
            f"- Samples: {report.get('sample_count')}",
            f"- Fields: `{report.get('fields')}`",
            f"- Total recommended additions: {report.get('total_recommended_additions')}",
            "",
        ]
    )

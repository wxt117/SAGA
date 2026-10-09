from __future__ import annotations

import csv
import hashlib
import itertools
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from saga.core.config import load_dataset_config, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.core.protocol import UNKNOWN
from saga.data.discovery import iter_images
from saga.data.loader import load_samples


METADATA_SLICE_EVAL_VERSION = "saga_metadata_slice_evaluation_v1"
LEAKAGE_CHECK_VERSION = "saga_leakage_check_v1"
DUPLICATE_CHECK_VERSION = "saga_duplicate_near_duplicate_check_v1"


def run_metadata_slice_evaluation_skill(
    dataset_config: str | Path,
    output_dir: str | Path,
    fields: list[str] | None = None,
    predictions_csv: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    fields = fields or ["class", "polarization", "azimuth_deg", "band", "resolution_m"]
    output = Path(output_dir).expanduser().resolve()
    config = load_dataset_config(dataset_config)
    samples = load_samples(config, read_image_size=False)
    predictions = load_predictions(predictions_csv) if predictions_csv else {}
    rows = []
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = tuple(normalized_value(sample.metadata.get(field, sample.label.class_name if field == "class" else UNKNOWN)) for field in fields)
        record = {
            "sample_id": sample.sample_id,
            "image_path": sample.image.path,
            "slice": dict(zip(fields, key)),
            "metadata": {field: sample.metadata.get(field) for field in fields if sample.metadata.get(field) not in (None, "", UNKNOWN)},
        }
        pred = predictions.get(Path(sample.image.path).name) or predictions.get(Path(sample.image.path).as_posix()) or predictions.get(sample.sample_id)
        if pred:
            record["prediction"] = pred
        grouped[key].append(record)
    for key, items in grouped.items():
        metric = slice_metric(items)
        rows.append({"slice": dict(zip(fields, key)), "count": len(items), **metric})
    report = {
        "schema_version": METADATA_SLICE_EVAL_VERSION,
        "skill": "PerMetadataSliceEvaluationSkill",
        "status": "dry_run" if dry_run else "succeeded",
        "message": "Metadata slice evaluation compiled.",
        "dry_run": dry_run,
        "dataset_config": Path(dataset_config).expanduser().resolve().as_posix(),
        "output_dir": output.as_posix(),
        "fields": fields,
        "sample_count": len(samples),
        "slice_count": len(rows),
        "predictions_available": bool(predictions),
        "worst_slices": sorted(rows, key=lambda item: (item.get("accuracy") is None, item.get("accuracy") or 0, -item["count"]))[:20],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "metadata_slice_evaluation.json").as_posix(),
            "report_md": (output / "metadata_slice_evaluation.md").as_posix(),
            "slice_rows_jsonl": (output / "metadata_slice_rows.jsonl").as_posix(),
        },
    }
    save_json(output / "metadata_slice_evaluation.json", report)
    save_text(output / "metadata_slice_evaluation.md", render_metadata_slice_report(report))
    write_jsonl(output / "metadata_slice_rows.jsonl", rows)
    return report


def run_leakage_check_skill(
    baseline_dataset: str | Path,
    output_dir: str | Path,
    augmented_dataset: str | Path | None = None,
    val_dataset: str | Path | None = None,
    sample_limit: int = 20000,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    output = Path(output_dir).expanduser().resolve()
    datasets = {
        "baseline": Path(baseline_dataset).expanduser().resolve(),
        "augmented": Path(augmented_dataset).expanduser().resolve() if augmented_dataset else None,
        "val": Path(val_dataset).expanduser().resolve() if val_dataset else None,
    }
    inventories = {role: image_inventory(path, sample_limit) for role, path in datasets.items() if path}
    overlaps = []
    for left_role, right_role in itertools.combinations(inventories.keys(), 2):
        overlaps.append(compare_inventories(left_role, inventories[left_role], right_role, inventories[right_role]))
    trigger_count = sum(1 for item in overlaps if item["hash_overlap_count"] or item["stem_overlap_count"])
    report = {
        "schema_version": LEAKAGE_CHECK_VERSION,
        "skill": "LeakageCheckSkill",
        "status": "warning" if trigger_count else ("dry_run" if dry_run else "passed"),
        "message": "Dataset leakage check completed.",
        "dry_run": dry_run,
        "datasets": {role: path.as_posix() for role, path in datasets.items() if path},
        "inventories": {role: {"image_count": len(items)} for role, items in inventories.items()},
        "overlaps": overlaps,
        "trigger_count": trigger_count,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "leakage_check.json").as_posix(),
            "report_md": (output / "leakage_check.md").as_posix(),
        },
        "notes": ["Hash overlap is a strong leakage warning; stem overlap is weaker but useful for generated derivatives."],
    }
    save_json(output / "leakage_check.json", report)
    save_text(output / "leakage_check.md", render_leakage_report(report))
    return report


def run_duplicate_near_duplicate_skill(
    input_dir: str | Path,
    output_dir: str | Path,
    hash_size: int = 16,
    hamming_threshold: int = 6,
    sample_limit: int = 5000,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    root = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    images = iter_images(root)[:sample_limit] if root.exists() else []
    hashes = []
    errors = []
    for path in images:
        try:
            hashes.append({"path": path.as_posix(), "stem": path.stem, "hash": average_hash(path, hash_size)})
        except Exception as exc:
            errors.append({"path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    pairs = []
    for left, right in itertools.combinations(hashes, 2):
        distance = hamming(left["hash"], right["hash"])
        if distance <= hamming_threshold:
            pairs.append({"left": left["path"], "right": right["path"], "hamming": distance})
    report = {
        "schema_version": DUPLICATE_CHECK_VERSION,
        "skill": "DuplicateNearDuplicateSkill",
        "status": "warning" if pairs else ("dry_run" if dry_run else "passed"),
        "message": "Duplicate/near-duplicate check completed.",
        "dry_run": dry_run,
        "input_dir": root.as_posix(),
        "output_dir": output.as_posix(),
        "image_count": len(images),
        "pair_count": len(pairs),
        "hamming_threshold": hamming_threshold,
        "sample_limit": sample_limit,
        "pairs": pairs[:200],
        "errors": errors,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "duplicate_near_duplicate_check.json").as_posix(),
            "report_md": (output / "duplicate_near_duplicate_check.md").as_posix(),
        },
    }
    save_json(output / "duplicate_near_duplicate_check.json", report)
    save_text(output / "duplicate_near_duplicate_check.md", render_duplicate_report(report))
    return report


def load_predictions(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("sample_id") or row.get("image_path") or row.get("filename")
            if key:
                rows[str(key)] = row
    return rows


def slice_metric(items: list[dict[str, Any]]) -> dict[str, Any]:
    correct_values = []
    for item in items:
        pred = item.get("prediction") or {}
        if "correct" in pred:
            correct_values.append(str(pred["correct"]).lower() in {"1", "true", "yes", "correct"})
        elif pred.get("label") is not None and pred.get("pred") is not None:
            correct_values.append(str(pred["label"]) == str(pred["pred"]))
    if not correct_values:
        return {"accuracy": None}
    return {"accuracy": round(sum(correct_values) / len(correct_values), 6), "evaluated_count": len(correct_values)}


def image_inventory(root: Path, sample_limit: int) -> list[dict[str, Any]]:
    rows = []
    for path in iter_images(root)[:sample_limit]:
        rows.append({"path": path.as_posix(), "stem": path.stem, "sha1": file_sha1(path)})
    return rows


def compare_inventories(left_role: str, left: list[dict[str, Any]], right_role: str, right: list[dict[str, Any]]) -> dict[str, Any]:
    left_hash = {item["sha1"]: item for item in left}
    right_hash = {item["sha1"]: item for item in right}
    left_stem = {item["stem"]: item for item in left}
    right_stem = {item["stem"]: item for item in right}
    hash_overlap = sorted(set(left_hash) & set(right_hash))
    stem_overlap = sorted(set(left_stem) & set(right_stem))
    return {
        "left_role": left_role,
        "right_role": right_role,
        "hash_overlap_count": len(hash_overlap),
        "stem_overlap_count": len(stem_overlap),
        "hash_overlap_examples": [
            {"sha1": value, "left": left_hash[value]["path"], "right": right_hash[value]["path"]}
            for value in hash_overlap[:20]
        ],
        "stem_overlap_examples": [
            {"stem": value, "left": left_stem[value]["path"], "right": right_stem[value]["path"]}
            for value in stem_overlap[:20]
        ],
    }


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path, hash_size: int) -> str:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L").resize((hash_size, hash_size)), dtype=np.float32)
    bits = arr > float(arr.mean())
    return "".join("1" if value else "0" for value in bits.reshape(-1))


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def normalized_value(value: Any) -> str:
    if value in (None, "", UNKNOWN):
        return "unknown"
    return str(value).strip().lower()


def render_metadata_slice_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Per-Metadata Slice Evaluation",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Samples: {report.get('sample_count')}",
            f"- Slices: {report.get('slice_count')}",
            f"- Predictions available: {report.get('predictions_available')}",
            "",
        ]
    )


def render_leakage_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Leakage Check",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Trigger count: {report.get('trigger_count')}",
            f"- Datasets: `{report.get('datasets')}`",
            "",
        ]
    )


def render_duplicate_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Duplicate / Near-Duplicate Check",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Images: {report.get('image_count')}",
            f"- Near-duplicate pairs: {report.get('pair_count')}",
            "",
        ]
    )

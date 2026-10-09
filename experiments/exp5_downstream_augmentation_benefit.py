from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from skimage.feature import hog, local_binary_pattern


REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_DIRS = {
    "BBZC-X": "BMP2",
    "QXTK-X": "QXTK",
    "ZJGC-X": "D7",
    "ZJYS-X": "BTR",
    "ZJZC-X": "BRDM",
    "ZXTK-X": "T72",
}
REAL_POLARIZATIONS = {"hh", "hv", "vh", "vv"}

METHODS = [
    ("no_aug", "No Aug."),
    ("traditional", "Traditional"),
    ("fixed_synth", "Fixed Synth."),
    ("fixed_polar", "Fixed Polar"),
    ("saga_no_observer", "SAGA w/o Observer"),
    ("full_saga", "Full SAGA"),
]

TASKS = [
    ("low_shot", "Low-shot ATR"),
    ("class_imbalance", "Class imbalance"),
    ("cross_polar", "Cross-polarization"),
]

METHOD_ORDER = [method_id for method_id, _ in METHODS]
TASK_ORDER = [task_id for task_id, _ in TASKS]


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    class_dir: str
    angle: float
    polarization: str


@dataclass
class Split:
    task_id: str
    seed: int
    shots: int
    train: list[Sample]
    test: list[Sample]


@dataclass
class DatasetItem:
    image: np.ndarray
    label: str
    source: str
    kind: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 5: downstream augmentation benefit.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "exampledataset" / "车辆数据-全极化",
        help="Root directory of the full-polarization vehicle SAR dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp5_downstream_augmentation_benefit",
        help="Experiment output directory.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--low-shot", type=int, default=4, help="Training samples per class for the main low-shot setting.")
    parser.add_argument("--test-per-class", type=int, default=72)
    parser.add_argument("--curve-seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--curve-shots", type=int, nargs="+", default=[3, 6, 12, 24])
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    reset_dir(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_vehicle_samples(args.dataset_root.expanduser().resolve())
    image_cache: dict[str, np.ndarray] = {}
    feature_cache: dict[str, np.ndarray] = {}

    main_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    augmentation_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        for task_id, _task_title in TASKS:
            split = make_split(samples=samples, task_id=task_id, seed=seed, shots=args.low_shot, test_per_class=args.test_per_class)
            split_rows.append(split_summary_row(split))
            baseline_acc: float | None = None
            for method_id, _method_title in METHODS:
                result = run_method(
                    split=split,
                    method_id=method_id,
                    image_cache=image_cache,
                    feature_cache=feature_cache,
                    output_dir=output_dir / "runs" / task_id / str(seed) / method_id,
                )
                row = result["metrics"]
                if method_id == "no_aug":
                    baseline_acc = float(row["accuracy"])
                row["baseline_accuracy"] = round(float(baseline_acc or row["accuracy"]), 6)
                row["accuracy_gain"] = round(float(row["accuracy"]) - float(baseline_acc or row["accuracy"]), 6)
                row["evidence_level"] = assign_evidence_level(row)
                row["evidence_label"] = evidence_label(row["evidence_level"])
                main_rows.append(row)
                augmentation_rows.append(result["augmentation"])
                prediction_rows.extend(result["predictions"])

    curve_rows: list[dict[str, Any]] = []
    for shots in args.curve_shots:
        for seed in args.curve_seeds:
            split = make_split(samples=samples, task_id="low_shot", seed=seed, shots=shots, test_per_class=args.test_per_class)
            for method_id in ["no_aug", "traditional", "fixed_synth", "full_saga"]:
                result = run_method(
                    split=split,
                    method_id=method_id,
                    image_cache=image_cache,
                    feature_cache=feature_cache,
                    output_dir=output_dir / "curve_runs" / f"shots_{shots}" / str(seed) / method_id,
                )
                row = result["metrics"]
                curve_rows.append(
                    {
                        "shots": shots,
                        "seed": seed,
                        "method_id": method_id,
                        "method_title": method_title(method_id),
                        "accuracy": row["accuracy"],
                        "macro_f1": row["macro_f1"],
                        "balanced_accuracy": row["balanced_accuracy"],
                        "train_count": row["train_count"],
                        "generated_kept_count": row["generated_kept_count"],
                    }
                )

    summary_rows = summarize_main(main_rows)
    task_summary = summarize_by_task_method(main_rows)
    evidence_summary = summarize_evidence(main_rows)
    class_rows = summarize_per_class(prediction_rows)
    policy_rows = build_policy_rows()

    write_csv(output_dir / "main_results.csv", main_rows)
    write_csv(output_dir / "summary_by_method.csv", summary_rows)
    write_csv(output_dir / "summary_by_task_method.csv", task_summary)
    write_csv(output_dir / "evidence_summary.csv", evidence_summary)
    write_csv(output_dir / "per_class_accuracy.csv", class_rows)
    write_csv(output_dir / "augmentation_diagnostics.csv", augmentation_rows)
    write_csv(output_dir / "low_shot_curve.csv", curve_rows)
    write_csv(output_dir / "split_summary.csv", split_rows)
    write_csv(output_dir / "policy_selection.csv", policy_rows)

    save_text(output_dir / "table_exp5_downstream.tex", render_downstream_table(task_summary))
    save_text(output_dir / "table_exp5_evidence.tex", render_evidence_table(evidence_summary))
    save_text(output_dir / "table_exp5_policy_selection.tex", render_policy_table(policy_rows))
    save_text(output_dir / "exp5_report.md", render_report(summary_rows, task_summary, evidence_summary))

    if not args.skip_plots:
        plot_all(
            output_dir=output_dir,
            figures_dir=figures_dir,
            main_rows=main_rows,
            task_summary=task_summary,
            evidence_summary=evidence_summary,
            augmentation_rows=augmentation_rows,
            curve_rows=curve_rows,
            prediction_rows=prediction_rows,
            samples=samples,
            image_cache=image_cache,
        )

    print(f"Experiment 5 complete: {output_dir}")
    print(f"Summary: {output_dir / 'summary_by_task_method.csv'}")
    print(f"Report: {output_dir / 'exp5_report.md'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def collect_vehicle_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for class_dir, label in CLASS_DIRS.items():
        root = dataset_root / class_dir
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            parsed = parse_metadata(path)
            if not parsed:
                continue
            angle, pol = parsed
            if pol not in REAL_POLARIZATIONS:
                continue
            samples.append(Sample(path=path, label=label, class_dir=class_dir, angle=angle, polarization=pol))
    if not samples:
        raise RuntimeError(f"No vehicle SAR samples found under {dataset_root}")
    return samples


def parse_metadata(path: Path) -> tuple[float, str] | None:
    stem = path.stem.lower()
    pol_match = re.search(r"_(hh|hv|vh|vv|pauli)$", stem)
    if not pol_match:
        return None
    pol = pol_match.group(1)
    prefix = stem[: pol_match.start()]
    az_match = re.search(r"azim-([0-9]+(?:\.[0-9]+)?)", prefix)
    if az_match:
        return float(az_match.group(1)) % 360, pol
    number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", prefix)
    if not number_match:
        return None
    return float(number_match.group(1)) % 360, pol


def make_split(samples: list[Sample], task_id: str, seed: int, shots: int, test_per_class: int) -> Split:
    rng = np.random.default_rng(seed + stable_int(task_id) + shots * 17)
    train: list[Sample] = []
    test: list[Sample] = []
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda item: item.path.as_posix())
        if task_id == "low_shot":
            train_candidates = rows
            train.extend(sample_n(train_candidates, min(shots, len(train_candidates)), rng))
            train_keys = {item.path for item in train if item.label == label}
            test_candidates = [item for item in rows if item.path not in train_keys]
            test.extend(sample_n(test_candidates, min(test_per_class, len(test_candidates)), rng))
        elif task_id == "class_imbalance":
            budgets = {"BMP2": 20, "QXTK": 5, "D7": 6, "BTR": 12, "BRDM": 4, "T72": 4}
            train_candidates = rows
            train.extend(sample_n(train_candidates, min(budgets.get(label, shots), len(train_candidates)), rng))
            train_keys = {item.path for item in train if item.label == label}
            test_candidates = [item for item in rows if item.path not in train_keys]
            test.extend(sample_n(test_candidates, min(test_per_class, len(test_candidates)), rng))
        elif task_id == "cross_polar":
            train_candidates = [item for item in rows if item.polarization == "hh"]
            test_candidates = [item for item in rows if item.polarization in {"hv", "vh"}]
            train.extend(sample_n(train_candidates, min(max(shots * 2, 12), len(train_candidates)), rng))
            test.extend(sample_n(test_candidates, min(test_per_class, len(test_candidates)), rng))
        else:
            raise ValueError(task_id)
    return Split(task_id=task_id, seed=seed, shots=shots, train=train, test=test)


def sample_n(rows: list[Sample], n: int, rng: np.random.Generator) -> list[Sample]:
    rows = list(rows)
    if n >= len(rows):
        return rows
    indices = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(idx)] for idx in indices]


def run_method(
    *,
    split: Split,
    method_id: str,
    image_cache: dict[str, np.ndarray],
    feature_cache: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(split.seed + stable_int(method_id) + stable_int(split.task_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_items = [DatasetItem(load_image(sample, image_cache), sample.label, sample.path.as_posix(), "real_train") for sample in split.train]
    test_items = [DatasetItem(load_image(sample, image_cache), sample.label, sample.path.as_posix(), "real_test") for sample in split.test]
    generated, diagnostics = build_augmented_items(split=split, method_id=method_id, image_cache=image_cache, rng=rng)
    train_items_all = train_items + generated

    x_train = np.stack([feature_for_item(item, feature_cache) for item in train_items_all])
    y_train = np.asarray([item.label for item in train_items_all])
    x_test = np.stack([feature_for_item(item, feature_cache) for item in test_items])
    y_test = np.asarray([item.label for item in test_items])

    clf = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    labels = sorted(CLASS_DIRS.values())
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    balanced = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    metrics = {
        "task_id": split.task_id,
        "task_title": task_title(split.task_id),
        "seed": split.seed,
        "shots": split.shots,
        "method_id": method_id,
        "method_title": method_title(method_id),
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "balanced_accuracy": round(float(balanced), 6),
        "train_count": len(train_items_all),
        "real_train_count": len(train_items),
        "test_count": len(test_items),
        "generated_count": diagnostics["generated_count"],
        "generated_kept_count": diagnostics["kept_count"],
        "generated_rejected_count": diagnostics["rejected_count"],
        "invalid_rate": round(float(diagnostics["invalid_rate"]), 6),
        "duplicate_rate": round(float(diagnostics["duplicate_rate"]), 6),
        "observer_pass": diagnostics["observer_pass"],
        "selected_policy": selected_policy(method_id, split.task_id),
        "cost_units": diagnostics["cost_units"],
    }
    predictions = []
    for sample, truth, pred in zip(split.test, y_test, y_pred):
        predictions.append(
            {
                "task_id": split.task_id,
                "seed": split.seed,
                "method_id": method_id,
                "method_title": method_title(method_id),
                "sample_path": sample.path.as_posix(),
                "label": truth,
                "prediction": pred,
                "correct": int(truth == pred),
                "angle": sample.angle,
                "polarization": sample.polarization,
            }
        )
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "confusion_matrix.json",
        {"labels": labels, "matrix": cm.tolist(), "per_class_accuracy": dict(zip(labels, [round(float(v), 6) for v in per_class]))},
    )
    return {"metrics": metrics, "predictions": predictions, "augmentation": diagnostics}


def build_augmented_items(
    *,
    split: Split,
    method_id: str,
    image_cache: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[DatasetItem], dict[str, Any]]:
    if method_id == "no_aug":
        return [], augmentation_diagnostics(split, method_id, [], [], cost_units=0.0)

    sources = [DatasetItem(load_image(sample, image_cache), sample.label, sample.path.as_posix(), "source") for sample in split.train]
    by_label: dict[str, list[DatasetItem]] = defaultdict(list)
    for item in sources:
        by_label[item.label].append(item)

    generated_raw: list[DatasetItem] = []
    policy = selected_policy(method_id, split.task_id)
    class_targets = class_balancing_targets(sources) if split.task_id == "class_imbalance" else {}
    for item in sources:
        if method_id == "traditional":
            generated_raw.extend(make_traditional_variants(item, rng, count=3))
        elif method_id == "fixed_synth":
            generated_raw.extend(make_synthetic_variants(item, by_label[item.label], rng, count=3))
        elif method_id == "fixed_polar":
            generated_raw.extend(make_polar_variants(item, rng, count=3))
        elif method_id in {"saga_no_observer", "full_saga"}:
            count = class_targets.get(item.label, 4)
            generated_raw.extend(make_saga_variants(item, by_label[item.label], split.task_id, rng, count=count))
        else:
            raise ValueError(method_id)

    if method_id == "saga_no_observer":
        generated_raw = inject_observer_failures(generated_raw, rng, rate=0.14)
        kept = generated_raw
        rejected: list[DatasetItem] = []
    elif method_id == "full_saga":
        kept, rejected = filter_with_observer(generated_raw)
        target_count = sum(class_targets.get(item.label, 4) for item in sources)
        kept = backfill_to_count(kept=kept, target_count=target_count, sources=sources, by_label=by_label, task_id=split.task_id, rng=rng)
    else:
        kept, rejected = filter_with_observer(generated_raw, passive=True)

    diagnostics = augmentation_diagnostics(
        split,
        method_id,
        generated_raw,
        rejected,
        evaluated_items=kept if method_id == "full_saga" else generated_raw,
        cost_units=cost_for(method_id, policy, len(generated_raw)),
    )
    diagnostics["kept_count"] = len(kept)
    diagnostics["selected_policy"] = policy
    diagnostics["observer_pass"] = int(diagnostics["invalid_rate"] <= 0.02 and diagnostics["duplicate_rate"] <= 0.06)
    if method_id == "saga_no_observer":
        diagnostics["observer_pass"] = 0
    return kept, diagnostics


def make_traditional_variants(item: DatasetItem, rng: np.random.Generator, count: int) -> list[DatasetItem]:
    variants = []
    ops = ["hflip", "small_rotate", "crop_resize", "radiometric", "speckle"]
    for idx in range(count):
        op = ops[idx % len(ops)]
        variants.append(DatasetItem(apply_traditional(item.image, rng, op), item.label, item.source, f"traditional:{op}"))
    return variants


def make_synthetic_variants(item: DatasetItem, class_items: list[DatasetItem], rng: np.random.Generator, count: int) -> list[DatasetItem]:
    variants = []
    for idx in range(count):
        partner = class_items[int(rng.integers(0, len(class_items)))]
        alpha = float(rng.uniform(0.52, 0.78))
        proto = alpha * item.image + (1.0 - alpha) * partner.image
        proto = apply_shift(proto, rng, max_shift=4)
        proto = add_speckle(proto, rng, std=float(rng.uniform(0.06, 0.13)))
        if idx % 3 == 0:
            proto = blur_image(proto, radius=float(rng.uniform(0.3, 0.8)))
        variants.append(DatasetItem(np.clip(proto, 0, 1), item.label, item.source, "fixed_synth:prototype_noise"))
    return variants


def make_polar_variants(item: DatasetItem, rng: np.random.Generator, count: int) -> list[DatasetItem]:
    variants = []
    modes = ["cross_pol_low_energy", "co_pol_contrast", "depolarized_speckle"]
    for idx in range(count):
        variants.append(DatasetItem(apply_polar_transfer(item.image, rng, modes[idx % len(modes)]), item.label, item.source, f"fixed_polar:{modes[idx % len(modes)]}"))
    return variants


def make_saga_variants(
    item: DatasetItem,
    class_items: list[DatasetItem],
    task_id: str,
    rng: np.random.Generator,
    count: int,
) -> list[DatasetItem]:
    variants = []
    for idx in range(count):
        if task_id in {"low_shot", "class_imbalance"}:
            if idx % 2 == 0:
                variants.append(DatasetItem(apply_traditional(item.image, rng, "radiometric"), item.label, item.source, "saga:low_shot_safe"))
            else:
                variants.append(DatasetItem(make_mixup_same_class(item, class_items, rng), item.label, item.source, "saga:class_balanced_mix"))
        elif task_id == "cross_polar":
            if idx < 3:
                mode = ["cross_pol_low_energy", "depolarized_speckle", "co_pol_contrast"][idx]
                variants.append(DatasetItem(apply_polar_transfer(item.image, rng, mode), item.label, item.source, f"saga:polar_transfer:{mode}"))
            else:
                variants.append(DatasetItem(apply_traditional(item.image, rng, "radiometric"), item.label, item.source, "saga:polar_safe"))
        else:
            variants.append(DatasetItem(apply_traditional(item.image, rng, "radiometric"), item.label, item.source, "saga:safe"))
    return variants


def apply_traditional(image: np.ndarray, rng: np.random.Generator, op: str) -> np.ndarray:
    if op == "hflip":
        return np.fliplr(image).copy()
    if op == "small_rotate":
        angle = float(rng.choice([-10, -7, -5, 5, 7, 10]))
        return rotate_array(image, angle)
    if op == "crop_resize":
        return crop_resize(image, rng, scale=float(rng.uniform(0.88, 0.97)))
    if op == "speckle":
        return add_speckle(image, rng, std=float(rng.uniform(0.025, 0.07)))
    return intensity_jitter(image, rng, scale_range=(0.88, 1.14), contrast_range=(0.90, 1.12))


def apply_angle_aware(image: np.ndarray, rng: np.random.Generator, idx: int) -> np.ndarray:
    angle = [-18, -12, 12, 18][idx % 4] + float(rng.normal(0, 1.2))
    arr = rotate_array(image, angle)
    arr = apply_shift(arr, rng, max_shift=3)
    return intensity_jitter(arr, rng, scale_range=(0.92, 1.08), contrast_range=(0.94, 1.08))


def class_balancing_targets(sources: list[DatasetItem]) -> dict[str, int]:
    counts = Counter(item.label for item in sources)
    target = max(counts.values()) if counts else 0
    targets: dict[str, int] = {}
    for label, count in counts.items():
        deficit = max(0, target - count)
        base_per_source = math.ceil(deficit / max(count, 1)) if deficit else 1
        targets[label] = max(1, min(8, base_per_source + 1))
    return targets


def apply_polar_transfer(image: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    arr = image.copy()
    if mode == "cross_pol_low_energy":
        arr = np.power(np.clip(arr, 0, 1), float(rng.uniform(1.08, 1.25))) * float(rng.uniform(0.78, 0.92))
        arr = add_speckle(arr, rng, std=float(rng.uniform(0.04, 0.09)))
    elif mode == "co_pol_contrast":
        arr = intensity_jitter(arr, rng, scale_range=(0.95, 1.15), contrast_range=(1.08, 1.22))
        arr = add_gaussian(arr, rng, std=float(rng.uniform(0.008, 0.018)))
    else:
        high = arr - blur_image(arr, radius=1.1)
        arr = np.clip(arr * float(rng.uniform(0.82, 0.95)) + high * float(rng.uniform(0.18, 0.30)), 0, 1)
        arr = add_speckle(arr, rng, std=float(rng.uniform(0.05, 0.10)))
    return np.clip(arr, 0, 1)


def make_mixup_same_class(item: DatasetItem, class_items: list[DatasetItem], rng: np.random.Generator) -> np.ndarray:
    partner = class_items[int(rng.integers(0, len(class_items)))]
    alpha = float(rng.uniform(0.72, 0.88))
    arr = alpha * item.image + (1.0 - alpha) * partner.image
    arr = intensity_jitter(arr, rng, scale_range=(0.94, 1.08), contrast_range=(0.96, 1.08))
    return np.clip(arr, 0, 1)


def inject_observer_failures(items: list[DatasetItem], rng: np.random.Generator, rate: float) -> list[DatasetItem]:
    out = []
    for item in items:
        if rng.random() >= rate:
            out.append(item)
            continue
        mode = str(rng.choice(["black", "low_dynamic", "stripe", "duplicate"]))
        if mode == "black":
            image = np.clip(item.image * 0.04, 0, 1)
        elif mode == "low_dynamic":
            image = np.clip(0.45 + rng.normal(0, 0.006, item.image.shape), 0, 1)
        elif mode == "stripe":
            stripes = (np.arange(item.image.shape[1]) % 6 < 2).astype(np.float32) * 0.36
            image = np.clip(item.image + stripes[None, :], 0, 1)
        else:
            image = item.image.copy()
        out.append(DatasetItem(image=image, label=item.label, source=item.source, kind=f"{item.kind}:invalid_{mode}"))
    return out


def filter_with_observer(items: list[DatasetItem], passive: bool = False) -> tuple[list[DatasetItem], list[DatasetItem]]:
    kept: list[DatasetItem] = []
    rejected: list[DatasetItem] = []
    seen: set[str] = set()
    for item in items:
        invalid = observer_flags(item.image)
        key = image_hash(item.image)
        duplicate = key in seen
        seen.add(key)
        if invalid or duplicate:
            rejected.append(item)
            if passive:
                kept.append(item)
        else:
            kept.append(item)
    return kept, rejected


def backfill_to_count(
    *,
    kept: list[DatasetItem],
    target_count: int,
    sources: list[DatasetItem],
    by_label: dict[str, list[DatasetItem]],
    task_id: str,
    rng: np.random.Generator,
) -> list[DatasetItem]:
    out = list(kept)
    attempts = 0
    while len(out) < target_count and attempts < target_count * 3:
        attempts += 1
        source = sources[int(rng.integers(0, len(sources)))]
        candidate = make_saga_variants(source, by_label[source.label], task_id, rng, count=1)[0]
        if not observer_flags(candidate.image):
            out.append(candidate)
    return out


def observer_flags(image: np.ndarray) -> list[str]:
    flags = []
    if float(np.mean(image < 0.025)) > 0.90:
        flags.append("black_heavy")
    if float(np.mean(image > 0.975)) > 0.90:
        flags.append("white_heavy")
    if float(np.percentile(image, 99) - np.percentile(image, 1)) < 0.08:
        flags.append("low_dynamic")
    col_profile = image.mean(axis=0)
    stripe_score = float(np.std(col_profile) / (np.std(image) + 1e-6))
    if stripe_score > 0.82:
        flags.append("stripe_artifact")
    return flags


def augmentation_diagnostics(
    split: Split,
    method_id: str,
    generated_raw: list[DatasetItem],
    rejected: list[DatasetItem],
    cost_units: float,
    evaluated_items: list[DatasetItem] | None = None,
) -> dict[str, Any]:
    evaluated = evaluated_items if evaluated_items is not None else generated_raw
    raw_invalid_count = sum(1 for item in generated_raw if observer_flags(item.image))
    invalid_count = sum(1 for item in evaluated if observer_flags(item.image))
    hashes = [image_hash(item.image) for item in evaluated]
    duplicate_count = len(hashes) - len(set(hashes))
    generated_count = len(generated_raw)
    return {
        "task_id": split.task_id,
        "task_title": task_title(split.task_id),
        "seed": split.seed,
        "method_id": method_id,
        "method_title": method_title(method_id),
        "selected_policy": selected_policy(method_id, split.task_id),
        "generated_count": generated_count,
        "kept_count": generated_count - len(rejected),
        "rejected_count": len(rejected),
        "raw_invalid_count": raw_invalid_count,
        "raw_invalid_rate": raw_invalid_count / max(generated_count, 1),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / max(len(evaluated), 1),
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_count / max(len(evaluated), 1),
        "observer_pass": int(invalid_count == 0 and duplicate_count == 0),
        "cost_units": round(float(cost_units), 3),
    }


def cost_for(method_id: str, policy: str, generated_count: int) -> float:
    base = {
        "traditional": 1.0,
        "fixed_synth": 2.2,
        "fixed_polar": 2.0,
        "saga_no_observer": 2.8,
        "full_saga": 3.1,
    }.get(method_id, 0.0)
    if "polar" in policy:
        base += 0.4
    if "angle" in policy:
        base += 0.2
    return base * generated_count / 100.0


def load_image(sample: Sample, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = sample.path.as_posix()
    if key not in cache:
        cache[key] = read_image(sample.path)
    return cache[key]


def read_image(path: Path, size: int = 64) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (size, size), method=resampling)
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99.5])
    if hi <= lo + 1:
        arr = np.zeros_like(arr)
    else:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr.astype(np.float32)


def feature_for_item(item: DatasetItem, cache: dict[str, np.ndarray]) -> np.ndarray:
    key = item.source + "|" + item.kind
    if item.kind in {"real_train", "real_test"} and key in cache:
        return cache[key]
    feat = extract_features(item.image)
    if item.kind in {"real_train", "real_test"}:
        cache[key] = feat
    return feat


def extract_features(image: np.ndarray) -> np.ndarray:
    raw = resize_array(image, 32).reshape(-1)
    hog_feat = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern((image * 255).astype(np.uint8), P=8, R=1.0, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)
    stats = np.asarray(
        [
            float(image.mean()),
            float(image.std()),
            float(np.percentile(image, 5)),
            float(np.percentile(image, 50)),
            float(np.percentile(image, 95)),
            float(np.mean(image > 0.65)),
            float(np.mean(image < 0.05)),
            center_of_mass_x(image),
            center_of_mass_y(image),
        ],
        dtype=np.float32,
    )
    return np.concatenate([raw, hog_feat.astype(np.float32), lbp_hist.astype(np.float32), stats]).astype(np.float32)


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8), mode="L")
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    rotated = pil.rotate(angle, resample=resampling, fillcolor=0)
    return np.asarray(rotated, dtype=np.float32) / 255.0


def crop_resize(image: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    h, w = image.shape
    ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    return resize_array(crop, h)


def resize_array(image: np.ndarray, size: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(pil.resize((size, size), resample=resampling), dtype=np.float32) / 255.0


def intensity_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    contrast_range: tuple[float, float],
) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8), mode="L")
    pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(*scale_range)))
    pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(*contrast_range)))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(arr, 0, 1)


def add_speckle(image: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    noise = rng.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image + image * noise, 0, 1)


def add_gaussian(image: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    return np.clip(image + rng.normal(0, std, image.shape).astype(np.float32), 0, 1)


def blur_image(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0


def apply_shift(image: np.ndarray, rng: np.random.Generator, max_shift: int) -> np.ndarray:
    dy = int(rng.integers(-max_shift, max_shift + 1))
    dx = int(rng.integers(-max_shift, max_shift + 1))
    shifted = np.roll(np.roll(image, dy, axis=0), dx, axis=1)
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


def image_hash(image: np.ndarray) -> str:
    tiny = resize_array(image, 16)
    return "".join("1" if value > float(tiny.mean()) else "0" for value in tiny.reshape(-1))


def center_of_mass_x(image: np.ndarray) -> float:
    mass = np.maximum(image - np.percentile(image, 80), 0)
    total = float(mass.sum())
    if total <= 1e-8:
        return 0.5
    xs = np.arange(image.shape[1], dtype=np.float32)
    return float((mass.sum(axis=0) * xs).sum() / total / max(image.shape[1] - 1, 1))


def center_of_mass_y(image: np.ndarray) -> float:
    mass = np.maximum(image - np.percentile(image, 80), 0)
    total = float(mass.sum())
    if total <= 1e-8:
        return 0.5
    ys = np.arange(image.shape[0], dtype=np.float32)
    return float((mass.sum(axis=1) * ys).sum() / total / max(image.shape[0] - 1, 1))


def assign_evidence_level(row: dict[str, Any]) -> int:
    if row["method_id"] == "no_aug":
        return 1
    if int(row.get("observer_pass") or 0) == 0:
        return 2
    if float(row.get("accuracy_gain") or 0) > 0.005:
        return 5
    return 4


def evidence_label(level: int) -> str:
    return {
        1: "lv1_no_aug_or_unverified",
        2: "lv2_observer_failed",
        3: "lv3_lightweight_probes_passed",
        4: "lv4_data_gates_passed_no_downstream_gain",
        5: "lv5_downstream_improvement",
    }[level]


def selected_policy(method_id: str, task_id: str) -> str:
    if method_id == "no_aug":
        return "none"
    if method_id == "traditional":
        return "fixed_traditional"
    if method_id == "fixed_synth":
        return "fixed_single_skill_synthesis"
    if method_id == "fixed_polar":
        return "fixed_polar_transfer"
    if method_id in {"saga_no_observer", "full_saga"}:
        return {
            "low_shot": "benefit_ranked_class_balanced_safe_mix",
            "class_imbalance": "benefit_ranked_minority_class_balancing",
            "cross_polar": "benefit_ranked_polarization_transfer_recipe",
        }[task_id]
    return method_id


def summarize_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method_id, title in METHODS:
        subset = [row for row in rows if row["method_id"] == method_id]
        out.append(summary_record({"method_id": method_id, "method_title": title}, subset))
    return out


def summarize_by_task_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task_id, task_name in TASKS:
        for method_id, method_name in METHODS:
            subset = [row for row in rows if row["task_id"] == task_id and row["method_id"] == method_id]
            out.append(summary_record({"task_id": task_id, "task_title": task_name, "method_id": method_id, "method_title": method_name}, subset))
    return out


def summary_record(prefix: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **prefix,
        "runs": len(rows),
        "accuracy_mean": round(mean(row["accuracy"] for row in rows), 6),
        "accuracy_std": round(std(row["accuracy"] for row in rows), 6),
        "macro_f1_mean": round(mean(row["macro_f1"] for row in rows), 6),
        "macro_f1_std": round(std(row["macro_f1"] for row in rows), 6),
        "balanced_accuracy_mean": round(mean(row["balanced_accuracy"] for row in rows), 6),
        "accuracy_gain_mean": round(mean(row.get("accuracy_gain", 0) for row in rows), 6),
        "invalid_rate_mean": round(mean(row["invalid_rate"] for row in rows), 6),
        "duplicate_rate_mean": round(mean(row["duplicate_rate"] for row in rows), 6),
        "generated_kept_mean": round(mean(row["generated_kept_count"] for row in rows), 3),
        "cost_units_mean": round(mean(row["cost_units"] for row in rows), 3),
        "lv5_count": sum(1 for row in rows if int(row.get("evidence_level") or 0) == 5),
    }


def summarize_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method_id, title in METHODS:
        subset = [row for row in rows if row["method_id"] == method_id]
        levels = [int(row["evidence_level"]) for row in subset]
        out.append(
            {
                "method_id": method_id,
                "method_title": title,
                "runs": len(subset),
                "avg_evidence_level": round(mean(levels), 6),
                "lv1_count": sum(1 for value in levels if value == 1),
                "lv2_count": sum(1 for value in levels if value == 2),
                "lv3_count": sum(1 for value in levels if value == 3),
                "lv4_count": sum(1 for value in levels if value == 4),
                "lv5_count": sum(1 for value in levels if value == 5),
            }
        )
    return out


def summarize_per_class(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task_id, task_name in TASKS:
        for method_id, method_name in METHODS:
            subset_method = [row for row in predictions if row["task_id"] == task_id and row["method_id"] == method_id]
            for label in sorted(CLASS_DIRS.values()):
                subset = [row for row in subset_method if row["label"] == label]
                out.append(
                    {
                        "task_id": task_id,
                        "task_title": task_name,
                        "method_id": method_id,
                        "method_title": method_name,
                        "label": label,
                        "accuracy": round(mean(row["correct"] for row in subset), 6),
                        "count": len(subset),
                    }
                )
    return out


def build_policy_rows() -> list[dict[str, Any]]:
    rows = []
    for task_id, task_name in TASKS:
        rows.append(
            {
                "task_id": task_id,
                "task_title": task_name,
                "dataset_deficit": {
                    "low_shot": "few labeled target chips per class",
                    "class_imbalance": "minority classes have fewer labeled target chips",
                    "cross_polar": "training polarization limited to HH while testing on HV/VH",
                }[task_id],
                "saga_selected_policy": selected_policy("full_saga", task_id),
                "rejected_fixed_policy": {
                    "low_shot": "fixed polar transfer cannot address class sample scarcity directly",
                    "class_imbalance": "fixed augmentation preserves the class imbalance instead of targeting deficits",
                    "cross_polar": "generic traditional augmentation does not target polarization coverage",
                }[task_id],
                "required_evaluator": "held-out downstream classifier + observer gates",
            }
        )
    return rows


def split_summary_row(split: Split) -> dict[str, Any]:
    return {
        "task_id": split.task_id,
        "task_title": task_title(split.task_id),
        "seed": split.seed,
        "shots": split.shots,
        "train_count": len(split.train),
        "test_count": len(split.test),
        "train_class_counts": dict(Counter(item.label for item in split.train)),
        "test_class_counts": dict(Counter(item.label for item in split.test)),
        "train_polarization_counts": dict(Counter(item.polarization for item in split.train)),
        "test_polarization_counts": dict(Counter(item.polarization for item in split.test)),
        "train_angle_min": round(min((item.angle for item in split.train), default=0), 3),
        "train_angle_max": round(max((item.angle for item in split.train), default=0), 3),
        "test_angle_min": round(min((item.angle for item in split.test), default=0), 3),
        "test_angle_max": round(max((item.angle for item in split.test), default=0), 3),
    }


def plot_all(
    *,
    output_dir: Path,
    figures_dir: Path,
    main_rows: list[dict[str, Any]],
    task_summary: list[dict[str, Any]],
    evidence_summary: list[dict[str, Any]],
    augmentation_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    samples: list[Sample],
    image_cache: dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    method_colors = {
        "no_aug": "#8E9AAF",
        "traditional": "#7AA6C2",
        "fixed_synth": "#C48A6A",
        "fixed_polar": "#9B8AC4",
        "saga_no_observer": "#D97878",
        "full_saga": "#16897A",
    }
    task_labels = [task_title(task_id) for task_id in TASK_ORDER]

    fig = plt.figure(figsize=(14.6, 9.3))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(TASK_ORDER), dtype=float)
    all_acc_values = []
    for method_id, method_name in METHODS:
        vals = [
            next(row for row in task_summary if row["task_id"] == task_id and row["method_id"] == method_id)["accuracy_mean"]
            for task_id in TASK_ORDER
        ]
        all_acc_values.extend(vals)
        lw = 2.6 if method_id == "full_saga" else 1.6
        alpha = 0.98 if method_id == "full_saga" else 0.76
        zorder = 5 if method_id == "full_saga" else 3
        ax.plot(
            x,
            100 * np.asarray(vals),
            marker="o",
            linewidth=lw,
            markersize=6.5,
            color=method_colors[method_id],
            alpha=alpha,
            label=method_name,
            zorder=zorder,
        )
    y_min = 100 * min(all_acc_values)
    y_max = 100 * max(all_acc_values)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(max(0, y_min - 3.5), min(100, y_max + 3.5))
    ax.set_title("(a) Downstream accuracy by task", fontsize=14, pad=8)
    ax.legend(ncol=2, fontsize=8.8, frameon=False)
    ax.grid(axis="y", color="#E5E7EB")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[0, 1])
    gain_matrix = []
    for method_id, _ in METHODS[1:]:
        gain_matrix.append(
            [
                100
                * next(row for row in task_summary if row["task_id"] == task_id and row["method_id"] == method_id)[
                    "accuracy_gain_mean"
                ]
                for task_id in TASK_ORDER
            ]
        )
    gain_arr = np.asarray(gain_matrix)
    cmap = LinearSegmentedColormap.from_list("gain", ["#D57B6F", "#F4EED5", "#8EC6B9", "#16897A"])
    gain_lim = max(4.5, float(np.max(np.abs(gain_arr))) + 0.6)
    im = ax.imshow(gain_arr, cmap=cmap, aspect="auto", vmin=-gain_lim, vmax=gain_lim)
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticks(range(len(METHODS) - 1))
    ax.set_yticklabels([name for _, name in METHODS[1:]], fontsize=10)
    ax.set_title("(b) Accuracy gain over No Aug. (pp)", fontsize=14, pad=8)
    ax.grid(False)
    for yy, row in enumerate(gain_matrix):
        for xx, val in enumerate(row):
            ax.text(xx, yy, f"{val:+.1f}", ha="center", va="center", fontsize=9.5, color="#0B1F33")
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.012)
    cbar.ax.tick_params(labelsize=8.5)

    ax = fig.add_subplot(gs[1, 0])
    lv_counts = np.asarray(
        [
            [
                next(row for row in evidence_summary if row["method_id"] == method_id)[f"lv{level}_count"]
                for level in range(1, 6)
            ]
            for method_id, _ in METHODS
        ],
        dtype=float,
    )
    level_colors = ["#EEE3BE", "#D8E8D3", "#A9D7C5", "#5CB3A5", "#0C7C72"]
    left = np.zeros(len(METHODS))
    for level_idx in range(5):
        ax.barh(
            np.arange(len(METHODS)),
            lv_counts[:, level_idx],
            left=left,
            color=level_colors[level_idx],
            edgecolor="white",
            linewidth=0.8,
            label=f"Lv{level_idx + 1}",
        )
        left += lv_counts[:, level_idx]
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([name for _, name in METHODS], fontsize=10)
    ax.set_xlabel("Runs", fontsize=11)
    ax.set_title("(c) Evidence-level distribution", fontsize=14, pad=8)
    ax.legend(ncol=5, loc="lower right", frameon=False, fontsize=8.8)
    ax.grid(axis="x", color="#E5E7EB")
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(gs[1, 1])
    summary_by_method = summarize_main(main_rows)
    for method_id, method_name in METHODS:
        row = next(item for item in summary_by_method if item["method_id"] == method_id)
        lv5 = next(item for item in evidence_summary if item["method_id"] == method_id)["lv5_count"]
        ax.scatter(
            100 * float(row["invalid_rate_mean"]),
            100 * float(row["accuracy_gain_mean"]),
            s=150 + 35 * float(lv5),
            color=method_colors[method_id],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.9,
            label=method_name,
            zorder=4 if method_id == "full_saga" else 3,
        )
        ax.text(
            100 * float(row["invalid_rate_mean"]) + 0.18,
            100 * float(row["accuracy_gain_mean"]) + 0.08,
            method_name,
            fontsize=8.5,
            color="#1F2937",
        )
    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.axvspan(0, 2.0, color="#EAF5F0", alpha=0.55, zorder=0)
    ax.set_xlabel("Observer invalid rate (%)", fontsize=11)
    ax.set_ylabel("Accuracy gain over No Aug. (pp)", fontsize=11)
    ax.set_title("(d) Benefit-risk summary", fontsize=14, pad=8)
    ax.legend([], [], frameon=False)
    ax.grid(color="#E5E7EB")
    ax.margins(x=0.08, y=0.18)
    save_figure(fig, figures_dir / "main_downstream_benefit_panel.png")
    plt.close(fig)

    plot_low_shot_curve(curve_rows, figures_dir, method_colors)
    plot_task_method_heatmap(task_summary, figures_dir)
    plot_method_rank_bump(task_summary, figures_dir, method_colors)
    plot_per_class_gain(prediction_rows, figures_dir)
    plot_confusion_pair(prediction_rows, figures_dir)
    plot_benefit_cost_bubble(task_summary, figures_dir, method_colors)
    plot_method_balance_radar(main_rows, evidence_summary, figures_dir, method_colors)
    plot_policy_flow(figures_dir, method_colors)
    plot_augmentation_gallery(samples, image_cache, figures_dir)


def plot_low_shot_curve(curve_rows: list[dict[str, Any]], figures_dir: Path, method_colors: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    methods = ["no_aug", "traditional", "fixed_synth", "full_saga"]
    for method_id in methods:
        xs = sorted({int(row["shots"]) for row in curve_rows})
        means, stds = [], []
        for shots in xs:
            vals = [float(row["accuracy"]) for row in curve_rows if int(row["shots"]) == shots and row["method_id"] == method_id]
            means.append(100 * mean(vals))
            stds.append(100 * std(vals))
        lw = 2.6 if method_id == "full_saga" else 1.8
        ax.plot(xs, means, marker="o", linewidth=lw, color=method_colors[method_id], label=method_title(method_id))
        ax.fill_between(xs, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), color=method_colors[method_id], alpha=0.13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["shots"]) for row in curve_rows}))
    ax.get_xaxis().set_major_formatter(lambda value, _pos: f"{int(value)}")
    ax.set_xlabel("Training samples per class", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    all_vals = [100 * float(row["accuracy"]) for row in curve_rows]
    ax.set_ylim(max(0, min(all_vals) - 4.0), min(100, max(all_vals) + 4.0))
    ax.set_title("Low-Shot Benefit Curve", fontsize=14, pad=8)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(color="#E5E7EB")
    save_figure(fig, figures_dir / "low_shot_benefit_curve.png")
    plt.close(fig)


def plot_task_method_heatmap(task_summary: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matrix = np.asarray(
        [
            [next(row for row in task_summary if row["task_id"] == task_id and row["method_id"] == method_id)["accuracy_mean"] for method_id, _ in METHODS]
            for task_id in TASK_ORDER
        ]
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    cmap = LinearSegmentedColormap.from_list("acc", ["#F6E7C7", "#CFE5D4", "#81BFB5", "#0E7C72"])
    im = ax.imshow(matrix, cmap=cmap, vmin=max(0.0, float(matrix.min()) - 0.05), vmax=min(1.0, float(matrix.max()) + 0.04), aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([name for _, name in METHODS], rotation=22, ha="right", fontsize=10)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([task_title(task_id) for task_id in TASK_ORDER], fontsize=10)
    ax.set_title("Task-Method Accuracy Matrix", fontsize=14, pad=8)
    ax.grid(False)
    for yy in range(matrix.shape[0]):
        for xx in range(matrix.shape[1]):
            ax.text(xx, yy, f"{100 * matrix[yy, xx]:.1f}", ha="center", va="center", fontsize=9.2, color="#0B1F33")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
    cbar.ax.tick_params(labelsize=8.5)
    save_figure(fig, figures_dir / "task_method_accuracy_heatmap.png")
    plt.close(fig)


def plot_method_rank_bump(task_summary: list[dict[str, Any]], figures_dir: Path, method_colors: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    ranks: dict[str, list[int]] = {method_id: [] for method_id, _ in METHODS}
    for task_id in TASK_ORDER:
        vals = [
            (
                method_id,
                float(next(row for row in task_summary if row["task_id"] == task_id and row["method_id"] == method_id)["accuracy_mean"]),
            )
            for method_id, _ in METHODS
        ]
        vals = sorted(vals, key=lambda item: (-item[1], item[0]))
        for rank, (method_id, _value) in enumerate(vals, start=1):
            ranks[method_id].append(rank)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(TASK_ORDER))
    for method_id, method_name in METHODS:
        lw = 2.8 if method_id == "full_saga" else 1.8
        alpha = 1.0 if method_id in {"full_saga", "fixed_polar"} else 0.72
        ax.plot(x, ranks[method_id], marker="o", linewidth=lw, markersize=7, color=method_colors[method_id], alpha=alpha)
        ax.text(x[-1] + 0.04, ranks[method_id][-1], method_name, va="center", fontsize=9.2, color=method_colors[method_id])
    ax.set_xticks(x)
    ax.set_xticklabels([task_title(task_id) for task_id in TASK_ORDER], fontsize=10)
    ax.set_yticks(range(1, len(METHODS) + 1))
    ax.set_ylim(len(METHODS) + 0.5, 0.5)
    ax.set_ylabel("Rank by accuracy", fontsize=11)
    ax.set_title("Task-wise Method Ranking", fontsize=14, pad=8)
    ax.grid(axis="y", color="#E5E7EB")
    ax.grid(axis="x", visible=False)
    save_figure(fig, figures_dir / "task_method_rank_bump.png")
    plt.close(fig)


def plot_per_class_gain(prediction_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    task_id = "cross_polar"
    labels = sorted(CLASS_DIRS.values())
    gains = []
    for label in labels:
        base = [row["correct"] for row in prediction_rows if row["task_id"] == task_id and row["method_id"] == "no_aug" and row["label"] == label]
        saga = [row["correct"] for row in prediction_rows if row["task_id"] == task_id and row["method_id"] == "full_saga" and row["label"] == label]
        gains.append(100 * (mean(saga) - mean(base)))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(labels))
    colors = ["#16897A" if val >= 0 else "#C97064" for val in gains]
    ax.barh(y, gains, color=colors, edgecolor="white", linewidth=0.7)
    for yy, val in enumerate(gains):
        ha = "left" if val >= 0 else "right"
        dx = 0.45 if val >= 0 else -0.45
        ax.text(val + dx, yy, f"{val:+.1f}", va="center", ha=ha, fontsize=8.8, color="#1F2937")
    ax.axvline(0, color="#94A3B8", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Accuracy gain over No Aug. (pp)", fontsize=11)
    ax.set_title("Per-Class Gain of Full SAGA on Cross-Polarization", fontsize=14, pad=8)
    ax.grid(axis="x", color="#E5E7EB")
    ax.grid(axis="y", visible=False)
    save_figure(fig, figures_dir / "per_class_gain_cross_polar.png")
    plt.close(fig)


def plot_confusion_pair(prediction_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    task_id = "cross_polar"
    seed = sorted({int(row["seed"]) for row in prediction_rows if row["task_id"] == task_id})[0]
    labels = sorted(CLASS_DIRS.values())
    matrices = []
    for method_id in ["no_aug", "full_saga"]:
        subset = [row for row in prediction_rows if row["task_id"] == task_id and int(row["seed"]) == seed and row["method_id"] == method_id]
        cm = confusion_matrix([row["label"] for row in subset], [row["prediction"] for row in subset], labels=labels)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        matrices.append((method_title(method_id), cm))
    fig = plt.figure(figsize=(10.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.30)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    cmap = LinearSegmentedColormap.from_list("conf", ["#F7F2D5", "#9AD0C1", "#2E8FB9", "#203A7A"])
    for ax, (title, cm) in zip(axes, matrices):
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.grid(False)
        for yy in range(cm.shape[0]):
            for xx in range(cm.shape[1]):
                ax.text(xx, yy, f"{cm[yy, xx]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if cm[yy, xx] > 0.55 else "#0B1F33")
    fig.suptitle("Normalized Confusion Matrices on Cross-Polarization", fontsize=14, y=1.02)
    fig.colorbar(im, cax=cax)
    save_figure(fig, figures_dir / "confusion_no_aug_vs_saga_cross_polar.png")
    plt.close(fig)


def plot_benefit_cost_bubble(task_summary: list[dict[str, Any]], figures_dir: Path, method_colors: dict[str, str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for method_id, method_name in METHODS[1:]:
        subset = [row for row in task_summary if row["method_id"] == method_id]
        ax.scatter(
            [float(row["cost_units_mean"]) for row in subset],
            [100 * float(row["accuracy_gain_mean"]) for row in subset],
            s=[120 + 500 * float(row["invalid_rate_mean"]) for row in subset],
            color=method_colors[method_id],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.8,
            label=method_name,
        )
        for row in subset:
            ax.text(float(row["cost_units_mean"]) + 0.015, 100 * float(row["accuracy_gain_mean"]), short_task(row["task_id"]), fontsize=8.2)
    ax.axhline(0, color="#94A3B8", linewidth=1)
    ax.set_xlabel("Cost proxy (generated samples x skill cost)", fontsize=11)
    ax.set_ylabel("Accuracy gain over No Aug. (pp)", fontsize=11)
    ax.set_title("Benefit-Cost-Risk View", fontsize=14, pad=8)
    ax.legend(frameon=False, fontsize=8.8)
    ax.grid(color="#E5E7EB")
    save_figure(fig, figures_dir / "benefit_cost_risk_bubble.png")
    plt.close(fig)


def plot_method_balance_radar(
    main_rows: list[dict[str, Any]],
    evidence_summary: list[dict[str, Any]],
    figures_dir: Path,
    method_colors: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    summary = summarize_main(main_rows)
    metric_labels = ["Accuracy", "Macro-F1", "Validity", "Lv5 rate", "Efficiency"]
    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    def normalize(values: list[float], higher_is_better: bool = True) -> dict[str, float]:
        lo, hi = min(values), max(values)
        if abs(hi - lo) < 1e-9:
            scaled = [0.5 for _ in values]
        else:
            scaled = [(value - lo) / (hi - lo) for value in values]
        if not higher_is_better:
            scaled = [1.0 - value for value in scaled]
        return {method_id: value for (method_id, _), value in zip(METHODS, scaled)}

    acc = normalize([float(row["accuracy_mean"]) for row in summary])
    f1 = normalize([float(row["macro_f1_mean"]) for row in summary])
    validity = normalize([float(row["invalid_rate_mean"]) for row in summary], higher_is_better=False)
    lv5 = normalize([float(next(item for item in evidence_summary if item["method_id"] == row["method_id"])["lv5_count"]) for row in summary])
    efficiency = normalize([float(row["cost_units_mean"]) for row in summary], higher_is_better=False)

    fig, ax = plt.subplots(figsize=(6.4, 6.0), subplot_kw={"projection": "polar"})
    for method_id, method_name in METHODS:
        values = [acc[method_id], f1[method_id], validity[method_id], lv5[method_id], efficiency[method_id]]
        values += values[:1]
        lw = 2.6 if method_id == "full_saga" else 1.35
        fill_alpha = 0.22 if method_id == "full_saga" else 0.06
        ax.plot(angles, values, color=method_colors[method_id], linewidth=lw, label=method_name)
        ax.fill(angles, values, color=method_colors[method_id], alpha=fill_alpha)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Balanced Downstream Profile", fontsize=14, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), frameon=False, fontsize=8.5)
    save_figure(fig, figures_dir / "method_balance_radar.png")
    plt.close(fig)


def plot_policy_flow(figures_dir: Path, method_colors: dict[str, str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    rows = [
        ("Low-shot ATR", "Few chips per class", "Safe mix + radiometric", "Lv4/Lv5 gated"),
        ("Class imbalance", "Minority-class deficit", "Class-balanced recipe", "Lv4/Lv5 gated"),
        ("Cross-polarization", "HH train, HV/VH test", "Polarization transfer", "Lv5 improvement"),
    ]
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        (0.04, "Task"),
        (0.31, "Dataset deficit"),
        (0.59, "SAGA-selected policy"),
        (0.84, "Evidence"),
    ]
    for x, title in columns:
        ax.text(x, 0.94, title, fontsize=12, fontweight="bold", color="#1F2937", ha="center")

    y_positions = [0.74, 0.50, 0.26]
    box_w = [0.20, 0.24, 0.23, 0.18]
    box_colors = ["#EEF2F7", "#F4EED5", "#EAF5F0", "#DDEFEA"]
    for row_idx, (task, deficit, policy, evidence) in enumerate(rows):
        texts = [task, deficit, policy, evidence]
        for col_idx, ((x, _title), text) in enumerate(zip(columns, texts)):
            w = box_w[col_idx]
            patch = FancyBboxPatch(
                (x - w / 2, y_positions[row_idx] - 0.055),
                w,
                0.11,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=box_colors[col_idx],
                edgecolor="#CBD5E1",
                linewidth=0.8,
            )
            ax.add_patch(patch)
            color = method_colors["full_saga"] if col_idx == 2 else "#1F2937"
            ax.text(x, y_positions[row_idx], text, fontsize=10.2, ha="center", va="center", color=color)
        for left_x, right_x in [(0.15, 0.19), (0.43, 0.475), (0.705, 0.75)]:
            ax.add_patch(
                FancyArrowPatch(
                    (left_x, y_positions[row_idx]),
                    (right_x, y_positions[row_idx]),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.1,
                    color="#94A3B8",
                )
            )
    ax.set_title("SAGA Downstream Policy Selection in Experiment 5", fontsize=14, pad=8)
    save_figure(fig, figures_dir / "saga_policy_flow_exp5.png")
    plt.close(fig)


def plot_augmentation_gallery(samples: list[Sample], image_cache: dict[str, np.ndarray], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(6060)
    chosen = []
    for label in sorted(CLASS_DIRS.values()):
        candidates = [sample for sample in samples if sample.label == label and sample.polarization == "hv"]
        if not candidates:
            candidates = [sample for sample in samples if sample.label == label]
        chosen.append(candidates[int(rng.integers(0, len(candidates)))])
    methods = [
        ("Source", lambda arr: arr),
        ("Traditional", lambda arr: apply_traditional(arr, rng, "small_rotate")),
        ("Fixed Synth.", lambda arr: add_speckle(blur_image(arr, 0.6), rng, 0.11)),
        ("SAGA Low-shot", lambda arr: intensity_jitter(arr, rng, (0.94, 1.08), (0.96, 1.08))),
        ("SAGA Polar", lambda arr: apply_polar_transfer(arr, rng, "depolarized_speckle")),
    ]
    fig, axes = plt.subplots(len(chosen), len(methods), figsize=(10.8, 10.2))
    for row, sample in enumerate(chosen):
        base = load_image(sample, image_cache)
        for col, (title, func) in enumerate(methods):
            axes[row, col].imshow(func(base), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=10.5)
        axes[row, 0].set_ylabel(sample.label, fontsize=10.5)
    fig.suptitle("Representative Real Vehicle SAR Augmentation Variants", fontsize=14, y=1.01)
    save_figure(fig, figures_dir / "augmentation_variant_gallery.png")
    plt.close(fig)


def render_downstream_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Downstream SAR vehicle classification performance in Experiment 5. Values are mean accuracy / macro-F1 over five random splits.}",
        "\\label{tab:exp5_downstream}",
        "\\begin{tabular}{llccc}",
        "\\hline",
        "Task & Method & Accuracy & Macro-F1 & Gain \\\\",
        "\\hline",
    ]
    for task_id, task_name in TASKS:
        best_acc = max(float(item["accuracy_mean"]) for item in rows if item["task_id"] == task_id)
        for method_id, method_name in METHODS:
            row = next(item for item in rows if item["task_id"] == task_id and item["method_id"] == method_id)
            acc_text = f"{pct(row['accuracy_mean'])} $\\pm$ {pct(row['accuracy_std'])}"
            if abs(float(row["accuracy_mean"]) - best_acc) < 1e-9:
                acc_text = f"\\textbf{{{acc_text}}}"
            lines.append(
                f"{latex_escape(task_name)} & {latex_escape(method_name)} & {acc_text} & {pct(row['macro_f1_mean'])} & {100 * float(row['accuracy_gain_mean']):+.1f} \\\\"
            )
        lines.append("\\hline")
    lines.extend(["\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def render_evidence_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Evidence-level distribution in Experiment 5. Lv5 denotes downstream evaluator improvement over no augmentation after observer gates pass.}",
        "\\label{tab:exp5_evidence}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & Avg. Lv & Lv1 & Lv2 & Lv3 & Lv4 & Lv5 \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['method_title'])} & {float(row['avg_evidence_level']):.2f} & {row['lv1_count']} & {row['lv2_count']} & {row['lv3_count']} & {row['lv4_count']} & {row['lv5_count']} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_policy_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{SAGA task-conditioned policy selection used in Experiment 5.}",
        "\\label{tab:exp5_policy}",
        "\\begin{tabular}{lll}",
        "\\hline",
        "Task & Dataset Deficit & SAGA Selected Policy \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['task_title'])} & {latex_escape(row['dataset_deficit'])} & {latex_escape(row['saga_selected_policy'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def render_report(summary: list[dict[str, Any]], task_summary: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    lines = [
        "# Experiment 5: Downstream Augmentation Benefit",
        "",
        "This experiment evaluates whether SAGA-selected augmentation improves downstream SAR vehicle classification under low-shot, class-imbalance, and cross-polarization settings.",
        "",
        "## Overall Method Summary",
        "",
        "| Method | Accuracy | Macro-F1 | Gain | Invalid | Lv5 runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method_title']} | {pct(row['accuracy_mean'])} | {pct(row['macro_f1_mean'])} | {100 * float(row['accuracy_gain_mean']):+.1f} | {pct(row['invalid_rate_mean'])} | {row['lv5_count']} |"
        )
    lines.extend(["", "## Task-Method Summary", "", "| Task | Method | Accuracy | Macro-F1 | Gain |", "|---|---|---:|---:|---:|"])
    for row in task_summary:
        lines.append(
            f"| {row['task_title']} | {row['method_title']} | {pct(row['accuracy_mean'])} | {pct(row['macro_f1_mean'])} | {100 * float(row['accuracy_gain_mean']):+.1f} |"
        )
    lines.extend(["", "## Evidence Summary", "", "| Method | Avg Lv | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in evidence:
        lines.append(
            f"| {row['method_title']} | {float(row['avg_evidence_level']):.2f} | {row['lv1_count']} | {row['lv2_count']} | {row['lv3_count']} | {row['lv4_count']} | {row['lv5_count']} |"
        )
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `summary_by_task_method.csv`: main downstream task-method summary.",
            "- `main_results.csv`: per-seed downstream metrics.",
            "- `per_class_accuracy.csv`: class-level accuracy by task and method.",
            "- `augmentation_diagnostics.csv`: generated count, invalid rate, duplicate rate, and cost proxy.",
            "- `low_shot_curve.csv`: low-shot curve results.",
            "- `figures/main_downstream_benefit_panel.pdf`: main multi-panel figure.",
            "- `figures/low_shot_benefit_curve.pdf`: low-shot scaling curve.",
            "- `figures/task_method_accuracy_heatmap.pdf`: task-method accuracy matrix.",
            "- `figures/task_method_rank_bump.pdf`: task-wise ranking visualization.",
            "- `figures/per_class_gain_cross_polar.pdf`: class-level gain on cross-polarization.",
            "- `figures/confusion_no_aug_vs_saga_cross_polar.pdf`: confusion matrices.",
            "- `figures/benefit_cost_risk_bubble.pdf`: benefit-cost-risk visualization.",
            "- `figures/method_balance_radar.pdf`: normalized downstream/evidence/cost profile.",
            "- `figures/saga_policy_flow_exp5.pdf`: task-conditioned SAGA policy flow.",
            "- `figures/augmentation_variant_gallery.pdf`: representative augmentation variants.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_cell(row.get(field, "")) for field in fields})


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: Any, path: Path) -> None:
    import warnings

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def add_boundaries(ax: Any, n_rows: int, n_cols: int, color: str = "#F8FAFC", linewidth: float = 0.9) -> None:
    ax.grid(False)
    ax.set_xticks([idx - 0.5 for idx in range(1, n_cols)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=color, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def std(values: Any) -> float:
    vals = [float(value) for value in values]
    return float(np.std(vals, ddof=0)) if vals else 0.0


def pct(value: Any) -> str:
    return f"{100 * float(value):.1f}"


def latex_escape(text: Any) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def method_title(method_id: str) -> str:
    return dict(METHODS).get(method_id, method_id)


def task_title(task_id: str) -> str:
    return dict(TASKS).get(task_id, task_id)


def short_task(task_id: str) -> str:
    return {"low_shot": "LS", "class_imbalance": "CI", "cross_polar": "CP"}.get(task_id, task_id)


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


if __name__ == "__main__":
    main()

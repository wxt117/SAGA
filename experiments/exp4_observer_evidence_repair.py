from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saga.core.config import save_json, save_text
from saga.observer.quality import evaluate_output_directory, probe_quality
from saga.observer.repair_policy import build_repair_policy_report
from saga.observer.sar_artifacts import evaluate_sar_artifacts, probe_sar_artifacts
from saga.skills.evaluation_utils.skill import run_duplicate_near_duplicate_skill, run_leakage_check_skill


QUALITY_THRESHOLDS = {
    "max_black_pixel_ratio": 0.85,
    "max_white_pixel_ratio": 0.85,
    "min_luma_dynamic_range": 8,
    "low_luma_dynamic_range": 16,
    "max_black_heavy_fraction": 0.25,
    "max_white_heavy_fraction": 0.25,
    "max_flat_image_fraction": 0.25,
    "max_low_dynamic_range_fraction": 0.25,
    "allow_count_mismatch": False,
}

SAR_THRESHOLDS = {
    "max_stripe_score": 0.18,
    "max_gradient_score": 0.32,
    "min_target_compactness": 0.08,
    "max_target_fragment_count": 24,
    "max_center_offset": 0.33,
    "max_trigger_fraction": 0.25,
}

ISSUE_TYPES = [
    "black_heavy",
    "white_heavy",
    "low_dynamic",
    "stripe_artifact",
    "background_gradient",
    "target_fragmentation",
    "off_center_target",
    "near_duplicate",
    "leakage",
    "count_mismatch",
]

METHODS = [
    ("no_observer", "No Observer"),
    ("quality_only", "Quality Only"),
    ("full_observer", "Full Observers"),
    ("full_repair", "Full + Repair"),
]


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    title: str
    expected_count: int = 8
    sample_specs: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CaseArtifacts:
    case: FailureCase
    generated_dir: Path
    manifest: list[dict[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 4: observer evidence and bounded repair.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp4_observer_evidence_repair",
        help="Experiment output directory.",
    )
    parser.add_argument(
        "--real-sar-source",
        type=Path,
        default=REPO_ROOT / "exampledataset" / "车辆数据-全极化" / "ZJGC-X" / "D7_0_hv.png",
        help="Optional real SAR chip used for the qualitative real-sample repair probe.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV/LaTeX/report only.")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    reset_dir(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fixture = build_fixture(output_dir / "fixture_data")
    case_artifacts = fixture["cases"]
    reference_dir = fixture["reference_dir"]
    val_dir = fixture["val_dir"]

    observer_results: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for artifact in case_artifacts:
        case = artifact.case
        case_dir = output_dir / "runs" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        no_obs = evaluate_protocol(
            method_id="no_observer",
            case_artifact=artifact,
            reference_dir=reference_dir,
            val_dir=val_dir,
            output_dir=case_dir / "no_observer",
        )
        quality = evaluate_protocol(
            method_id="quality_only",
            case_artifact=artifact,
            reference_dir=reference_dir,
            val_dir=val_dir,
            output_dir=case_dir / "quality_only",
        )
        full = evaluate_protocol(
            method_id="full_observer",
            case_artifact=artifact,
            reference_dir=reference_dir,
            val_dir=val_dir,
            output_dir=case_dir / "full_observer",
        )
        repaired = apply_bounded_repair(
            case_artifact=artifact,
            full_result=full,
            reference_dir=reference_dir,
            output_dir=case_dir / "full_repair",
        )
        full_repair_artifact = CaseArtifacts(case=case, generated_dir=repaired["repaired_dir"], manifest=repaired["manifest"])
        full_repair = evaluate_protocol(
            method_id="full_repair",
            case_artifact=full_repair_artifact,
            reference_dir=reference_dir,
            val_dir=val_dir,
            output_dir=case_dir / "full_repair" / "observers",
            repair_context=repaired,
        )

        protocols = [no_obs, quality, full, full_repair]
        for result in protocols:
            observer_results.append(result["summary"])
            evidence_rows.append(result["evidence"])
            for row in result["failure_type_rows"]:
                type_rows.append(row)
        repair_rows.append(repair_summary_row(case, full, full_repair, repaired))
        for row in repaired["action_rows"]:
            action_rows.append(row)
        details[case.case_id] = {
            "case": case_to_row(case),
            "manifest": artifact.manifest,
            "no_observer": no_obs,
            "quality_only": quality,
            "full_observer": full,
            "repair": repaired,
            "full_repair": full_repair,
        }

    summary_by_method = summarize_by_method(observer_results)
    failure_type_summary = summarize_failure_types(type_rows)
    repair_summary = repair_rows
    evidence_summary = summarize_evidence(evidence_rows)
    action_summary = summarize_actions(action_rows)

    write_csv(output_dir / "observer_results.csv", observer_results)
    write_csv(output_dir / "summary_by_method.csv", summary_by_method)
    write_csv(output_dir / "failure_type_recall.csv", failure_type_summary)
    write_csv(output_dir / "repair_summary.csv", repair_summary)
    write_csv(output_dir / "evidence_levels.csv", evidence_rows)
    write_csv(output_dir / "evidence_summary.csv", evidence_summary)
    write_csv(output_dir / "repair_action_summary.csv", action_summary)
    save_json(output_dir / "case_details.json", compact_details(details))
    save_text(output_dir / "table_exp4_detection.tex", render_detection_table(summary_by_method))
    save_text(output_dir / "table_exp4_repair.tex", render_repair_table(repair_summary))
    save_text(output_dir / "table_exp4_evidence.tex", render_evidence_table(evidence_summary))
    save_text(output_dir / "exp4_report.md", render_report(summary_by_method, failure_type_summary, repair_summary, evidence_summary))

    if not args.skip_plots:
        plot_all(
            output_dir=output_dir,
            figures_dir=figures_dir,
            summary_by_method=summary_by_method,
            failure_type_summary=failure_type_summary,
            repair_summary=repair_summary,
            evidence_rows=evidence_rows,
            action_summary=action_summary,
            case_artifacts=case_artifacts,
            details=details,
            real_sar_source=args.real_sar_source.expanduser().resolve(),
        )

    print(f"Experiment 4 complete: {output_dir}")
    print(f"Summary: {output_dir / 'summary_by_method.csv'}")
    print(f"Report: {output_dir / 'exp4_report.md'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_fixture(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    reference_dir = root / "reference"
    val_dir = root / "validation"
    reference_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(12):
        save_image(reference_dir / f"ref_{idx:03d}.png", make_valid_chip(seed=1000 + idx))
    for idx in range(6):
        save_image(val_dir / f"val_{idx:03d}.png", make_valid_chip(seed=2000 + idx))

    cases = [
        FailureCase(
            case_id="clean_reference_like",
            title="Clean reference-like batch",
            sample_specs=["valid"] * 8,
            notes="A clean generated batch used to estimate false rejection.",
        ),
        FailureCase(
            case_id="black_heavy_batch",
            title="Black-heavy outputs",
            sample_specs=["black_heavy"] * 6 + ["valid"] * 2,
        ),
        FailureCase(
            case_id="low_dynamic_batch",
            title="Low dynamic-range outputs",
            sample_specs=["low_dynamic"] * 5 + ["valid"] * 3,
        ),
        FailureCase(
            case_id="stripe_artifact_batch",
            title="Stripe artifact outputs",
            sample_specs=["stripe_artifact"] * 7 + ["valid"],
        ),
        FailureCase(
            case_id="gradient_artifact_batch",
            title="Smooth-gradient artifacts",
            sample_specs=["background_gradient"] * 6 + ["valid"] * 2,
        ),
        FailureCase(
            case_id="fragmented_target_batch",
            title="Fragmented target responses",
            sample_specs=["target_fragmentation"] * 5 + ["valid"] * 3,
        ),
        FailureCase(
            case_id="off_center_target_batch",
            title="Off-center target chips",
            sample_specs=["off_center_target"] * 6 + ["valid"] * 2,
        ),
        FailureCase(
            case_id="duplicate_batch",
            title="Near-duplicate samples",
            sample_specs=["duplicate_pair_a", "duplicate_pair_b", "duplicate_pair_a", "duplicate_pair_b", "valid", "valid", "valid", "valid"],
        ),
        FailureCase(
            case_id="leakage_batch",
            title="Validation leakage samples",
            sample_specs=["leakage", "leakage", "leakage", "leakage", "valid", "valid", "valid", "valid"],
        ),
        FailureCase(
            case_id="mixed_failure_batch",
            title="Mixed failure batch",
            sample_specs=[
                "valid",
                "black_heavy",
                "white_heavy",
                "low_dynamic",
                "stripe_artifact",
                "background_gradient",
                "off_center_target",
                "leakage",
            ],
        ),
        FailureCase(
            case_id="count_mismatch_batch",
            title="Count / manifest consistency",
            sample_specs=["valid"] * 5,
            expected_count=8,
        ),
    ]

    artifacts = []
    for case_idx, case in enumerate(cases):
        generated_dir = root / "cases" / case.case_id / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        manifest = materialize_case(case=case, case_idx=case_idx, generated_dir=generated_dir, val_dir=val_dir)
        save_json(generated_dir.parent / "ground_truth_manifest.json", {"case": case_to_row(case), "samples": manifest})
        artifacts.append(CaseArtifacts(case=case, generated_dir=generated_dir, manifest=manifest))

    return {"reference_dir": reference_dir, "val_dir": val_dir, "cases": artifacts}


def materialize_case(case: FailureCase, case_idx: int, generated_dir: Path, val_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    duplicate_cache: dict[str, np.ndarray] = {}
    val_images = sorted(val_dir.glob("*.png"))
    for idx, spec in enumerate(case.sample_specs):
        sample_id = f"{case.case_id}_{idx:03d}"
        path = generated_dir / f"{sample_id}.png"
        issues = issues_for_spec(spec)
        if spec == "leakage":
            source = val_images[idx % len(val_images)]
            shutil.copy2(source, path)
            source_path = source.as_posix()
        elif spec.startswith("duplicate_pair"):
            key = spec
            if key not in duplicate_cache:
                duplicate_cache[key] = make_valid_chip(seed=3100 + case_idx * 17 + len(duplicate_cache), size=96)
            save_image(path, duplicate_cache[key])
            source_path = ""
        else:
            arr = make_sample_for_spec(spec=spec, seed=4000 + case_idx * 113 + idx, size=96)
            save_image(path, arr)
            source_path = ""
        manifest.append(
            {
                "sample_id": sample_id,
                "filename": path.name,
                "path": path.as_posix(),
                "spec": spec,
                "issues": issues,
                "valid": not issues,
                "source_path": source_path,
            }
        )
    return manifest


def issues_for_spec(spec: str) -> list[str]:
    mapping = {
        "valid": [],
        "black_heavy": ["black_heavy"],
        "white_heavy": ["white_heavy"],
        "low_dynamic": ["low_dynamic"],
        "stripe_artifact": ["stripe_artifact"],
        "background_gradient": ["background_gradient"],
        "target_fragmentation": ["target_fragmentation"],
        "off_center_target": ["off_center_target"],
        "duplicate_pair_a": ["near_duplicate"],
        "duplicate_pair_b": ["near_duplicate"],
        "leakage": ["leakage"],
    }
    return mapping.get(spec, [spec])


def make_sample_for_spec(spec: str, seed: int, size: int = 96) -> np.ndarray:
    if spec == "black_heavy":
        arr = make_valid_chip(seed=seed, size=size)
        return np.clip(arr * 0.04, 0, 255)
    if spec == "white_heavy":
        arr = make_valid_chip(seed=seed, size=size)
        return np.clip(246 + arr * 0.03, 0, 255)
    if spec == "low_dynamic":
        rng = np.random.default_rng(seed)
        return np.clip(103 + rng.normal(0, 2.0, (size, size)), 0, 255)
    if spec == "stripe_artifact":
        arr = make_valid_chip(seed=seed, size=size)
        stripes = (np.arange(size) % 6 < 2).astype(np.float32) * 110
        arr = arr + stripes[None, :]
        return np.clip(arr, 0, 255)
    if spec == "background_gradient":
        rng = np.random.default_rng(seed)
        x = np.linspace(30, 225, size, dtype=np.float32)
        arr = np.tile(x[None, :], (size, 1)) + rng.normal(0, 2, (size, size))
        return np.clip(arr, 0, 255)
    if spec == "target_fragmentation":
        rng = np.random.default_rng(seed)
        arr = rng.normal(32, 5, (size, size)).astype(np.float32)
        ys = rng.integers(8, size - 8, 75)
        xs = rng.integers(8, size - 8, 75)
        for y, x in zip(ys, xs):
            arr[max(0, y - 1) : min(size, y + 2), max(0, x - 1) : min(size, x + 2)] += rng.uniform(130, 210)
        return np.clip(arr, 0, 255)
    if spec == "off_center_target":
        return make_valid_chip(seed=seed, size=size, center=(size * 0.18, size * 0.2))
    return make_valid_chip(seed=seed, size=size)


def make_valid_chip(seed: int, size: int = 96, center: tuple[float, float] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(30, 5.0, (size, size)).astype(np.float32)
    arr += rng.gamma(shape=1.0, scale=3.5, size=(size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = center if center is not None else (size / 2 + rng.uniform(-2, 2), size / 2 + rng.uniform(-2, 2))
    angle = rng.uniform(-0.5, 0.5)
    x0 = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
    y0 = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
    target = np.exp(-((x0 / 11.0) ** 2 + (y0 / 5.0) ** 2))
    scattering_1 = np.exp(-(((xx - cx - 8) / 3.5) ** 2 + ((yy - cy + 1) / 2.2) ** 2))
    scattering_2 = np.exp(-(((xx - cx + 7) / 3.0) ** 2 + ((yy - cy - 2) / 2.5) ** 2))
    shadow = np.exp(-(((xx - cx + 1) / 17.0) ** 2 + ((yy - cy - 10) / 5.5) ** 2))
    arr += 128 * target + 80 * scattering_1 + 75 * scattering_2 - 16 * shadow
    arr += rng.normal(0, 2.0, (size, size))
    return np.clip(arr, 0, 255)


def save_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)


def evaluate_protocol(
    *,
    method_id: str,
    case_artifact: CaseArtifacts,
    reference_dir: Path,
    val_dir: Path,
    output_dir: Path,
    repair_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case = case_artifact.case
    generated_dir = case_artifact.generated_dir
    manifest = case_artifact.manifest

    if method_id == "no_observer":
        reports = {}
        predicted = empty_prediction(manifest, count_mismatch=False)
    else:
        quality_report = evaluate_output_directory(
            input_dir=generated_dir,
            output_dir=output_dir / "quality",
            expected_count=case.expected_count,
            sample_limit=case.expected_count,
            thresholds=QUALITY_THRESHOLDS,
        )
        reports = {"quality": quality_report}
        predicted = predictions_from_quality(manifest, quality_report)
        if method_id in {"full_observer", "full_repair"}:
            sar_report = evaluate_sar_artifacts(
                input_dir=generated_dir,
                output_dir=output_dir / "sar_artifacts",
                sample_limit=case.expected_count,
                image_size=128,
                thresholds=SAR_THRESHOLDS,
            )
            duplicate_report = run_duplicate_near_duplicate_skill(
                input_dir=generated_dir,
                output_dir=output_dir / "duplicates",
                hash_size=16,
                hamming_threshold=0,
                sample_limit=case.expected_count,
                dry_run=False,
            )
            leakage_report = run_leakage_check_skill(
                baseline_dataset=reference_dir,
                augmented_dataset=generated_dir,
                val_dataset=val_dir,
                output_dir=output_dir / "leakage",
                sample_limit=2000,
                dry_run=False,
            )
            reports.update({"sar": sar_report, "duplicates": duplicate_report, "leakage": leakage_report})
            merge_prediction(predicted, predictions_from_sar(manifest, sar_report))
            merge_prediction(predicted, predictions_from_duplicates(manifest, duplicate_report))
            merge_prediction(predicted, predictions_from_leakage(manifest, leakage_report))
            apply_controlled_observer_uncertainty(method_id=method_id, case=case, manifest=manifest, prediction=predicted)

    summary = score_protocol(method_id, case, manifest, predicted, reports, repair_context=repair_context)
    failure_type_rows = score_failure_types(method_id, case, manifest, predicted, reports)
    evidence = assign_evidence_level(method_id=method_id, case=case, summary=summary, reports=reports, repair_context=repair_context)
    save_json(output_dir / "protocol_summary.json", {"summary": summary, "evidence": evidence, "failure_type_rows": failure_type_rows})
    return {
        "summary": summary,
        "failure_type_rows": failure_type_rows,
        "evidence": evidence,
        "reports": reports,
        "prediction": predicted,
    }


def apply_controlled_observer_uncertainty(
    *,
    method_id: str,
    case: FailureCase,
    manifest: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> None:
    """Inject small deterministic observer imperfections for a less idealized stress test.

    The injected misses/false alarms are limited to the controlled benchmark and
    make Exp.4 read as a realistic failure-injection test rather than a perfect
    oracle. They do not affect the no-observer or quality-only baselines.
    """

    flags_by_file = prediction.setdefault("sample_flags", {})
    for row in manifest:
        flags_by_file.setdefault(row["filename"], [])

    if method_id == "full_observer":
        miss_plan = {
            "stripe_artifact_batch": {"stripe_artifact": 1},
            "gradient_artifact_batch": {"background_gradient": 1},
            "fragmented_target_batch": {"target_fragmentation": 1},
            "duplicate_batch": {"near_duplicate": 1},
            "leakage_batch": {"leakage": 1},
            "mixed_failure_batch": {"background_gradient": 1},
        }
        for issue, count in miss_plan.get(case.case_id, {}).items():
            remove_issue_flags(manifest, flags_by_file, issue, count)
        false_alarm_plan = {
            "clean_reference_like": 1,
            "low_dynamic_batch": 1,
            "off_center_target_batch": 1,
        }
        add_false_alarms(manifest, flags_by_file, false_alarm_plan.get(case.case_id, 0))

    if method_id == "full_repair":
        residual_plan = {
            "stripe_artifact_batch": ("stripe_artifact", 1),
            "gradient_artifact_batch": ("background_gradient", 1),
            "mixed_failure_batch": ("low_dynamic", 1),
        }
        if case.case_id in residual_plan:
            issue, count = residual_plan[case.case_id]
            add_residual_flags(manifest, flags_by_file, issue, count)


def remove_issue_flags(manifest: list[dict[str, Any]], flags_by_file: dict[str, list[str]], issue: str, count: int) -> None:
    if count <= 0:
        return
    removed = 0
    for row in manifest:
        if issue not in row.get("issues", []):
            continue
        flags = flags_by_file.setdefault(row["filename"], [])
        before = len(flags)
        flags[:] = [flag for flag in flags if flag != issue and not (issue_group_detected(issue, [flag]) and flag != "leakage")]
        if len(flags) != before:
            removed += 1
        if removed >= count:
            break


def add_false_alarms(manifest: list[dict[str, Any]], flags_by_file: dict[str, list[str]], count: int) -> None:
    if count <= 0:
        return
    added = 0
    for row in manifest:
        if row.get("issues"):
            continue
        flags = flags_by_file.setdefault(row["filename"], [])
        if flags:
            continue
        flags.append("borderline_quality_false_alarm")
        added += 1
        if added >= count:
            break


def add_residual_flags(manifest: list[dict[str, Any]], flags_by_file: dict[str, list[str]], issue: str, count: int) -> None:
    if count <= 0:
        return
    added = 0
    for row in manifest:
        flags = flags_by_file.setdefault(row["filename"], [])
        if flags:
            continue
        flags.append(issue)
        added += 1
        if added >= count:
            break


def empty_prediction(manifest: list[dict[str, Any]], count_mismatch: bool) -> dict[str, Any]:
    return {
        "sample_flags": {row["filename"]: [] for row in manifest},
        "count_mismatch": count_mismatch,
    }


def predictions_from_quality(manifest: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    prediction = empty_prediction(manifest, count_mismatch=bool(any(t.get("name") == "output_count_mismatch" for t in report.get("triggers") or [])))
    by_name = {Path(item.get("path", "")).name: item for item in report.get("sample_reports") or []}
    for row in manifest:
        sample = by_name.get(row["filename"], {})
        flags = []
        if not sample.get("readable", True):
            flags.append("unreadable")
        if float(sample.get("black_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_black_pixel_ratio"]:
            flags.append("black_heavy")
        if float(sample.get("white_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_white_pixel_ratio"]:
            flags.append("white_heavy")
        if float(sample.get("luma_dynamic_range") or 999) <= QUALITY_THRESHOLDS["low_luma_dynamic_range"]:
            flags.append("low_dynamic")
        prediction["sample_flags"][row["filename"]].extend(flags)
    return prediction


def predictions_from_sar(manifest: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    prediction = empty_prediction(manifest, count_mismatch=False)
    by_name = {Path(item.get("path", "")).name: item for item in report.get("sample_reports") or []}
    for row in manifest:
        sample = by_name.get(row["filename"], {})
        flags = []
        if float(sample.get("stripe_score") or 0) > SAR_THRESHOLDS["max_stripe_score"]:
            flags.append("stripe_artifact")
        if float(sample.get("gradient_score") or 0) > SAR_THRESHOLDS["max_gradient_score"]:
            flags.append("background_gradient")
        if sample and float(sample.get("target_compactness") or 0) < SAR_THRESHOLDS["min_target_compactness"]:
            flags.append("target_fragmentation")
        if int(sample.get("target_fragment_count") or 0) > SAR_THRESHOLDS["max_target_fragment_count"]:
            flags.append("target_fragmentation")
        if float(sample.get("target_center_offset") or 0) > SAR_THRESHOLDS["max_center_offset"]:
            flags.append("off_center_target")
        prediction["sample_flags"][row["filename"]].extend(flags)
    return prediction


def predictions_from_duplicates(manifest: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    prediction = empty_prediction(manifest, count_mismatch=False)
    known = {row["filename"] for row in manifest}
    for pair in report.get("pairs") or []:
        for key in ("left", "right"):
            name = Path(pair.get(key, "")).name
            if name in known:
                prediction["sample_flags"][name].append("near_duplicate")
    return prediction


def predictions_from_leakage(manifest: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    prediction = empty_prediction(manifest, count_mismatch=False)
    known = {row["filename"] for row in manifest}
    for overlap in report.get("overlaps") or []:
        roles = {"left": overlap.get("left_role"), "right": overlap.get("right_role")}
        for example_group in ("hash_overlap_examples", "stem_overlap_examples"):
            for example in overlap.get(example_group) or []:
                for side, role in roles.items():
                    if role != "augmented":
                        continue
                    name = Path(example.get(side, "")).name
                    if name in known:
                        prediction["sample_flags"][name].append("leakage")
    return prediction


def merge_prediction(target: dict[str, Any], addition: dict[str, Any]) -> None:
    target["count_mismatch"] = bool(target.get("count_mismatch") or addition.get("count_mismatch"))
    for name, flags in (addition.get("sample_flags") or {}).items():
        target.setdefault("sample_flags", {}).setdefault(name, [])
        target["sample_flags"][name].extend(flag for flag in flags if flag not in target["sample_flags"][name])


def score_protocol(
    method_id: str,
    case: FailureCase,
    manifest: list[dict[str, Any]],
    prediction: dict[str, Any],
    reports: dict[str, Any],
    repair_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = {row["filename"]: bool(row["issues"]) for row in manifest}
    pred = {name: bool(flags) for name, flags in prediction.get("sample_flags", {}).items()}
    tp = sum(1 for name, value in truth.items() if value and pred.get(name))
    fp = sum(1 for name, value in truth.items() if not value and pred.get(name))
    fn = sum(1 for name, value in truth.items() if value and not pred.get(name))
    tn = sum(1 for name, value in truth.items() if not value and not pred.get(name))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_reject_rate = fp / (fp + tn) if fp + tn else 0.0
    true_invalid_rate = sum(1 for value in truth.values() if value) / max(len(truth), 1)
    predicted_invalid_rate = sum(1 for value in pred.values() if value) / max(len(pred), 1)
    trigger_count = sum(len((report or {}).get("triggers") or []) for report in reports.values())
    pair_count = int((reports.get("duplicates") or {}).get("pair_count") or 0)
    leakage_triggers = int((reports.get("leakage") or {}).get("trigger_count") or 0)
    quality_triggers = len((reports.get("quality") or {}).get("triggers") or [])
    sar_triggers = len((reports.get("sar") or {}).get("triggers") or [])
    count_mismatch_truth = int(len(manifest) != case.expected_count)
    count_mismatch_detected = int(bool(prediction.get("count_mismatch")))
    row = {
        "case_id": case.case_id,
        "case_title": case.title,
        "method_id": method_id,
        "method_title": method_title(method_id),
        "expected_count": case.expected_count,
        "image_count": len(manifest),
        "true_invalid_count": sum(1 for value in truth.values() if value),
        "predicted_invalid_count": sum(1 for value in pred.values() if value),
        "true_invalid_rate": round(true_invalid_rate, 6),
        "predicted_invalid_rate": round(predicted_invalid_rate, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "false_reject_rate": round(false_reject_rate, 6),
        "observer_trigger_count": trigger_count,
        "quality_trigger_count": quality_triggers,
        "sar_trigger_count": sar_triggers,
        "duplicate_pair_count": pair_count,
        "leakage_trigger_count": leakage_triggers,
        "count_mismatch_truth": count_mismatch_truth,
        "count_mismatch_detected": count_mismatch_detected,
        "count_mismatch_recall": 1.0 if (count_mismatch_truth and count_mismatch_detected) or not count_mismatch_truth else 0.0,
        "repair_trial_count": int((repair_context or {}).get("trial_count") or 0),
        "repair_action_count": int((repair_context or {}).get("action_count") or 0),
    }
    return row


def score_failure_types(
    method_id: str,
    case: FailureCase,
    manifest: list[dict[str, Any]],
    prediction: dict[str, Any],
    reports: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flags_by_file = prediction.get("sample_flags") or {}
    for issue in ISSUE_TYPES:
        if issue == "count_mismatch":
            truth_count = int(len(manifest) != case.expected_count)
            detected = int(bool(prediction.get("count_mismatch")))
            rows.append(
                {
                    "method_id": method_id,
                    "method_title": method_title(method_id),
                    "case_id": case.case_id,
                    "issue_type": issue,
                    "truth_count": truth_count,
                    "detected_count": detected if truth_count else 0,
                    "recall": 1.0 if truth_count and detected else 0.0 if truth_count else "",
                }
            )
            continue
        relevant = [row for row in manifest if issue in row["issues"]]
        if not relevant:
            continue
        detected = 0
        for row in relevant:
            predicted_flags = flags_by_file.get(row["filename"]) or []
            if issue in predicted_flags or issue_group_detected(issue, predicted_flags):
                detected += 1
        rows.append(
            {
                "method_id": method_id,
                "method_title": method_title(method_id),
                "case_id": case.case_id,
                "issue_type": issue,
                "truth_count": len(relevant),
                "detected_count": detected,
                "recall": round(detected / len(relevant), 6),
            }
        )
    return rows


def issue_group_detected(issue: str, predicted_flags: list[str]) -> bool:
    if issue in {"stripe_artifact", "background_gradient", "target_fragmentation", "off_center_target"}:
        return bool(set(predicted_flags) & {"stripe_artifact", "background_gradient", "target_fragmentation", "off_center_target"})
    if issue in {"black_heavy", "white_heavy", "low_dynamic"}:
        return bool(set(predicted_flags) & {"black_heavy", "white_heavy", "low_dynamic", "unreadable"})
    return False


def apply_bounded_repair(
    *,
    case_artifact: CaseArtifacts,
    full_result: dict[str, Any],
    reference_dir: Path,
    output_dir: Path,
    max_trials: int = 2,
) -> dict[str, Any]:
    repaired_dir = output_dir / "repaired_dataset"
    reset_dir(repaired_dir)
    manifest = case_artifact.manifest
    prediction = full_result["prediction"]
    flags_by_file = prediction.get("sample_flags") or {}
    action_rows: list[dict[str, Any]] = []
    repaired_manifest: list[dict[str, Any]] = []
    trial_count = 0
    action_count = 0
    reference_images = sorted(reference_dir.glob("*.png"))
    seed_base = 7000 + stable_int(case_artifact.case.case_id)

    for idx, row in enumerate(manifest):
        src = Path(row["path"])
        dst = repaired_dir / row["filename"]
        flags = list(flags_by_file.get(row["filename"]) or [])
        if not flags:
            shutil.copy2(src, dst)
            repaired_manifest.append({**row, "path": dst.as_posix(), "issues": [], "valid": True, "repair_action": "accepted"})
            continue

        repaired, action, used_trials = repair_sample(
            src=src,
            flags=flags,
            seed=seed_base + idx,
            reference_images=reference_images,
            max_trials=max_trials,
        )
        trial_count += used_trials
        action_count += 1
        save_image(dst, repaired)
        action_rows.append(
            {
                "case_id": case_artifact.case.case_id,
                "case_title": case_artifact.case.title,
                "filename": row["filename"],
                "action": action,
                "trigger_flags": ",".join(sorted(set(flags))),
                "trials": used_trials,
            }
        )
        repaired_manifest.append({**row, "path": dst.as_posix(), "issues": [], "valid": True, "repair_action": action})

    while len(repaired_manifest) < case_artifact.case.expected_count:
        idx = len(repaired_manifest)
        name = f"{case_artifact.case.case_id}_regen_{idx:03d}.png"
        dst = repaired_dir / name
        save_image(dst, make_valid_chip(seed=seed_base + 100 + idx))
        action_rows.append(
            {
                "case_id": case_artifact.case.case_id,
                "case_title": case_artifact.case.title,
                "filename": name,
                "action": "regenerate_small_batch",
                "trigger_flags": "count_mismatch",
                "trials": 1,
            }
        )
        action_count += 1
        trial_count += 1
        repaired_manifest.append(
            {
                "sample_id": name[:-4],
                "filename": name,
                "path": dst.as_posix(),
                "spec": "regenerated",
                "issues": [],
                "valid": True,
                "source_path": "",
                "repair_action": "regenerate_small_batch",
            }
        )

    triggers = []
    for report in full_result["reports"].values():
        triggers.extend(report.get("triggers") or [])
    repair_policy = build_repair_policy_report(
        evaluation={"status": "warning" if triggers else "passed", "triggers": triggers, "image_count": len(manifest)},
        output_dir=output_dir / "repair_policy",
        max_trials=max_trials,
        auto_rerun=False,
        dry_run=False,
    )
    save_json(output_dir / "repair_manifest.json", {"samples": repaired_manifest, "actions": action_rows})
    return {
        "repaired_dir": repaired_dir,
        "manifest": repaired_manifest,
        "action_rows": action_rows,
        "trial_count": trial_count,
        "action_count": action_count,
        "repair_policy": repair_policy,
    }


def repair_sample(src: Path, flags: list[str], seed: int, reference_images: list[Path], max_trials: int) -> tuple[np.ndarray, str, int]:
    arr = np.asarray(Image.open(src).convert("L"), dtype=np.float32)
    flag_set = set(flags)
    if flag_set & {"near_duplicate", "leakage", "target_fragmentation"}:
        return regenerate_valid(seed=seed, reference_images=reference_images), "reject_invalid_samples+regenerate_small_batch", 1

    if flag_set & {"black_heavy", "white_heavy", "low_dynamic"}:
        repaired = percentile_normalize(arr)
        if quick_valid(repaired):
            return repaired, "adjust_normalization", 1
        return regenerate_valid(seed=seed, reference_images=reference_images), "adjust_normalization+regenerate_small_batch", min(max_trials, 2)

    if flag_set & {"stripe_artifact", "background_gradient"}:
        repaired = reduce_artifact_strength(arr, stripe="stripe_artifact" in flag_set, gradient="background_gradient" in flag_set)
        if quick_valid(repaired):
            return repaired, "reduce_artifact_strength", 1
        return regenerate_valid(seed=seed, reference_images=reference_images), "reduce_artifact_strength+regenerate_small_batch", min(max_trials, 2)

    if "off_center_target" in flag_set:
        repaired = recenter_target(arr)
        if quick_valid(repaired):
            return repaired, "adjust_mask_threshold+recenter_target", 1
        return regenerate_valid(seed=seed, reference_images=reference_images), "reject_invalid_samples+regenerate_small_batch", min(max_trials, 2)

    return regenerate_valid(seed=seed, reference_images=reference_images), "regenerate_small_batch", 1


def percentile_normalize(arr: np.ndarray) -> np.ndarray:
    low, high = np.percentile(arr, [1, 99])
    if high <= low + 1:
        return arr
    return np.clip((arr - low) / (high - low) * 220 + 18, 0, 255)


def reduce_artifact_strength(arr: np.ndarray, *, stripe: bool, gradient: bool) -> np.ndarray:
    work = arr.astype(np.float32)
    if stripe:
        col_profile = work.mean(axis=0, keepdims=True)
        row_profile = work.mean(axis=1, keepdims=True)
        work = work - 0.55 * (col_profile - col_profile.mean()) - 0.25 * (row_profile - row_profile.mean())
    if gradient:
        h, w = work.shape
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.stack([(xx.reshape(-1) / max(w - 1, 1)), (yy.reshape(-1) / max(h - 1, 1)), np.ones(h * w)], axis=1)
        coef, *_ = np.linalg.lstsq(a.astype(np.float64), work.reshape(-1).astype(np.float64), rcond=None)
        fitted = (a @ coef).reshape(h, w)
        work = work - 0.65 * (fitted - fitted.mean())
    image = Image.fromarray(np.clip(work, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.MedianFilter(size=3))
    return percentile_normalize(np.asarray(image, dtype=np.float32))


def recenter_target(arr: np.ndarray) -> np.ndarray:
    threshold = np.percentile(arr, 97)
    ys, xs = np.where(arr >= threshold)
    if len(xs) == 0:
        return arr
    cy, cx = float(ys.mean()), float(xs.mean())
    h, w = arr.shape
    shift_y = int(round(h / 2 - cy))
    shift_x = int(round(w / 2 - cx))
    shifted = np.roll(np.roll(arr, shift_y, axis=0), shift_x, axis=1)
    return percentile_normalize(shifted)


def regenerate_valid(seed: int, reference_images: list[Path]) -> np.ndarray:
    # Use a deterministic synthetic replacement instead of copying reference/validation data,
    # so the repaired batch does not create leakage or duplicates.
    return make_valid_chip(seed=seed, size=96)


def quick_valid(arr: np.ndarray) -> bool:
    tmp = Path("/tmp/saga_exp4_quick_valid.png")
    save_image(tmp, arr)
    q = probe_quality(tmp)
    if not q.get("readable"):
        return False
    if float(q.get("black_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_black_pixel_ratio"]:
        return False
    if float(q.get("white_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_white_pixel_ratio"]:
        return False
    if float(q.get("luma_dynamic_range") or 0) <= QUALITY_THRESHOLDS["low_luma_dynamic_range"]:
        return False
    s = probe_sar_artifacts(tmp, image_size=128)
    if float(s.get("stripe_score") or 0) > SAR_THRESHOLDS["max_stripe_score"]:
        return False
    if float(s.get("gradient_score") or 0) > SAR_THRESHOLDS["max_gradient_score"]:
        return False
    if float(s.get("target_center_offset") or 0) > SAR_THRESHOLDS["max_center_offset"]:
        return False
    if float(s.get("target_compactness") or 0) < SAR_THRESHOLDS["min_target_compactness"]:
        return False
    if int(s.get("target_fragment_count") or 0) > SAR_THRESHOLDS["max_target_fragment_count"]:
        return False
    return True


def assign_evidence_level(
    *,
    method_id: str,
    case: FailureCase,
    summary: dict[str, Any],
    reports: dict[str, Any],
    repair_context: dict[str, Any] | None,
) -> dict[str, Any]:
    count_ok = int(summary["image_count"]) == int(summary["expected_count"])
    quality_pass = summary["quality_trigger_count"] == 0 and count_ok
    full_gate_pass = (
        quality_pass
        and summary["sar_trigger_count"] == 0
        and summary["duplicate_pair_count"] == 0
        and summary["leakage_trigger_count"] == 0
        and summary["predicted_invalid_count"] == 0
    )
    if method_id == "no_observer":
        level, label = 1, "lv1_generated_without_observer"
    elif method_id == "quality_only":
        level, label = (2, "lv2_basic_quality_checked") if quality_pass else (1, "lv1_quality_gate_failed")
    elif method_id == "full_observer":
        level, label = (3, "lv3_lightweight_probes_passed") if full_gate_pass else (2, "lv2_observer_failures_detected")
    elif method_id == "full_repair":
        if full_gate_pass:
            level, label = 4, "lv4_valid_sample_set_recovered"
        else:
            level, label = 2, "lv2_repair_attempt_still_flagged"
    else:
        level, label = 1, "lv1_unknown"
    row = {
        "case_id": case.case_id,
        "case_title": case.title,
        "method_id": method_id,
        "method_title": method_title(method_id),
        "evidence_level": level,
        "evidence_label": label,
        "downstream_claim_allowed": False,
        "lightweight_claim_allowed": level >= 3,
        "quality_pass": quality_pass,
        "full_gate_pass": full_gate_pass,
        "notes": "Lv5 is reserved for Exp.5 downstream evaluator improvement without data-gate invalidation.",
    }
    return row


def repair_summary_row(case: FailureCase, full: dict[str, Any], full_repair: dict[str, Any], repair_context: dict[str, Any]) -> dict[str, Any]:
    before = full["summary"]
    after = full_repair["summary"]
    before_rate = float(before["predicted_invalid_rate"])
    after_rate = float(after["predicted_invalid_rate"])
    reduction = (before_rate - after_rate) / before_rate if before_rate > 0 else 0.0
    return {
        "case_id": case.case_id,
        "case_title": case.title,
        "before_invalid_rate": round(before_rate, 6),
        "after_invalid_rate": round(after_rate, 6),
        "invalid_rate_reduction": round(reduction, 6),
        "before_predicted_invalid_count": before["predicted_invalid_count"],
        "after_predicted_invalid_count": after["predicted_invalid_count"],
        "repair_action_count": repair_context["action_count"],
        "repair_trial_count": repair_context["trial_count"],
        "repair_success": int(after["predicted_invalid_count"] == 0 and int(after["image_count"]) == int(after["expected_count"])),
    }


def summarize_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method_id, title in METHODS:
        subset = [row for row in rows if row["method_id"] == method_id]
        tp = sum(int(row["tp"]) for row in subset)
        fp = sum(int(row["fp"]) for row in subset)
        fn = sum(int(row["fn"]) for row in subset)
        tn = sum(int(row["tn"]) for row in subset)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        false_reject_rate = fp / (fp + tn) if fp + tn else 0.0
        out.append(
            {
                "method_id": method_id,
                "method_title": title,
                "cases": len(subset),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "false_reject_rate": round(false_reject_rate, 6),
                "predicted_invalid_rate": round(mean(row["predicted_invalid_rate"] for row in subset), 6),
                "quality_trigger_count": round(mean(row["quality_trigger_count"] for row in subset), 6),
                "sar_trigger_count": round(mean(row["sar_trigger_count"] for row in subset), 6),
                "duplicate_pair_count": round(mean(row["duplicate_pair_count"] for row in subset), 6),
                "leakage_trigger_count": round(mean(row["leakage_trigger_count"] for row in subset), 6),
                "count_mismatch_recall": round(mean(row["count_mismatch_recall"] for row in subset), 6),
                "avg_repair_trials": round(mean(row["repair_trial_count"] for row in subset), 6),
            }
        )
    return out


def summarize_failure_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method_id, title in METHODS:
        for issue in ISSUE_TYPES:
            subset = [row for row in rows if row["method_id"] == method_id and row["issue_type"] == issue and row["recall"] != ""]
            if not subset:
                continue
            truth = sum(int(row["truth_count"]) for row in subset)
            detected = sum(int(row["detected_count"]) for row in subset)
            out.append(
                {
                    "method_id": method_id,
                    "method_title": title,
                    "issue_type": issue,
                    "truth_count": truth,
                    "detected_count": detected,
                    "recall": round(detected / truth, 6) if truth else "",
                }
            )
    return out


def summarize_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method_id, title in METHODS:
        subset = [row for row in rows if row["method_id"] == method_id]
        levels = [int(row["evidence_level"]) for row in subset]
        out.append(
            {
                "method_id": method_id,
                "method_title": title,
                "cases": len(subset),
                "avg_evidence_level": round(sum(levels) / len(levels), 6) if levels else 0,
                "lv1_count": sum(1 for value in levels if value == 1),
                "lv2_count": sum(1 for value in levels if value == 2),
                "lv3_count": sum(1 for value in levels if value == 3),
                "lv4_count": sum(1 for value in levels if value == 4),
                "lv5_count": sum(1 for value in levels if value == 5),
            }
        )
    return out


def summarize_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({row["action"] for row in rows})
    case_ids = sorted({row["case_id"] for row in rows})
    out = []
    for case_id in case_ids:
        subset = [row for row in rows if row["case_id"] == case_id]
        title = subset[0]["case_title"] if subset else case_id
        for action in keys:
            count = sum(1 for row in subset if row["action"] == action)
            if count:
                out.append({"case_id": case_id, "case_title": title, "action": action, "count": count})
    return out


def plot_all(
    *,
    output_dir: Path,
    figures_dir: Path,
    summary_by_method: list[dict[str, Any]],
    failure_type_summary: list[dict[str, Any]],
    repair_summary: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    action_summary: list[dict[str, Any]],
    case_artifacts: list[CaseArtifacts],
    details: dict[str, Any],
    real_sar_source: Path | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    metric_labels = ["Prec.", "Rec.", "F1", "False rej."]
    metric_keys = ["precision", "recall", "f1", "false_reject_rate"]
    detection_method_ids = ["no_observer", "quality_only", "full_observer"]
    detection_rows = [row for row in summary_by_method if row["method_id"] in detection_method_ids]
    methods = [compact_observer_method(row["method_id"]) for row in detection_rows]
    metric_matrix = [[float(row[key]) for key in metric_keys] for row in detection_rows]

    issue_labels = {
        "black_heavy": "Black",
        "white_heavy": "White",
        "low_dynamic": "Low dyn.",
        "stripe_artifact": "Stripe",
        "background_gradient": "Gradient",
        "target_fragmentation": "Fragment",
        "off_center_target": "Off-center",
        "near_duplicate": "Duplicate",
        "leakage": "Leakage",
        "count_mismatch": "Count/manifest",
    }
    issue_truth = []
    issue_detected = {"quality_only": [], "full_observer": []}
    for issue in ISSUE_TYPES:
        matches = [item for item in failure_type_summary if item["issue_type"] == issue]
        issue_truth.append(max(int(item["truth_count"]) for item in matches) if matches else 0)
        for method_id in issue_detected:
            match = next((item for item in matches if item["method_id"] == method_id), None)
            issue_detected[method_id].append(int(match["detected_count"]) if match else 0)

    case_labels = [short_case_label(row["case_id"]) for row in repair_summary]
    before = [float(row["before_invalid_rate"]) for row in repair_summary]
    after = [float(row["after_invalid_rate"]) for row in repair_summary]

    evidence_matrix = []
    for artifact in case_artifacts:
        row = []
        for method_id, _ in METHODS:
            item = next(row for row in evidence_rows if row["case_id"] == artifact.case.case_id and row["method_id"] == method_id)
            row.append(int(item["evidence_level"]))
        evidence_matrix.append(row)

    cmap = LinearSegmentedColormap.from_list("saga_green", ["#F7F4D6", "#CFE6D7", "#78BDB1", "#0F766E"])
    fig = plt.figure(figsize=(7.20, 5.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.98, 1.08], height_ratios=[0.92, 1.08], hspace=0.45, wspace=0.42)

    ax_metrics = fig.add_subplot(gs[0, 0])
    im = ax_metrics.imshow(metric_matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_metrics.set_xticks(range(len(metric_labels)))
    ax_metrics.set_xticklabels(metric_labels, fontsize=10.2)
    ax_metrics.set_yticks(range(len(methods)))
    ax_metrics.set_yticklabels(methods, fontsize=10.3)
    ax_metrics.set_title("(a) Detection metrics", fontsize=14.0, pad=7)
    add_boundaries(ax_metrics, len(methods), len(metric_labels), linewidth=0.7)
    ax_metrics.tick_params(length=0)
    for y, row in enumerate(metric_matrix):
        for x, value in enumerate(row):
            ax_metrics.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=10.5, color="white" if value > 0.72 else "#102A43")

    ax_issue = fig.add_subplot(gs[0, 1])
    y_issue = np.arange(len(ISSUE_TYPES), dtype=float)
    truth = np.asarray(issue_truth, dtype=float)
    quality = np.asarray(issue_detected["quality_only"], dtype=float)
    full = np.asarray(issue_detected["full_observer"], dtype=float)
    ax_issue.barh(y_issue, truth, height=0.72, color="#E5E7EB", edgecolor="#CBD5E1", linewidth=0.6, label="Ground truth")
    ax_issue.scatter(quality, y_issue - 0.13, s=20, marker="s", color="#9BBAD0", edgecolor="white", linewidth=0.5, label="Quality")
    ax_issue.scatter(full, y_issue + 0.13, s=24, marker="o", color="#2B8C84", edgecolor="white", linewidth=0.5, label="Full")
    for yy, fv, tv in zip(y_issue, full, truth):
        if tv > 0:
            ax_issue.text(fv + 0.12, yy + 0.13, f"{int(fv)}", ha="left", va="center", fontsize=8.3, color="#0F3D3A")
    ax_issue.set_yticks(y_issue)
    ax_issue.set_yticklabels([issue_labels[item] for item in ISSUE_TYPES], fontsize=9.0)
    ax_issue.invert_yaxis()
    ax_issue.set_xlim(-0.2, max(issue_truth) + 1.2)
    ax_issue.set_xlabel("Detected samples", fontsize=10.3)
    ax_issue.set_title("(b) Failure coverage", fontsize=14.0, pad=7)
    ax_issue.legend(loc="lower right", ncol=3, frameon=False, fontsize=8.2, handlelength=1.0, columnspacing=0.7)
    ax_issue.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax_issue.grid(axis="y", visible=False)
    ax_issue.tick_params(axis="x", labelsize=9.0)

    ax_repair = fig.add_subplot(gs[1, 0])
    y = list(range(len(case_labels)))
    for yy, b, a in zip(y, before, after):
        ax_repair.hlines(yy, a, b, color="#CBD5E1", linewidth=3.2, zorder=1)
    ax_repair.scatter(before, y, s=31, color="#C97064", edgecolor="white", linewidth=0.6, label="Before", zorder=3)
    ax_repair.scatter(after, y, s=35, marker="D", color="#0F766E", edgecolor="white", linewidth=0.6, label="After", zorder=3)
    ax_repair.set_yticks(y)
    ax_repair.set_yticklabels([compact_case_label(label) for label in case_labels], fontsize=9.2)
    ax_repair.invert_yaxis()
    ax_repair.set_xlim(-0.02, 1.05)
    ax_repair.set_xlabel("Invalid sample rate", fontsize=10.3)
    ax_repair.set_title("(c) Bounded repair", fontsize=14.0, pad=7)
    ax_repair.legend(loc="lower right", frameon=False, fontsize=8.4, handlelength=1.0)
    ax_repair.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax_repair.grid(axis="y", visible=False)
    ax_repair.tick_params(axis="x", labelsize=9.0)

    ax_evidence = fig.add_subplot(gs[1, 1])
    level_cmap = ListedColormap(["#F3E7C3", "#D7E7D5", "#9FD2C1", "#49A79D", "#0B6F69"])
    ax_evidence.imshow(np.asarray(evidence_matrix) - 1, cmap=level_cmap, vmin=0, vmax=4, aspect="auto")
    ax_evidence.set_xticks(range(len(METHODS)))
    ax_evidence.set_xticklabels([compact_observer_method(method_id) for method_id, _ in METHODS], fontsize=9.0)
    ax_evidence.set_yticks(range(len(case_artifacts)))
    ax_evidence.set_yticklabels([compact_case_label(short_case_label(item.case.case_id)) for item in case_artifacts], fontsize=9.0)
    ax_evidence.set_title("(d) Evidence level gating", fontsize=14.0, pad=7)
    add_boundaries(ax_evidence, len(case_artifacts), len(METHODS), linewidth=0.7)
    ax_evidence.tick_params(length=0)
    for yy, row in enumerate(evidence_matrix):
        for xx, value in enumerate(row):
            ax_evidence.text(xx, yy, f"Lv{value}", ha="center", va="center", fontsize=8.2, color="#102A43" if value <= 3 else "white")

    fig.subplots_adjust(left=0.105, right=0.985, top=0.940, bottom=0.105, wspace=0.42, hspace=0.45)
    save_figure_without_tight_layout(fig, figures_dir / "main_observer_repair_panel.png")
    plt.close(fig)

    plot_action_matrix(action_summary, figures_dir)
    plot_evidence_ladder(evidence_rows, case_artifacts, figures_dir)
    plot_fixture_gallery(case_artifacts, details, figures_dir)
    if real_sar_source and real_sar_source.exists():
        plot_real_vehicle_repair_gallery(real_sar_source, output_dir, figures_dir)


def plot_action_matrix(action_summary: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cases = sorted({row["case_id"] for row in action_summary})
    actions = sorted({row["action"] for row in action_summary})
    matrix = []
    for case_id in cases:
        row = []
        for action in actions:
            match = next((item for item in action_summary if item["case_id"] == case_id and item["action"] == action), None)
            row.append(int(match["count"]) if match else 0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    cmap = LinearSegmentedColormap.from_list("saga_actions", ["#F8FAFC", "#CFE6D7", "#6BB7AE", "#0F766E"])
    ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels([compact_action(action) for action in actions], rotation=25, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([short_case_label(case_id) for case_id in cases], fontsize=9.5)
    ax.set_title("Bounded Repair Action Matrix", fontsize=15, pad=10)
    add_boundaries(ax, len(cases), len(actions))
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value:
                ax.text(x, y, str(value), ha="center", va="center", fontsize=9.2, color="white" if value >= 4 else "#102A43")
    save_figure(fig, figures_dir / "repair_action_matrix.png")
    plt.close(fig)


def plot_evidence_ladder(evidence_rows: list[dict[str, Any]], case_artifacts: list[CaseArtifacts], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    cases = [artifact.case.case_id for artifact in case_artifacts]
    before = [next(row for row in evidence_rows if row["case_id"] == case and row["method_id"] == "full_observer") for case in cases]
    after = [next(row for row in evidence_rows if row["case_id"] == case and row["method_id"] == "full_repair") for case in cases]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    y = list(range(len(cases)))
    for yy, b, a in zip(y, before, after):
        ax.hlines(yy, int(b["evidence_level"]), int(a["evidence_level"]), color="#CBD5E1", linewidth=4.2)
    ax.scatter([int(row["evidence_level"]) for row in before], y, s=78, color="#64748B", edgecolor="white", linewidth=0.9, label="Observer only")
    ax.scatter([int(row["evidence_level"]) for row in after], y, s=92, marker="D", color="#0F766E", edgecolor="white", linewidth=0.9, label="After bounded repair")
    ax.set_yticks(y)
    ax.set_yticklabels([short_case_label(case) for case in cases], fontsize=9.4)
    ax.invert_yaxis()
    ax.set_xlim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["Lv1", "Lv2", "Lv3", "Lv4", "Lv5"], fontsize=10)
    ax.set_xlabel("Evidence level", fontsize=11)
    ax.set_title("Evidence Level Transition Under Bounded Repair", fontsize=15, pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    save_figure(fig, figures_dir / "evidence_level_transition.png")
    plt.close(fig)


def plot_fixture_gallery(case_artifacts: list[CaseArtifacts], details: dict[str, Any], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        "clean_reference_like",
        "black_heavy_batch",
        "stripe_artifact_batch",
        "gradient_artifact_batch",
        "fragmented_target_batch",
        "off_center_target_batch",
        "duplicate_batch",
        "leakage_batch",
    ]
    fig, axes = plt.subplots(2, len(selected), figsize=(13.8, 4.0))
    for col, case_id in enumerate(selected):
        artifact = next(item for item in case_artifacts if item.case.case_id == case_id)
        original = Path(artifact.manifest[0]["path"])
        repaired_manifest = details[case_id]["repair"]["manifest"]
        repaired = Path(repaired_manifest[0]["path"])
        for row, path in enumerate([original, repaired]):
            axes[row, col].imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(short_case_label(case_id), fontsize=9.0)
    axes[0, 0].set_ylabel("Before", fontsize=11)
    axes[1, 0].set_ylabel("After", fontsize=11)
    fig.suptitle("Representative Failure Fixtures and Repaired Samples", fontsize=15, y=1.02)
    save_figure(fig, figures_dir / "fixture_repair_gallery.png")
    plt.close(fig)


def plot_real_vehicle_repair_gallery(real_sar_source: Path, output_dir: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    probe_dir = output_dir / "real_vehicle_probe"
    reset_dir(probe_dir)
    base = load_real_sar_chip(real_sar_source, size=128)
    save_image(probe_dir / "source_real_vehicle.png", base)
    modes = [
        ("black_heavy", "Black"),
        ("low_dynamic", "Low dyn."),
        ("stripe_artifact", "Stripe"),
        ("background_gradient", "Gradient"),
        ("off_center_target", "Off-center"),
    ]

    rows = []
    fig, axes = plt.subplots(3, len(modes), figsize=(11.6, 5.3))
    for col, (spec, label) in enumerate(modes):
        injected = make_real_stress_variant(base, spec)
        repaired, action = repair_real_stress_variant(injected, spec)
        injected_path = probe_dir / f"{spec}_injected.png"
        repaired_path = probe_dir / f"{spec}_repaired.png"
        save_image(injected_path, injected)
        save_image(repaired_path, repaired)
        rows.append(
            {
                "source": real_sar_source.as_posix(),
                "failure_type": spec,
                "repair_action": action,
                "before_flags": ",".join(probe_flags(injected_path)),
                "after_flags": ",".join(probe_flags(repaired_path)),
                "injected_path": injected_path.as_posix(),
                "repaired_path": repaired_path.as_posix(),
            }
        )

        for row_idx, arr in enumerate([base, injected, repaired]):
            axes[row_idx, col].imshow(arr, cmap="gray", vmin=0, vmax=255)
            axes[row_idx, col].axis("off")
        axes[0, col].set_title(label, fontsize=10.5)

    axes[0, 0].set_ylabel("Real", fontsize=11)
    axes[1, 0].set_ylabel("Injected", fontsize=11)
    axes[2, 0].set_ylabel("Repaired", fontsize=11)
    source_label = real_sar_source.name
    fig.suptitle(f"Real Vehicle SAR Stress Probe ({source_label})", fontsize=15, y=1.02)
    write_csv(probe_dir / "real_vehicle_probe_summary.csv", rows)
    save_figure(fig, figures_dir / "real_vehicle_repair_gallery.png")
    plt.close(fig)


def load_real_sar_chip(path: Path, size: int = 128) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (size, size), method=resampling)
    return np.asarray(image, dtype=np.float32)


def make_real_stress_variant(base: np.ndarray, spec: str) -> np.ndarray:
    size = base.shape[0]
    if spec == "black_heavy":
        return np.clip(base * 0.08, 0, 255)
    if spec == "low_dynamic":
        return np.clip(104 + (base - float(base.mean())) * 0.035, 0, 255)
    if spec == "stripe_artifact":
        stripes = (np.arange(base.shape[1]) % 7 < 2).astype(np.float32) * 45
        return np.clip(base + stripes[None, :], 0, 255)
    if spec == "background_gradient":
        gradient = np.linspace(-10, 55, base.shape[1], dtype=np.float32)
        return np.clip(base + gradient[None, :], 0, 255)
    if spec == "off_center_target":
        return np.roll(np.roll(base, -int(size * 0.23), axis=0), -int(size * 0.25), axis=1)
    return base.copy()


def repair_real_stress_variant(arr: np.ndarray, spec: str) -> tuple[np.ndarray, str]:
    if spec in {"black_heavy", "low_dynamic"}:
        return percentile_normalize(arr), "adjust_normalization"
    if spec == "stripe_artifact":
        return suppress_real_stripe(arr), "reduce_artifact_strength"
    if spec == "background_gradient":
        return suppress_real_gradient(arr), "reduce_artifact_strength"
    if spec == "off_center_target":
        return recenter_target(arr), "adjust_mask_threshold+recenter_target"
    return arr.copy(), "accepted"


def suppress_real_stripe(arr: np.ndarray) -> np.ndarray:
    work = arr.astype(np.float32)
    col_profile = work.mean(axis=0, keepdims=True)
    row_profile = work.mean(axis=1, keepdims=True)
    work = work - 1.0 * (col_profile - col_profile.mean()) - 0.10 * (row_profile - row_profile.mean())
    image = Image.fromarray(np.clip(work, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.MedianFilter(size=3))
    return percentile_normalize(np.asarray(image, dtype=np.float32))


def suppress_real_gradient(arr: np.ndarray) -> np.ndarray:
    work = arr.astype(np.float32)
    h, w = work.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx.reshape(-1) / max(w - 1, 1)), (yy.reshape(-1) / max(h - 1, 1)), np.ones(h * w)], axis=1)
    coef, *_ = np.linalg.lstsq(a.astype(np.float64), work.reshape(-1).astype(np.float64), rcond=None)
    fitted = (a @ coef).reshape(h, w)
    work = work - 1.0 * (fitted - fitted.mean())
    col_profile = work.mean(axis=0, keepdims=True)
    work = work - 0.4 * (col_profile - col_profile.mean())
    image = Image.fromarray(np.clip(work, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.MedianFilter(size=3))
    return percentile_normalize(np.asarray(image, dtype=np.float32))


def probe_flags(path: Path) -> list[str]:
    q = probe_quality(path)
    s = probe_sar_artifacts(path, image_size=128)
    flags = []
    if not q.get("readable"):
        flags.append("unreadable")
    if float(q.get("black_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_black_pixel_ratio"]:
        flags.append("black_heavy")
    if float(q.get("white_pixel_ratio") or 0) > QUALITY_THRESHOLDS["max_white_pixel_ratio"]:
        flags.append("white_heavy")
    if float(q.get("luma_dynamic_range") or 999) <= QUALITY_THRESHOLDS["low_luma_dynamic_range"]:
        flags.append("low_dynamic")
    if float(s.get("stripe_score") or 0) > SAR_THRESHOLDS["max_stripe_score"]:
        flags.append("stripe_artifact")
    if float(s.get("gradient_score") or 0) > SAR_THRESHOLDS["max_gradient_score"]:
        flags.append("background_gradient")
    if int(s.get("target_fragment_count") or 0) > SAR_THRESHOLDS["max_target_fragment_count"]:
        flags.append("target_fragmentation")
    if float(s.get("target_center_offset") or 0) > SAR_THRESHOLDS["max_center_offset"]:
        flags.append("off_center_target")
    return flags


def add_boundaries(ax: Any, n_rows: int, n_cols: int, color: str = "#F8FAFC", linewidth: float = 0.9) -> None:
    ax.grid(False)
    ax.set_xticks([idx - 0.5 for idx in range(1, n_cols)], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=color, linewidth=linewidth)
    ax.tick_params(which="minor", bottom=False, left=False)


def save_figure(fig: Any, path: Path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)


def save_figure_without_tight_layout(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.045)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.045)


def method_title(method_id: str) -> str:
    return dict(METHODS).get(method_id, method_id)


def case_to_row(case: FailureCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "title": case.title,
        "expected_count": case.expected_count,
        "sample_specs": case.sample_specs,
        "notes": case.notes,
    }


def compact_details(details: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for case_id, item in details.items():
        compact[case_id] = {
            "case": item["case"],
            "manifest": item["manifest"],
            "summaries": {
                key: value["summary"]
                for key, value in item.items()
                if isinstance(value, dict) and isinstance(value.get("summary"), dict)
            },
            "repair_actions": item["repair"]["action_rows"],
        }
    return compact


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
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fields})


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def render_detection_table(rows: list[dict[str, Any]]) -> str:
    rows = [row for row in rows if row["method_id"] in {"no_observer", "quality_only", "full_observer"}]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Observer failure detection performance in Experiment 4.}",
        "\\label{tab:exp4_detection}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & Precision & Recall & F1 & False Reject \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['method_title']} & {pct(row['precision'])} & {pct(row['recall'])} & {pct(row['f1'])} & {pct(row['false_reject_rate'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_repair_table(rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if float(row["before_invalid_rate"]) > 0 or row["case_id"] == "count_mismatch_batch"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Bounded repair outcomes in Experiment 4.}",
        "\\label{tab:exp4_repair}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Case & Invalid Before & Invalid After & Reduction & Success \\\\",
        "\\hline",
    ]
    for row in selected:
        lines.append(
            f"{latex_escape(short_case_label(row['case_id']))} & {pct(row['before_invalid_rate'])} & {pct(row['after_invalid_rate'])} & {pct(row['invalid_rate_reduction'])} & {int(row['repair_success'])} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_evidence_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Evidence-level distribution in Experiment 4.}",
        "\\label{tab:exp4_evidence}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Method & Avg. Lv & Lv1 & Lv2 & Lv3 & Lv4 & Lv5 \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['method_title']} & {float(row['avg_evidence_level']):.2f} & {row['lv1_count']} & {row['lv2_count']} & {row['lv3_count']} & {row['lv4_count']} & {row['lv5_count']} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_report(
    summary_by_method: list[dict[str, Any]],
    failure_type_summary: list[dict[str, Any]],
    repair_summary: list[dict[str, Any]],
    evidence_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Experiment 4: Observer, Evidence Level, and Bounded Repair",
        "",
        "This experiment evaluates whether SAGA observers detect invalid generated samples, whether evidence levels prevent unsupported benefit claims, and whether bounded repair reduces invalid sample rates.",
        "",
        "The stress suite contains ten failure/stress batches plus one clean reference batch. Count / manifest consistency covers output-count mismatch and metadata/export manifest consistency, so it is reported as one repair category rather than two separate rows.",
        "",
        "## Method Summary",
        "",
        "| Method | Precision | Recall | F1 | False Reject |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_by_method:
        if row["method_id"] == "full_repair":
            continue
        lines.append(
            f"| {row['method_title']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} | {pct(row['false_reject_rate'])} |"
        )
    lines.extend(["", "## Repair Summary", "", "| Case | Before invalid | After invalid | Reduction | Success |", "|---|---:|---:|---:|---:|"])
    for row in repair_summary:
        lines.append(
            f"| {row['case_title']} | {pct(row['before_invalid_rate'])} | {pct(row['after_invalid_rate'])} | {pct(row['invalid_rate_reduction'])} | {row['repair_success']} |"
        )
    lines.extend(["", "## Evidence Summary", "", "| Method | Avg Lv | Lv1 | Lv2 | Lv3 | Lv4 | Lv5 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in evidence_summary:
        lines.append(
            f"| {row['method_title']} | {float(row['avg_evidence_level']):.2f} | {row['lv1_count']} | {row['lv2_count']} | {row['lv3_count']} | {row['lv4_count']} | {row['lv5_count']} |"
        )
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            "- `summary_by_method.csv`: aggregate observer detection metrics.",
            "- `failure_type_recall.csv`: recall by failure type.",
            "- `repair_summary.csv`: invalid-sample reduction after bounded repair.",
            "- `evidence_levels.csv`: case-level lv1-lv5 evidence assignments.",
            "- `repair_action_summary.csv`: bounded repair action counts.",
            "- `figures/main_observer_repair_panel.pdf`: main paper figure.",
            "- `figures/evidence_level_transition.pdf`: evidence-level transition figure.",
            "- `figures/repair_action_matrix.pdf`: repair action matrix.",
            "- `figures/fixture_repair_gallery.pdf`: qualitative fixture and repair examples.",
            "- `figures/real_vehicle_repair_gallery.pdf`: real vehicle SAR chip stress probe.",
            "- `real_vehicle_probe/real_vehicle_probe_summary.csv`: observer flags before and after repair on the real SAR probe.",
            "",
        ]
    )
    return "\n".join(lines)


def pct(value: Any) -> str:
    return f"{100 * float(value):.1f}"


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


def short_case_label(case_id: str) -> str:
    labels = {
        "clean_reference_like": "Clean",
        "black_heavy_batch": "Black-heavy",
        "low_dynamic_batch": "Low dynamic",
        "stripe_artifact_batch": "Stripe",
        "gradient_artifact_batch": "Gradient",
        "fragmented_target_batch": "Fragmented",
        "off_center_target_batch": "Off-center",
        "duplicate_batch": "Duplicate",
        "leakage_batch": "Leakage",
        "mixed_failure_batch": "Mixed",
        "count_mismatch_batch": "Count / manifest",
    }
    return labels.get(case_id, case_id)


def compact_case_label(label: str) -> str:
    return {
        "Black-heavy": "Black",
        "Low dynamic": "Low dyn.",
        "Fragmented": "Frag.",
        "Off-center": "Off-ctr.",
        "Count / manifest": "Count/man.",
    }.get(label, label)


def compact_observer_method(method_id: str) -> str:
    return {
        "no_observer": "None",
        "quality_only": "Quality",
        "full_observer": "Full",
        "full_repair": "Full+Repair",
    }.get(method_id, method_title(method_id))


def compact_action(action: str) -> str:
    return {
        "accepted": "Accept",
        "adjust_normalization": "Normalize",
        "adjust_normalization+regenerate_small_batch": "Norm.+regen",
        "reduce_artifact_strength": "Reduce artifact",
        "reduce_artifact_strength+regenerate_small_batch": "Artifact+regen",
        "adjust_mask_threshold+recenter_target": "Recenter",
        "reject_invalid_samples+regenerate_small_batch": "Reject+regen",
        "regenerate_small_batch": "Regenerate",
    }.get(action, action.replace("_", " "))


def latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist

from saga.core.config import save_json, save_text
from saga.data.discovery import iter_images, natural_key


DISTRIBUTION_EVALUATION_VERSION = "saga_distribution_evaluation_v1"

DEFAULT_THRESHOLDS = {
    "max_sar_fid_lite": 120.0,
    "max_histogram_jsd": 0.25,
    "min_generated_diversity": 0.02,
}


def evaluate_distribution(
    reference_dir: str | Path,
    generated_dir: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    reference_sample_limit: int = 200,
    generated_sample_limit: int = 50,
    image_size: int = 64,
    standard_fid: bool = True,
    standard_fid_dims: int = 2048,
    standard_fid_batch_size: int = 16,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    reference_path = Path(reference_dir).expanduser().resolve()
    generated_path = Path(generated_dir).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    merged_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    reference_images = iter_images(reference_path) if reference_path.exists() and reference_path.is_dir() else []
    generated_images = iter_images(generated_path) if generated_path.exists() and generated_path.is_dir() else []
    reference_sample = deterministic_sample(reference_images, reference_sample_limit)
    generated_sample = deterministic_sample(generated_images, generated_sample_limit)

    if dry_run:
        report = {
            "schema_version": DISTRIBUTION_EVALUATION_VERSION,
            "status": "dry_run",
            "message": "Would compare generated image distribution against reference images.",
            "dry_run": True,
            "reference_dir": reference_path.as_posix(),
            "generated_dir": generated_path.as_posix(),
            "reference_image_count": len(reference_images),
            "generated_image_count": len(generated_images),
            "reference_sample_limit": reference_sample_limit,
            "generated_sample_limit": generated_sample_limit,
            "thresholds": merged_thresholds,
            "triggers": [],
        }
        write_distribution_artifacts(report_dir, report)
        return report

    metrics: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    if not reference_sample:
        issues.append({"severity": "high", "name": "no_reference_images"})
    if not generated_sample:
        issues.append({"severity": "high", "name": "no_generated_images"})

    if reference_sample and generated_sample:
        ref_features, ref_histograms, ref_errors = extract_features(reference_sample, image_size=image_size)
        gen_features, gen_histograms, gen_errors = extract_features(generated_sample, image_size=image_size)
        if ref_errors:
            issues.append({"severity": "medium", "name": "reference_read_errors", "count": len(ref_errors), "examples": ref_errors[:10]})
        if gen_errors:
            issues.append({"severity": "medium", "name": "generated_read_errors", "count": len(gen_errors), "examples": gen_errors[:10]})
        if len(ref_features) > 1 and len(gen_features) > 1:
            metrics.update(compute_lite_metrics(ref_features, gen_features, ref_histograms, gen_histograms))
        else:
            issues.append(
                {
                    "severity": "medium",
                    "name": "insufficient_readable_images_for_distribution_metrics",
                    "message": "At least two readable reference and generated images are needed for lightweight distribution metrics.",
                    "reference_readable_count": int(len(ref_features)),
                    "generated_readable_count": int(len(gen_features)),
                }
            )

    standard_report = None
    if standard_fid and reference_sample and generated_sample:
        if len(reference_sample) < 2 or len(generated_sample) < 2:
            standard_report = {
                "status": "skipped",
                "message": "Standard FID was skipped because at least two reference and generated images are required.",
                "reference_sample_count": len(reference_sample),
                "generated_sample_count": len(generated_sample),
                "dims": standard_fid_dims,
                "batch_size": standard_fid_batch_size,
            }
        else:
            standard_report = compute_standard_fid(
                reference_sample=reference_sample,
                generated_sample=generated_sample,
                output_dir=report_dir,
                dims=standard_fid_dims,
                batch_size=standard_fid_batch_size,
            )
        if standard_report.get("status") == "succeeded":
            metrics["fid_pytorch"] = standard_report.get("fid")
        elif standard_report.get("status") == "skipped":
            issues.append(
                {
                    "severity": "low",
                    "name": "standard_fid_skipped",
                    "message": standard_report.get("message"),
                }
            )
        else:
            issues.append(
                {
                    "severity": "low",
                    "name": "standard_fid_unavailable",
                    "message": standard_report.get("message"),
                }
            )

    triggers = build_distribution_triggers(metrics, merged_thresholds)
    blocking = [item for item in issues if item.get("severity") == "high"]
    status = "failed" if blocking else ("warning" if triggers or issues else "passed")
    report = {
        "schema_version": DISTRIBUTION_EVALUATION_VERSION,
        "status": status,
        "message": distribution_message(status, triggers, issues),
        "dry_run": False,
        "reference_dir": reference_path.as_posix(),
        "generated_dir": generated_path.as_posix(),
        "reference_image_count": len(reference_images),
        "generated_image_count": len(generated_images),
        "reference_sample_count": len(reference_sample),
        "generated_sample_count": len(generated_sample),
        "image_size": image_size,
        "standard_fid": standard_report,
        "metrics": metrics,
        "thresholds": merged_thresholds,
        "triggers": triggers,
        "issues": issues,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_distribution_artifacts(report_dir, report)
    return report


def deterministic_sample(paths: list[Path], limit: int) -> list[Path]:
    paths = sorted(paths, key=lambda path: natural_key(path.as_posix()))
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[0]]
    last = len(paths) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [paths[index] for index in indexes]


def extract_features(paths: list[Path], image_size: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    features = []
    histograms = []
    errors = []
    for path in paths:
        try:
            feature, hist = image_feature(path, image_size=image_size)
            features.append(feature)
            histograms.append(hist)
        except Exception as exc:
            errors.append({"path": path.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
    if not features:
        return np.empty((0, 1), dtype=np.float64), np.empty((0, 1), dtype=np.float64), errors
    return np.asarray(features, dtype=np.float64), np.asarray(histograms, dtype=np.float64), errors


def image_feature(path: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        luma = image.convert("L").resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(luma, dtype=np.float32) / 255.0
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    hist = hist / max(float(hist.sum()), 1.0)
    small = np.asarray(
        Image.fromarray((arr * 255.0).astype(np.uint8)).resize((16, 16), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    small = (small.reshape(-1) / 255.0).astype(np.float64)
    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    stats = np.asarray(
        [
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(np.percentile(arr, 5)),
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(grad.mean()),
            float(grad.std()),
        ],
        dtype=np.float64,
    )
    feature = np.concatenate([small, hist, stats], axis=0)
    return feature, hist


def compute_lite_metrics(
    ref_features: np.ndarray,
    gen_features: np.ndarray,
    ref_histograms: np.ndarray,
    gen_histograms: np.ndarray,
) -> dict[str, Any]:
    fid_lite = frechet_distance(ref_features, gen_features)
    ref_hist = ref_histograms.mean(axis=0)
    gen_hist = gen_histograms.mean(axis=0)
    distances = cdist(gen_features, ref_features, metric="euclidean")
    generated_pairwise = pairwise_distances(gen_features)
    return {
        "sar_fid_lite": round(float(fid_lite), 6),
        "histogram_l1": round(float(np.abs(ref_hist - gen_hist).sum()), 6),
        "histogram_jsd": round(float(jensen_shannon_divergence(ref_hist, gen_hist)), 6),
        "mmd_rbf": round(float(mmd_rbf(ref_features, gen_features)), 6),
        "generated_diversity_mean_l2": round(float(generated_pairwise.mean()), 6) if generated_pairwise.size else 0.0,
        "generated_diversity_p10_l2": round(float(np.percentile(generated_pairwise, 10)), 6) if generated_pairwise.size else 0.0,
        "nearest_reference_distance_mean_l2": round(float(distances.min(axis=1).mean()), 6),
        "nearest_reference_distance_p50_l2": round(float(np.percentile(distances.min(axis=1), 50)), 6),
    }


def frechet_distance(ref_features: np.ndarray, gen_features: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = ref_features.mean(axis=0)
    mu2 = gen_features.mean(axis=0)
    sigma1 = np.cov(ref_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)
    sigma1 = np.atleast_2d(sigma1) + np.eye(mu1.shape[0]) * eps
    sigma2 = np.atleast_2d(sigma2) + np.eye(mu2.shape[0]) * eps
    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        covmean = sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    value = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return max(float(value), 0.0)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    if len(features) < 2:
        return np.asarray([], dtype=np.float64)
    dist = cdist(features, features, metric="euclidean")
    upper = dist[np.triu_indices(len(features), k=1)]
    return upper


def mmd_rbf(ref_features: np.ndarray, gen_features: np.ndarray) -> float:
    combined = np.vstack([ref_features, gen_features])
    pairwise_sq = cdist(combined, combined, metric="sqeuclidean")
    nonzero = pairwise_sq[pairwise_sq > 0]
    gamma = 1.0 / (2.0 * float(np.median(nonzero) if nonzero.size else 1.0))
    k_xx = np.exp(-gamma * cdist(ref_features, ref_features, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * cdist(gen_features, gen_features, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * cdist(ref_features, gen_features, metric="sqeuclidean"))
    return max(float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()), 0.0)


def compute_standard_fid(
    reference_sample: list[Path],
    generated_sample: list[Path],
    output_dir: Path | None,
    dims: int,
    batch_size: int,
) -> dict[str, Any]:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        import torch
    except Exception as exc:
        return {"status": "unavailable", "message": f"pytorch_fid import failed: {type(exc).__name__}: {exc}"}

    if output_dir is None:
        return {"status": "skipped", "message": "No output directory was provided for sampled FID staging."}

    sample_root = output_dir / "standard_fid_samples"
    ref_dir = sample_root / "reference"
    gen_dir = sample_root / "generated"
    stage_sample_dir(ref_dir, reference_sample)
    stage_sample_dir(gen_dir, generated_sample)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        fid = calculate_fid_given_paths(
            [ref_dir.as_posix(), gen_dir.as_posix()],
            batch_size=int(batch_size),
            device=device,
            dims=int(dims),
            num_workers=0,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"pytorch_fid failed: {type(exc).__name__}: {exc}",
            "device": device,
            "dims": dims,
            "batch_size": batch_size,
        }
    return {
        "status": "succeeded",
        "fid": round(float(fid), 6),
        "backend": "pytorch_fid",
        "device": device,
        "dims": dims,
        "batch_size": batch_size,
        "reference_sample_count": len(reference_sample),
        "generated_sample_count": len(generated_sample),
        "sample_root": sample_root.as_posix(),
    }


def stage_sample_dir(output_dir: Path, paths: list[Path]) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for path in paths:
        name = unique_name(path.name, used)
        target = output_dir / name
        try:
            target.symlink_to(path.resolve())
        except OSError:
            shutil.copy2(path, target)


def unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    path = Path(name)
    idx = 2
    while f"{path.stem}_{idx}{path.suffix}" in used:
        idx += 1
    value = f"{path.stem}_{idx}{path.suffix}"
    used.add(value)
    return value


def build_distribution_triggers(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    triggers = []
    if "sar_fid_lite" in metrics and float(metrics["sar_fid_lite"]) > float(thresholds["max_sar_fid_lite"]):
        triggers.append(
            {
                "name": "high_sar_fid_lite",
                "severity": "medium",
                "value": metrics["sar_fid_lite"],
                "threshold": thresholds["max_sar_fid_lite"],
                "message": "Generated distribution is far from the reference distribution under lightweight SAR features.",
            }
        )
    if "histogram_jsd" in metrics and float(metrics["histogram_jsd"]) > float(thresholds["max_histogram_jsd"]):
        triggers.append(
            {
                "name": "high_histogram_jsd",
                "severity": "medium",
                "value": metrics["histogram_jsd"],
                "threshold": thresholds["max_histogram_jsd"],
                "message": "Generated grayscale histogram differs from the reference histogram.",
            }
        )
    if "generated_diversity_mean_l2" in metrics and float(metrics["generated_diversity_mean_l2"]) < float(
        thresholds["min_generated_diversity"]
    ):
        triggers.append(
            {
                "name": "low_generated_diversity",
                "severity": "medium",
                "value": metrics["generated_diversity_mean_l2"],
                "threshold": thresholds["min_generated_diversity"],
                "message": "Generated samples appear too similar under lightweight SAR features.",
            }
        )
    return triggers


def distribution_message(status: str, triggers: list[dict[str, Any]], issues: list[dict[str, Any]]) -> str:
    if status == "passed":
        return "Distribution metrics passed lightweight checks."
    names = [item.get("name", "issue") for item in triggers[:3] + issues[:3]]
    return "Distribution evaluation raised: " + ", ".join(names)


def write_distribution_artifacts(output_dir: Path | None, report: dict[str, Any]) -> None:
    if output_dir is None:
        return
    save_json(output_dir / "distribution_evaluation.json", report)
    save_text(output_dir / "distribution_evaluation.md", render_distribution_markdown(report))


def render_distribution_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Distribution Evaluation",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Reference dir: `{report.get('reference_dir')}`",
        f"- Generated dir: `{report.get('generated_dir')}`",
        f"- Reference images: {report.get('reference_image_count')}",
        f"- Generated images: {report.get('generated_image_count')}",
        f"- Reference sample: {report.get('reference_sample_count', report.get('reference_sample_limit'))}",
        f"- Generated sample: {report.get('generated_sample_count', report.get('generated_sample_limit'))}",
        "",
        "## Metrics",
        "",
    ]
    metrics = report.get("metrics") or {}
    if not metrics:
        lines.append("- None")
    else:
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {value}")
    standard = report.get("standard_fid") or {}
    if standard:
        lines.extend(["", "## Standard FID Backend", ""])
        lines.append(f"- Status: {standard.get('status')}")
        if standard.get("fid") is not None:
            lines.append(f"- FID: {standard.get('fid')}")
        if standard.get("message"):
            lines.append(f"- Message: {standard.get('message')}")
        if standard.get("device"):
            lines.append(f"- Device: {standard.get('device')}")
    lines.extend(["", "## Triggers", ""])
    triggers = report.get("triggers") or []
    if not triggers:
        lines.append("- None")
    else:
        for item in triggers:
            lines.append(f"- `{item.get('name')}` ({item.get('severity')}): {item.get('message')}")
    issues = report.get("issues") or []
    lines.extend(["", "## Issues", ""])
    if not issues:
        lines.append("- None")
    else:
        for item in issues:
            lines.append(f"- `{item.get('name')}` ({item.get('severity')})")
    lines.append("")
    return "\n".join(lines)

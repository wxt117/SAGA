from __future__ import annotations

from typing import Any


EVALUATOR_REPORT_VERSION = "saga_evaluator_report_v1"


def build_evaluator_report(
    *,
    evaluator: str,
    task: str,
    status: str,
    dry_run: bool,
    metrics: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    issues: list[dict[str, Any]] | None = None,
    leakage_risk: str | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    primary = metrics.get("primary") if isinstance(metrics.get("primary"), dict) else {}
    return {
        "schema_version": EVALUATOR_REPORT_VERSION,
        "evaluator": evaluator,
        "task": task,
        "status": status,
        "dry_run": dry_run,
        "primary_metric": primary.get("metric"),
        "baseline": primary.get("baseline"),
        "augmented": primary.get("augmented"),
        "delta": primary.get("delta"),
        "verdict": primary.get("verdict") or primary.get("status") or "unavailable",
        "confidence": evaluator_confidence(primary=primary, dry_run=dry_run, issues=issues or []),
        "leakage_risk": leakage_risk or infer_leakage_risk(benchmark or {}, issues or []),
        "metrics": metrics,
        "benchmark_summary": summarize_benchmark(benchmark or {}),
        "issues": issues or [],
        "artifacts": artifacts or {},
        "notes": [
            "EvaluatorReport is the task-level evidence protocol used by SAGA policy memory.",
            "Dry-run reports prepare protocols and commands but do not claim downstream benefit.",
        ],
    }


def evaluator_confidence(primary: dict[str, Any], dry_run: bool, issues: list[dict[str, Any]]) -> str:
    if dry_run or not primary or primary.get("status") == "unavailable":
        return "none"
    if any(issue.get("severity") == "high" for issue in issues):
        return "low"
    if any(issue.get("severity") == "medium" for issue in issues):
        return "medium"
    return "high"


def infer_leakage_risk(benchmark: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    issue_names = {issue.get("name") for issue in issues}
    if "heldout_missing" in issue_names:
        return "medium"
    if "single_class_baseline" in issue_names:
        return "not_applicable"
    split_policy = benchmark.get("split_policy") or {}
    if split_policy.get("include_augmented_in_validation"):
        return "high"
    return "low"


def summarize_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    datasets = benchmark.get("datasets") or {}
    return {
        "datasets": {
            name: {
                "path": summary.get("path"),
                "counts": summary.get("counts"),
                "source": summary.get("source"),
            }
            for name, summary in datasets.items()
            if isinstance(summary, dict)
        },
        "split_policy": benchmark.get("split_policy"),
    }

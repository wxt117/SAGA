from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Iterable

from saga.core.protocol import SagaSample, UNKNOWN


def build_machine_report(dataset_name: str, samples: list[SagaSample]) -> dict[str, Any]:
    sample_rows = [sample.to_dict() for sample in samples]
    class_counts = Counter(sample.label.class_name for sample in samples)
    parser_counts = Counter(str(sample.metadata.get("parser_schema") or sample.metadata.get("filename_parser_schema") or UNKNOWN) for sample in samples)
    parse_status_counts = Counter(str(sample.metadata.get("parse_status", UNKNOWN)) for sample in samples)

    report: dict[str, Any] = {
        "dataset_id": dataset_name,
        "task": "classification",
        "summary": {
            "total_images": len(samples),
            "total_samples": len(samples),
            "classes": dict(sorted(class_counts.items())),
            "parser_schemas": dict(sorted(parser_counts.items())),
            "parse_status": dict(sorted(parse_status_counts.items())),
        },
        "metadata_coverage": metadata_coverage(samples),
        "distributions": {
            "band": value_counts(samples, "band"),
            "polarization": value_counts(samples, "polarization"),
            "incidence_angle_deg": numeric_counts(samples, "incidence_angle_deg"),
            "azimuth_deg": numeric_counts(samples, "azimuth_deg"),
            "resolution_m": numeric_counts(samples, "resolution_m"),
        },
        "azimuth_analysis": azimuth_analysis(samples),
        "issues": issue_report(samples),
        "recommended_skill_candidates": recommended_skills(samples),
        "sample_preview": sample_rows[:20],
    }
    return report


def metadata_coverage(samples: list[SagaSample]) -> dict[str, float]:
    fields = [
        "class",
        "azimuth_deg",
        "incidence_angle_deg",
        "polarization",
        "band",
        "resolution_m",
        "caption",
    ]
    total = len(samples) or 1
    coverage: dict[str, float] = {}
    for field in fields:
        count = sum(has_value(sample.metadata.get(field) or getattr(sample.label, "caption", None) if field == "caption" else sample.metadata.get(field)) for sample in samples)
        coverage[field] = round(count / total, 4)
    return coverage


def has_value(value: Any) -> bool:
    return value not in {None, "", UNKNOWN}


def value_counts(samples: list[SagaSample], field: str) -> dict[str, int]:
    counter = Counter(str(sample.metadata.get(field)) for sample in samples if has_value(sample.metadata.get(field)))
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def numeric_counts(samples: list[SagaSample], field: str) -> dict[str, Any]:
    values = [float(sample.metadata[field]) for sample in samples if is_number(sample.metadata.get(field))]
    if not values:
        return {"count": 0}
    counter = Counter(format_number(value) for value in values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "unique": len(counter),
        "values": dict(sorted(counter.items(), key=lambda item: float(item[0]))),
    }


def azimuth_analysis(samples: list[SagaSample]) -> dict[str, Any]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        azimuth = sample.metadata.get("azimuth_deg")
        if is_number(azimuth):
            by_class[sample.label.class_name].append(float(azimuth))

    analysis: dict[str, Any] = {}
    for class_name, values in sorted(by_class.items()):
        unique_values = sorted(set(values))
        diffs = [b - a for a, b in zip(unique_values, unique_values[1:]) if b >= a]
        step = median(diffs) if diffs else None
        large_gaps = []
        if step and step > 0:
            large_gaps = [
                {"from": a, "to": b, "gap": b - a}
                for a, b in zip(unique_values, unique_values[1:])
                if b - a > step * 1.5
            ]
        analysis[class_name] = {
            "sample_count_with_azimuth": len(values),
            "unique_count": len(unique_values),
            "min": min(unique_values) if unique_values else None,
            "max": max(unique_values) if unique_values else None,
            "step_deg_estimate": step,
            "large_gaps": large_gaps[:20],
            "coverage_score_360": round(len(unique_values) / 360, 4),
            "unique_values_preview": [format_number(value) for value in unique_values[:72]],
        }
    return analysis


def issue_report(samples: list[SagaSample]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    failed = [sample for sample in samples if sample.metadata.get("parse_status") == "failed"]
    partial = [sample for sample in samples if sample.metadata.get("parse_status") == "partial"]
    unknown_class = [sample for sample in samples if sample.label.class_name == UNKNOWN]
    missing_azimuth = [sample for sample in samples if not has_value(sample.metadata.get("azimuth_deg"))]

    if failed:
        issues.append(make_issue("parse_failure", "high", "Some files could not be parsed by known schemas.", failed))
    if partial:
        issues.append(make_issue("partial_parse", "medium", "Some files only produced directory-level labels.", partial))
    if unknown_class:
        issues.append(make_issue("unknown_class", "high", "Some samples have unknown class labels.", unknown_class))
    if missing_azimuth:
        severity = "medium" if len(missing_azimuth) < len(samples) else "high"
        issues.append(make_issue("missing_azimuth", severity, "Some samples lack azimuth metadata.", missing_azimuth))

    for class_name, info in azimuth_analysis(samples).items():
        if info["large_gaps"]:
            issues.append(
                {
                    "type": "sparse_azimuth",
                    "severity": "medium",
                    "class": class_name,
                    "message": "Azimuth distribution contains gaps larger than the estimated step.",
                    "count": len(info["large_gaps"]),
                    "examples": info["large_gaps"][:5],
                }
            )
    return issues


def make_issue(issue_type: str, severity: str, message: str, samples: list[SagaSample]) -> dict[str, Any]:
    return {
        "type": issue_type,
        "severity": severity,
        "message": message,
        "count": len(samples),
        "examples": [sample.metadata.get("original_relpath", sample.image.path) for sample in samples[:5]],
    }


def recommended_skills(samples: list[SagaSample]) -> list[dict[str, str]]:
    coverage = metadata_coverage(samples)
    recommendations: list[dict[str, str]] = []
    if coverage.get("azimuth_deg", 0) > 0.5:
        recommendations.append(
            {
                "skill": "GeoDiffSARSkill",
                "reason": "Dataset contains azimuth metadata suitable for target sparse-azimuth completion.",
            }
        )
        recommendations.append(
            {
                "skill": "GaussianSplattingCompletionSkill",
                "reason": "Azimuth metadata can drive fast target view completion candidates.",
            }
        )
    if coverage.get("band", 0) > 0.2 or coverage.get("polarization", 0) > 0.2:
        recommendations.append(
            {
                "skill": "StyleTransferSkill",
                "reason": "Band or polarization metadata is available for cross-domain/style transfer planning.",
            }
        )
    recommendations.append(
        {
            "skill": "TraditionalAugSkill",
            "reason": "Classification labels can be preserved under conservative SAR-aware image augmentations.",
        }
    )
    return recommendations


def is_number(value: Any) -> bool:
    try:
        float(value)
        return value not in {None, ""}
    except (TypeError, ValueError):
        return False


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# SAGA Dataset Diagnosis: {report['dataset_id']}",
        "",
        "## Summary",
        "",
        f"- Task: {report.get('task', 'classification')}",
        f"- Total images: {summary['total_images']}",
        f"- Total samples: {summary['total_samples']}",
        f"- Classes: {', '.join(f'{k} ({v})' for k, v in summary['classes'].items()) or 'none'}",
        f"- Parser schemas: {', '.join(f'{k} ({v})' for k, v in summary['parser_schemas'].items()) or 'none'}",
        "",
        "## Metadata Coverage",
        "",
    ]
    for field, value in report["metadata_coverage"].items():
        lines.append(f"- {field}: {value:.2%}")
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- [{issue['severity']}] {issue['type']}: {issue['message']} Count: {issue['count']}")
    else:
        lines.append("- No major issues found.")
    lines.extend(["", "## Recommended Skill Candidates", ""])
    for item in report["recommended_skill_candidates"]:
        lines.append(f"- {item['skill']}: {item['reason']}")
    lines.extend(["", "## Azimuth Analysis", ""])
    for class_name, info in report["azimuth_analysis"].items():
        lines.append(
            f"- {class_name}: {info['unique_count']} unique azimuths, "
            f"estimated step {info['step_deg_estimate']}, gaps {len(info['large_gaps'])}"
        )
    lines.append("")
    return "\n".join(lines)

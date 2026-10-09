from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.core.protocol import DatasetConfig, UNKNOWN
from saga.data.loader import load_samples
from saga.data.path_patterns import path_signature


DEFAULT_FIELDS = [
    "class",
    "azimuth_deg",
    "incidence_angle_deg",
    "depression_angle_deg",
    "elevation_angle_deg",
    "band",
    "polarization",
    "resolution_m",
]


def validate_dataset_format(
    config: DatasetConfig,
    output_dir: str | Path,
    sample_limit: int = 200,
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    samples = load_samples(config, read_image_size=False)
    sampled = samples[:sample_limit] if sample_limit > 0 else samples
    required_fields = ["class"] if required_fields is None else required_fields

    field_coverage = compute_field_coverage(sampled, DEFAULT_FIELDS)
    missing_required: list[dict[str, Any]] = []
    for sample in sampled:
        for field in required_fields:
            value = sample.label.class_name if field == "class" else sample.metadata.get(field)
            if not has_value(value):
                missing_required.append(
                    {
                        "sample_id": sample.sample_id,
                        "field": field,
                        "path": sample.metadata.get("original_relpath", sample.image.path),
                    }
                )

    parse_status = Counter(str(sample.metadata.get("parse_status", UNKNOWN)) for sample in sampled)
    rules = Counter(str(sample.metadata.get("format_rule", sample.metadata.get("parser_schema", UNKNOWN))) for sample in sampled)

    preview = []
    for sample in sampled[:50]:
        preview.append(
            {
                "path": sample.metadata.get("original_relpath", sample.image.path),
                "class": sample.label.class_name,
                "parse_status": sample.metadata.get("parse_status"),
                "format_rule": sample.metadata.get("format_rule"),
                "azimuth_deg": sample.metadata.get("azimuth_deg", UNKNOWN),
                "incidence_angle_deg": sample.metadata.get("incidence_angle_deg", UNKNOWN),
                "depression_angle_deg": sample.metadata.get("depression_angle_deg", UNKNOWN),
                "elevation_angle_deg": sample.metadata.get("elevation_angle_deg", UNKNOWN),
                "band": sample.metadata.get("band", UNKNOWN),
                "polarization": sample.metadata.get("polarization", UNKNOWN),
                "resolution_m": sample.metadata.get("resolution_m", UNKNOWN),
            }
        )

    report = {
        "dataset": config.name,
        "format_spec": config.format_spec.as_posix() if config.format_spec else None,
        "total_samples_seen": len(samples),
        "validated_samples": len(sampled),
        "valid": not missing_required,
        "parse_status": dict(sorted(parse_status.items())),
        "format_rules": dict(rules.most_common()),
        "field_coverage": field_coverage,
        "required_fields": required_fields,
        "missing_required_count": len(missing_required),
        "missing_required_examples": missing_required[:100],
        "missing_required_groups": group_missing_required(missing_required),
        "preview": preview,
    }
    save_json(output_path / "format_validation.json", report)
    save_text(output_path / "format_validation.md", render_validation_markdown(report))
    return report


def compute_field_coverage(samples: list, fields: list[str]) -> dict[str, float]:
    total = len(samples) or 1
    coverage: dict[str, float] = {}
    for field in fields:
        count = 0
        for sample in samples:
            value = sample.label.class_name if field == "class" else sample.metadata.get(field)
            if has_value(value):
                count += 1
        coverage[field] = round(count / total, 4)
    return coverage


def has_value(value: Any) -> bool:
    return value not in {None, "", UNKNOWN}


def group_missing_required(missing_required: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in missing_required:
        path = str(item.get("path", ""))
        field = str(item.get("field", UNKNOWN))
        signature = path_signature(path)
        key = (field, signature)
        if key not in grouped:
            grouped[key] = {
                "field": field,
                "signature": signature,
                "count": 0,
                "examples": [],
            }
        group = grouped[key]
        group["count"] += 1
        if len(group["examples"]) < 10:
            group["examples"].append(path)
    return sorted(grouped.values(), key=lambda value: (-int(value["count"]), value["field"], value["signature"]))


def render_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SAGA Format Validation: {report['dataset']}",
        "",
        f"- Format spec: `{report.get('format_spec')}`",
        f"- Total samples seen: {report['total_samples_seen']}",
        f"- Validated samples: {report['validated_samples']}",
        f"- Valid: {report['valid']}",
        f"- Missing required count: {report['missing_required_count']}",
        "",
        "## Parse Status",
        "",
    ]
    for key, value in report["parse_status"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Format Rules", ""])
    for key, value in report["format_rules"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Field Coverage", ""])
    for key, value in report["field_coverage"].items():
        lines.append(f"- {key}: {value:.2%}")

    lines.extend(["", "## Missing Required Examples", ""])
    if report["missing_required_examples"]:
        for item in report["missing_required_examples"][:20]:
            lines.append(f"- `{item['path']}` missing `{item['field']}`")
    else:
        lines.append("- None.")

    lines.extend(["", "## Missing Required Groups", ""])
    groups = report.get("missing_required_groups", [])
    if groups:
        for group in groups[:20]:
            examples = ", ".join(f"`{item}`" for item in group.get("examples", [])[:3])
            lines.append(
                f"- {group['count']} x missing `{group['field']}` in `{group['signature']}`: {examples}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Preview", ""])
    for item in report["preview"][:30]:
        lines.append(
            "- "
            f"`{item['path']}` -> class={item['class']}, "
            f"azimuth={item['azimuth_deg']}, incidence={item['incidence_angle_deg']}, "
            f"depression={item['depression_angle_deg']}, elevation={item['elevation_angle_deg']}, "
            f"band={item['band']}, pol={item['polarization']}, "
            f"rule={item['format_rule']}"
        )
    lines.append("")
    return "\n".join(lines)

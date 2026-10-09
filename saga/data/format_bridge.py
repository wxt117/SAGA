from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from saga.core.config import load_mapping, save_json, save_text
from saga.core.protocol import DatasetConfig
from saga.data.format_builder import dedupe_rules, heuristic_rules_from_inspection, regex_for_field
from saga.data.format_validation import validate_dataset_format


FORMAT_BRIDGE_VERSION = "saga_dataset_format_bridge_v1"


def compile_and_validate_format_from_profile_dir(
    profile_dir: str | Path,
    output_dir: str | Path | None = None,
    sample_limit: int = 500,
    required_fields: list[str] | None = None,
    include_heuristics: bool = True,
) -> dict[str, Any]:
    profile_path = Path(profile_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve() if output_dir else profile_path
    raw_profile = load_mapping(profile_path / "raw_profile.json")
    intent_spec = load_mapping(profile_path / "intent_spec.json") if (profile_path / "intent_spec.json").exists() else {}
    format_hints = load_mapping(profile_path / "format_hints.json") if (profile_path / "format_hints.json").exists() else []
    if not isinstance(format_hints, list):
        raise ValueError("format_hints.json must contain a list.")
    return compile_and_validate_format(
        root=raw_profile["root"],
        output_dir=output_path,
        raw_profile=raw_profile,
        intent_spec=intent_spec,
        format_hints=format_hints,
        sample_limit=sample_limit,
        required_fields=required_fields,
        include_heuristics=include_heuristics,
    )


def compile_and_validate_format(
    root: str | Path,
    output_dir: str | Path,
    raw_profile: dict[str, Any],
    intent_spec: dict[str, Any] | None = None,
    format_hints: list[dict[str, Any]] | None = None,
    sample_limit: int = 500,
    required_fields: list[str] | None = None,
    include_heuristics: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    root_path = Path(root).expanduser().resolve()
    intent_spec = intent_spec or {}
    format_hints = format_hints or []

    spec = compile_format_spec_from_profile(
        raw_profile=raw_profile,
        format_hints=format_hints,
        name=f"{root_path.name}_compiled_format",
        include_heuristics=include_heuristics,
    )
    format_spec_path = output_path / "compiled_format_spec.yaml"
    dataset_config_path = output_path / "compiled_dataset.yaml"
    save_text(format_spec_path, yaml.safe_dump(spec, allow_unicode=True, sort_keys=False))
    dataset_config = build_dataset_config_mapping(
        root=root_path,
        format_spec_path=format_spec_path,
        intent_spec=intent_spec,
    )
    save_text(dataset_config_path, yaml.safe_dump(dataset_config, allow_unicode=True, sort_keys=False))

    fields = required_fields or infer_required_fields(intent_spec=intent_spec, format_hints=format_hints)
    validation = validate_dataset_format(
        config=DatasetConfig(
            name=dataset_config["dataset"]["name"],
            root=root_path,
            task=dataset_config["dataset"]["task"],
            format=dataset_config["dataset"]["format"],
            splits=dataset_config["dataset"]["splits"],
            image=dataset_config["dataset"].get("image") or {},
            label=dataset_config["dataset"].get("label") or {},
            format_spec=format_spec_path,
            format_options={},
            classes=[],
            normalization={},
        ),
        output_dir=output_path,
        sample_limit=sample_limit,
        required_fields=fields,
    )
    report = build_bridge_report(
        root=root_path,
        format_spec_path=format_spec_path,
        dataset_config_path=dataset_config_path,
        spec=spec,
        validation=validation,
        required_fields=fields,
        sample_limit=sample_limit,
        format_hints=format_hints,
    )
    save_json(output_path / "format_bridge.json", report)
    save_json(output_path / "validated_dataset_profile.json", build_validated_dataset_profile(report))
    save_text(output_path / "format_bridge.md", render_bridge_markdown(report))
    return report


def compile_format_spec_from_profile(
    raw_profile: dict[str, Any],
    format_hints: list[dict[str, Any]] | None = None,
    name: str = "compiled_dataset_format",
    include_heuristics: bool = True,
) -> dict[str, Any]:
    rules: list[dict[str, Any]] = []
    for hint in format_hints or []:
        if hint.get("type") == "filename_anchor_fields":
            rule = compile_anchor_hint_rule(hint, raw_profile)
            if rule:
                rules.append(rule)
        elif hint.get("type") == "filename_suffix_field":
            rule = compile_suffix_field_hint_rule(hint, raw_profile)
            if rule:
                rules.append(rule)
    if include_heuristics:
        rules.extend(heuristic_rules_from_inspection(raw_profile))
    rules = dedupe_rules(rules)
    return {
        "format_spec": {
            "name": sanitize_name(name),
            "version": "saga_format_spec_v1",
            "description": (
                "Compiled by SAGA from RawDatasetProfile and FormatHint. "
                "This spec is only trusted after deterministic validation."
            ),
            "rules": rules,
            "fallback": {"class_from_parent": True},
            "compile_notes": [
                "Rules from user FormatHint are semantic candidates, not facts.",
                "Validation coverage determines whether this spec can promote the dataset profile to Level 2.",
            ],
        }
    }


def compile_anchor_hint_rule(hint: dict[str, Any], raw_profile: dict[str, Any]) -> dict[str, Any] | None:
    rule = hint.get("rule") or {}
    fields_after_anchor = normalize_hint_fields(rule.get("fields_after_anchor") or [])
    if not fields_after_anchor:
        return None
    anchor = rule.get("anchor")
    if anchor is None:
        return None

    fields: dict[str, str] = {}
    used_groups: set[str] = set()
    field_patterns: list[str] = []
    for field in fields_after_anchor:
        group = unique_group_name(field, used_groups)
        used_groups.add(group)
        fields[group] = field
        field_patterns.append(f"(?P<{group}>{regex_for_field(field)})")

    separator = r"[_\-\s]+"
    anchor_pattern = anchor_regex(anchor)
    pattern = (
        r"^(?:.*[/_\-\s])?"
        + anchor_pattern
        + separator
        + separator.join(field_patterns)
        + r"\.[^.]+$"
    )
    support = count_rule_support(raw_profile, pattern)
    return {
        "name": f"anchor_{sanitize_name(str(anchor))}_fields",
        "target": "relative_path",
        "pattern": pattern,
        "fields": fields,
        "constants": {"anchor_token": anchor},
        "confidence": 0.85 if support else 0.5,
        "support_in_profile": support,
        "source": "format_hint",
        "source_text": hint.get("source_text", ""),
    }


def compile_suffix_field_hint_rule(hint: dict[str, Any], raw_profile: dict[str, Any]) -> dict[str, Any] | None:
    field_info = (hint.get("rule") or {}).get("field") or {}
    field = field_info.get("normalized") or field_info.get("raw")
    if not field:
        return None
    field = str(field)
    group = unique_group_name(field, set())
    pattern = (
        r"^(?:.*/)?(?P<target_name>.+?)_"
        f"(?P<{group}>{regex_for_field(field)})"
        r"\.[^.]+$"
    )
    support = count_rule_support(raw_profile, pattern)
    return {
        "name": f"suffix_{sanitize_name(field)}",
        "target": "relative_path",
        "pattern": pattern,
        "fields": {"target_name": "target_name", group: field},
        "confidence": 0.85 if support else 0.5,
        "support_in_profile": support,
        "source": "format_hint",
        "source_text": hint.get("source_text", ""),
    }


def build_dataset_config_mapping(
    root: Path,
    format_spec_path: Path,
    intent_spec: dict[str, Any],
) -> dict[str, Any]:
    task = intent_spec.get("intent", {}).get("task") or "classification"
    if task == "style_transfer":
        # Dataset ingestion is still classification-like for selecting SAR target images.
        dataset_task = "classification"
    elif task in {"object_detection", "segmentation"}:
        dataset_task = task
    else:
        dataset_task = "classification"
    return {
        "dataset": {
            "name": root.name,
            "root": root.as_posix(),
            "format": "custom",
            "format_spec": format_spec_path.as_posix(),
            "task": dataset_task,
            "splits": {"train": "."},
            "label": {"source": "mixed"},
            "image": {"domain": "unknown"},
        }
    }


def infer_required_fields(
    intent_spec: dict[str, Any],
    format_hints: list[dict[str, Any]],
) -> list[str]:
    task = intent_spec.get("intent", {}).get("task")
    if task == "raysar_synthesis" and not format_hints:
        return []
    fields = ["class"]
    for hint in format_hints:
        rule = hint.get("rule") or {}
        if hint.get("type") == "filename_anchor_fields":
            fields.extend(normalize_hint_fields(rule.get("fields_after_anchor") or []))
        elif hint.get("type") == "filename_suffix_field":
            fields.extend(normalize_hint_fields([rule.get("field") or {}]))
    filters = intent_spec.get("intent", {}).get("filters") or {}
    fields.extend(str(field) for field in filters.keys())
    exclude_filters = intent_spec.get("intent", {}).get("exclude_filters") or {}
    fields.extend(str(field) for field in exclude_filters.keys())
    return sorted(dict.fromkeys(fields), key=lambda item: (item != "class", item))


def build_bridge_report(
    root: Path,
    format_spec_path: Path,
    dataset_config_path: Path,
    spec: dict[str, Any],
    validation: dict[str, Any],
    required_fields: list[str],
    sample_limit: int,
    format_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    rules = spec.get("format_spec", {}).get("rules", [])
    valid = bool(validation.get("valid"))
    return {
        "schema_version": FORMAT_BRIDGE_VERSION,
        "root": root.as_posix(),
        "format_spec_path": format_spec_path.as_posix(),
        "dataset_config_path": dataset_config_path.as_posix(),
        "validation_report_path": (format_spec_path.parent / "format_validation.json").as_posix(),
        "compiled_rules": [
            {
                "name": rule.get("name"),
                "source": rule.get("source", "heuristic"),
                "target": rule.get("target"),
                "fields": rule.get("fields"),
                "confidence": rule.get("confidence"),
                "support_in_profile": rule.get("support_in_profile", rule.get("support_in_inspection")),
            }
            for rule in rules
        ],
        "format_hints_used": format_hints,
        "required_fields": required_fields,
        "sample_limit": sample_limit,
        "valid": valid,
        "profile_level_after_validation": 2 if valid else 1,
        "validation": {
            "validated_samples": validation.get("validated_samples"),
            "total_samples_seen": validation.get("total_samples_seen"),
            "parse_status": validation.get("parse_status"),
            "format_rules": validation.get("format_rules"),
            "field_coverage": validation.get("field_coverage"),
            "missing_required_count": validation.get("missing_required_count"),
            "missing_required_groups": validation.get("missing_required_groups", [])[:20],
        },
        "next_step": (
            "Use compiled_dataset.yaml for diagnosis/planning."
            if valid
            else "Inspect missing_required_groups, ask for a better format hint, then recompile."
        ),
    }


def render_bridge_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA DatasetFormatSpec Bridge",
        "",
        "This report connects Dataset Profiler output to deterministic format validation.",
        "",
        f"- Root: `{report['root']}`",
        f"- Format spec: `{report['format_spec_path']}`",
        f"- Dataset config: `{report['dataset_config_path']}`",
        f"- Valid: {report['valid']}",
        f"- Profile level after validation: {report['profile_level_after_validation']}",
        f"- Required fields: `{report['required_fields']}`",
        f"- Next step: {report['next_step']}",
        "",
        "## Compiled Rules",
        "",
    ]
    for rule in report.get("compiled_rules", []):
        lines.append(
            "- "
            f"`{rule.get('name')}` source={rule.get('source')} "
            f"fields={rule.get('fields')} support={rule.get('support_in_profile')}"
        )
    validation = report.get("validation", {})
    lines.extend(["", "## Validation Summary", ""])
    lines.append(f"- Total samples seen: {validation.get('total_samples_seen')}")
    lines.append(f"- Validated samples: {validation.get('validated_samples')}")
    lines.append(f"- Parse status: `{validation.get('parse_status')}`")
    lines.append(f"- Format rules: `{validation.get('format_rules')}`")
    lines.append(f"- Field coverage: `{validation.get('field_coverage')}`")
    lines.append(f"- Missing required count: {validation.get('missing_required_count')}")
    groups = validation.get("missing_required_groups") or []
    if groups:
        lines.extend(["", "## Missing Required Groups", ""])
        for group in groups[:20]:
            examples = ", ".join(f"`{item}`" for item in group.get("examples", [])[:3])
            lines.append(
                f"- {group.get('count')} x missing `{group.get('field')}` in `{group.get('signature')}`: {examples}"
            )
    lines.append("")
    return "\n".join(lines)


def build_validated_dataset_profile(report: dict[str, Any]) -> dict[str, Any]:
    validation = report.get("validation", {})
    return {
        "schema_version": "saga_validated_dataset_profile_v1",
        "root": report.get("root"),
        "status": "validated_semantic_profile" if report.get("valid") else "validation_failed",
        "profile_level": report.get("profile_level_after_validation"),
        "dataset_config": report.get("dataset_config_path"),
        "format_spec": report.get("format_spec_path"),
        "required_fields": report.get("required_fields", []),
        "field_coverage": validation.get("field_coverage", {}),
        "parse_status": validation.get("parse_status", {}),
        "format_rules": validation.get("format_rules", {}),
        "validated_samples": validation.get("validated_samples"),
        "total_samples_seen": validation.get("total_samples_seen"),
        "missing_required_count": validation.get("missing_required_count"),
        "missing_required_groups": validation.get("missing_required_groups", []),
        "planner_ready": bool(report.get("valid")),
        "next_step": report.get("next_step"),
    }


def normalize_hint_fields(fields_after_anchor: list[Any]) -> list[str]:
    fields: list[str] = []
    for item in fields_after_anchor:
        if isinstance(item, dict):
            field = item.get("normalized") or item.get("field") or item.get("raw")
        else:
            field = str(item)
        if field:
            fields.append(str(field))
    return fields


def anchor_regex(anchor: Any) -> str:
    if isinstance(anchor, (int, float)):
        value = float(anchor)
        if value.is_integer():
            return re.escape(str(int(value))) + r"(?:\.0+)?"
    return re.escape(str(anchor))


def unique_group_name(field: str, used: set[str]) -> str:
    base = re.sub(r"\W+", "_", field).strip("_") or "field"
    if base not in used:
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    return f"{base}_{idx}"


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_")
    return sanitized or "compiled_format"


def count_rule_support(raw_profile: dict[str, Any], pattern: str) -> int:
    relpaths = []
    seen: set[str] = set()
    for sample in raw_profile.get("samples", []):
        relpath = sample.get("relative_path")
        if relpath and relpath not in seen:
            seen.add(relpath)
            relpaths.append(relpath)
    for group in raw_profile.get("path_groups", []):
        for relpath in group.get("examples", []):
            if relpath and relpath not in seen:
                seen.add(relpath)
                relpaths.append(relpath)
    return sum(1 for relpath in relpaths if re.search(pattern, relpath))

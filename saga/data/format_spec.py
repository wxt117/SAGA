from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping
from saga.core.protocol import UNKNOWN


FIELD_ALIASES = {
    "class_name": "class",
    "class_dir": "class",
    "class_directory": "class",
    "parent_class": "class",
    "azimuth": "azimuth_deg",
    "azim": "azimuth_deg",
    "incidence": "incidence_angle_deg",
    "inci": "incidence_angle_deg",
    "elevation": "elevation_angle_deg",
    "elev": "elevation_angle_deg",
    "resolution": "resolution_m",
    "pol": "polarization",
}


@dataclass
class FormatRule:
    name: str
    pattern: str
    target: str = "relative_path"
    fields: dict[str, str] = field(default_factory=dict)
    constants: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DatasetFormatSpec:
    name: str
    version: str = "saga_format_spec_v1"
    description: str = ""
    rules: list[FormatRule] = field(default_factory=list)
    fallback: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "DatasetFormatSpec":
        raw = load_mapping(path)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DatasetFormatSpec":
        if "format_spec" in raw and isinstance(raw["format_spec"], dict):
            raw = raw["format_spec"]
        rules = [
            FormatRule(
                name=rule.get("name", f"rule_{idx}"),
                pattern=rule["pattern"],
                target=rule.get("target", "relative_path"),
                fields=rule.get("fields") or {},
                constants=rule.get("constants") or {},
                confidence=float(rule.get("confidence", 1.0)),
            )
            for idx, rule in enumerate(raw.get("rules") or [])
        ]
        return cls(
            name=raw.get("name", "custom_format"),
            version=raw.get("version", "saga_format_spec_v1"),
            description=raw.get("description", ""),
            rules=rules,
            fallback=raw.get("fallback") or {},
        )


def parse_with_format_spec(path: Path, dataset_root: Path, spec: DatasetFormatSpec) -> dict[str, Any]:
    relpath = path.relative_to(dataset_root).as_posix()
    targets = {
        "relative_path": relpath,
        "filename": path.name,
        "stem": path.stem,
        "parent": path.parent.name,
    }
    for rule in spec.rules:
        value = targets.get(rule.target, relpath)
        match = re.search(rule.pattern, value)
        if not match:
            continue
        metadata: dict[str, Any] = {
            "parse_status": "ok",
            "parser_schema": spec.name,
            "format_rule": rule.name,
            "format_rule_confidence": rule.confidence,
            "original_relpath": relpath,
            "image_ext": path.suffix.lower().lstrip("."),
        }
        groups = match.groupdict()
        for source, dest in rule.fields.items():
            if source in groups and groups[source] is not None:
                normalized_dest = normalize_field_name(dest)
                metadata[normalized_dest] = coerce_metadata_value(normalized_dest, groups[source])
        for key, value in rule.constants.items():
            metadata.setdefault(key, value)
        if "class" not in metadata and spec.fallback.get("class_from_parent", False):
            metadata["class"] = path.parent.name
            metadata["class_source"] = "parent_dir_fallback_after_match"
        return metadata

    return fallback_metadata(path=path, dataset_root=dataset_root, spec=spec)


def fallback_metadata(path: Path, dataset_root: Path, spec: DatasetFormatSpec | None = None) -> dict[str, Any]:
    relpath = path.relative_to(dataset_root).as_posix()
    fallback = spec.fallback if spec else {}
    metadata = {
        "parse_status": "partial",
        "parser_schema": spec.name if spec else "fallback_directory",
        "original_relpath": relpath,
        "image_ext": path.suffix.lower().lstrip("."),
    }
    class_from_parent = fallback.get("class_from_parent", True)
    if class_from_parent:
        metadata["class"] = path.parent.name
        metadata["class_source"] = "parent_dir_fallback"
    return metadata


def coerce_metadata_value(field: str, value: str) -> Any:
    value = value.strip()
    if not value:
        return UNKNOWN
    numeric_fields = {
        "azimuth_deg",
        "incidence_angle_deg",
        "elevation_angle_deg",
        "resolution_m",
        "range_m",
        "sample_index",
        "index",
    }
    if field in numeric_fields or field.endswith("_deg") or field.endswith("_m"):
        try:
            number = float(value)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    return value


def normalize_field_name(field: str) -> str:
    return FIELD_ALIASES.get(field, field)

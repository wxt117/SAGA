from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from saga.core.config import load_mapping, save_text
from saga.data.format_spec import normalize_field_name


PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
NUMERIC_FIELDS = {
    "azimuth_deg",
    "incidence_angle_deg",
    "elevation_angle_deg",
    "resolution_m",
    "range_m",
    "sample_index",
    "frame_index",
    "index",
    "sea_state",
    "param_1",
}


def build_format_spec(
    output_path: str | Path,
    understanding_path: str | Path | None = None,
    inspection_path: str | Path | None = None,
    name: str | None = None,
    include_heuristics: bool = True,
) -> dict[str, Any]:
    understanding = load_mapping(understanding_path) if understanding_path else {}
    if "format_spec" in understanding:
        spec = understanding
    else:
        body = understanding.get("format_understanding", understanding) if understanding else {}
        rules = rules_from_understanding(body)
        if include_heuristics and inspection_path:
            rules = heuristic_rules_from_inspection(load_mapping(inspection_path)) + rules
        rules = dedupe_rules(rules)
        spec = {
            "format_spec": {
                "name": name or body.get("name", "saga_generated_format"),
                "version": "saga_format_spec_v1",
                "description": body.get(
                    "summary",
                    "Generated from SAGA format understanding. Validate before diagnosis.",
                ),
                "rules": rules,
                "fallback": body.get("fallback") or {"class_from_parent": True},
            }
        }
        unresolved = body.get("unresolved")
        if unresolved:
            spec["format_spec"]["unresolved"] = unresolved

    rendered = yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)
    save_text(output_path, rendered)
    return spec


def rules_from_understanding(understanding: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for idx, item in enumerate(understanding.get("rules") or understanding.get("path_groups") or []):
        template = item.get("template") or item.get("path_template")
        if not template:
            continue
        try:
            rule = rule_from_template(
                template=template,
                name=item.get("name") or f"template_rule_{idx + 1}",
                target=item.get("target", "relative_path"),
                constants=item.get("constants") or {},
                confidence=float(item.get("confidence", 0.7)),
            )
        except ValueError:
            continue
        rules.append(rule)
    return rules


def rule_from_template(
    template: str,
    name: str,
    target: str = "relative_path",
    constants: dict[str, Any] | None = None,
    confidence: float = 0.7,
) -> dict[str, Any]:
    fields: dict[str, str] = {}
    pattern_parts: list[str] = []
    last = 0
    used_groups: set[str] = set()
    for match in PLACEHOLDER_RE.finditer(template):
        pattern_parts.append(re.escape(template[last : match.start()]))
        raw_field = match.group(1)
        field = normalize_field_name(raw_field)
        if field == "ext":
            pattern_parts.append(r"[^./]+")
        else:
            group = unique_group_name(field, used_groups)
            used_groups.add(group)
            fields[group] = field
            pattern_parts.append(f"(?P<{group}>{regex_for_field(field)})")
        last = match.end()
    pattern_parts.append(re.escape(template[last:]))
    if not fields:
        raise ValueError(f"Template has no metadata placeholders: {template}")
    constants = normalize_constants(constants or {})
    return {
        "name": sanitize_rule_name(name),
        "target": target,
        "pattern": "^" + "".join(pattern_parts) + "$",
        "fields": fields,
        "constants": constants,
        "confidence": confidence,
    }


def regex_for_field(field: str) -> str:
    if field in NUMERIC_FIELDS or field.endswith("_deg") or field.endswith("_m"):
        return r"-?\d+(?:\.\d+)?"
    if field == "polarization":
        return r"[A-Za-z0-9]+"
    if field == "band":
        return r"[A-Za-z0-9]+"
    return r"[^/]+?"


def unique_group_name(field: str, used: set[str]) -> str:
    base = re.sub(r"\W+", "_", field).strip("_") or "field"
    if base not in used:
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    return f"{base}_{idx}"


def sanitize_rule_name(name: str) -> str:
    value = re.sub(r"\W+", "_", str(name)).strip("_")
    return value or "format_rule"


def normalize_constants(constants: dict[str, Any]) -> dict[str, Any]:
    return {normalize_field_name(str(key)): value for key, value in constants.items()}


def heuristic_rules_from_inspection(inspection: dict[str, Any]) -> list[dict[str, Any]]:
    relpaths = collect_inspected_relpaths(inspection)
    candidates = [
        heuristic_rule(
            name="saga_inci_azim_pol_with_band",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/（(]+)[（(](?P<class_description>.*?)[）)]/"
                r"(?P<band>[^/]+)/inci-(?P<incidence>-?\d+(?:\.\d+)?)-"
                r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)-(?P<polarization>[A-Za-z0-9]+)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "class_description": "class_description",
                "band": "band",
                "incidence": "incidence_angle_deg",
                "azimuth": "azimuth_deg",
                "polarization": "polarization",
            },
        ),
        heuristic_rule(
            name="saga_inci_azim_pol_plain",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/]+)/(?P<band>[^/]+)/"
                r"inci-(?P<incidence>-?\d+(?:\.\d+)?)-azim-(?P<azimuth>-?\d+(?:\.\d+)?)-"
                r"(?P<polarization>[A-Za-z0-9]+)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "band": "band",
                "incidence": "incidence_angle_deg",
                "azimuth": "azimuth_deg",
                "polarization": "polarization",
            },
        ),
        heuristic_rule(
            name="saga_elev_azim_pol",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/]+)/elev-(?P<elevation>-?\d+(?:\.\d+)?)-"
                r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<polarization>hh|hv|vh|vv|pauli)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "elevation": "elevation_angle_deg",
                "azimuth": "azimuth_deg",
                "polarization": "polarization",
            },
            constants={"band": "X"},
        ),
        heuristic_rule(
            name="saga_prefixed_elev_azim_band",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/]+)/(?P<target_name>[^/]*?)"
                r"elev-(?P<elevation>-?\d+(?:\.\d+)?)-"
                r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)-(?P<band>[A-Za-z0-9]+)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "target_name": "target_name",
                "elevation": "elevation_angle_deg",
                "azimuth": "azimuth_deg",
                "band": "band",
            },
        ),
        heuristic_rule(
            name="saga_filename_prefixed_elev_azim_band",
            pattern=(
                r"^(?P<target_name>[^/]*?)elev-(?P<elevation>-?\d+(?:\.\d+)?)-"
                r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)-(?P<band>[A-Za-z0-9]+)\.[^.]+$"
            ),
            fields={
                "target_name": "target_name",
                "elevation": "elevation_angle_deg",
                "azimuth": "azimuth_deg",
                "band": "band",
            },
        ),
        heuristic_rule(
            name="saga_name_attr_azim_pol",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/]+)/(?P<target_name>.+?)[(（](?P<target_attr>.*?)[)）]_"
                r"(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<polarization>hh|hv|vh|vv|pauli)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "target_name": "target_name",
                "target_attr": "target_serial_or_attr",
                "azimuth": "azimuth_deg",
                "polarization": "polarization",
            },
            constants={"band": "X"},
        ),
        heuristic_rule(
            name="saga_name_azim_pol",
            pattern=(
                r"^(?:.+?/)?(?P<class>[^/]+)/(?P<target_name>[^/()（）]+)_"
                r"(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<polarization>hh|hv|vh|vv|pauli)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "target_name": "target_name",
                "azimuth": "azimuth_deg",
                "polarization": "polarization",
            },
            constants={"band": "X"},
        ),
        heuristic_rule(
            name="saga_sea_state_aircraft",
            pattern=(
                r"^(?P<class>[^/]+)/海况(?P<sea_state>\d+)级_(?P<processing>[^_]+)_"
                r"(?P<target_name_cn>.+?)[(（](?P<target_group>.*?)[)）]_"
                r"(?P<param_1>-?\d+(?:\.\d+)?)_(?P<incidence>-?\d+(?:\.\d+)?)_"
                r"(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<resolution>-?\d+(?:\.\d+)?)_"
                r"(?P<band>[A-Za-z0-9]+)\.[^.]+$"
            ),
            fields={
                "class": "class",
                "sea_state": "sea_state",
                "processing": "processing",
                "target_name_cn": "target_name_cn",
                "target_group": "target_group",
                "param_1": "param_1",
                "incidence": "incidence_angle_deg",
                "azimuth": "azimuth_deg",
                "resolution": "resolution_m",
                "band": "band",
            },
        ),
    ]
    rules = []
    for candidate in candidates:
        support = sum(1 for relpath in relpaths if re.search(candidate["pattern"], relpath))
        if support:
            candidate["confidence"] = round(min(0.95, 0.55 + support / max(len(relpaths), 1)), 3)
            candidate["support_in_inspection"] = support
            rules.append(candidate)
    return rules


def heuristic_rule(
    name: str,
    pattern: str,
    fields: dict[str, str],
    constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "target": "relative_path",
        "pattern": pattern,
        "fields": fields,
        "constants": constants or {},
        "confidence": 0.6,
    }


def collect_inspected_relpaths(inspection: dict[str, Any]) -> list[str]:
    relpaths: list[str] = []
    seen: set[str] = set()
    for item in inspection.get("samples", []):
        relpath = item.get("relative_path")
        if relpath and relpath not in seen:
            relpaths.append(relpath)
            seen.add(relpath)
    for group in inspection.get("path_groups", []):
        for relpath in group.get("examples", []):
            if relpath and relpath not in seen:
                relpaths.append(relpath)
                seen.add(relpath)
    return relpaths


def dedupe_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        key = (rule.get("target", "relative_path"), rule.get("pattern", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped

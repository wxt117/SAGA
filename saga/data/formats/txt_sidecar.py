from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from saga.core.protocol import UNKNOWN


KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s*:\s*(.*?)\s*$")


def parse_txt_sidecar(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {"caption": "", "parse_status": "empty_txt", "parser_schema": "txt_sidecar_empty"}

    fields: dict[str, Any] = {}
    has_kv = False
    for line in text.splitlines():
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        has_kv = True
        key, raw_value = match.groups()
        fields[key.strip()] = coerce_value(raw_value.strip())

    if has_kv:
        fields.setdefault("caption", text)
        fields.setdefault("parser_schema", fields.get("schema", "saga_kv_v1"))
        fields["parse_status"] = "ok"
        normalize_field_aliases(fields)
        return fields

    fields = parse_caption(text)
    fields["caption"] = text
    fields["parser_schema"] = "caption_v1"
    fields["parse_status"] = "ok"
    return fields


def coerce_value(value: str) -> Any:
    if value.lower() in {"unknown", "none", "null", ""}:
        return UNKNOWN
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def normalize_field_aliases(fields: dict[str, Any]) -> None:
    aliases = {
        "azimuth": "azimuth_deg",
        "target_azimuth": "azimuth_deg",
        "inci": "incidence_angle_deg",
        "incidence": "incidence_angle_deg",
        "resolution": "resolution_m",
        "pol": "polarization",
    }
    for old, new in aliases.items():
        if old in fields and new not in fields:
            fields[new] = fields[old]


def parse_caption(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    lower = text.lower()

    class_patterns = [
        r"\bclass\s*[:=]\s*([A-Za-z0-9_\-]+)",
        r"\b(aircraft|ship|vehicle|tank|b747|bmp2|d7|btr60|hms)\b",
    ]
    for pattern in class_patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            fields["class"] = match.group(1)
            break

    patterns = {
        "resolution_m": r"([0-9]+(?:\.[0-9]+)?)\s*m\b",
        "azimuth_deg": r"(?:azimuth|azim|aspect|方位角)\s*[-:= ]\s*([0-9]+(?:\.[0-9]+)?)",
        "incidence_angle_deg": r"(?:incidence|inci|入射角)\s*[-:= ]\s*([0-9]+(?:\.[0-9]+)?)",
        "band": r"\b([A-Za-z]+)\s*[- ]?band\b|\b([XCKKu]+)\s*频段\b",
        "polarization": r"\b(HH|HV|VH|VV|AHH|pauli)\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = next(group for group in match.groups() if group is not None)
            fields[key] = coerce_value(value)
    return fields

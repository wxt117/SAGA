from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from saga.core.protocol import DatasetConfig


def resolve_resource_path(path: str | Path) -> Path:
    """Resolve repository-relative configuration paths after wheel installation."""

    candidate = Path(path).expanduser()
    if candidate.exists() or candidate.is_absolute():
        return candidate
    installed = Path(sys.prefix) / "share" / "saga" / candidate
    return installed if installed.exists() else candidate


def load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = resolve_resource_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".json"}:
            return json.load(handle)
        return yaml.safe_load(handle) or {}


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = load_mapping(path)
    if "dataset" in raw and isinstance(raw["dataset"], dict):
        raw = raw["dataset"]
    root = Path(raw.get("root", ".")).expanduser()
    if not root.is_absolute():
        root = (Path(path).parent / root).resolve()
    else:
        root = root.resolve()
    format_spec = raw.get("format_spec")
    format_spec_path = None
    if format_spec:
        format_spec_path = Path(format_spec).expanduser()
        if not format_spec_path.is_absolute():
            format_spec_path = (Path(path).parent / format_spec_path).resolve()
    return DatasetConfig(
        name=raw.get("name") or root.name,
        root=root,
        task=raw.get("task", "classification"),
        format=raw.get("format", "mixed"),
        splits=raw.get("splits") or {"train": "."},
        image=raw.get("image") or {},
        label=raw.get("label") or {},
        format_spec=format_spec_path,
        format_options=raw.get("format_options") or {},
        classes=raw.get("classes") or [],
        normalization=raw.get("normalization") or {},
    )


def load_llm_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_mapping(path)

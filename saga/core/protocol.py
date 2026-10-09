from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


UNKNOWN = "unknown"


@dataclass
class ImageInfo:
    path: str
    width: int | None = None
    height: int | None = None
    domain: str = UNKNOWN


@dataclass
class SagaLabel:
    class_name: str = UNKNOWN
    class_id: int | None = None
    caption: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaSample:
    sample_id: str
    split: str
    image: ImageInfo
    label: SagaLabel
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    name: str
    root: Path
    task: str = "classification"
    format: str = "mixed"
    splits: dict[str, str] = field(default_factory=lambda: {"train": "."})
    image: dict[str, Any] = field(default_factory=dict)
    label: dict[str, Any] = field(default_factory=dict)
    format_spec: Path | None = None
    format_options: dict[str, Any] = field(default_factory=dict)
    classes: list[dict[str, Any]] = field(default_factory=list)
    normalization: dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    type: str
    severity: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Drop keys with null-like values while preserving explicit unknown fields."""
    return {k: v for k, v in value.items() if v is not None and v != [] and v != {}}

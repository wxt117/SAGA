from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from saga.core.protocol import DatasetConfig, ImageInfo, SagaLabel, SagaSample, UNKNOWN
from saga.data.discovery import iter_images, split_roots
from saga.data.format_spec import DatasetFormatSpec, fallback_metadata, parse_with_format_spec
from saga.data.formats.txt_sidecar import parse_txt_sidecar


def load_samples(config: DatasetConfig, read_image_size: bool = True) -> list[SagaSample]:
    samples: list[SagaSample] = []
    seen_paths: set[Path] = set()
    label_source = config.label.get("source", "mixed")
    format_spec = DatasetFormatSpec.from_path(config.format_spec) if config.format_spec else None
    split_paths = split_roots(config.root, config.splits)

    for split, split_root in split_paths.items():
        if not split_root.exists():
            continue
        for image_path in iter_images(split_root):
            resolved = image_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            metadata = parse_sample_metadata(
                image_path=image_path,
                dataset_root=config.root,
                label_source=label_source,
                format_spec=format_spec,
            )
            width, height = read_dimensions(image_path) if read_image_size else (None, None)
            class_name = str(metadata.get("class") or metadata.get("class_name") or UNKNOWN)
            caption = build_caption(class_name=class_name, metadata=metadata)
            sample_id = build_sample_id(split=split, relpath=image_path.relative_to(config.root).as_posix())
            attributes = {
                "target_azimuth_deg": metadata.get("azimuth_deg", UNKNOWN),
                "synthetic": metadata.get("source") == "generated",
            }
            samples.append(
                SagaSample(
                    sample_id=sample_id,
                    split=split,
                    image=ImageInfo(
                        path=image_path.as_posix(),
                        width=width,
                        height=height,
                        domain=config.image.get("domain", UNKNOWN),
                    ),
                    label=SagaLabel(
                        class_name=class_name,
                        class_id=class_id_for(class_name, config.classes),
                        caption=str(metadata.get("caption") or caption),
                        attributes=attributes,
                    ),
                    metadata=metadata,
                    provenance={"source": "user", "synthetic": False},
                )
            )
    return samples


def parse_sample_metadata(
    image_path: Path,
    dataset_root: Path,
    label_source: str,
    format_spec: DatasetFormatSpec | None,
) -> dict[str, Any]:
    txt_path = image_path.with_suffix(".txt")
    metadata: dict[str, Any] = {}
    if label_source in {"txt_sidecar", "mixed"} and txt_path.exists():
        metadata.update(parse_txt_sidecar(txt_path))
        metadata["txt_path"] = txt_path.as_posix()
    if label_source in {"filename", "directory", "mixed"}:
        if format_spec:
            path_metadata = parse_with_format_spec(image_path, dataset_root=dataset_root, spec=format_spec)
        else:
            path_metadata = fallback_metadata(image_path, dataset_root=dataset_root)
        # Sidecar labels win for semantic fields, but path provenance is still useful.
        for key, value in path_metadata.items():
            metadata.setdefault(key, value)
        metadata.setdefault("path_parser_schema", path_metadata.get("parser_schema"))
    if not metadata:
        metadata = fallback_metadata(image_path, dataset_root=dataset_root, spec=format_spec)
    metadata.setdefault("original_relpath", image_path.relative_to(dataset_root).as_posix())
    metadata.setdefault("parse_status", "ok" if metadata.get("class") else "partial")
    return metadata


def read_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def build_sample_id(split: str, relpath: str) -> str:
    digest = hashlib.sha1(relpath.encode("utf-8")).hexdigest()[:12]
    return f"{split}_{digest}"


def class_id_for(class_name: str, classes: list[dict[str, Any]]) -> int | None:
    for item in classes:
        if str(item.get("name")) == class_name:
            return int(item["id"]) if "id" in item else None
    return None


def build_caption(class_name: str, metadata: dict[str, Any]) -> str:
    pieces = ["SAR image"]
    band = metadata.get("band")
    if band and band != UNKNOWN:
        pieces.append(f"{band}-band")
    if class_name and class_name != UNKNOWN:
        pieces.append(str(class_name))
    incidence = metadata.get("incidence_angle_deg")
    if incidence not in {None, UNKNOWN}:
        pieces.append(f"incidence {incidence} deg")
    azimuth = metadata.get("azimuth_deg")
    if azimuth not in {None, UNKNOWN}:
        pieces.append(f"azimuth {azimuth} deg")
    pol = metadata.get("polarization")
    if pol and pol != UNKNOWN:
        pieces.append(f"{pol} polarization")
    resolution = metadata.get("resolution_m")
    if resolution not in {None, UNKNOWN}:
        pieces.append(f"{resolution}m resolution")
    return ", ".join(pieces)

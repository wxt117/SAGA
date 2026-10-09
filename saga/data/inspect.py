from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.data.discovery import iter_images
from saga.data.path_patterns import build_path_groups, stratified_path_sample


def inspect_dataset_format(
    root: str | Path,
    output_dir: str | Path,
    sample_limit: int = 120,
    txt_preview_chars: int = 300,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir)
    image_paths = iter_images(root_path)
    sampled = stratified_path_sample(image_paths, root=root_path, limit=sample_limit)

    suffix_counts = Counter(path.suffix.lower() for path in image_paths)
    depth_counts = Counter(len(path.relative_to(root_path).parts) for path in image_paths)
    parent_counts = Counter(path.parent.relative_to(root_path).as_posix() for path in image_paths)
    path_groups = build_path_groups(image_paths, root=root_path)

    samples: list[dict[str, Any]] = []
    for path in sampled:
        relpath = path.relative_to(root_path).as_posix()
        txt_path = path.with_suffix(".txt")
        item: dict[str, Any] = {
            "relative_path": relpath,
            "filename": path.name,
            "stem": path.stem,
            "suffix": path.suffix.lower(),
            "parent": path.parent.relative_to(root_path).as_posix(),
            "depth": len(path.relative_to(root_path).parts),
            "has_txt_sidecar": txt_path.exists(),
        }
        if txt_path.exists():
            item["txt_preview"] = txt_path.read_text(encoding="utf-8", errors="replace")[:txt_preview_chars]
        samples.append(item)

    report = {
        "root": root_path.as_posix(),
        "total_images": len(image_paths),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "depth_counts": dict(sorted(depth_counts.items())),
        "top_parent_dirs": dict(parent_counts.most_common(50)),
        "path_groups": path_groups,
        "samples": samples,
        "notes": [
            "This is only a format inspection report, not a dataset diagnosis.",
            "Use it with the user's natural-language description to create a DatasetFormatSpec.",
            "Samples are stratified by structural path signatures to expose mixed dataset layouts.",
        ],
    }
    save_json(output_path / "format_inspection.json", report)
    save_text(output_path / "format_inspection.md", render_inspection_markdown(report))
    return report


def spread_sample(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    if limit <= 0:
        return []
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def render_inspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Dataset Format Inspection",
        "",
        f"- Root: {report['root']}",
        f"- Total images: {report['total_images']}",
        f"- Suffixes: {report['suffix_counts']}",
        f"- Depths: {report['depth_counts']}",
        "",
        "## Top Parent Directories",
        "",
    ]
    for parent, count in report["top_parent_dirs"].items():
        lines.append(f"- {parent}: {count}")
    lines.extend(["", "## Path Pattern Groups", ""])
    for group in report.get("path_groups", [])[:40]:
        examples = ", ".join(f"`{item}`" for item in group.get("examples", [])[:3])
        lines.append(f"- {group['count']} x `{group['signature']}`: {examples}")
    lines.extend(["", "## Sample Paths", ""])
    for item in report["samples"][:80]:
        txt_note = " + txt" if item["has_txt_sidecar"] else ""
        lines.append(f"- `{item['relative_path']}`{txt_note}")
    lines.append("")
    return "\n".join(lines)
def build_format_spec_prompt(
    inspection: dict[str, Any],
    user_description: str = "",
    sample_limit: int = 40,
    txt_preview_limit: int = 10,
) -> str:
    compact = {
        "total_images": inspection.get("total_images"),
        "suffix_counts": inspection.get("suffix_counts"),
        "depth_counts": inspection.get("depth_counts"),
        "path_groups": inspection.get("path_groups", [])[: min(sample_limit, 40)],
        "sample_relative_paths": [
            item.get("relative_path") for item in inspection.get("samples", [])[:sample_limit]
        ],
        "txt_previews": [
            {
                "relative_path": item.get("relative_path"),
                "txt_preview": item.get("txt_preview"),
            }
            for item in inspection.get("samples", [])[:txt_preview_limit]
            if item.get("txt_preview")
        ],
    }
    return (
        "You are configuring SAGA dataset ingestion. "
        "Given the user's dataset description and sampled paths, infer a DatasetFormatSpec. "
        "Return only JSON. Use Python regular expressions with named groups. "
        "Do not infer fields that are neither visible in the paths/text nor stated by the user.\n\n"
        f"User description:\n{user_description or '(none)'}\n\n"
        f"Compact inspection JSON:\n{compact}\n\n"
        "Return a JSON object with this exact top-level key:\n"
        "{\n"
        '  "format_spec": {\n'
        '    "name": "custom_dataset_format",\n'
        '    "version": "saga_format_spec_v1",\n'
        '    "description": "short description",\n'
        '    "rules": [\n'
        "      {\n"
        '        "name": "rule_name",\n'
        '        "target": "relative_path",\n'
        '        "pattern": "python_regex_with_named_groups",\n'
        '        "fields": {"named_group": "normalized_field_name"},\n'
        '        "constants": {"optional_field": "value"}\n'
        "      }\n"
        "    ],\n"
        '    "fallback": {"class_from_parent": true}\n'
        "  }\n"
        "}\n"
    )

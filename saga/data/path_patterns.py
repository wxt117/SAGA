from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any


POLARIZATION_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:ahh|ahv|avh|avv|hh|hv|vh|vv|pauli)(?![A-Za-z0-9])"
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")


def path_signature(relpath: str) -> str:
    """Create a compact structural signature for grouping similar paths."""
    parts = []
    for part in relpath.split("/"):
        parts.append(component_signature(part))
    return "/".join(parts)


def component_signature(component: str) -> str:
    suffix = ""
    stem = component
    if "." in component and not component.startswith("."):
        stem, suffix = component.rsplit(".", 1)
        suffix = "." + suffix.lower()
    stem = POLARIZATION_PATTERN.sub("<pol>", stem)
    stem = NUMBER_PATTERN.sub("<num>", stem)
    return stem + suffix


def build_path_groups(
    paths: list[Path],
    root: Path,
    max_groups: int = 80,
    examples_per_group: int = 5,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        relpath = path.relative_to(root).as_posix()
        grouped[path_signature(relpath)].append(path)

    groups: list[dict[str, Any]] = []
    for signature, group_paths in grouped.items():
        examples = [path.relative_to(root).as_posix() for path in spread_sample(group_paths, examples_per_group)]
        suffix_counts: dict[str, int] = {}
        depth_counts: dict[str, int] = {}
        parent_examples: list[str] = []
        seen_parents: set[str] = set()
        for path in group_paths:
            suffix = path.suffix.lower()
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            depth = str(len(path.relative_to(root).parts))
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            parent = path.parent.relative_to(root).as_posix()
            if parent not in seen_parents and len(parent_examples) < examples_per_group:
                seen_parents.add(parent)
                parent_examples.append(parent)
        groups.append(
            {
                "signature": signature,
                "count": len(group_paths),
                "depth_counts": dict(sorted(depth_counts.items())),
                "suffix_counts": dict(sorted(suffix_counts.items())),
                "parent_examples": parent_examples,
                "examples": examples,
            }
        )

    return sorted(groups, key=lambda item: (-int(item["count"]), item["signature"]))[:max_groups]


def stratified_path_sample(paths: list[Path], root: Path, limit: int) -> list[Path]:
    if limit <= 0:
        return []
    if len(paths) <= limit:
        return paths

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path_signature(path.relative_to(root).as_posix())].append(path)

    selected: list[Path] = []
    seen: set[Path] = set()
    groups = sorted(grouped.values(), key=lambda group: (-len(group), group[0].as_posix()))

    # First take the start and end of each structural group, which helps expose
    # rare layouts without flooding the prompt with adjacent filenames.
    for group in groups:
        for path in spread_sample(group, 2):
            if path not in seen:
                selected.append(path)
                seen.add(path)
                if len(selected) >= limit:
                    return selected

    # Then fill remaining slots with a global spread sample.
    for path in spread_sample(paths, limit):
        if path not in seen:
            selected.append(path)
            seen.add(path)
            if len(selected) >= limit:
                return selected
    return selected


def spread_sample(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    if limit <= 0:
        return []
    if limit == 1:
        return [paths[0]]
    step = (len(paths) - 1) / (limit - 1)
    return [paths[round(i * step)] for i in range(limit)]

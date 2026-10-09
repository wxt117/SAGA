from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.data.profile import profile_dataset


MULTI_DATASET_PROFILE_VERSION = "saga_multi_dataset_profile_v1"


def build_multi_dataset_profile(
    *,
    intent_spec: dict[str, Any],
    primary_raw_profile: dict[str, Any],
    primary_dataset_profile: dict[str, Any],
    output_dir: str | Path,
    request: str,
    format_hints: list[dict[str, Any]],
    sample_limit: int = 80,
    image_probe_limit: int = 120,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    profile_dir = output_path / "multi_dataset_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    roles = collect_dataset_roles(intent_spec)
    datasets: dict[str, Any] = {
        "primary": summarize_profile(
            role="primary",
            root=primary_raw_profile.get("root"),
            raw_profile=primary_raw_profile,
            dataset_profile=primary_dataset_profile,
            artifact_dir=output_path,
        )
    }
    primary_root = Path(str(primary_raw_profile.get("root"))).expanduser().resolve().as_posix()
    for role, raw_path in roles.items():
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        if not path.exists() or not path.is_dir():
            datasets[role] = {
                "role": role,
                "root": str(raw_path),
                "status": "unprofiled",
                "reason": "path_missing_or_not_directory",
            }
            continue
        resolved = path.resolve().as_posix()
        if resolved == primary_root:
            datasets[role] = {
                **datasets["primary"],
                "role": role,
                "alias_of": "primary",
            }
            continue
        role_dir = profile_dir / safe_role_name(role)
        report = profile_dataset(
            root=path,
            output_dir=role_dir,
            request=request,
            intent_spec=intent_spec,
            format_hints=format_hints,
            sample_limit=sample_limit,
            image_probe_limit=image_probe_limit,
        )
        raw_profile = load_json(role_dir / "raw_profile.json")
        datasets[role] = summarize_profile(
            role=role,
            root=resolved,
            raw_profile=raw_profile,
            dataset_profile=report,
            artifact_dir=role_dir,
        )

    alignment = build_alignment_report(datasets)
    report = {
        "schema_version": MULTI_DATASET_PROFILE_VERSION,
        "dataset_count": len(datasets),
        "datasets": datasets,
        "alignment": alignment,
        "notes": [
            "Primary dataset is the root used for DatasetFormatSpec compilation.",
            "Auxiliary datasets are raw-profiled for planning and alignment checks; they do not become trusted semantic datasets until validated.",
        ],
    }
    save_json(output_path / "multi_dataset_profile.json", report)
    save_text(output_path / "multi_dataset_profile.md", render_multi_dataset_profile_markdown(report))
    return report


def collect_dataset_roles(intent_spec: dict[str, Any]) -> dict[str, str]:
    intent = intent_spec.get("intent") or {}
    role_keys = {
        "baseline": "baseline_dataset",
        "augmented": "augmented_dataset",
        "validation": "val_dataset",
        "dataset_source": "dataset_source",
        "content": "content_source",
        "style": "style_source",
    }
    roles: dict[str, str] = {}
    for role, key in role_keys.items():
        value = intent.get(key)
        if value:
            roles[role] = str(value)
    return roles


def summarize_profile(
    *,
    role: str,
    root: str | None,
    raw_profile: dict[str, Any],
    dataset_profile: dict[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    images = raw_profile.get("images") or {}
    probe = images.get("probe_summary") or {}
    filesystem = raw_profile.get("filesystem") or {}
    return {
        "role": role,
        "status": "profiled",
        "root": root,
        "artifact_dir": Path(artifact_dir).expanduser().resolve().as_posix(),
        "num_images": images.get("total_images", 0),
        "suffix_counts": images.get("suffix_counts", {}),
        "image_modes": probe.get("mode_counts", {}),
        "image_sizes": probe.get("size_counts", {}),
        "total_files": filesystem.get("total_files"),
        "path_group_count": len(raw_profile.get("path_groups") or []),
        "task_candidates": ((dataset_profile.get("dataset_profile") or {}).get("task_candidates") or [])[:8],
    }


def build_alignment_report(datasets: dict[str, Any]) -> dict[str, Any]:
    issues = []
    primary = datasets.get("primary") or {}
    primary_suffixes = set((primary.get("suffix_counts") or {}).keys())
    primary_sizes = set((primary.get("image_sizes") or {}).keys())
    for role, summary in datasets.items():
        if role == "primary" or summary.get("status") != "profiled":
            continue
        suffixes = set((summary.get("suffix_counts") or {}).keys())
        sizes = set((summary.get("image_sizes") or {}).keys())
        if primary_suffixes and suffixes and not (primary_suffixes & suffixes):
            issues.append(
                {
                    "severity": "medium",
                    "name": "image_suffix_mismatch",
                    "role": role,
                    "primary_suffixes": sorted(primary_suffixes),
                    "role_suffixes": sorted(suffixes),
                }
            )
        if primary_sizes and sizes and not (primary_sizes & sizes):
            issues.append(
                {
                    "severity": "low",
                    "name": "image_size_mismatch",
                    "role": role,
                    "primary_sizes": sorted(primary_sizes)[:10],
                    "role_sizes": sorted(sizes)[:10],
                }
            )
        if int(summary.get("num_images") or 0) == 0:
            issues.append({"severity": "high", "name": "empty_auxiliary_dataset", "role": role})
    return {
        "status": "passed" if not [item for item in issues if item.get("severity") == "high"] else "warning",
        "issue_count": len(issues),
        "issues": issues,
    }


def render_multi_dataset_profile_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Multi-Dataset Profile",
        "",
        f"- Dataset count: {report.get('dataset_count')}",
        f"- Alignment status: `{(report.get('alignment') or {}).get('status')}`",
        "",
        "## Datasets",
        "",
    ]
    for role, summary in (report.get("datasets") or {}).items():
        lines.extend(
            [
                f"### {role}",
                "",
                f"- Status: `{summary.get('status')}`",
                f"- Root: `{summary.get('root')}`",
                f"- Images: {summary.get('num_images')}",
                f"- Suffixes: `{summary.get('suffix_counts')}`",
                f"- Sizes: `{summary.get('image_sizes')}`",
                "",
            ]
        )
    lines.extend(["## Alignment Issues", ""])
    issues = (report.get("alignment") or {}).get("issues") or []
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('name')}` ({issue.get('severity')}): `{issue}`")
    lines.append("")
    return "\n".join(lines)


def safe_role_name(role: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in role) or "dataset"


def load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

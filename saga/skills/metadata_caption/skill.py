from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from saga.core.config import load_dataset_config, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.core.protocol import UNKNOWN
from saga.data.discovery import iter_images
from saga.data.format_spec import fallback_metadata
from saga.data.loader import build_caption, load_samples


METADATA_CAPTION_VERSION = "saga_metadata_caption_run_v1"


def run_metadata_caption_skill(
    dataset_root: str | Path,
    output_dir: str | Path,
    dataset_config: str | Path | None = None,
    template: str | None = None,
    mode: str = "copy",
    overwrite: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    staged = output / "captioned_dataset"
    rows = build_caption_rows(root=root, dataset_config=dataset_config, template=template)
    if dry_run:
        report = build_report(
            status="dry_run",
            message="Metadata caption staging plan generated.",
            root=root,
            output=output,
            staged=staged,
            rows=rows,
            materialized_count=0,
            dry_run=True,
            mode=mode,
            started=started,
        )
        write_artifacts(output, report, rows)
        return report

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)
    materialized = []
    errors = []
    for row in rows:
        try:
            src = Path(row["source_path"])
            rel = Path(row["relative_path"])
            dst = staged / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                dst.symlink_to(src)
            elif mode == "copy":
                shutil.copy2(src, dst)
            elif mode != "manifest_only":
                raise ValueError(f"Unsupported caption staging mode: {mode}")
            txt = dst.with_suffix(".txt")
            if overwrite or not txt.exists():
                txt.write_text(str(row["caption"]).strip() + "\n", encoding="utf-8")
            done = dict(row)
            done["output_path"] = dst.as_posix()
            done["caption_path"] = txt.as_posix()
            done["status"] = "materialized"
            materialized.append(done)
        except Exception as exc:
            errors.append({"source_path": row["source_path"], "error": f"{type(exc).__name__}: {exc}"})
    status = "succeeded" if not errors else ("warning" if materialized else "failed")
    report = build_report(
        status=status,
        message="Metadata captions materialized.",
        root=root,
        output=output,
        staged=staged,
        rows=materialized,
        materialized_count=len(materialized),
        dry_run=False,
        mode=mode,
        started=started,
        errors=errors,
    )
    write_artifacts(output, report, materialized)
    return report


def build_caption_rows(root: Path, dataset_config: str | Path | None, template: str | None) -> list[dict[str, Any]]:
    rows = []
    if dataset_config:
        config = load_dataset_config(dataset_config)
        samples = load_samples(config, read_image_size=False)
        for sample in samples:
            path = Path(sample.image.path).expanduser().resolve()
            caption = render_caption(template, sample.metadata) if template else sample.label.caption
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_path": path.as_posix(),
                    "relative_path": path.relative_to(config.root).as_posix(),
                    "caption": caption,
                    "metadata": compact_metadata(sample.metadata),
                    "status": "planned",
                }
            )
        return rows
    for path in iter_images(root):
        metadata = fallback_metadata(path, dataset_root=root)
        class_name = str(metadata.get("class") or metadata.get("class_name") or path.parent.name or UNKNOWN)
        caption = render_caption(template, metadata) if template else build_caption(class_name=class_name, metadata=metadata)
        rows.append(
            {
                "sample_id": path.relative_to(root).as_posix(),
                "source_path": path.as_posix(),
                "relative_path": path.relative_to(root).as_posix(),
                "caption": caption,
                "metadata": compact_metadata(metadata),
                "status": "planned",
            }
        )
    return rows


def render_caption(template: str, metadata: dict[str, Any]) -> str:
    safe = {key: value for key, value in metadata.items() if isinstance(key, str)}
    safe.setdefault("class", metadata.get("class") or metadata.get("class_name") or "target")
    try:
        return template.format(**safe)
    except KeyError:
        return build_caption(class_name=str(safe.get("class") or "target"), metadata=metadata)


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = ("class", "class_name", "azimuth_deg", "depression_angle", "incidence_angle_deg", "polarization", "band", "resolution_m", "resolution")
    return {key: metadata.get(key) for key in keys if metadata.get(key) not in (None, "", UNKNOWN)}


def build_report(
    *,
    status: str,
    message: str,
    root: Path,
    output: Path,
    staged: Path,
    rows: list[dict[str, Any]],
    materialized_count: int,
    dry_run: bool,
    mode: str,
    started: float,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": METADATA_CAPTION_VERSION,
        "skill": "MetadataCaptionSkill",
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "dataset_root": root.as_posix(),
        "output_dir": output.as_posix(),
        "captioned_dataset_dir": staged.as_posix(),
        "mode": mode,
        "planned_count": len(rows),
        "materialized_count": materialized_count,
        "errors": errors or [],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "metadata_caption_report.json").as_posix(),
            "report_md": (output / "metadata_caption_report.md").as_posix(),
            "manifest_jsonl": (output / "caption_manifest.jsonl").as_posix(),
        },
    }


def write_artifacts(output: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    save_json(output / "metadata_caption_report.json", report)
    save_text(output / "metadata_caption_report.md", render_report(report))
    write_jsonl(output / "caption_manifest.jsonl", rows)


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Metadata Caption Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Dataset: `{report.get('dataset_root')}`",
            f"- Captioned dataset: `{report.get('captioned_dataset_dir')}`",
            f"- Planned captions: {report.get('planned_count')}",
            f"- Materialized: {report.get('materialized_count')}",
            "",
        ]
    )

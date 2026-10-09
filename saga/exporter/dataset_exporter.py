from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.data.discovery import iter_images


EXPORT_REPORT_VERSION = "saga_export_report_v1"


def export_augmented_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    recipe_id: str,
    task: str,
    dry_run: bool = False,
    mode: str = "copy",
    selection_manifest: str | Path | None = None,
    quality_report: dict[str, Any] | None = None,
    repair_policy: dict[str, Any] | None = None,
    include_originals: bool = False,
    require_quality_pass: bool = False,
    max_items: int | None = None,
    write_caption_sidecars: bool = False,
) -> dict[str, Any]:
    """Package skill outputs as a traceable augmented dataset artifact."""

    started = time.time()
    source_path = Path(source_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    images = iter_images(source_path) if source_path.exists() and source_path.is_dir() else []
    if max_items is not None and max_items > 0:
        images = images[:max_items]

    quality_status = (quality_report or {}).get("status")
    quality_triggers = (quality_report or {}).get("triggers") or []
    blocked = bool(require_quality_pass and quality_status not in {None, "passed", "dry_run"})
    if quality_triggers and require_quality_pass:
        blocked = True

    plan = {
        "source_dir": source_path.as_posix(),
        "output_dir": output_path.as_posix(),
        "mode": mode,
        "image_count": len(images),
        "include_originals": include_originals,
        "require_quality_pass": require_quality_pass,
        "quality_status": quality_status,
        "quality_trigger_count": len(quality_triggers),
        "write_caption_sidecars": write_caption_sidecars,
    }

    if dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        report = build_report(
            recipe_id=recipe_id,
            task=task,
            source_dir=source_path,
            output_dir=output_path,
            images=images,
            rows=[],
            mode=mode,
            dry_run=True,
            status="dry_run",
            message="Would export augmented dataset.",
            plan=plan,
            selection_manifest_path=selection_manifest,
            quality_report=quality_report,
            repair_policy=repair_policy,
            elapsed_seconds=time.time() - started,
        )
        write_export_artifacts(output_path, report, rows=[])
        return report

    if blocked:
        output_path.mkdir(parents=True, exist_ok=True)
        report = build_report(
            recipe_id=recipe_id,
            task=task,
            source_dir=source_path,
            output_dir=output_path,
            images=images,
            rows=[],
            mode=mode,
            dry_run=False,
            status="blocked",
            message="Export blocked by quality gate.",
            plan=plan,
            selection_manifest_path=selection_manifest,
            quality_report=quality_report,
            repair_policy=repair_policy,
            elapsed_seconds=time.time() - started,
        )
        write_export_artifacts(output_path, report, rows=[])
        return report

    if mode not in {"copy", "symlink", "manifest_only"}:
        raise ValueError(f"Unsupported export mode: {mode}")

    image_output_dir = output_path / "images"
    if mode in {"copy", "symlink"}:
        if image_output_dir.exists():
            shutil.rmtree(image_output_dir)
        image_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    rows = build_manifest_rows(
        images=images,
        source_dir=source_path,
        output_dir=output_path,
        mode=mode,
        recipe_id=recipe_id,
        task=task,
    )
    if mode in {"copy", "symlink"}:
        materialize_rows(rows, mode=mode)

    if include_originals:
        append_original_rows(rows, selection_manifest=selection_manifest, output_dir=output_path, recipe_id=recipe_id)

    caption_sidecar_count = 0
    if write_caption_sidecars:
        caption_sidecar_count = materialize_caption_sidecars(rows=rows, output_dir=output_path, mode=mode)

    status = "succeeded"
    message = "Exported augmented dataset."
    if not images:
        status = "warning"
        message = "No generated images were exported."

    report = build_report(
        recipe_id=recipe_id,
        task=task,
        source_dir=source_path,
        output_dir=output_path,
        images=images,
        rows=rows,
        mode=mode,
        dry_run=False,
        status=status,
        message=message,
        plan=plan,
        selection_manifest_path=selection_manifest,
        quality_report=quality_report,
        repair_policy=repair_policy,
        caption_sidecar_count=caption_sidecar_count,
        elapsed_seconds=time.time() - started,
    )
    write_export_artifacts(output_path, report, rows=rows)
    return report


def build_manifest_rows(
    images: list[Path],
    source_dir: Path,
    output_dir: Path,
    mode: str,
    recipe_id: str,
    task: str,
) -> list[dict[str, Any]]:
    rows = []
    used_names: set[str] = set()
    for idx, source in enumerate(images):
        relpath = safe_relative(source, source_dir)
        target_relpath = unique_export_relpath(relpath, used_names)
        target_path = output_dir / "images" / target_relpath
        caption = read_sidecar_caption(source)
        caption_metadata = metadata_from_caption(caption)
        rows.append(
            {
                "sample_id": build_sample_id(recipe_id=recipe_id, relpath=relpath, index=idx),
                "split": "train",
                "task": task,
                "image": {
                    "path": target_path.as_posix() if mode in {"copy", "symlink"} else source.as_posix(),
                    "relative_path": f"images/{target_relpath}" if mode in {"copy", "symlink"} else relpath,
                    "source_path": source.as_posix(),
                },
                "label": {
                    "class_name": infer_class_from_caption(caption) or infer_class_from_generated_name(source),
                    "caption": caption,
                    "attributes": caption_metadata,
                },
                "metadata": {
                    "augmentation": augmentation_type_for_task(task),
                    "generated": True,
                    "export_mode": mode,
                    "source_relative_path": relpath,
                    **caption_metadata,
                },
                "provenance": {
                    "source": "saga_skill_output",
                    "recipe_id": recipe_id,
                    "synthetic": True,
                },
                "_target_path": target_path.as_posix(),
            }
        )
    return rows


def read_sidecar_caption(image_path: Path) -> str:
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.exists():
        return ""
    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def infer_class_from_caption(caption: str) -> str | None:
    for token in caption_tokens(caption):
        lowered = token.lower()
        if lowered == "sar image":
            continue
        if lowered.endswith("polarization"):
            continue
        if lowered.endswith("-band"):
            continue
        if lowered.startswith(("azimuth ", "incidence ", "resolution ")):
            continue
        return token
    return None


def metadata_from_caption(caption: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for token in caption_tokens(caption):
        lowered = token.lower()
        if lowered.endswith(" polarization"):
            metadata["polarization"] = lowered[: -len(" polarization")].strip()
        elif lowered.endswith("-band"):
            metadata["band"] = token[: -len("-band")]
        elif lowered.startswith("azimuth ") and lowered.endswith(" deg"):
            metadata["azimuth_deg"] = coerce_number(token[len("azimuth ") : -len(" deg")])
        elif lowered.startswith("incidence ") and lowered.endswith(" deg"):
            metadata["incidence_angle_deg"] = coerce_number(token[len("incidence ") : -len(" deg")])
        elif lowered.endswith("m resolution"):
            metadata["resolution_m"] = coerce_number(token[: -len("m resolution")])
    return metadata


def caption_tokens(caption: str) -> list[str]:
    return [token.strip() for token in str(caption or "").split(",") if token.strip()]


def coerce_number(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def augmentation_type_for_task(task: str) -> str:
    if task == "style_transfer":
        return "style_transfer"
    if task == "diffusion_lora_generation":
        return "diffusion_lora_generation"
    return task or "augmentation"


def materialize_rows(rows: list[dict[str, Any]], mode: str) -> None:
    for row in rows:
        source = Path(row["image"]["source_path"])
        target = Path(row["_target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "symlink":
            target.symlink_to(source.resolve())


def materialize_caption_sidecars(rows: list[dict[str, Any]], output_dir: Path, mode: str) -> int:
    if mode not in {"copy", "symlink"}:
        return 0
    count = 0
    for row in rows:
        image = row.get("image") if isinstance(row.get("image"), dict) else {}
        label = row.get("label") if isinstance(row.get("label"), dict) else {}
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        caption = str(label.get("caption") or "").strip()
        if not caption:
            continue
        target_raw = image.get("path")
        if not target_raw:
            continue
        target = Path(str(target_raw))
        sidecar = target.with_suffix(".txt")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        save_text(sidecar, caption + "\n")
        relative_sidecar = safe_relative(sidecar, output_dir)
        image["caption_path"] = sidecar.as_posix()
        image["caption_relative_path"] = relative_sidecar
        label["caption_sidecar_path"] = sidecar.as_posix()
        label["caption_sidecar_relative_path"] = relative_sidecar
        metadata["caption_sidecar"] = relative_sidecar
        row["image"] = image
        row["label"] = label
        row["metadata"] = metadata
        count += 1
    return count


def append_original_rows(
    rows: list[dict[str, Any]],
    selection_manifest: str | Path | None,
    output_dir: Path,
    recipe_id: str,
) -> None:
    if not selection_manifest:
        return
    manifest_path = Path(selection_manifest).expanduser().resolve()
    if not manifest_path.exists():
        return
    manifest = load_mapping(manifest_path)
    sources = manifest.get("sources") or manifest.get("staged") or []
    originals_dir = output_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    used_names = {Path(row["image"]["relative_path"]).name for row in rows}
    for idx, source_raw in enumerate(sources):
        source = Path(str(source_raw)).expanduser().resolve()
        if not source.exists():
            continue
        target_name = unique_name(source.name, used_names)
        target = originals_dir / target_name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
        rows.append(
            {
                "sample_id": build_sample_id(recipe_id=recipe_id, relpath=source.as_posix(), index=idx, prefix="original"),
                "split": "train",
                "task": "classification",
                "image": {
                    "path": target.as_posix(),
                    "relative_path": f"originals/{target_name}",
                    "source_path": source.as_posix(),
                },
                "label": {
                    "class_name": source.parent.name,
                    "caption": "",
                    "attributes": {},
                },
                "metadata": {
                    "augmentation": "none",
                    "generated": False,
                },
                "provenance": {
                    "source": "user_selected_content",
                    "recipe_id": recipe_id,
                    "synthetic": False,
                },
            }
        )


def build_report(
    recipe_id: str,
    task: str,
    source_dir: Path,
    output_dir: Path,
    images: list[Path],
    rows: list[dict[str, Any]],
    mode: str,
    dry_run: bool,
    status: str,
    message: str,
    plan: dict[str, Any],
    selection_manifest_path: str | Path | None,
    quality_report: dict[str, Any] | None,
    repair_policy: dict[str, Any] | None,
    elapsed_seconds: float,
    caption_sidecar_count: int = 0,
) -> dict[str, Any]:
    quality_report = quality_report or {}
    repair_policy = repair_policy or {}
    return {
        "schema_version": EXPORT_REPORT_VERSION,
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "recipe_id": recipe_id,
        "task": task,
        "source_dir": source_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "mode": mode,
        "generated_image_count": len(images),
        "manifest_count": len(rows),
        "caption_sidecar_count": caption_sidecar_count,
        "plan": plan,
        "artifacts": {
            "images_dir": (output_dir / "images").as_posix(),
            "manifest_jsonl": (output_dir / "manifest.jsonl").as_posix(),
            "dataset_card_json": (output_dir / "dataset_card.json").as_posix(),
            "provenance_json": (output_dir / "provenance.json").as_posix(),
            "export_report_json": (output_dir / "export_report.json").as_posix(),
            "export_report_md": (output_dir / "export_report.md").as_posix(),
        },
        "quality_gate": {
            "status": quality_report.get("status"),
            "trigger_count": len(quality_report.get("triggers") or []),
            "triggers": quality_report.get("triggers") or [],
        },
        "repair_policy": {
            "status": repair_policy.get("status"),
            "suggested_action_count": len(repair_policy.get("suggested_actions") or []),
        },
        "selection_manifest": Path(selection_manifest_path).expanduser().resolve().as_posix()
        if selection_manifest_path
        else None,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "examples": [path.as_posix() for path in images[:20]],
    }


def write_export_artifacts(output_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    cleaned_rows = [strip_internal_fields(row) for row in rows]
    save_json(output_dir / "export_report.json", report)
    save_text(output_dir / "export_report.md", render_export_markdown(report))
    write_jsonl(output_dir / "manifest.jsonl", cleaned_rows)
    save_json(output_dir / "dataset_card.json", build_dataset_card(report))
    save_json(output_dir / "provenance.json", build_provenance(report))


def build_dataset_card(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"{report.get('recipe_id')}_augmented",
        "task": report.get("task"),
        "created_by": "SAGA",
        "schema_version": "saga_augmented_dataset_card_v1",
        "source_dir": report.get("source_dir"),
        "generated_image_count": report.get("generated_image_count"),
        "manifest_count": report.get("manifest_count"),
        "caption_sidecar_count": report.get("caption_sidecar_count"),
        "quality_gate": report.get("quality_gate"),
        "recommended_use": [
            "Inspect quality_evaluation artifacts before downstream training.",
            "Treat this export as a traceable candidate dataset, not an automatically validated performance improvement.",
        ],
    }


def build_provenance(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "saga_export_provenance_v1",
        "recipe_id": report.get("recipe_id"),
        "task": report.get("task"),
        "source_dir": report.get("source_dir"),
        "output_dir": report.get("output_dir"),
        "mode": report.get("mode"),
        "quality_gate": report.get("quality_gate"),
        "repair_policy": report.get("repair_policy"),
        "selection_manifest": report.get("selection_manifest"),
        "caption_sidecar_count": report.get("caption_sidecar_count"),
    }


def render_export_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Dataset Export",
        "",
        f"- Status: {report.get('status')}",
        f"- Message: {report.get('message')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Recipe: `{report.get('recipe_id')}`",
        f"- Task: `{report.get('task')}`",
        f"- Source dir: `{report.get('source_dir')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Mode: {report.get('mode')}",
        f"- Generated images: {report.get('generated_image_count')}",
        f"- Manifest rows: {report.get('manifest_count')}",
        f"- Caption sidecars: {report.get('caption_sidecar_count')}",
        "",
        "## Quality Gate",
        "",
        f"- Status: {report.get('quality_gate', {}).get('status')}",
        f"- Trigger count: {report.get('quality_gate', {}).get('trigger_count')}",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in (report.get("artifacts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def unique_export_relpath(relpath: str, used_names: set[str]) -> str:
    path = Path(relpath)
    parts = list(path.parts)
    if not parts:
        parts = [path.name]
    candidate = Path(*parts).as_posix()
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    stem = path.stem
    suffix = path.suffix
    parent = path.parent if str(path.parent) != "." else Path("")
    idx = 2
    while True:
        candidate_path = parent / f"{stem}_{idx}{suffix}"
        candidate = candidate_path.as_posix()
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    path = Path(name)
    idx = 2
    while f"{path.stem}_{idx}{path.suffix}" in used:
        idx += 1
    value = f"{path.stem}_{idx}{path.suffix}"
    used.add(value)
    return value


def build_sample_id(recipe_id: str, relpath: str, index: int, prefix: str = "aug") -> str:
    digest = hashlib.sha1(f"{recipe_id}|{relpath}|{index}".encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def infer_class_from_generated_name(path: Path) -> str:
    stem = path.stem
    if "_styliedby_" in stem:
        return stem.split("_styliedby_", 1)[0]
    if "_styledby_" in stem:
        return stem.split("_styledby_", 1)[0]
    return path.parent.name

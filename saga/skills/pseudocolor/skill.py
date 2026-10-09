from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from saga.core.config import save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.data.discovery import iter_images


PSEUDOCOLOR_VERSION = "saga_pseudocolor_run_v1"


COLORMAPS = {
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "hot": [(0, 0, 0), (180, 0, 0), (255, 120, 0), (255, 255, 128), (255, 255, 255)],
    "ocean": [(0, 0, 30), (0, 80, 120), (0, 160, 180), (120, 220, 210), (255, 255, 255)],
    "sar": [(5, 8, 24), (38, 70, 83), (42, 157, 143), (233, 196, 106), (244, 162, 97)],
}


def run_pseudocolor_skill(
    input_dir: str | Path,
    output_dir: str | Path,
    colormap: str = "viridis",
    preserve_tree: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    root = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    pseudo_dir = output / "pseudocolor_images"
    images = iter_images(root) if root.exists() else []
    rows = [plan_row(path, root, pseudo_dir, colormap, preserve_tree) for path in images]
    if dry_run:
        report = build_report("dry_run", root, output, pseudo_dir, colormap, rows, 0, True, started)
        write_artifacts(output, report, rows)
        return report
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if pseudo_dir.exists():
        shutil.rmtree(pseudo_dir)
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    done = []
    errors = []
    for row in rows:
        try:
            image = pseudocolor_image(Path(row["source_path"]), colormap)
            dst = Path(row["output_path"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.save(dst)
            done_row = dict(row)
            done_row["status"] = "materialized"
            done.append(done_row)
        except Exception as exc:
            errors.append({"source_path": row["source_path"], "error": f"{type(exc).__name__}: {exc}"})
    status = "succeeded" if not errors else ("warning" if done else "failed")
    report = build_report(status, root, output, pseudo_dir, colormap, done, len(done), False, started, errors=errors)
    write_artifacts(output, report, done)
    return report


def pseudocolor_image(path: Path, colormap: str) -> Image.Image:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    stops = np.asarray(COLORMAPS.get(colormap, COLORMAPS["viridis"]), dtype=np.float32)
    pos = gray * (len(stops) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def plan_row(path: Path, root: Path, pseudo_dir: Path, colormap: str, preserve_tree: bool) -> dict[str, Any]:
    rel = path.relative_to(root) if preserve_tree else Path(path.name)
    out = pseudo_dir / rel.with_suffix(".png")
    return {
        "source_path": path.as_posix(),
        "relative_path": rel.as_posix(),
        "output_path": out.as_posix(),
        "colormap": colormap,
        "status": "planned",
    }


def build_report(
    status: str,
    root: Path,
    output: Path,
    pseudo_dir: Path,
    colormap: str,
    rows: list[dict[str, Any]],
    materialized_count: int,
    dry_run: bool,
    started: float,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PSEUDOCOLOR_VERSION,
        "skill": "PseudocolorSkill",
        "status": status,
        "message": "Pseudocolor plan generated." if dry_run else "Pseudocolor conversion completed.",
        "dry_run": dry_run,
        "input_dir": root.as_posix(),
        "output_dir": output.as_posix(),
        "pseudocolor_output_dir": pseudo_dir.as_posix(),
        "colormap": colormap,
        "planned_count": len(rows),
        "materialized_count": materialized_count,
        "errors": errors or [],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "pseudocolor_report.json").as_posix(),
            "report_md": (output / "pseudocolor_report.md").as_posix(),
            "manifest_jsonl": (output / "pseudocolor_manifest.jsonl").as_posix(),
        },
    }


def write_artifacts(output: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    save_json(output / "pseudocolor_report.json", report)
    save_text(output / "pseudocolor_report.md", render_report(report))
    write_jsonl(output / "pseudocolor_manifest.jsonl", rows)


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA Pseudocolor Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Input: `{report.get('input_dir')}`",
            f"- Output: `{report.get('pseudocolor_output_dir')}`",
            f"- Colormap: `{report.get('colormap')}`",
            f"- Planned: {report.get('planned_count')}",
            f"- Materialized: {report.get('materialized_count')}",
            "",
        ]
    )

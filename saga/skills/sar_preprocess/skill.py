from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from saga.core.config import load_mapping, save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.data.discovery import iter_images


SAR_PREPROCESS_VERSION = "saga_sar_preprocess_run_v1"


@dataclass
class SARPreprocessConfig:
    mode: str = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    log_scale: bool = False
    output_bit_depth: str = "uint8"
    output_image_format: str = "png"
    resize: int | None = None
    preserve_tree: bool = True
    copy_sidecars: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SARPreprocessConfig":
        raw = raw or {}
        body = raw.get("sar_preprocess", raw)
        default = cls()
        resize = body.get("resize", default.resize)
        return cls(
            mode=str(body.get("mode", default.mode)),
            percentile_low=float(body.get("percentile_low", default.percentile_low)),
            percentile_high=float(body.get("percentile_high", default.percentile_high)),
            log_scale=parse_bool(body.get("log_scale", default.log_scale)),
            output_bit_depth=str(body.get("output_bit_depth", default.output_bit_depth)),
            output_image_format=str(body.get("output_image_format", default.output_image_format)).lower().lstrip("."),
            resize=int(resize) if resize not in (None, "", 0) else None,
            preserve_tree=parse_bool(body.get("preserve_tree", default.preserve_tree)),
            copy_sidecars=parse_bool(body.get("copy_sidecars", default.copy_sidecars)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "SARPreprocessConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


def run_sar_preprocess_skill(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    mode: str | None = None,
    resize: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    config = SARPreprocessConfig.from_path(config_path)
    if mode:
        config.mode = mode
    if resize:
        config.resize = int(resize)
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    images = iter_images(input_path) if input_path.exists() else []
    processed_dir = output_path / "processed_images"
    rows = [plan_row(path, input_path, processed_dir, config) for path in images]

    if dry_run:
        report = build_report(
            status="dry_run",
            message="SAR preprocessing plan generated.",
            input_path=input_path,
            output_path=output_path,
            processed_dir=processed_dir,
            config=config,
            rows=rows,
            processed_count=0,
            dry_run=True,
            started=started,
        )
        write_artifacts(output_path, report, rows)
        return report

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_rows = []
    errors = []
    for row in rows:
        try:
            src = Path(row["source_path"])
            dst = Path(row["output_path"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            image = preprocess_image(src, config)
            image.save(dst)
            if config.copy_sidecars:
                copy_sidecars(src, dst)
            done = dict(row)
            done["status"] = "processed"
            processed_rows.append(done)
        except Exception as exc:
            errors.append({"source_path": row["source_path"], "error": f"{type(exc).__name__}: {exc}"})
    status = "succeeded" if not errors else ("warning" if processed_rows else "failed")
    report = build_report(
        status=status,
        message="SAR preprocessing completed.",
        input_path=input_path,
        output_path=output_path,
        processed_dir=processed_dir,
        config=config,
        rows=processed_rows,
        processed_count=len(processed_rows),
        dry_run=False,
        started=started,
        errors=errors,
    )
    write_artifacts(output_path, report, processed_rows)
    return report


def preprocess_image(path: Path, config: SARPreprocessConfig) -> Image.Image:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("F"), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if config.log_scale or config.mode == "log_percentile":
        arr = np.log1p(np.maximum(arr - float(arr.min()), 0.0))
    if config.mode in {"percentile", "log_percentile"}:
        lo, hi = np.percentile(arr, [config.percentile_low, config.percentile_high])
    elif config.mode == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    elif config.mode in {"none", "copy"}:
        lo, hi = float(arr.min()), float(arr.max())
    else:
        raise ValueError(f"Unsupported SAR preprocess mode: {config.mode}")
    if hi <= lo:
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if config.output_bit_depth == "uint16":
        out = (scaled * 65535.0).round().astype(np.uint16)
        result = Image.fromarray(out)
    else:
        out = (scaled * 255.0).round().astype(np.uint8)
        result = Image.fromarray(out, mode="L")
    if config.resize:
        result = result.resize((config.resize, config.resize), Image.BICUBIC)
    return result


def plan_row(path: Path, root: Path, processed_dir: Path, config: SARPreprocessConfig) -> dict[str, Any]:
    rel = path.relative_to(root) if config.preserve_tree else Path(path.name)
    out_rel = rel.with_suffix(f".{config.output_image_format}")
    return {
        "source_path": path.as_posix(),
        "relative_path": rel.as_posix(),
        "output_path": (processed_dir / out_rel).as_posix(),
        "mode": config.mode,
        "status": "planned",
    }


def copy_sidecars(src: Path, dst: Path) -> None:
    for suffix in (".txt", ".json", ".xml"):
        sidecar = src.with_suffix(suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, dst.with_suffix(suffix))


def build_report(
    *,
    status: str,
    message: str,
    input_path: Path,
    output_path: Path,
    processed_dir: Path,
    config: SARPreprocessConfig,
    rows: list[dict[str, Any]],
    processed_count: int,
    dry_run: bool,
    started: float,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SAR_PREPROCESS_VERSION,
        "skill": "SARPreprocessSkill",
        "status": status,
        "message": message,
        "dry_run": dry_run,
        "input_dir": input_path.as_posix(),
        "output_dir": output_path.as_posix(),
        "processed_output_dir": processed_dir.as_posix(),
        "input_count": len(rows) if dry_run else len(rows) + len(errors or []),
        "planned_count": len(rows),
        "processed_count": processed_count,
        "config": asdict(config),
        "errors": errors or [],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output_path / "sar_preprocess_report.json").as_posix(),
            "report_md": (output_path / "sar_preprocess_report.md").as_posix(),
            "manifest_jsonl": (output_path / "preprocess_manifest.jsonl").as_posix(),
        },
    }


def write_artifacts(output_path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    save_json(output_path / "sar_preprocess_report.json", report)
    save_text(output_path / "sar_preprocess_report.md", render_report(report))
    write_jsonl(output_path / "preprocess_manifest.jsonl", rows)


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA SAR Preprocess Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Input: `{report.get('input_dir')}`",
            f"- Output: `{report.get('processed_output_dir')}`",
            f"- Planned: {report.get('planned_count')}",
            f"- Processed: {report.get('processed_count')}",
            f"- Mode: `{(report.get('config') or {}).get('mode')}`",
            "",
        ]
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

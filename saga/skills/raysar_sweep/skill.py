from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.data.discovery import iter_images
from saga.skills.model_to_pov_scene.skill import run_model_to_pov_scene_compiler_skill
from saga.skills.raysar.skill import run_raysar_synthesis_skill


RAYSAR_SWEEP_VERSION = "saga_raysar_sweep_synthesis_run_v1"


@dataclass
class RaySARSweepSynthesisSkillConfig:
    name: str = "RaySARSweepSynthesisSkill"
    description: str = "Multi-view RaySAR simulation from a 3D model by sweeping target azimuth/aspect angles."
    model_to_pov_config: str = "configs/skills/model_to_pov_scene.yaml"
    raysar_config: str = "configs/skills/raysar.yaml"
    max_views: int = 72
    width: int = 512
    height: int = 512
    map_products: list[str] = field(
        default_factory=lambda: [
            "All_Reflections_Fr.tif",
            "Single_Bounce_Fr.tif",
            "Double_Bounce_Fr.tif",
        ]
    )
    view_dir_template: str = "az_{azimuth_label}"
    caption_template: str = (
        "SAR image, RaySAR simulation, azimuth {azimuth_deg:g} deg, "
        "incidence {incidence_angle_deg:g} deg, {map_product}"
    )
    copy_maps: bool = True
    render: bool = True
    postprocess: bool = True
    sar_intersection: bool = True
    auto_fix_scene: bool = True
    build_if_missing: bool = True
    keep_intermediate: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RaySARSweepSynthesisSkillConfig":
        raw = raw or {}
        body = raw.get("raysar_sweep", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            model_to_pov_config=str(body.get("model_to_pov_config", default.model_to_pov_config)),
            raysar_config=str(body.get("raysar_config", default.raysar_config)),
            max_views=int(body.get("max_views", default.max_views)),
            width=int(body.get("width", default.width)),
            height=int(body.get("height", default.height)),
            map_products=[str(item) for item in body.get("map_products", default.map_products)],
            view_dir_template=str(body.get("view_dir_template", default.view_dir_template)),
            caption_template=str(body.get("caption_template", default.caption_template)),
            copy_maps=parse_bool(body.get("copy_maps", default.copy_maps)),
            render=parse_bool(body.get("render", default.render)),
            postprocess=parse_bool(body.get("postprocess", default.postprocess)),
            sar_intersection=parse_bool(body.get("sar_intersection", default.sar_intersection)),
            auto_fix_scene=parse_bool(body.get("auto_fix_scene", default.auto_fix_scene)),
            build_if_missing=parse_bool(body.get("build_if_missing", default.build_if_missing)),
            keep_intermediate=parse_bool(body.get("keep_intermediate", default.keep_intermediate)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "RaySARSweepSynthesisSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class RaySARSweepSynthesisSkill:
    def __init__(self, config: RaySARSweepSynthesisSkillConfig | None = None) -> None:
        self.config = config or RaySARSweepSynthesisSkillConfig()

    def run(
        self,
        model_file: str | Path,
        output_dir: str | Path,
        azimuth_values: list[float] | None = None,
        azimuth_sweep: dict[str, Any] | None = None,
        incidence_angle_deg: float | None = None,
        depression_angle_deg: float | None = None,
        target_extent_m: float | None = None,
        scale_factor: float | None = None,
        input_up_axis: str | None = None,
        pov_height_axis: str | None = None,
        range_distance_m: float | None = None,
        sensor_plane_m: float | None = None,
        parameters_file: str | Path | None = None,
        adapted_povray: str | Path | None = None,
        width: int | None = None,
        height: int | None = None,
        render: bool | None = None,
        postprocess: bool | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        model_path = Path(model_file).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            raise FileNotFoundError(f"3D model file does not exist: {model_path}")

        views = resolve_azimuth_values(azimuth_values=azimuth_values, azimuth_sweep=azimuth_sweep)
        if not views:
            raise ValueError("RaySARSweepSynthesisSkill needs azimuth_values or azimuth_sweep.")
        if len(views) > int(self.config.max_views):
            raise ValueError(f"Requested {len(views)} views exceeds max_views={self.config.max_views}.")

        requested_width = int(width or self.config.width)
        requested_height = int(height or self.config.height)
        requested_render = self.config.render if render is None else bool(render)
        requested_postprocess = self.config.postprocess if postprocess is None else bool(postprocess)
        aggregate_dir = output_path / "sweep_maps"
        if not dry_run and aggregate_dir.exists():
            shutil.rmtree(aggregate_dir)
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        view_reports = []
        copied_maps = []
        failures = []
        for index, azimuth in enumerate(views):
            view_id = self.view_id(azimuth, index)
            view_root = output_path / "views" / view_id
            compile_dir = view_root / "compile"
            synthesis_dir = view_root / "raysar"
            try:
                compile_report = run_model_to_pov_scene_compiler_skill(
                    model_file=model_path,
                    output_dir=compile_dir,
                    config_path=self.config.model_to_pov_config,
                    incidence_angle_deg=incidence_angle_deg,
                    depression_angle_deg=depression_angle_deg,
                    azimuth_deg=azimuth,
                    target_extent_m=target_extent_m,
                    scale_factor=scale_factor,
                    input_up_axis=input_up_axis,
                    pov_height_axis=pov_height_axis,
                    range_distance_m=range_distance_m,
                    sensor_plane_m=sensor_plane_m,
                    dry_run=dry_run,
                )
                raysar_report = run_raysar_synthesis_skill(
                    output_dir=synthesis_dir,
                    pov_scene=compile_report.get("compiled_scene"),
                    parameters_file=parameters_file,
                    adapted_povray=adapted_povray,
                    width=requested_width,
                    height=requested_height,
                    render=requested_render,
                    postprocess=requested_postprocess,
                    auto_fix_scene=self.config.auto_fix_scene,
                    sar_intersection=self.config.sar_intersection,
                    build_if_missing=self.config.build_if_missing,
                    config_path=self.config.raysar_config,
                    dry_run=dry_run,
                )
                maps = collect_view_maps(
                    raysar_report=raysar_report,
                    map_products=self.config.map_products,
                    aggregate_dir=aggregate_dir,
                    view_id=view_id,
                    azimuth=azimuth,
                    incidence_angle=effective_incidence(incidence_angle_deg, depression_angle_deg),
                    caption_template=self.config.caption_template,
                    dry_run=dry_run,
                    copy_maps=self.config.copy_maps,
                )
                copied_maps.extend(maps)
                view_reports.append(
                    {
                        "index": index,
                        "view_id": view_id,
                        "azimuth_deg": azimuth,
                        "status": view_status(compile_report, raysar_report),
                        "compiled_scene": compile_report.get("compiled_scene"),
                        "raysar_output_dir": raysar_report.get("output_dir"),
                        "generated_output_dir": raysar_report.get("generated_output_dir"),
                        "map_count": len(maps),
                        "compile_report": summarize_nested_report(compile_report),
                        "raysar_report": summarize_nested_report(raysar_report),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "index": index,
                        "azimuth_deg": azimuth,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                view_reports.append(
                    {
                        "index": index,
                        "view_id": view_id,
                        "azimuth_deg": azimuth,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if dry_run:
            status = "dry_run"
            message = "RaySAR azimuth sweep dry-run generated compile/render plans."
        elif failures:
            status = "warning" if copied_maps else "failed"
            message = "RaySAR azimuth sweep completed with failed views." if copied_maps else "RaySAR azimuth sweep failed."
        else:
            status = "succeeded"
            message = "RaySAR azimuth sweep completed."

        expected_map_count = len(views) * len(self.config.map_products) if requested_postprocess else len(views)
        report = {
            "schema_version": RAYSAR_SWEEP_VERSION,
            "skill": "RaySARSweepSynthesisSkill",
            "status": status,
            "message": message,
            "dry_run": dry_run,
            "model_file": model_path.as_posix(),
            "output_dir": output_path.as_posix(),
            "generated_output_dir": aggregate_dir.as_posix(),
            "azimuth_values": views,
            "view_count": len(views),
            "width": requested_width,
            "height": requested_height,
            "render": requested_render,
            "postprocess": requested_postprocess,
            "expected_count": expected_map_count,
            "image_count": len(copied_maps),
            "map_count": len(copied_maps),
            "view_reports": view_reports,
            "failures": failures,
            "metrics": {
                "view_count": len(views),
                "expected_map_count": expected_map_count,
                "map_count": len(copied_maps),
                "failed_view_count": len(failures),
            },
            "warnings": [
                {
                    "type": "domain_gap",
                    "message": "RaySAR sweep outputs are synthetic physical simulations; downstream use must evaluate real-synthetic domain gap.",
                }
            ],
            "artifacts": {
                "aggregate_dir": aggregate_dir.as_posix(),
                "map_images": [path.as_posix() for path in copied_maps],
                "report_json": (output_path / "raysar_sweep_report.json").as_posix(),
                "report_md": (output_path / "raysar_sweep_report.md").as_posix(),
            },
            "notes": [
                "Each view compiles a fresh POV scene with its own azimuth/aspect angle.",
                "The aggregate directory contains renamed RaySAR reflection maps and txt sidecar captions for export metadata.",
                "Observer warnings should be interpreted differently from learned image generation: flat or sparse maps can be physically meaningful for simple geometries.",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(output_path / "raysar_sweep_report.json", report)
        save_text(output_path / "raysar_sweep_report.md", render_report(report))
        return report

    def view_id(self, azimuth: float, index: int) -> str:
        label = azimuth_label(azimuth)
        return self.config.view_dir_template.format(index=index, azimuth_label=label, azimuth_deg=azimuth)


def run_raysar_sweep_synthesis_skill(
    model_file: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    azimuth_values: list[float] | None = None,
    azimuth_sweep: dict[str, Any] | None = None,
    incidence_angle_deg: float | None = None,
    depression_angle_deg: float | None = None,
    target_extent_m: float | None = None,
    scale_factor: float | None = None,
    input_up_axis: str | None = None,
    pov_height_axis: str | None = None,
    range_distance_m: float | None = None,
    sensor_plane_m: float | None = None,
    parameters_file: str | Path | None = None,
    adapted_povray: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    render: bool | None = None,
    postprocess: bool | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = RaySARSweepSynthesisSkillConfig.from_path(config_path)
    return RaySARSweepSynthesisSkill(config).run(
        model_file=model_file,
        output_dir=output_dir,
        azimuth_values=azimuth_values,
        azimuth_sweep=azimuth_sweep,
        incidence_angle_deg=incidence_angle_deg,
        depression_angle_deg=depression_angle_deg,
        target_extent_m=target_extent_m,
        scale_factor=scale_factor,
        input_up_axis=input_up_axis,
        pov_height_axis=pov_height_axis,
        range_distance_m=range_distance_m,
        sensor_plane_m=sensor_plane_m,
        parameters_file=parameters_file,
        adapted_povray=adapted_povray,
        width=width,
        height=height,
        render=render,
        postprocess=postprocess,
        dry_run=dry_run,
    )


def resolve_azimuth_values(
    azimuth_values: list[float] | None,
    azimuth_sweep: dict[str, Any] | None,
) -> list[float]:
    if azimuth_values:
        return unique_angles([float(item) for item in azimuth_values])
    if not azimuth_sweep:
        return []
    start = float(azimuth_sweep.get("start", azimuth_sweep.get("from", 0)))
    stop = float(azimuth_sweep.get("stop", azimuth_sweep.get("to", start)))
    step = float(azimuth_sweep.get("step", azimuth_sweep.get("interval", 0)))
    if step == 0:
        raise ValueError("azimuth_sweep step must be non-zero.")
    if start < stop and step < 0:
        step = abs(step)
    if start > stop and step > 0:
        step = -step
    values = []
    current = start
    epsilon = abs(step) * 1e-6
    if step > 0:
        while current <= stop + epsilon:
            values.append(round_angle(current))
            current += step
    else:
        while current >= stop - epsilon:
            values.append(round_angle(current))
            current += step
    return unique_angles(values)


def unique_angles(values: list[float]) -> list[float]:
    seen = set()
    result = []
    for value in values:
        rounded = round_angle(value)
        key = f"{rounded:.6f}"
        if key in seen:
            continue
        seen.add(key)
        result.append(rounded)
    return result


def round_angle(value: float) -> float:
    rounded = round(float(value), 6)
    return int(rounded) if float(rounded).is_integer() else rounded


def azimuth_label(value: float) -> str:
    number = round_angle(value)
    if isinstance(number, int):
        raw = f"{number:+04d}" if number < 0 else f"{number:03d}"
    else:
        raw = f"{number:+08.3f}" if number < 0 else f"{number:07.3f}"
    return raw.replace("+", "p").replace("-", "m").replace(".", "p")


def collect_view_maps(
    *,
    raysar_report: dict[str, Any],
    map_products: list[str],
    aggregate_dir: Path,
    view_id: str,
    azimuth: float,
    incidence_angle: float,
    caption_template: str,
    dry_run: bool,
    copy_maps: bool,
) -> list[Path]:
    if dry_run:
        return []
    maps_dir = Path(str(raysar_report.get("generated_output_dir") or "")).expanduser()
    if not maps_dir.exists():
        return []
    products = []
    for name in map_products:
        path = maps_dir / name
        if path.exists():
            products.append(path)
    if not products:
        products = iter_images(maps_dir)
    copied = []
    for source in products:
        target = aggregate_dir / f"{view_id}_{source.name}"
        if target.exists():
            target.unlink()
        if copy_maps:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())
        caption = caption_template.format(
            azimuth_deg=float(azimuth),
            incidence_angle_deg=float(incidence_angle),
            map_product=source.stem,
        )
        target.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        copied.append(target)
    return copied


def effective_incidence(incidence_angle_deg: float | None, depression_angle_deg: float | None) -> float:
    if incidence_angle_deg is not None:
        return float(incidence_angle_deg)
    if depression_angle_deg is not None:
        return 90.0 - float(depression_angle_deg)
    return 35.0


def view_status(compile_report: dict[str, Any], raysar_report: dict[str, Any]) -> str:
    statuses = {compile_report.get("status"), raysar_report.get("status")}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    if "dry_run" in statuses:
        return "dry_run"
    return "succeeded"


def summarize_nested_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "message": report.get("message"),
        "output_dir": report.get("output_dir"),
        "generated_output_dir": report.get("generated_output_dir"),
        "metrics": report.get("metrics", {}),
        "artifacts": report.get("artifacts", {}),
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# RaySAR Sweep Synthesis",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Dry run: {report.get('dry_run')}",
        f"- Model: `{report.get('model_file')}`",
        f"- Views: {report.get('view_count')}",
        f"- Azimuth values: `{report.get('azimuth_values')}`",
        f"- Generated output: `{report.get('generated_output_dir')}`",
        f"- Maps: {report.get('map_count')} / {report.get('expected_count')}",
        f"- Failed views: {report.get('metrics', {}).get('failed_view_count')}",
        "",
        "## Views",
        "",
    ]
    for item in report.get("view_reports") or []:
        lines.append(
            f"- `{item.get('view_id')}` azimuth={item.get('azimuth_deg')} "
            f"status={item.get('status')} maps={item.get('map_count')}"
        )
    lines.extend(["", "## Failures", ""])
    if report.get("failures"):
        for item in report["failures"][:20]:
            lines.append(f"- azimuth={item.get('azimuth_deg')}: {item.get('error_type')} {item.get('message')}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)

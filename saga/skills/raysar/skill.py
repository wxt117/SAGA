from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.data.discovery import iter_images


RAYSAR_SYNTHESIS_VERSION = "saga_raysar_synthesis_run_v1"


DEFAULT_RAYSAR_ROOT = Path("myproject/raysar/RaySAR_1.2")
DEFAULT_SOURCE_DIR = (
    DEFAULT_RAYSAR_ROOT
    / "POV-Ray/POV-Ray_3.7_Source_Code_for_RaySAR/povray-3.7.0.0_for_raysar-loc_conwriter"
)
DEFAULT_PYTHON_MODULE = DEFAULT_RAYSAR_ROOT / "MATLAB_part/Python_module/RaySAR_forbs"
DEFAULT_PARAMETERS_FILE = DEFAULT_RAYSAR_ROOT / "MATLAB_part/Python_module/parameters.txt"


@dataclass
class RaySARSynthesisSkillConfig:
    name: str = "RaySARSynthesisSkill"
    description: str = "Physics-based SAR simulation through RaySAR-adapted POV-Ray and Python map post-processing."
    raysar_root: str = DEFAULT_RAYSAR_ROOT.as_posix()
    source_dir: str = DEFAULT_SOURCE_DIR.as_posix()
    adapted_povray: str | None = None
    build_root: str = "runs/raysar_build"
    auto_build: bool = True
    postprocess_module: str = DEFAULT_PYTHON_MODULE.as_posix()
    default_parameters_file: str = DEFAULT_PARAMETERS_FILE.as_posix()
    width: int = 512
    height: int = 512
    output_format: str = "png"
    sar_output_data: bool = True
    sar_intersection: bool = True
    auto_fix_scene: bool = True
    orthographic_angle: float = 45.0
    provide_minimal_includes: bool = True
    postprocess: bool = True
    make_jobs: int = 2
    build_compiled_by: str = "SAGA RaySAR skill"
    extra_library_paths: list[str] = field(default_factory=list)
    domain_gap_warning: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RaySARSynthesisSkillConfig":
        raw = raw or {}
        body = raw.get("raysar", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            raysar_root=str(body.get("raysar_root", default.raysar_root)),
            source_dir=str(body.get("source_dir", default.source_dir)),
            adapted_povray=optional_str(body.get("adapted_povray", default.adapted_povray)),
            build_root=str(body.get("build_root", default.build_root)),
            auto_build=parse_bool(body.get("auto_build", default.auto_build)),
            postprocess_module=str(body.get("postprocess_module", default.postprocess_module)),
            default_parameters_file=str(body.get("default_parameters_file", default.default_parameters_file)),
            width=int(body.get("width", default.width)),
            height=int(body.get("height", default.height)),
            output_format=str(body.get("output_format", default.output_format)).lower().lstrip("."),
            sar_output_data=parse_bool(body.get("sar_output_data", default.sar_output_data)),
            sar_intersection=parse_bool(body.get("sar_intersection", default.sar_intersection)),
            auto_fix_scene=parse_bool(body.get("auto_fix_scene", default.auto_fix_scene)),
            orthographic_angle=float(body.get("orthographic_angle", default.orthographic_angle)),
            provide_minimal_includes=parse_bool(body.get("provide_minimal_includes", default.provide_minimal_includes)),
            postprocess=parse_bool(body.get("postprocess", default.postprocess)),
            make_jobs=int(body.get("make_jobs", default.make_jobs)),
            build_compiled_by=str(body.get("build_compiled_by", default.build_compiled_by)),
            extra_library_paths=[str(item) for item in body.get("extra_library_paths", default.extra_library_paths)],
            domain_gap_warning=parse_bool(body.get("domain_gap_warning", default.domain_gap_warning)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "RaySARSynthesisSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class RaySARSynthesisSkill:
    def __init__(self, config: RaySARSynthesisSkillConfig | None = None) -> None:
        self.config = config or RaySARSynthesisSkillConfig()

    def run(
        self,
        output_dir: str | Path,
        pov_scene: str | Path | None = None,
        parameters_file: str | Path | None = None,
        simulation_parameters: dict[str, Any] | None = None,
        contributions_txt: str | Path | None = None,
        adapted_povray: str | Path | None = None,
        width: int | None = None,
        height: int | None = None,
        postprocess: bool | None = None,
        render: bool = True,
        auto_fix_scene: bool | None = None,
        sar_intersection: bool | None = None,
        build_if_missing: bool | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        run_dir = output_path / "raytracing"
        run_dir.mkdir(parents=True, exist_ok=True)

        requested_postprocess = self.config.postprocess if postprocess is None else bool(postprocess)
        requested_render = bool(render)
        requested_width = int(width or self.config.width)
        requested_height = int(height or self.config.height)
        requested_intersection = self.config.sar_intersection if sar_intersection is None else bool(sar_intersection)
        requested_autofix = self.config.auto_fix_scene if auto_fix_scene is None else bool(auto_fix_scene)
        requested_build = self.config.auto_build if build_if_missing is None else bool(build_if_missing)

        warnings = []
        if self.config.domain_gap_warning:
            warnings.append(
                {
                    "type": "domain_gap",
                    "message": "RaySAR outputs are physically interpretable synthetic SAR maps; downstream use should account for real-synthetic domain shift.",
                }
            )
        if requested_postprocess and not requested_intersection:
            warnings.append(
                {
                    "type": "sar_intersection_for_postprocess",
                    "message": "RaySAR Python post-processing expects intersection columns; SAGA will prefer SAR_Intersection 1 for normal synthesis.",
                }
            )

        pov_path = Path(pov_scene).expanduser().resolve() if pov_scene else None
        contribution_input = Path(contributions_txt).expanduser().resolve() if contributions_txt else None
        if requested_render and not pov_path:
            raise ValueError("RaySARSynthesisSkill needs pov_scene unless render=False with contributions_txt.")
        if pov_path and not pov_path.exists():
            raise FileNotFoundError(f"POV scene does not exist: {pov_path}")
        if contribution_input and not contribution_input.exists():
            raise FileNotFoundError(f"contributions_txt does not exist: {contribution_input}")

        include_dir = prepare_minimal_includes(output_path) if self.config.provide_minimal_includes else None
        prepared_scene = None
        scene_report: dict[str, Any] = {}
        if pov_path:
            prepared_scene = output_path / "prepared_scene.pov"
            scene_report = prepare_scene(
                source=pov_path,
                output=prepared_scene,
                auto_fix=requested_autofix,
                sar_output_data=self.config.sar_output_data,
                sar_intersection=requested_intersection,
                orthographic_angle=self.config.orthographic_angle,
            )

        parameters_path = resolve_parameters_file(
            output_dir=output_path,
            parameters_file=parameters_file,
            default_parameters_file=self.config.default_parameters_file,
            simulation_parameters=simulation_parameters,
        )
        povray_resolution = resolve_povray(
            explicit=adapted_povray,
            config=self.config,
            output_dir=output_path,
            dry_run=dry_run,
            build_if_missing=requested_build,
        )
        povray_raw_path = povray_resolution.get("path")
        povray_path = Path(str(povray_raw_path)).expanduser().resolve() if povray_raw_path else None

        preview_path = run_dir / f"render_preview.{self.config.output_format}"
        render_command = build_render_command(
            povray=povray_path or Path("<adapted-povray>"),
            scene=prepared_scene,
            output_image=preview_path,
            width=requested_width,
            height=requested_height,
            output_format=self.config.output_format,
            include_dir=include_dir,
            extra_library_paths=[Path(item).expanduser().resolve() for item in self.config.extra_library_paths],
        ) if prepared_scene else []

        render_result: dict[str, Any] = {"status": "skipped", "command": render_command}
        contribution_path = run_dir / "Contributions.txt"
        if contribution_input and not requested_render:
            contribution_path = output_path / "Contributions.txt"
            if not dry_run:
                shutil.copy2(contribution_input, contribution_path)
            render_result = {
                "status": "dry_run" if dry_run else "succeeded",
                "message": "Using provided contributions_txt for postprocess-only mode.",
                "source_contributions": contribution_input.as_posix(),
            }
        elif requested_render:
            if dry_run:
                render_result = {
                    "status": "dry_run",
                    "message": "RaySAR render command prepared.",
                    "command": render_command,
                    "cwd": run_dir.as_posix(),
                }
            else:
                if povray_path is None:
                    raise FileNotFoundError("No RaySAR-adapted POV-Ray executable was resolved.")
                if contribution_path.exists():
                    contribution_path.unlink()
                proc = subprocess.run(render_command, cwd=run_dir.as_posix(), text=True, capture_output=True)
                render_result = {
                    "status": "succeeded" if proc.returncode == 0 else "failed",
                    "message": "RaySAR rendering completed." if proc.returncode == 0 else "RaySAR rendering failed.",
                    "command": render_command,
                    "cwd": run_dir.as_posix(),
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-6000:],
                    "stderr_tail": proc.stderr[-6000:],
                }

        postprocess_dir = output_path / "postprocess"
        maps_dir = postprocess_dir / "Maps"
        postprocess_result: dict[str, Any] = {"status": "skipped"}
        if requested_postprocess:
            if dry_run:
                postprocess_result = {
                    "status": "dry_run",
                    "message": "RaySAR Python postprocess would convert Contributions.txt into reflection maps.",
                    "module": Path(self.config.postprocess_module).expanduser().resolve().as_posix(),
                    "parameters_file": parameters_path.as_posix(),
                    "output_dir": postprocess_dir.as_posix(),
                }
            elif contribution_path.exists() and contribution_path.stat().st_size > 0:
                postprocess_result = run_python_postprocess(
                    module_path=Path(self.config.postprocess_module).expanduser().resolve(),
                    contributions=contribution_path,
                    output_dir=postprocess_dir,
                    parameters_file=parameters_path,
                )
            else:
                postprocess_result = {
                    "status": "failed",
                    "message": "Postprocess requested, but Contributions.txt is missing or empty.",
                    "contributions_txt": contribution_path.as_posix(),
                }

        map_images = iter_images(maps_dir) if maps_dir.exists() else []
        contribution_rows = count_nonempty_lines(contribution_path) if contribution_path.exists() else 0
        status = determine_status(dry_run=dry_run, render_result=render_result, postprocess_result=postprocess_result)
        message = status_message(status=status, dry_run=dry_run, requested_postprocess=requested_postprocess)
        report = {
            "schema_version": RAYSAR_SYNTHESIS_VERSION,
            "skill": "RaySARSynthesisSkill",
            "status": status,
            "message": message,
            "dry_run": dry_run,
            "pov_scene": pov_path.as_posix() if pov_path else None,
            "prepared_scene": prepared_scene.as_posix() if prepared_scene else None,
            "parameters_file": parameters_path.as_posix(),
            "contributions_txt": contribution_path.as_posix(),
            "adapted_povray": povray_path.as_posix() if povray_path else None,
            "output_dir": output_path.as_posix(),
            "run_dir": run_dir.as_posix(),
            "generated_output_dir": maps_dir.as_posix() if requested_postprocess else run_dir.as_posix(),
            "preview_image": preview_path.as_posix(),
            "width": requested_width,
            "height": requested_height,
            "render": requested_render,
            "postprocess": requested_postprocess,
            "sar_intersection": requested_intersection,
            "scene_report": scene_report,
            "povray_resolution": povray_resolution,
            "render_result": render_result,
            "postprocess_result": postprocess_result,
            "image_count": len(map_images),
            "map_count": len(map_images),
            "metrics": {
                "contribution_rows": contribution_rows,
                "map_count": len(map_images),
                "render_returncode": render_result.get("returncode"),
            },
            "warnings": warnings,
            "elapsed_seconds": round(time.time() - started, 3),
            "artifacts": {
                "report_json": (output_path / "raysar_synthesis_report.json").as_posix(),
                "report_md": (output_path / "raysar_synthesis_report.md").as_posix(),
                "prepared_scene": prepared_scene.as_posix() if prepared_scene else None,
                "parameters_file": parameters_path.as_posix(),
                "contributions_txt": contribution_path.as_posix(),
                "preview_image": preview_path.as_posix() if preview_path.exists() or dry_run else None,
                "maps_dir": maps_dir.as_posix(),
                "map_images": [path.as_posix() for path in map_images],
            },
            "notes": [
                "Primary executable input is a RaySAR-compatible .pov scene.",
                "Common OBJ/STL/PLY/GLB model support is provided upstream by ModelToPOVSceneCompilerSkill.",
                "RaySAR is physically interpretable but may have a real-to-synthetic domain gap.",
            ],
        }
        save_json(output_path / "raysar_synthesis_report.json", report)
        save_text(output_path / "raysar_synthesis_report.md", render_report(report))
        return report


def run_raysar_synthesis_skill(
    output_dir: str | Path,
    pov_scene: str | Path | None = None,
    parameters_file: str | Path | None = None,
    simulation_parameters: dict[str, Any] | None = None,
    contributions_txt: str | Path | None = None,
    adapted_povray: str | Path | None = None,
    width: int | None = None,
    height: int | None = None,
    postprocess: bool | None = None,
    render: bool = True,
    auto_fix_scene: bool | None = None,
    sar_intersection: bool | None = None,
    build_if_missing: bool | None = None,
    config_path: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = RaySARSynthesisSkillConfig.from_path(config_path)
    return RaySARSynthesisSkill(config).run(
        output_dir=output_dir,
        pov_scene=pov_scene,
        parameters_file=parameters_file,
        simulation_parameters=simulation_parameters,
        contributions_txt=contributions_txt,
        adapted_povray=adapted_povray,
        width=width,
        height=height,
        postprocess=postprocess,
        render=render,
        auto_fix_scene=auto_fix_scene,
        sar_intersection=sar_intersection,
        build_if_missing=build_if_missing,
        dry_run=dry_run,
    )


def resolve_povray(
    *,
    explicit: str | Path | None,
    config: RaySARSynthesisSkillConfig,
    output_dir: Path,
    dry_run: bool,
    build_if_missing: bool,
) -> dict[str, Any]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if config.adapted_povray:
        candidates.append(Path(config.adapted_povray).expanduser())
    if os.environ.get("SAGA_RAYSAR_POVRAY"):
        candidates.append(Path(os.environ["SAGA_RAYSAR_POVRAY"]).expanduser())
    candidates.extend(
        [
            Path("runs/raysar_build_probe/source/unix/povray"),
            Path(config.build_root) / "source/unix/povray",
            Path(config.source_dir) / "unix/povray",
        ]
    )
    for candidate in candidates:
        path = candidate.resolve()
        if path.exists() and os.access(path, os.X_OK):
            return {"status": "resolved", "path": path.as_posix(), "source": "existing_binary"}

    source_dir = Path(config.source_dir).expanduser().resolve()
    build_root = Path(config.build_root).expanduser().resolve()
    build_commands = build_povray_build_commands(source_dir=source_dir, build_root=build_root, config=config)
    if dry_run:
        return {
            "status": "dry_run_missing_binary",
            "path": None,
            "source_dir": source_dir.as_posix(),
            "build_root": build_root.as_posix(),
            "build_commands": build_commands,
        }
    if not build_if_missing:
        return {
            "status": "missing_binary",
            "path": None,
            "source_dir": source_dir.as_posix(),
            "build_commands": build_commands,
        }
    build_report = build_povray_from_source(source_dir=source_dir, build_root=build_root, config=config, output_dir=output_dir)
    binary = build_root / "source/unix/povray"
    if binary.exists() and os.access(binary, os.X_OK):
        return {"status": "built", "path": binary.resolve().as_posix(), "build_report": build_report}
    return {"status": "build_failed", "path": None, "build_report": build_report}


def build_povray_build_commands(source_dir: Path, build_root: Path, config: RaySARSynthesisSkillConfig) -> list[list[str]]:
    prefix = build_root / "install"
    return [
        ["cp", "-a", source_dir.as_posix(), (build_root / "source").as_posix()],
        ["bash", "-lc", "chmod +x unix/prebuild.sh configure_raysar.sh 2>/dev/null || true"],
        ["bash", "prebuild.sh"],
        [
            "./configure",
            f"COMPILED_BY={config.build_compiled_by}",
            "LIBS=-lboost_system -lboost_thread",
            "--disable-io-restrictions",
            f"--prefix={prefix.as_posix()}",
        ],
        ["make", f"-j{int(config.make_jobs)}"],
    ]


def build_povray_from_source(
    *,
    source_dir: Path,
    build_root: Path,
    config: RaySARSynthesisSkillConfig,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    build_root.mkdir(parents=True, exist_ok=True)
    build_source = build_root / "source"
    if build_source.exists():
        shutil.rmtree(build_source)
    shutil.copytree(source_dir, build_source)
    runs = []

    def run(command: list[str], cwd: Path) -> bool:
        proc = subprocess.run(command, cwd=cwd.as_posix(), text=True, capture_output=True)
        runs.append(
            {
                "command": command,
                "cwd": cwd.as_posix(),
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
            }
        )
        return proc.returncode == 0

    subprocess.run(["bash", "-lc", "chmod +x unix/prebuild.sh configure_raysar.sh 2>/dev/null || true"], cwd=build_source.as_posix())
    ok = run(["bash", "prebuild.sh"], build_source / "unix")
    if ok:
        ok = run(
            [
                "./configure",
                f"COMPILED_BY={config.build_compiled_by}",
                "LIBS=-lboost_system -lboost_thread",
                "--disable-io-restrictions",
                f"--prefix={(build_root / 'install').as_posix()}",
            ],
            build_source,
        )
    if ok:
        ok = run(["make", f"-j{int(config.make_jobs)}"], build_source)
    report = {
        "status": "succeeded" if ok else "failed",
        "source_dir": source_dir.as_posix(),
        "build_root": build_root.as_posix(),
        "binary": (build_source / "unix/povray").as_posix(),
        "commands": runs,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(output_dir / "raysar_povray_build.json", report)
    return report


def prepare_scene(
    *,
    source: Path,
    output: Path,
    auto_fix: bool,
    sar_output_data: bool,
    sar_intersection: bool,
    orthographic_angle: float,
) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    changes = []
    if auto_fix:
        updated, changed = ensure_sar_global_settings(text, sar_output_data=sar_output_data, sar_intersection=sar_intersection)
        if changed:
            changes.append(changed)
        text = updated
        updated, count = ensure_orthographic_angle(text, angle=orthographic_angle)
        if count:
            changes.append({"type": "orthographic_angle", "camera_blocks_updated": count, "angle": orthographic_angle})
        text = updated
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return {
        "source": source.as_posix(),
        "prepared_scene": output.as_posix(),
        "auto_fix": auto_fix,
        "changes": changes,
        "sar_output_data": sar_output_data,
        "sar_intersection": sar_intersection,
    }


def ensure_sar_global_settings(text: str, sar_output_data: bool, sar_intersection: bool) -> tuple[str, dict[str, Any] | None]:
    changed = False
    output_value = "1" if sar_output_data else "0"
    intersection_value = "1" if sar_intersection else "0"
    if re.search(r"\bSAR_Output_Data\s+[-+]?\d+(?:\.\d+)?", text):
        text = re.sub(r"\bSAR_Output_Data\s+[-+]?\d+(?:\.\d+)?", f"SAR_Output_Data {output_value}", text)
        changed = True
    if re.search(r"\bSAR_Intersection\s+[-+]?\d+(?:\.\d+)?", text):
        text = re.sub(r"\bSAR_Intersection\s+[-+]?\d+(?:\.\d+)?", f"SAR_Intersection {intersection_value}", text)
        changed = True
    if "SAR_Output_Data" not in text or "SAR_Intersection" not in text:
        block = f"global_settings{{SAR_Output_Data {output_value} SAR_Intersection {intersection_value}}}\n\n"
        version_match = re.search(r"(?m)^\s*#version[^\n]*\n", text)
        if version_match:
            insert_at = version_match.end()
            text = text[:insert_at] + "\n" + block + text[insert_at:]
        else:
            text = block + text
        changed = True
    if not changed:
        return text, None
    return text, {"type": "sar_global_settings", "sar_output_data": sar_output_data, "sar_intersection": sar_intersection}


def ensure_orthographic_angle(text: str, angle: float) -> tuple[str, int]:
    spans = find_keyword_blocks(text, "camera")
    updated = text
    replacements: list[tuple[int, int, str]] = []
    for start, end in spans:
        block = text[start:end]
        lowered = block.lower()
        if "orthographic" not in lowered or re.search(r"\bangle\b", block, flags=re.IGNORECASE):
            continue
        replacement = re.sub(
            r"\borthographic\b",
            f"orthographic\n  angle {float(angle):g}",
            block,
            count=1,
            flags=re.IGNORECASE,
        )
        replacements.append((start, end, replacement))
    for start, end, replacement in reversed(replacements):
        updated = updated[:start] + replacement + updated[end:]
    return updated, len(replacements)


def find_keyword_blocks(text: str, keyword: str) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(rf"\b{re.escape(keyword)}\s*\{{", text, flags=re.IGNORECASE):
        brace = text.find("{", match.start())
        if brace < 0:
            continue
        depth = 0
        for idx in range(brace, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), idx + 1))
                    break
    return spans


def prepare_minimal_includes(output_dir: Path) -> Path:
    include_dir = output_dir / "povray_include"
    include_dir.mkdir(parents=True, exist_ok=True)
    colors = include_dir / "colors.inc"
    finish = include_dir / "finish.inc"
    if not colors.exists():
        colors.write_text(
            "\n".join(
                [
                    "#declare White = rgb <1, 1, 1>;",
                    "#declare Black = rgb <0, 0, 0>;",
                    "#declare Red = rgb <1, 0, 0>;",
                    "#declare Green = rgb <0, 1, 0>;",
                    "#declare Blue = rgb <0, 0, 1>;",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if not finish.exists():
        finish.write_text("// Minimal finish include placeholder generated by SAGA RaySAR skill.\n", encoding="utf-8")
    return include_dir


def resolve_parameters_file(
    *,
    output_dir: Path,
    parameters_file: str | Path | None,
    default_parameters_file: str,
    simulation_parameters: dict[str, Any] | None,
) -> Path:
    if simulation_parameters:
        target = output_dir / "parameters.txt"
        base = {}
        default = Path(default_parameters_file).expanduser().resolve()
        if default.exists():
            base = parse_parameters_file(default)
        base.update({str(key): value for key, value in simulation_parameters.items()})
        write_parameters_file(target, base)
        return target
    if parameters_file:
        path = Path(parameters_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"RaySAR parameters file does not exist: {path}")
        return path
    default = Path(default_parameters_file).expanduser().resolve()
    if not default.exists():
        raise FileNotFoundError(f"Default RaySAR parameters file does not exist: {default}")
    return default


def parse_parameters_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_parameters_file(path: Path, values: dict[str, Any]) -> None:
    order = [
        "a_pix",
        "r_pix",
        "a_min",
        "a_max",
        "r_min",
        "r_max",
        "bounce_level",
        "db_min",
        "db_max",
        "range_dir",
        "filt",
        "r_geom",
        "ang",
        "clip",
        "mode",
        "a_rem",
        "r_rem",
        "Output_path",
        "elevation_scale",
    ]
    lines = []
    seen = set()
    for key in order:
        if key in values:
            lines.append(f"{key}={values[key]}")
            seen.add(key)
    for key in sorted(set(values) - seen):
        lines.append(f"{key}={values[key]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_render_command(
    *,
    povray: Path,
    scene: Path | None,
    output_image: Path,
    width: int,
    height: int,
    output_format: str,
    include_dir: Path | None,
    extra_library_paths: list[Path],
) -> list[str]:
    command = [
        povray.as_posix(),
        f"+I{scene.as_posix() if scene else '<scene.pov>'}",
        f"+O{output_image.name}",
        f"+W{int(width)}",
        f"+H{int(height)}",
        "-D",
        output_format_switch(output_format),
    ]
    if include_dir:
        command.append(f"+L{include_dir.as_posix()}")
    for path in extra_library_paths:
        command.append(f"+L{path.as_posix()}")
    return command


def output_format_switch(value: str) -> str:
    normalized = value.lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        return "+FJ"
    if normalized in {"tif", "tiff"}:
        return "+FT"
    if normalized == "bmp":
        return "+FB"
    return "+FN"


def run_python_postprocess(module_path: Path, contributions: Path, output_dir: Path, parameters_file: Path) -> dict[str, Any]:
    started = time.time()
    if not module_path.exists():
        return {"status": "failed", "message": f"RaySAR Python postprocess module not found: {module_path}"}
    try:
        module = SourceFileLoader("saga_raysar_forbs", module_path.as_posix()).load_module()
        output_dir.mkdir(parents=True, exist_ok=True)
        module.raysar_function(contributions.as_posix(), output_dir.as_posix(), parameters_file.as_posix())
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "module": module_path.as_posix(),
            "contributions_txt": contributions.as_posix(),
            "parameters_file": parameters_file.as_posix(),
            "elapsed_seconds": round(time.time() - started, 3),
        }
    maps_dir = output_dir / "Maps"
    maps = iter_images(maps_dir) if maps_dir.exists() else []
    return {
        "status": "succeeded" if maps else "warning",
        "message": "RaySAR Python postprocess completed." if maps else "Postprocess completed but no map images were detected.",
        "module": module_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "maps_dir": maps_dir.as_posix(),
        "map_images": [path.as_posix() for path in maps],
        "map_count": len(maps),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def determine_status(dry_run: bool, render_result: dict[str, Any], postprocess_result: dict[str, Any]) -> str:
    if dry_run:
        return "dry_run"
    statuses = {str(render_result.get("status")), str(postprocess_result.get("status"))}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return "succeeded"


def status_message(status: str, dry_run: bool, requested_postprocess: bool) -> str:
    if dry_run:
        return "RaySAR synthesis plan generated."
    if status == "succeeded":
        return "RaySAR synthesis completed." if requested_postprocess else "RaySAR contribution rendering completed."
    if status == "warning":
        return "RaySAR synthesis completed with warnings."
    return "RaySAR synthesis failed."


def count_nonempty_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA RaySAR Synthesis Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Dry run: {report.get('dry_run')}",
            f"- POV scene: `{report.get('pov_scene')}`",
            f"- Prepared scene: `{report.get('prepared_scene')}`",
            f"- Adapted POV-Ray: `{report.get('adapted_povray')}`",
            f"- Contributions: `{report.get('contributions_txt')}`",
            f"- Generated maps: `{report.get('generated_output_dir')}`",
            f"- Contribution rows: {report.get('metrics', {}).get('contribution_rows')}",
            f"- Map count: {report.get('map_count')}",
            "",
            "## Notes",
            "",
            "- RaySAR is a physical simulation skill; it needs a RaySAR-compatible `.pov` scene or an existing `Contributions.txt`.",
            "- The default output maps are single-bounce, double-bounce, and all-reflection TIFF files after Python post-processing.",
            "- Domain shift against real SAR collections should be evaluated before writing benefit into memory.",
            "",
        ]
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)

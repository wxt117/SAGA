from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from saga.core.config import load_mapping, save_json, save_text


MODEL_TO_POV_SCENE_VERSION = "saga_model_to_pov_scene_compiler_v1"
SUPPORTED_MODEL_EXTS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae", ".3ds", ".txt", ".xyz"}


@dataclass
class ModelToPOVSceneCompilerSkillConfig:
    name: str = "ModelToPOVSceneCompilerSkill"
    description: str = "Compile common 3D target assets into RaySAR-compatible POV-Ray scenes."
    output_scene_name: str = "generated_scene.pov"
    object_name: str = "SAGA_Target"
    input_up_axis: str = "z"
    pov_height_axis: str = "z"
    center_model: bool = True
    place_on_ground: bool = True
    scene_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_extent_m: float | None = None
    scale_factor: float = 1.0
    max_faces: int = 250000
    max_vertices: int = 250000
    simplify_if_needed: bool = False
    incidence_angle_deg: float = 35.0
    depression_angle_deg: float | None = None
    azimuth_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    range_distance_m: float = 500.0
    sensor_plane_m: float = 100.0
    orthographic_angle: float = 45.0
    add_ground: bool = True
    sar_output_data: bool = True
    sar_intersection: bool = True
    material_reflection: float = 0.3
    material_diffuse: float = 0.3
    material_ambient: float = 0.0
    material_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ground_reflection: float = 0.05
    ground_diffuse: float = 0.2
    ground_ambient: float = 0.0
    pointcloud_delaunay: bool = True
    dry_run_writes_scene: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ModelToPOVSceneCompilerSkillConfig":
        raw = raw or {}
        body = raw.get("model_to_pov_scene", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            output_scene_name=str(body.get("output_scene_name", default.output_scene_name)),
            object_name=str(body.get("object_name", default.object_name)),
            input_up_axis=str(body.get("input_up_axis", default.input_up_axis)).lower(),
            pov_height_axis=str(body.get("pov_height_axis", default.pov_height_axis)).lower(),
            center_model=parse_bool(body.get("center_model", default.center_model)),
            place_on_ground=parse_bool(body.get("place_on_ground", default.place_on_ground)),
            scene_center=parse_vec3(body.get("scene_center"), default.scene_center),
            target_extent_m=parse_optional_float(body.get("target_extent_m", default.target_extent_m)),
            scale_factor=float(body.get("scale_factor", default.scale_factor)),
            max_faces=int(body.get("max_faces", default.max_faces)),
            max_vertices=int(body.get("max_vertices", default.max_vertices)),
            simplify_if_needed=parse_bool(body.get("simplify_if_needed", default.simplify_if_needed)),
            incidence_angle_deg=float(body.get("incidence_angle_deg", default.incidence_angle_deg)),
            depression_angle_deg=parse_optional_float(body.get("depression_angle_deg", default.depression_angle_deg)),
            azimuth_deg=float(body.get("azimuth_deg", default.azimuth_deg)),
            pitch_deg=float(body.get("pitch_deg", default.pitch_deg)),
            roll_deg=float(body.get("roll_deg", default.roll_deg)),
            range_distance_m=float(body.get("range_distance_m", default.range_distance_m)),
            sensor_plane_m=float(body.get("sensor_plane_m", default.sensor_plane_m)),
            orthographic_angle=float(body.get("orthographic_angle", default.orthographic_angle)),
            add_ground=parse_bool(body.get("add_ground", default.add_ground)),
            sar_output_data=parse_bool(body.get("sar_output_data", default.sar_output_data)),
            sar_intersection=parse_bool(body.get("sar_intersection", default.sar_intersection)),
            material_reflection=float(body.get("material_reflection", default.material_reflection)),
            material_diffuse=float(body.get("material_diffuse", default.material_diffuse)),
            material_ambient=float(body.get("material_ambient", default.material_ambient)),
            material_color=parse_vec3(body.get("material_color"), default.material_color),
            ground_reflection=float(body.get("ground_reflection", default.ground_reflection)),
            ground_diffuse=float(body.get("ground_diffuse", default.ground_diffuse)),
            ground_ambient=float(body.get("ground_ambient", default.ground_ambient)),
            pointcloud_delaunay=parse_bool(body.get("pointcloud_delaunay", default.pointcloud_delaunay)),
            dry_run_writes_scene=parse_bool(body.get("dry_run_writes_scene", default.dry_run_writes_scene)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ModelToPOVSceneCompilerSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class ModelToPOVSceneCompilerSkill:
    def __init__(self, config: ModelToPOVSceneCompilerSkillConfig | None = None) -> None:
        self.config = config or ModelToPOVSceneCompilerSkillConfig()

    def run(
        self,
        model_file: str | Path,
        output_dir: str | Path,
        incidence_angle_deg: float | None = None,
        depression_angle_deg: float | None = None,
        azimuth_deg: float | None = None,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
        target_extent_m: float | None = None,
        scale_factor: float | None = None,
        input_up_axis: str | None = None,
        pov_height_axis: str | None = None,
        range_distance_m: float | None = None,
        sensor_plane_m: float | None = None,
        scene_center: tuple[float, float, float] | list[float] | None = None,
        add_ground: bool | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        model_path = Path(model_file).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        scene_path = output_path / self.config.output_scene_name
        if not model_path.exists():
            raise FileNotFoundError(f"3D model file does not exist: {model_path}")
        if model_path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
            raise ValueError(f"Unsupported model extension {model_path.suffix!r}; supported: {sorted(SUPPORTED_MODEL_EXTS)}")

        effective = self.effective_params(
            incidence_angle_deg=incidence_angle_deg,
            depression_angle_deg=depression_angle_deg,
            azimuth_deg=azimuth_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            target_extent_m=target_extent_m,
            scale_factor=scale_factor,
            input_up_axis=input_up_axis,
            pov_height_axis=pov_height_axis,
            range_distance_m=range_distance_m,
            sensor_plane_m=sensor_plane_m,
            scene_center=scene_center,
            add_ground=add_ground,
        )
        mesh = load_model_as_mesh(
            model_path=model_path,
            pointcloud_delaunay=self.config.pointcloud_delaunay,
        )
        mesh = ensure_mesh_budget(
            mesh,
            max_faces=self.config.max_faces,
            max_vertices=self.config.max_vertices,
            simplify_if_needed=self.config.simplify_if_needed,
        )
        vertices = np.asarray(mesh["vertices"], dtype=float)
        faces = np.asarray(mesh["faces"], dtype=np.int64)
        vertices = transform_vertices(vertices=vertices, params=effective, config=self.config)
        stats = mesh_stats(vertices=vertices, faces=faces)
        warnings = build_warnings(stats=stats, config=self.config)
        should_write_scene = (not dry_run) or self.config.dry_run_writes_scene
        if should_write_scene:
            scene_text = render_pov_scene(
                vertices=vertices,
                faces=faces,
                params=effective,
                config=self.config,
                source_model=model_path,
                stats=stats,
            )
            save_text(scene_path, scene_text)

        report = {
            "schema_version": MODEL_TO_POV_SCENE_VERSION,
            "skill": "ModelToPOVSceneCompilerSkill",
            "status": "dry_run" if dry_run else "succeeded",
            "message": (
                "Compiled RaySAR-compatible POV scene from 3D model."
                if should_write_scene
                else "Dry-run inspected model; scene writing was disabled."
            ),
            "dry_run": dry_run,
            "model_file": model_path.as_posix(),
            "output_dir": output_path.as_posix(),
            "compiled_scene": scene_path.as_posix() if should_write_scene else None,
            "generated_output_dir": output_path.as_posix(),
            "effective_params": effective,
            "mesh_stats": stats,
            "metrics": {
                "vertex_count": int(stats["vertex_count"]),
                "face_count": int(stats["face_count"]),
                "extent_max_m": stats["extent_max_m"],
                "incidence_angle_deg": effective["incidence_angle_deg"],
                "azimuth_deg": effective["azimuth_deg"],
            },
            "warnings": warnings,
            "artifacts": {
                "compiled_scene": scene_path.as_posix() if should_write_scene else None,
                "report_json": (output_path / "model_to_pov_scene_report.json").as_posix(),
                "report_md": (output_path / "model_to_pov_scene_report.md").as_posix(),
            },
            "notes": [
                "This compiler fills the gap between common 3D mesh assets and RaySAR .pov scenes.",
                "It reuses RaySAR-style orthographic camera, parallel light source, and SAR_Output_Data/SAR_Intersection switches.",
                "Material reflectivity is a coarse POV-Ray/RaySAR proxy; real electromagnetic material modeling remains outside this skill.",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        save_json(output_path / "model_to_pov_scene_report.json", report)
        save_text(output_path / "model_to_pov_scene_report.md", render_report(report))
        return report

    def effective_params(self, **overrides: Any) -> dict[str, Any]:
        input_up_axis = str(overrides.get("input_up_axis") or self.config.input_up_axis).lower()
        pov_height_axis = str(overrides.get("pov_height_axis") or self.config.pov_height_axis).lower()
        validate_axis(input_up_axis)
        validate_axis(pov_height_axis)
        depression = choose_optional_float(overrides.get("depression_angle_deg"), self.config.depression_angle_deg)
        incidence = choose_optional_float(overrides.get("incidence_angle_deg"), self.config.incidence_angle_deg)
        if depression is not None and overrides.get("incidence_angle_deg") is None:
            incidence = 90.0 - depression
        if incidence is None:
            incidence = self.config.incidence_angle_deg
        incidence = clamp(float(incidence), 1.0, 89.0)
        if depression is None:
            depression = 90.0 - incidence
        center_value = overrides.get("scene_center")
        center = parse_vec3(center_value, self.config.scene_center) if center_value is not None else self.config.scene_center
        return {
            "incidence_angle_deg": round(float(incidence), 6),
            "depression_angle_deg": round(float(depression), 6),
            "azimuth_deg": float(overrides.get("azimuth_deg") if overrides.get("azimuth_deg") is not None else self.config.azimuth_deg),
            "pitch_deg": float(overrides.get("pitch_deg") if overrides.get("pitch_deg") is not None else self.config.pitch_deg),
            "roll_deg": float(overrides.get("roll_deg") if overrides.get("roll_deg") is not None else self.config.roll_deg),
            "target_extent_m": choose_optional_float(overrides.get("target_extent_m"), self.config.target_extent_m),
            "scale_factor": float(overrides.get("scale_factor") if overrides.get("scale_factor") is not None else self.config.scale_factor),
            "input_up_axis": input_up_axis,
            "pov_height_axis": pov_height_axis,
            "range_distance_m": float(
                overrides.get("range_distance_m") if overrides.get("range_distance_m") is not None else self.config.range_distance_m
            ),
            "sensor_plane_m": float(
                overrides.get("sensor_plane_m") if overrides.get("sensor_plane_m") is not None else self.config.sensor_plane_m
            ),
            "scene_center": tuple(float(item) for item in center),
            "add_ground": bool(overrides.get("add_ground") if overrides.get("add_ground") is not None else self.config.add_ground),
        }


def run_model_to_pov_scene_compiler_skill(
    model_file: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    incidence_angle_deg: float | None = None,
    depression_angle_deg: float | None = None,
    azimuth_deg: float | None = None,
    pitch_deg: float | None = None,
    roll_deg: float | None = None,
    target_extent_m: float | None = None,
    scale_factor: float | None = None,
    input_up_axis: str | None = None,
    pov_height_axis: str | None = None,
    range_distance_m: float | None = None,
    sensor_plane_m: float | None = None,
    scene_center: tuple[float, float, float] | list[float] | None = None,
    add_ground: bool | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = ModelToPOVSceneCompilerSkillConfig.from_path(config_path)
    return ModelToPOVSceneCompilerSkill(config).run(
        model_file=model_file,
        output_dir=output_dir,
        incidence_angle_deg=incidence_angle_deg,
        depression_angle_deg=depression_angle_deg,
        azimuth_deg=azimuth_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        target_extent_m=target_extent_m,
        scale_factor=scale_factor,
        input_up_axis=input_up_axis,
        pov_height_axis=pov_height_axis,
        range_distance_m=range_distance_m,
        sensor_plane_m=sensor_plane_m,
        scene_center=scene_center,
        add_ground=add_ground,
        dry_run=dry_run,
    )


def load_model_as_mesh(model_path: Path, pointcloud_delaunay: bool) -> dict[str, np.ndarray]:
    suffix = model_path.suffix.lower()
    if suffix in {".txt", ".xyz"}:
        return load_pointcloud_as_mesh(model_path, pointcloud_delaunay=pointcloud_delaunay)
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("ModelToPOVSceneCompilerSkill requires trimesh for mesh formats.") from exc
    loaded = trimesh.load(model_path.as_posix(), force="scene" if suffix in {".glb", ".gltf", ".dae"} else None)
    meshes = []
    if hasattr(loaded, "geometry"):
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices") and hasattr(geom, "faces"):
                meshes.append(geom)
    elif hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
        meshes.append(loaded)
    if not meshes:
        raise ValueError(f"No triangular mesh geometry found in {model_path}")
    if len(meshes) > 1:
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = meshes[0]
    if getattr(combined, "faces", None) is None or len(combined.faces) == 0:
        raise ValueError(f"Mesh has no faces: {model_path}")
    return {
        "vertices": np.asarray(combined.vertices, dtype=float),
        "faces": np.asarray(combined.faces, dtype=np.int64),
    }


def load_pointcloud_as_mesh(model_path: Path, pointcloud_delaunay: bool) -> dict[str, np.ndarray]:
    points = np.loadtxt(model_path.as_posix(), delimiter=None)
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"Point cloud must contain at least three columns: {model_path}")
    vertices = points[:, :3]
    if not pointcloud_delaunay:
        raise ValueError("Point-cloud input needs pointcloud_delaunay=true to create POV triangles.")
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise RuntimeError("Point-cloud triangulation requires scipy.") from exc
    tri = Delaunay(vertices[:, :2])
    faces = np.asarray(tri.simplices, dtype=np.int64)
    return {"vertices": vertices, "faces": faces}


def ensure_mesh_budget(
    mesh: dict[str, np.ndarray],
    max_faces: int,
    max_vertices: int,
    simplify_if_needed: bool,
) -> dict[str, np.ndarray]:
    faces = mesh["faces"]
    vertices = mesh["vertices"]
    if len(faces) <= max_faces and len(vertices) <= max_vertices:
        return mesh
    if not simplify_if_needed:
        raise ValueError(
            f"Mesh size exceeds budget: vertices={len(vertices)}/{max_vertices}, faces={len(faces)}/{max_faces}. "
            "Raise max_vertices/max_faces or enable simplify_if_needed in the skill config."
        )
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh simplification requires trimesh.") from exc
    tri_mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=False)
    simplified = tri_mesh.simplify_quadric_decimation(int(max_faces))
    return {
        "vertices": np.asarray(simplified.vertices, dtype=float),
        "faces": np.asarray(simplified.faces, dtype=np.int64),
    }


def transform_vertices(
    vertices: np.ndarray,
    params: dict[str, Any],
    config: ModelToPOVSceneCompilerSkillConfig,
) -> np.ndarray:
    transformed = map_up_axis(vertices, input_up_axis=params["input_up_axis"], pov_height_axis=params["pov_height_axis"])
    transformed = transformed * float(params["scale_factor"])
    extent = np.ptp(transformed, axis=0)
    target_extent = params.get("target_extent_m")
    max_extent = float(np.max(extent)) if extent.size else 0.0
    if target_extent and max_extent > 0:
        transformed = transformed * (float(target_extent) / max_extent)
    transformed = apply_euler_rotations(
        transformed,
        height_axis=params["pov_height_axis"],
        azimuth_deg=float(params["azimuth_deg"]),
        pitch_deg=float(params["pitch_deg"]),
        roll_deg=float(params["roll_deg"]),
    )
    center = np.asarray(params["scene_center"], dtype=float)
    height_index = axis_index(params["pov_height_axis"])
    if config.center_model:
        bbox_min = transformed.min(axis=0)
        bbox_max = transformed.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        transformed = transformed - bbox_center
    if config.place_on_ground:
        transformed[:, height_index] -= transformed[:, height_index].min()
        center_for_ground = center.copy()
        center_for_ground[height_index] = 0.0
        transformed = transformed + center_for_ground
    else:
        transformed = transformed + center
    return transformed


def map_up_axis(vertices: np.ndarray, input_up_axis: str, pov_height_axis: str) -> np.ndarray:
    validate_axis(input_up_axis)
    validate_axis(pov_height_axis)
    if input_up_axis == pov_height_axis:
        return vertices.copy()
    input_idx = axis_index(input_up_axis)
    output_idx = axis_index(pov_height_axis)
    order = [0, 1, 2]
    order[output_idx] = input_idx
    order[input_idx] = output_idx
    return vertices[:, order].copy()


def apply_euler_rotations(
    vertices: np.ndarray,
    height_axis: str,
    azimuth_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    rotated = vertices.copy()
    if azimuth_deg:
        rotated = rotated @ rotation_matrix(axis=height_axis, deg=azimuth_deg).T
    if pitch_deg:
        rotated = rotated @ rotation_matrix(axis="x", deg=pitch_deg).T
    if roll_deg:
        rotated = rotated @ rotation_matrix(axis="y", deg=roll_deg).T
    return rotated


def rotation_matrix(axis: str, deg: float) -> np.ndarray:
    theta = math.radians(deg)
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == "x":
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(f"Unsupported axis: {axis}")


def render_pov_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    params: dict[str, Any],
    config: ModelToPOVSceneCompilerSkillConfig,
    source_model: Path,
    stats: dict[str, Any],
) -> str:
    header = render_raysar_header(params=params, config=config)
    mesh = render_mesh2(vertices=vertices, faces=faces, object_name=config.object_name)
    color = format_vec(config.material_color)
    lines = [
        "// SAGA-generated RaySAR POV scene",
        f"// Source model: {source_model.as_posix()}",
        f"// Vertex count: {stats['vertex_count']}; face count: {stats['face_count']}",
        "",
        header,
        "",
        mesh,
        "",
    ]
    if params.get("add_ground"):
        lines.extend(
            [
                "object { ground",
                "  pigment { color rgb <1,1,1> }",
                f"  finish {{ reflection{{{config.ground_reflection:g}}} ambient {config.ground_ambient:g} diffuse {config.ground_diffuse:g} }}",
                "}",
                "",
            ]
        )
    lines.extend(
        [
            f"object {{ {config.object_name}",
            f"  pigment {{ color rgb {color} }}",
            (
                "  finish {"
                f" reflection{{{config.material_reflection:g}}}"
                f" ambient {config.material_ambient:g}"
                f" diffuse {config.material_diffuse:g}"
                " }"
            ),
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def render_raysar_header(params: dict[str, Any], config: ModelToPOVSceneCompilerSkillConfig) -> str:
    center = np.asarray(params["scene_center"], dtype=float)
    height_axis = params["pov_height_axis"]
    incidence_rad = math.radians(float(params["incidence_angle_deg"]))
    horizontal = float(params["range_distance_m"])
    vertical = horizontal / math.tan(incidence_rad)
    satellite = center.copy()
    ground_normal = np.zeros(3, dtype=float)
    if height_axis == "z":
        satellite[1] = center[1] - horizontal
        satellite[2] = center[2] + vertical
        ground_normal[2] = 1.0
        up_axis = "z"
    elif height_axis == "y":
        satellite[1] = center[1] + vertical
        satellite[2] = center[2] - horizontal
        ground_normal[1] = 1.0
        up_axis = "y"
    else:
        satellite[0] = center[0] + vertical
        satellite[1] = center[1] - horizontal
        ground_normal[0] = 1.0
        up_axis = "x"
    output_data = 1 if config.sar_output_data else 0
    intersection = 1 if config.sar_intersection else 0
    plane = float(params["sensor_plane_m"])
    return "\n".join(
        [
            '#include "colors.inc"',
            '#include "finish.inc"',
            "",
            f"global_settings{{SAR_Output_Data {output_data} SAR_Intersection {intersection}}}",
            "",
            f"// Radar sensor (angle of incidence: {float(params['incidence_angle_deg']):g} degrees)",
            "#declare Cam = camera {",
            "  orthographic",
            f"  angle {float(config.orthographic_angle):g}",
            f"  location {format_vec(satellite)}",
            f"  look_at {format_vec(center)}",
            f"  right {plane:g}*x",
            f"  up {plane:g}*{up_axis}",
            "}",
            "camera{Cam}",
            "",
            "light_source {",
            "  0*x",
            "  color rgb <1,1,1>",
            "  parallel",
            f"  translate {format_vec(satellite)}",
            f"  point_at {format_vec(center)}",
            "}",
            "",
            "#declare ground = plane {",
            f"  {format_vec(ground_normal)}",
            "  0",
            "}",
        ]
    )


def render_mesh2(vertices: np.ndarray, faces: np.ndarray, object_name: str) -> str:
    lines = [f"#declare {object_name} = mesh2 {{", "  vertex_vectors {", f"    {len(vertices)},"]
    for vertex in vertices:
        lines.append(f"    {format_vec(vertex)},")
    lines.extend(["  }", "  face_indices {", f"    {len(faces)},"])
    for face in faces:
        lines.append(f"    <{int(face[0])},{int(face[1])},{int(face[2])}>,")
    lines.extend(["  }", "}"])
    return "\n".join(lines)


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "bbox_min": [round(float(item), 6) for item in bbox_min],
        "bbox_max": [round(float(item), 6) for item in bbox_max],
        "extent_m": [round(float(item), 6) for item in extent],
        "extent_max_m": round(float(np.max(extent)), 6),
    }


def build_warnings(stats: dict[str, Any], config: ModelToPOVSceneCompilerSkillConfig) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = [
        {
            "type": "domain_gap",
            "message": "RaySAR scenes are physically interpretable but still synthetic; validate real-to-synthetic domain gap downstream.",
        }
    ]
    if stats["face_count"] > 100000:
        warnings.append(
            {
                "type": "large_mesh",
                "message": "Large mesh may make POV-Ray/RaySAR rendering slow; consider simplification or lower render resolution.",
            }
        )
    if config.material_reflection <= 0:
        warnings.append(
            {
                "type": "low_reflection",
                "message": "Material reflection is near zero; RaySAR scattering contributions may be weak.",
            }
        )
    return warnings


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Model To POV Scene Compiler",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Dry run: {report.get('dry_run')}",
        f"- Model: `{report.get('model_file')}`",
        f"- Compiled scene: `{report.get('compiled_scene')}`",
        f"- Vertex count: {report.get('mesh_stats', {}).get('vertex_count')}",
        f"- Face count: {report.get('mesh_stats', {}).get('face_count')}",
        f"- Extent: `{report.get('mesh_stats', {}).get('extent_m')}`",
        f"- Incidence angle: {report.get('effective_params', {}).get('incidence_angle_deg')}",
        f"- Depression angle: {report.get('effective_params', {}).get('depression_angle_deg')}",
        f"- Azimuth: {report.get('effective_params', {}).get('azimuth_deg')}",
        "",
        "## Warnings",
        "",
    ]
    for warning in report.get("warnings") or []:
        lines.append(f"- `{warning.get('type')}`: {warning.get('message')}")
    if not report.get("warnings"):
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def parse_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def choose_optional_float(override: Any, default: Any) -> float | None:
    if override is not None and override != "":
        return float(override)
    if default is None or default == "":
        return None
    return float(default)


def parse_vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"Expected 3 vector values, got {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def format_vec(value: Any) -> str:
    values = [float(item) for item in value]
    return f"<{values[0]:.8g},{values[1]:.8g},{values[2]:.8g}>"


def validate_axis(axis: str) -> None:
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Axis must be one of x/y/z, got {axis!r}")


def axis_index(axis: str) -> int:
    validate_axis(axis)
    return {"x": 0, "y": 1, "z": 2}[axis]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

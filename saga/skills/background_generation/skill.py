from __future__ import annotations

import json
import subprocess
import sys
import time
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.data.discovery import iter_images


BACKGROUND_GENERATION_VERSION = "saga_background_generation_run_v1"


@dataclass
class BackgroundGenerationSkillConfig:
    name: str = "BackgroundGenerationSkill"
    description: str = "Generate SAR background scenes with the packaged segmentation-controlled FLUX ControlNet model."
    project_dir: str = "myproject/scene_gen_segment_large"
    num_images: int = 4
    split: str = "all"
    top_k: int = 64
    selection_seed: int = 1026
    seed: int = 1026
    cns: float = 1.3
    steps: int = 20
    scale: float = 3.5
    width: int = 512
    height: int = 512
    cuda: str | None = None
    cpu_threads: int = 8
    extra_prompt: str = ""
    clear_cache_each_image: bool = False
    show_top: int = 8

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "BackgroundGenerationSkillConfig":
        raw = raw or {}
        body = raw.get("background_generation", raw)
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            project_dir=str(body.get("project_dir", default.project_dir)),
            num_images=int(body.get("num_images", default.num_images)),
            split=str(body.get("split", default.split)),
            top_k=int(body.get("top_k", default.top_k)),
            selection_seed=int(body.get("selection_seed", default.selection_seed)),
            seed=int(body.get("seed", default.seed)),
            cns=float(body.get("cns", default.cns)),
            steps=int(body.get("steps", default.steps)),
            scale=float(body.get("scale", default.scale)),
            width=int(body.get("width", default.width)),
            height=int(body.get("height", default.height)),
            cuda=parse_optional_str(body.get("cuda", default.cuda)),
            cpu_threads=int(body.get("cpu_threads", default.cpu_threads)),
            extra_prompt=str(body.get("extra_prompt", default.extra_prompt)),
            clear_cache_each_image=parse_bool(body.get("clear_cache_each_image", default.clear_cache_each_image)),
            show_top=int(body.get("show_top", default.show_top)),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "BackgroundGenerationSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class BackgroundGenerationSkill:
    def __init__(self, config: BackgroundGenerationSkillConfig | None = None) -> None:
        self.config = config or BackgroundGenerationSkillConfig()

    def run(
        self,
        scene_prompt: str,
        output_dir: str | Path,
        project_dir: str | Path | None = None,
        num_images: int | None = None,
        split: str | None = None,
        top_k: int | None = None,
        selection_seed: int | None = None,
        seed: int | None = None,
        cns: float | None = None,
        steps: int | None = None,
        scale: float | None = None,
        width: int | None = None,
        height: int | None = None,
        cuda: str | None = None,
        cpu_threads: int | None = None,
        extra_prompt: str | None = None,
        clear_cache_each_image: bool | None = None,
        show_top: int | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        project = Path(project_dir or self.config.project_dir).expanduser().resolve()
        generated_dir = output / "background_images"
        run_dir = output / "scene_gen_run"

        params = {
            "scene_prompt": scene_prompt.strip(),
            "num_images": positive_int(num_images, self.config.num_images),
            "split": split or self.config.split,
            "top_k": positive_int(top_k, self.config.top_k),
            "selection_seed": positive_int(selection_seed, self.config.selection_seed),
            "seed": positive_int(seed, self.config.seed),
            "cns": float(cns if cns is not None else self.config.cns),
            "steps": positive_int(steps, self.config.steps),
            "scale": float(scale if scale is not None else self.config.scale),
            "width": positive_int(width, self.config.width),
            "height": positive_int(height, self.config.height),
            "cuda": cuda if cuda is not None else self.config.cuda,
            "cpu_threads": positive_int(cpu_threads, self.config.cpu_threads),
            "extra_prompt": extra_prompt if extra_prompt is not None else self.config.extra_prompt,
            "clear_cache_each_image": self.config.clear_cache_each_image
            if clear_cache_each_image is None
            else bool(clear_cache_each_image),
            "show_top": int(show_top if show_top is not None else self.config.show_top),
        }
        validate_project(project)
        command = build_command(
            project=project,
            generated_dir=generated_dir,
            run_dir=run_dir,
            params=params,
            dry_run=dry_run,
        )
        status_path = output / "background_generation_status.json"
        log_dir = output / "logs"
        save_json(
            status_path,
            {
                "schema_version": BACKGROUND_GENERATION_VERSION,
                "status": "running",
                "dry_run": bool(dry_run),
                "project_dir": project.as_posix(),
                "scene_prompt": params["scene_prompt"],
                "target_count": params["num_images"],
                "generated_output_dir": generated_dir.as_posix(),
                "run_dir": run_dir.as_posix(),
                "command": command,
                "logs": {
                    "stdout": (log_dir / "background_generation.stdout.log").as_posix(),
                    "stderr": (log_dir / "background_generation.stderr.log").as_posix(),
                    "command": (log_dir / "background_generation.command.txt").as_posix(),
                },
                "started_at": wall_time(started),
                "updated_at": wall_time(),
            },
        )
        proc = run_logged_command(
            command=command,
            cwd=project,
            log_dir=log_dir,
            prefix="background_generation",
        )
        selection = load_selection(run_dir / "selection.json")
        generated_images = iter_images(generated_dir) if generated_dir.exists() else []
        status = "dry_run" if dry_run and proc["returncode"] == 0 else "succeeded" if proc["returncode"] == 0 else "failed"
        message = (
            "Background generation dry-run selected segmentation controls."
            if status == "dry_run"
            else "Background generation completed."
            if status == "succeeded"
            else "Background generation failed."
        )
        report = {
            "schema_version": BACKGROUND_GENERATION_VERSION,
            "skill": "BackgroundGenerationSkill",
            "status": status,
            "message": message,
            "dry_run": bool(dry_run),
            "project_dir": project.as_posix(),
            "scene_prompt": params["scene_prompt"],
            "output_dir": output.as_posix(),
            "generated_output_dir": generated_dir.as_posix(),
            "run_dir": run_dir.as_posix(),
            "target_count": params["num_images"],
            "generated_count": len(generated_images),
            "params": params,
            "selection": selection,
            "commands": [command],
            "command_results": [
                {
                    "command": command,
                    "returncode": proc["returncode"],
                    "stdout_path": proc["stdout_path"],
                    "stderr_path": proc["stderr_path"],
                    "command_path": proc["command_path"],
                    "stdout_tail": proc["stdout_tail"],
                    "stderr_tail": proc["stderr_tail"],
                }
            ],
            "elapsed_seconds": round(time.time() - started, 3),
            "artifacts": {
                "report_json": (output / "background_generation_report.json").as_posix(),
                "report_md": (output / "background_generation_report.md").as_posix(),
                "status_json": status_path.as_posix(),
                "selection_json": (run_dir / "selection.json").as_posix(),
                "prompts_txt": (run_dir / "prompts.txt").as_posix(),
                "generated_dir": generated_dir.as_posix(),
                "run_dir": run_dir.as_posix(),
                "log_dir": log_dir.as_posix(),
            },
            "notes": [
                "This skill uses the existing trained segmentation-control background generator; it does not train user models.",
                "Generated backgrounds are scene assets for background diversity or later composition, not default target-chip classification augmentation.",
                "Downstream benefit should be evaluated with task-specific composition/detection/classification protocols before being written into memory.",
            ],
        }
        save_json(output / "background_generation_report.json", report)
        save_text(output / "background_generation_report.md", render_report(report))
        save_json(
            status_path,
            {
                **report,
                "updated_at": wall_time(),
                "completed_at": wall_time(),
            },
        )
        return report


def run_background_generation_skill(
    scene_prompt: str,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    num_images: int | None = None,
    split: str | None = None,
    top_k: int | None = None,
    selection_seed: int | None = None,
    seed: int | None = None,
    cns: float | None = None,
    steps: int | None = None,
    scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    cuda: str | None = None,
    cpu_threads: int | None = None,
    extra_prompt: str | None = None,
    clear_cache_each_image: bool | None = None,
    show_top: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = BackgroundGenerationSkillConfig.from_path(config_path)
    return BackgroundGenerationSkill(config).run(
        scene_prompt=scene_prompt,
        output_dir=output_dir,
        project_dir=project_dir,
        num_images=num_images,
        split=split,
        top_k=top_k,
        selection_seed=selection_seed,
        seed=seed,
        cns=cns,
        steps=steps,
        scale=scale,
        width=width,
        height=height,
        cuda=cuda,
        cpu_threads=cpu_threads,
        extra_prompt=extra_prompt,
        clear_cache_each_image=clear_cache_each_image,
        show_top=show_top,
        dry_run=dry_run,
    )


def build_command(project: Path, generated_dir: Path, run_dir: Path, params: dict[str, Any], dry_run: bool) -> list[str]:
    command = [
        sys.executable,
        (project / "generate_scene.py").as_posix(),
        params["scene_prompt"] or "SAR background scene",
        "--num-images",
        str(params["num_images"]),
        "--split",
        str(params["split"]),
        "--top-k",
        str(params["top_k"]),
        "--selection-seed",
        str(params["selection_seed"]),
        "--seed",
        str(params["seed"]),
        "--cns",
        str(params["cns"]),
        "--steps",
        str(params["steps"]),
        "--scale",
        str(params["scale"]),
        "--width",
        str(params["width"]),
        "--height",
        str(params["height"]),
        "--output-dir",
        generated_dir.as_posix(),
        "--run-dir",
        run_dir.as_posix(),
        "--cpu-threads",
        str(params["cpu_threads"]),
        "--show-top",
        str(params["show_top"]),
    ]
    if params.get("cuda") not in (None, ""):
        command.extend(["--cuda", str(params["cuda"])])
    if params.get("extra_prompt"):
        command.extend(["--extra-prompt", str(params["extra_prompt"])])
    if params.get("clear_cache_each_image"):
        command.append("--clear-cache-each-image")
    if dry_run:
        command.append("--dry-run")
    return command


def validate_project(project: Path) -> None:
    required = [
        project / "generate_scene.py",
        project / "config/defaults.json",
        project / "data/control_index.jsonl",
        project / "models/seg_hh_sarlora_e36r065_semblend_oldrecipe_c6s4.safetensors",
        project / "models/flux1-dev2pro_sarhh_full1x_e36_lora0p65.safetensors",
        project / "models/clip_l.safetensors",
        project / "models/t5xxl_fp16.safetensors",
        project / "models/ae.sft",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Background generation project is incomplete: " + "; ".join(missing))


def load_selection(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def positive_int(value: Any, default: int) -> int:
    parsed = int(value if value is not None else default)
    if parsed <= 0:
        raise ValueError("Expected a positive integer.")
    return parsed


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def run_logged_command(command: list[str], cwd: Path, log_dir: Path, prefix: str) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{prefix}.stdout.log"
    stderr_path = log_dir / f"{prefix}.stderr.log"
    command_path = log_dir / f"{prefix}.command.txt"
    command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=cwd.as_posix(),
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    return {
        "returncode": completed.returncode,
        "stdout_path": stdout_path.as_posix(),
        "stderr_path": stderr_path.as_posix(),
        "command_path": command_path.as_posix(),
        "stdout_tail": read_tail(stdout_path),
        "stderr_tail": read_tail(stderr_path),
    }


def read_tail(path: Path, max_chars: int = 6000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_chars))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def wall_time(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or time.time()))


def render_report(report: dict[str, Any]) -> str:
    selected = report.get("selection", {}).get("selected", [])
    lines = [
        "# SAGA Background Generation Skill",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Prompt: {report.get('scene_prompt')}",
        f"- Target count: {report.get('target_count')}",
        f"- Generated count: {report.get('generated_count')}",
        f"- Output: `{report.get('generated_output_dir')}`",
        f"- Run dir: `{report.get('run_dir')}`",
        "",
        "## Selected Controls",
        "",
    ]
    if selected:
        for row in selected[:20]:
            record = row.get("record") or {}
            lines.append(f"- `{record.get('id')}` score={row.get('score')} seed={row.get('generation_seed')}")
    else:
        lines.append("- No selection metadata was available.")
    lines.extend(["", "## Command", "", f"`{report.get('commands', [[]])[0]}`", ""])
    return "\n".join(lines)

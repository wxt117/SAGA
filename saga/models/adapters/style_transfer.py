from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class StyleTransferAdapterConfig:
    entrypoint: str = "myproject/stytransfer/style_transfer_general.py"
    conda_env: str = "sd3"
    model_path: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda:0"
    image_size: int = 512
    steps: int = 200
    lr: float = 0.05
    content_weight: float = 0.25
    seed: int = 2025
    self_layers: str = "10,16"
    limit: int = 0
    overwrite: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "StyleTransferAdapterConfig":
        raw = raw or {}
        if "style_transfer" in raw and isinstance(raw["style_transfer"], dict):
            raw = raw["style_transfer"]
        if "adapter" in raw and isinstance(raw["adapter"], dict):
            raw = raw["adapter"]
        return cls(
            entrypoint=str(raw.get("entrypoint", "myproject/stytransfer/style_transfer_general.py")),
            conda_env=str(raw.get("conda_env", "sd3")),
            model_path=str(raw.get("model_path", "runwayml/stable-diffusion-v1-5")),
            device=str(raw.get("device", "cuda:0")),
            image_size=int(raw.get("image_size", 512)),
            steps=int(raw.get("steps", 200)),
            lr=float(raw.get("lr", 0.05)),
            content_weight=float(raw.get("content_weight", raw.get("weight", 0.25))),
            seed=int(raw.get("seed", 2025)),
            self_layers=str(raw.get("self_layers", "10,16")),
            limit=int(raw.get("limit", 0)),
            overwrite=parse_bool(raw.get("overwrite", False)),
        )


class StyleTransferAdapter:
    def __init__(self, config: StyleTransferAdapterConfig | None = None) -> None:
        self.config = config or StyleTransferAdapterConfig()

    def build_command(self, content: str | Path, style: str | Path, output_dir: str | Path) -> list[str]:
        cfg = self.config
        entrypoint = resolve_repo_path(cfg.entrypoint)
        command = [
            "conda",
            "run",
            "-n",
            cfg.conda_env,
            "python",
            entrypoint.as_posix(),
            "--cnt",
            str(content),
            "--sty",
            str(style),
            "--output-dir",
            str(output_dir),
            "--model-path",
            cfg.model_path,
            "--device",
            cfg.device,
            "--image-size",
            str(cfg.image_size),
            "--steps",
            str(cfg.steps),
            "--lr",
            str(cfg.lr),
            "--weight",
            str(cfg.content_weight),
            "--seed",
            str(cfg.seed),
            "--self-layers",
            cfg.self_layers,
        ]
        if cfg.limit > 0:
            command.extend(["--limit", str(cfg.limit)])
        if cfg.overwrite:
            command.append("--overwrite")
        return command

    def run(self, content: str | Path, style: str | Path, output_dir: str | Path, dry_run: bool = False) -> dict:
        command = self.build_command(content=content, style=style, output_dir=output_dir)
        if dry_run:
            return {"status": "dry_run", "command": command}
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        return {
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (REPO_ROOT / value).resolve()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

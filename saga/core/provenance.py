from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from saga.core.config import save_json


RUN_PROVENANCE_VERSION = "saga_run_provenance_v1"


def build_run_provenance(output_dir: str | Path | None = None) -> dict[str, Any]:
    cwd = Path.cwd().resolve()
    report = {
        "schema_version": RUN_PROVENANCE_VERSION,
        "created_at": round(time.time(), 3),
        "cwd": cwd.as_posix(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": git_summary(cwd),
    }
    if output_dir:
        save_json(Path(output_dir).expanduser().resolve() / "run_provenance.json", report)
    return report


def git_summary(cwd: Path) -> dict[str, Any]:
    root = run_git(cwd, ["rev-parse", "--show-toplevel"])
    if not root:
        return {"available": False}
    return {
        "available": True,
        "root": root,
        "commit": run_git(cwd, ["rev-parse", "HEAD"]),
        "status_short": run_git(cwd, ["status", "--short"]),
    }


def run_git(cwd: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()

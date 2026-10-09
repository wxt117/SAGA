"""Reject credentials and accidental heavyweight artifacts before publication."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
)
PRIVATE_PATH = re.compile(r"(?:^|[\s\"'])/(?:home|archive|mnt|opt)/[A-Za-z0-9_.+\-]+")
MAX_TRACKED_BYTES = 100 * 1024 * 1024


def tracked_files() -> list[Path]:
    try:
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=False, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    return [ROOT / raw for raw in output.decode().split("\0") if raw]


def main() -> int:
    files = tracked_files()
    findings: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            findings.append(f"large tracked file ({path.stat().st_size} bytes): {path.relative_to(ROOT)}")
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            findings.append(f"credential-like text in {path.relative_to(ROOT)}")
        if PRIVATE_PATH.search(content):
            findings.append(f"absolute machine-local path in {path.relative_to(ROOT)}")
    if findings:
        print("Publication check failed:")
        print("\n".join(f"- {finding}" for finding in sorted(set(findings))))
        return 1
    print(f"Publication check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

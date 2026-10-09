from __future__ import annotations

from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TXT_EXTS = {".txt"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_images(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if is_image(p)), key=lambda p: natural_key(p.as_posix()))


def natural_key(value: str) -> list[object]:
    parts: list[object] = []
    current = ""
    is_digit = False
    for char in value:
        if char.isdigit():
            if current and not is_digit:
                parts.append(current)
                current = ""
            current += char
            is_digit = True
        else:
            if current and is_digit:
                parts.append(int(current))
                current = ""
            current += char
            is_digit = False
    if current:
        parts.append(int(current) if is_digit else current)
    return parts


def split_roots(dataset_root: Path, splits: dict[str, str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for split, rel in splits.items():
        root = (dataset_root / rel).resolve() if rel != "." else dataset_root.resolve()
        roots[split] = root
    return roots

"""Create a tiny synthetic SAR-like dataset for the README quickstart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parent / "demo_dataset"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        rng = np.random.default_rng(1000 + index)
        image = np.clip(rng.normal(22, 7, (64, 64)), 0, 255).astype(np.uint8)
        yy, xx = np.ogrid[:64, :64]
        target = (xx - (20 + index * 2)) ** 2 + (yy - 32) ** 2 < 8**2
        image[target] = np.clip(image[target] + 150, 0, 255)
        polarization = ("hh", "hv", "vv")[index % 3]
        path = root / f"vehicle_850_20_{index * 30}_0.5_{polarization}.png"
        Image.fromarray(image, mode="L").save(path)
        path.with_suffix(".txt").write_text(
            f"SAR image, vehicle, 0.5m, Ku-band, polarization {polarization}, depression 20, azimuth {index * 30}\n",
            encoding="utf-8",
        )
    print(root)


if __name__ == "__main__":
    main()

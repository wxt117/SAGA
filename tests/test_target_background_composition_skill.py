from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from saga.skills.target_background_composition.skill import run_target_background_composition_skill


try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - unittest skip handles this.
    Image = None
    ImageDraw = None


@unittest.skipIf(Image is None, "Pillow is required for image fixture generation")
class TargetBackgroundCompositionSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.targets = self.root / "targets"
        self.backgrounds = self.root / "backgrounds"
        self.targets.mkdir(parents=True)
        self.backgrounds.mkdir(parents=True)
        create_target(self.targets / "target_001.png")
        create_background(self.backgrounds / "background_001.png", base=28)
        create_background(self.backgrounds / "background_002.png", base=42)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dry_run_writes_plan(self) -> None:
        out = self.root / "dry"
        report = run_target_background_composition_skill(
            target_dir=self.targets,
            background_dir=self.backgrounds,
            output_dir=out,
            target_count=3,
            dry_run=True,
        )
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["target_count"], 3)
        self.assertTrue((out / "composition_plan.json").exists())
        self.assertTrue((out / "target_background_composition_report.json").exists())

    def test_real_run_writes_images_masks_and_manifest(self) -> None:
        out = self.root / "real"
        report = run_target_background_composition_skill(
            target_dir=self.targets,
            background_dir=self.backgrounds,
            output_dir=out,
            target_count=2,
            blend_mode="feather",
            placement_policy="center",
            dry_run=False,
        )
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["generated_count"], 2)
        self.assertEqual(len(list((out / "composed_images").glob("*.png"))), 2)
        self.assertEqual(len(list((out / "masks").glob("*.png"))), 2)
        self.assertTrue((out / "composition_manifest.jsonl").exists())
        self.assertTrue((out / "previews" / "composition_000001_preview.png").exists())


def create_target(path: Path) -> None:
    image = Image.new("L", (64, 64), color=4)
    draw = ImageDraw.Draw(image)
    draw.ellipse((16, 20, 48, 44), fill=220)
    draw.rectangle((28, 12, 36, 52), fill=180)
    image.save(path)


def create_background(path: Path, *, base: int) -> None:
    image = Image.new("L", (128, 128), color=base)
    draw = ImageDraw.Draw(image)
    for x in range(0, 128, 8):
        shade = base + (x % 24)
        draw.line((x, 0, x, 127), fill=shade)
    for y in range(0, 128, 11):
        draw.line((0, y, 127, y), fill=base + 8)
    image.save(path)


if __name__ == "__main__":
    unittest.main()

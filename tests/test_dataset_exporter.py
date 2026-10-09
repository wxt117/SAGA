from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from saga.exporter.dataset_exporter import export_augmented_dataset


class DatasetExporterTest(unittest.TestCase):
    def test_export_can_write_caption_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "generated"
            image_dir = source / "000001"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "sample.png"
            Image.new("L", (16, 16), color=32).save(image_path)
            image_path.with_suffix(".txt").write_text("SAR image, ZJGC-X, hh polarization\n", encoding="utf-8")

            output = root / "export"
            report = export_augmented_dataset(
                source_dir=source,
                output_dir=output,
                recipe_id="test_recipe",
                task="diffusion_lora_generation",
                write_caption_sidecars=True,
            )

            exported_image = output / "images" / "000001" / "sample.png"
            exported_caption = exported_image.with_suffix(".txt")
            self.assertTrue(exported_image.exists())
            self.assertTrue(exported_caption.exists())
            self.assertEqual(exported_caption.read_text(encoding="utf-8").strip(), "SAR image, ZJGC-X, hh polarization")
            self.assertEqual(report["caption_sidecar_count"], 1)

            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"]["caption"], "SAR image, ZJGC-X, hh polarization")
            self.assertEqual(rows[0]["metadata"]["polarization"], "hh")
            self.assertEqual(rows[0]["image"]["caption_relative_path"], "images/000001/sample.txt")

    def test_export_does_not_write_caption_sidecars_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "generated"
            source.mkdir()
            image_path = source / "sample.png"
            Image.new("L", (16, 16), color=32).save(image_path)
            image_path.with_suffix(".txt").write_text("SAR image, target\n", encoding="utf-8")

            output = root / "export"
            report = export_augmented_dataset(
                source_dir=source,
                output_dir=output,
                recipe_id="test_recipe",
                task="augmentation",
            )

            self.assertTrue((output / "images" / "sample.png").exists())
            self.assertFalse((output / "images" / "sample.txt").exists())
            self.assertEqual(report["caption_sidecar_count"], 0)


if __name__ == "__main__":
    unittest.main()

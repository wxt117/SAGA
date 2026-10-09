from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from saga.core.config import load_mapping, resolve_resource_path
from saga.core.saga_config import default_saga_config


class PackagedConfigTest(unittest.TestCase):
    def test_builtin_config_includes_public_composition_skill(self) -> None:
        self.assertEqual(
            default_saga_config()["skill_configs"]["target_background_composition"],
            "configs/skills/target_background_composition.yaml",
        )

    def test_resolves_installed_share_config_when_cwd_has_no_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            installed = prefix / "share" / "saga" / "configs" / "saga.yaml"
            installed.parent.mkdir(parents=True)
            installed.write_text("saga:\n  execution:\n    default_dry_run: true\n", encoding="utf-8")
            with patch("saga.core.config.sys.prefix", prefix), patch("saga.core.config.Path.exists", autospec=True) as exists:
                exists.side_effect = lambda path: Path(path) == installed
                resolved = resolve_resource_path("configs/saga.yaml")

            self.assertEqual(resolved, installed)
            self.assertTrue(load_mapping(installed)["saga"]["execution"]["default_dry_run"])


if __name__ == "__main__":
    unittest.main()

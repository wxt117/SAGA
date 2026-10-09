from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from saga.agent.runtime import run_agent
from saga.core.config import load_mapping


try:
    from PIL import Image
except Exception:  # pragma: no cover - unittest skip handles this.
    Image = None


@unittest.skipIf(Image is None, "Pillow is required for image fixture generation")
class AgentRunMatrixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.memory_dir = self.root / "memory"
        self.dataset = self.root / "vehicle" / "ZJGC-X"
        self.style = self.root / "style" / "Boeing707"
        self.content = self.root / "content" / "B747"
        self.background = self.root / "backgrounds" / "scene_bank"
        self.model_dir = self.root / "models"
        self.model_file = self.model_dir / "target.obj"
        self.saga_config = self.root / "saga_test_config.yaml"
        self.saga_config.write_text(
            "\n".join(
                [
                    "saga:",
                    "  llm:",
                    "    config: null",
                    "    use_intent_by_default: false",
                    "    use_planner_by_default: false",
                    "  skill_configs:",
                    "    style_transfer: configs/skills/style_transfer.yaml",
                    "    diffusion_lora: configs/skills/diffusion_lora.yaml",
                    "    geodiff_sar: configs/skills/geodiff_sar.yaml",
                    "    traditional_augmentation: configs/skills/traditional_augmentation.yaml",
                    "    classification_evaluation: configs/skills/classification_evaluation.yaml",
                    "    background_generation: configs/skills/background_generation.yaml",
                    "    target_background_composition: configs/skills/target_background_composition.yaml",
                    "    model_to_pov_scene: configs/skills/model_to_pov_scene.yaml",
                    "    raysar_sweep: configs/skills/raysar_sweep.yaml",
                    "    raysar: configs/skills/raysar.yaml",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        create_captioned_images(self.dataset, prefix="ZJGC", count=6)
        create_captioned_images(self.style, prefix="B707", count=3)
        create_captioned_images(self.content, prefix="B747", count=4)
        create_captioned_images(self.background, prefix="BG", count=3)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_file.write_text(
            "\n".join(
                [
                    "o target",
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "f 1 2 3",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_natural_language_agent_run_matrix(self) -> None:
        cases = [
            {
                "id": "traditional",
                "request": f"对 {self.dataset} 做快速传统SAR增广，生成3张",
                "task": "traditional_augmentation",
                "skill": "TraditionalAugmentationSkill",
                "recipe_task": "traditional_augmentation",
            },
            {
                "id": "diffusion_lora",
                "request": f"用 {self.dataset} 训练LoRA做有极化方式的扩散模型生成，后面的字母代表极化方式，pauli不作为数据集，生成3张，训练1轮",
                "task": "diffusion_lora_generation",
                "skill": "DiffusionLoRAGenerationSkill",
                "recipe_task": "diffusion_lora_generation",
            },
            {
                "id": "gan",
                "request": f"用 {self.dataset} 做GAN图生图快速生成，生成3张，训练1轮",
                "task": "gan_generation",
                "skill": "GANImageToImageSkill",
                "recipe_task": "gan_generation",
            },
            {
                "id": "style_transfer",
                "request": f"把 {self.style} 作为指导图，将707的特征迁移到 {self.content} 的数据上",
                "task": "style_transfer",
                "skill": "StyleTransferSkill",
                "recipe_task": "style_transfer",
            },
            {
                "id": "geodiff",
                "request": f"用 {self.dataset} 作为真实SAR训练数据，用 {self.model_file} 作为3D模型，使用GeoDiff-SAR做目标稀疏方位角补全，生成2张，方位角0和30，下视角20，训练1轮",
                "task": "geodiff_sar_generation",
                "skill": "GeoDiffSARSkill",
                "recipe_task": "geodiff_sar_generation",
            },
            {
                "id": "raysar",
                "request": f"用 {self.model_file} 做RaySAR物理仿真，方位角0和30，下视角20，生成2张",
                "task": "raysar_synthesis",
                "skill": "RaySARSweepSynthesisSkill",
                "recipe_task": "raysar_synthesis",
            },
            {
                "id": "target_background_composition",
                "request": f"把 {self.dataset} 作为目标图，{self.background} 作为背景图，做目标和背景图像融合，生成2张，使用羽化融合",
                "task": "target_background_composition",
                "skill": "TargetBackgroundCompositionSkill",
                "recipe_task": "target_background_composition",
            },
        ]
        for case in cases:
            with self.subTest(case=case["id"]):
                state = run_agent(
                    request=case["request"],
                    output_dir=self.root / "runs" / case["id"],
                    memory_dir=self.memory_dir,
                    saga_config_path=self.saga_config,
                    use_memory=False,
                    dry_run=True,
                    sample_limit=8,
                    image_probe_limit=8,
                    bridge_sample_limit=16,
                    execute=True,
                )
                self.assertEqual(state["task"], case["task"])
                self.assertEqual(state["augmentation_selected_skill"], case["skill"])
                self.assertEqual(state["augmentation_selected_recipe_task"], case["recipe_task"])
                self.assertTrue(Path(state["recipe_path"]).exists())
                self.assertNotEqual(state.get("plan_critic_status"), "blocked")
                execution_path = Path(state["execution_report_path"])
                self.assertTrue(execution_path.exists())
                execution = load_mapping(execution_path)
                self.assertTrue(execution.get("skill_report_validation", {}).get("valid"), execution.get("skill_report_validation"))
                if case["id"] == "geodiff":
                    recipe = load_mapping(state["recipe_path"])
                    geometry = recipe.get("inputs", {}).get("raysar_geometry", {})
                    self.assertEqual(geometry.get("azimuth_values"), [0, 30])
                    self.assertEqual(geometry.get("depressions"), [20])


def create_captioned_images(root: Path, *, prefix: str, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    polarizations = ["HH", "HV", "VH", "VV", "HH", "HV"]
    for idx in range(count):
        pol = polarizations[idx % len(polarizations)]
        stem = f"{prefix}_850_20_{idx * 10}_0.5_{pol}"
        image = Image.new("L", (32, 32), color=20 + idx * 20)
        image.save(root / f"{stem}.png")
        (root / f"{stem}.txt").write_text(
            f"SAR image, {prefix}, 0.5m, Ku-band, polarization {pol}, depression 20, azimuth {idx * 10}",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

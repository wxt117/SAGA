from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, resolve_resource_path


DEFAULT_SAGA_CONFIG = "configs/saga.yaml"


def load_saga_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_resource_path(path or DEFAULT_SAGA_CONFIG)
    if not config_path.exists():
        return default_saga_config()
    raw = load_mapping(config_path)
    base = default_saga_config()
    return deep_merge(base, raw.get("saga", raw))


def default_saga_config() -> dict[str, Any]:
    return {
        "planner": {
            "auto_apply_selected_augmentation_plan": True,
            "candidate_plan_limit": 8,
            "prefer_executable_skills": True,
        },
        "llm": {
            "config": None,
            "use_intent_by_default": False,
            "use_planner_by_default": False,
        },
        "execution": {
            "default_dry_run": True,
            "record_provenance": True,
            "max_repair_trials": 3,
        },
        "evaluation_policy": {
            "default_benefit_probe_mode": "lightweight_metrics",
            "classification_evaluation_requires_explicit_request": True,
            "lightweight_metrics": [
                "quality",
                "distribution",
                "sar_artifacts",
                "duplicate_near_duplicate",
                "leakage_when_reference_available",
            ],
        },
        "skill_configs": {
            "style_transfer": "configs/skills/style_transfer.yaml",
            "diffusion_lora": "configs/skills/diffusion_lora.yaml",
            "geodiff_sar": "configs/skills/geodiff_sar.yaml",
            "gaussian_splatting": "configs/skills/gaussian_splatting.yaml",
            "traditional_augmentation": "configs/skills/traditional_augmentation.yaml",
            "classification_evaluation": "configs/skills/classification_evaluation.yaml",
            "background_generation": "configs/skills/background_generation.yaml",
            "target_background_composition": "configs/skills/target_background_composition.yaml",
            "model_to_pov_scene": "configs/skills/model_to_pov_scene.yaml",
            "raysar_sweep": "configs/skills/raysar_sweep.yaml",
            "raysar": "configs/skills/raysar.yaml",
        },
        "memory": {
            "enabled": True,
            "dir": "runs/memory",
            "retrieval_limit": 5,
        },
        "hardware": {
            "preferred_conda_env": "sd3",
            "default_cuda_devices": None,
        },
    }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

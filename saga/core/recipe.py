from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping


RECIPE_VERSION = "saga_recipe_v1"
EXECUTION_REPORT_VERSION = "saga_recipe_execution_report_v1"


@dataclass
class RecipeStep:
    id: str
    skill: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    condition: str | None = None
    description: str = ""

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "RecipeStep":
        return cls(
            id=str(raw.get("id") or raw.get("step_id")),
            skill=str(raw["skill"]),
            params=raw.get("params") or {},
            depends_on=list(raw.get("depends_on") or []),
            condition=raw.get("condition"),
            description=str(raw.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SagaRecipe:
    recipe_id: str
    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    pipeline: list[RecipeStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RECIPE_VERSION
    planner: str = "saga_rule_recipe_generator_v1"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SagaRecipe":
        body = raw.get("recipe", raw)
        return cls(
            recipe_id=str(body["recipe_id"]),
            task=str(body.get("task", "unknown")),
            inputs=body.get("inputs") or {},
            filters=body.get("filters") or {},
            pipeline=[RecipeStep.from_mapping(item) for item in body.get("pipeline", [])],
            metadata=body.get("metadata") or {},
            schema_version=body.get("schema_version", RECIPE_VERSION),
            planner=body.get("planner", "saga_rule_recipe_generator_v1"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SagaRecipe":
        return cls.from_mapping(load_mapping(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "task": self.task,
            "planner": self.planner,
            "inputs": self.inputs,
            "filters": self.filters,
            "pipeline": [step.to_dict() for step in self.pipeline],
            "metadata": self.metadata,
        }


def recipe_artifact_path(run_dir: str | Path, relative: str) -> str:
    return (Path(run_dir).expanduser().resolve() / relative).as_posix()

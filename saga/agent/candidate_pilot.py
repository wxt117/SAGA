from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from saga.agent.benefit_evidence import build_benefit_evidence_report
from saga.agent.recipe_generator import generate_recipe, write_recipe
from saga.core.config import load_mapping, save_json, save_text
from saga.executor.recipe_executor import execute_recipe


CANDIDATE_PILOT_VERSION = "saga_candidate_recipe_pilot_v1"


def build_candidate_recipe_pilot(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    top_k: int = 3,
    pilot_sample_count: int = 8,
    execute: bool = False,
    dry_run: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else run_path / "candidate_pilot"
    out.mkdir(parents=True, exist_ok=True)

    state = load_optional_mapping(run_path / "agent_state.json")
    augmentation_plan = load_optional_mapping(run_path / "augmentation_plan.json")
    intent_spec = (
        load_optional_mapping(run_path / "effective_intent_spec.json")
        or load_optional_mapping(run_path / "intent_effective_spec.json")
        or load_optional_mapping(run_path / "intent_spec.json")
    )
    validated_profile = load_optional_mapping(run_path / "validated_dataset_profile.json")
    if run_path.joinpath("validated_dataset_profile.json").exists():
        validated_profile["_validated_profile_path"] = (run_path / "validated_dataset_profile.json").as_posix()
    bridge_report = load_optional_mapping(run_path / "format_bridge.json")
    saga_config = load_optional_mapping(run_path / "saga_config_effective.json")
    skill_configs = saga_config.get("skill_configs") or {}

    candidates = executable_candidate_plans(augmentation_plan, top_k=top_k)
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate_dir = out / f"{rank:02d}_{candidate.get('plan_id')}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_intent = intent_for_candidate(
            intent_spec=intent_spec,
            candidate=candidate,
            pilot_sample_count=pilot_sample_count,
        )
        save_json(candidate_dir / "candidate_intent_spec.json", candidate_intent)
        save_json(candidate_dir / "candidate_plan.json", candidate)
        try:
            recipe = generate_recipe(
                request_text=str(state.get("request") or intent_spec.get("raw_text") or ""),
                intent_spec=candidate_intent,
                dataset_profile_path=run_path / "dataset_profile.json",
                validated_profile=validated_profile,
                bridge_report=bridge_report,
                output_dir=candidate_dir,
                style_transfer_config=skill_configs.get("style_transfer"),
                diffusion_lora_config=skill_configs.get("diffusion_lora"),
                traditional_aug_config=skill_configs.get("traditional_augmentation"),
                model_to_pov_config=skill_configs.get("model_to_pov_scene"),
                background_generation_config=skill_configs.get("background_generation"),
                classification_eval_config=skill_configs.get("classification_evaluation"),
                max_repair_trials=int((saga_config.get("execution") or {}).get("max_repair_trials", 3)),
                planner_mode="candidate_pilot",
                planner_proposal={"plan_id": candidate.get("plan_id"), "strategy": "top_k_candidate_pilot"},
                planner_guardrail={"decision": "candidate_recipe", "execution_allowed": execute},
            )
            recipe_path = candidate_dir / "recipe.yaml"
            write_recipe(recipe, recipe_path)
            execution_report = None
            benefit = None
            if execute:
                execution_report = execute_recipe(
                    recipe_path=recipe_path,
                    output_dir=candidate_dir / "execution",
                    dry_run=dry_run,
                    stop_on_error=not continue_on_error,
                )
                candidate_state = {
                    "request": state.get("request"),
                    "task": recipe.task,
                    "dry_run": dry_run,
                    "augmentation_selected_skill": candidate.get("skill"),
                    "augmentation_selected_recipe_task": candidate.get("recipe_task"),
                    "execution_status": execution_report.get("status"),
                }
                save_json(candidate_dir / "agent_state.json", candidate_state)
                save_json(candidate_dir / "augmentation_plan.json", candidate_augmentation_plan_stub(candidate, augmentation_plan))
                benefit = build_benefit_evidence_report(
                    run_dir=candidate_dir,
                    output_dir=candidate_dir,
                    execution_report=execution_report,
                    state=candidate_state,
                )
            rows.append(
                {
                    "rank": rank,
                    "plan_id": candidate.get("plan_id"),
                    "skill": candidate.get("skill"),
                    "recipe_task": candidate.get("recipe_task"),
                    "planner_score": candidate.get("planner_score"),
                    "candidate_dir": candidate_dir.as_posix(),
                    "recipe_path": recipe_path.as_posix(),
                    "execution_status": execution_report.get("status") if execution_report else None,
                    "evidence_level": benefit.get("evidence_level") if benefit else None,
                    "evidence_label": benefit.get("evidence_label") if benefit else None,
                    "status": "executed" if execution_report else "recipe_written",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "rank": rank,
                    "plan_id": candidate.get("plan_id"),
                    "skill": candidate.get("skill"),
                    "recipe_task": candidate.get("recipe_task"),
                    "planner_score": candidate.get("planner_score"),
                    "candidate_dir": candidate_dir.as_posix(),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report = {
        "schema_version": CANDIDATE_PILOT_VERSION,
        "source_run_dir": run_path.as_posix(),
        "output_dir": out.as_posix(),
        "top_k": int(top_k),
        "pilot_sample_count": int(pilot_sample_count),
        "execute": bool(execute),
        "dry_run": bool(dry_run),
        "candidate_count": len(rows),
        "candidates": rows,
        "ranking": rank_candidate_rows(rows),
        "notes": [
            "Candidate pilot compiles top-k executable planner candidates into separate SagaRecipe DAGs.",
            "Pilot execution defaults to dry-run so expensive skills are not launched without explicit user intent.",
            "Use pilot evidence to select a final recipe before scaling expensive generation.",
        ],
    }
    save_json(out / "candidate_pilot.json", report)
    save_text(out / "candidate_pilot.md", render_candidate_pilot_markdown(report))
    return report


def executable_candidate_plans(augmentation_plan: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for item in augmentation_plan.get("ranked_plans") or []:
        if not (item.get("execution_policy") or {}).get("executable"):
            continue
        plan_id = str(item.get("plan_id") or item.get("skill") or "")
        if not plan_id or plan_id in seen:
            continue
        seen.add(plan_id)
        rows.append(item)
        if len(rows) >= int(top_k):
            break
    return rows


def intent_for_candidate(
    *,
    intent_spec: dict[str, Any],
    candidate: dict[str, Any],
    pilot_sample_count: int,
) -> dict[str, Any]:
    effective = copy.deepcopy(intent_spec)
    intent = effective.setdefault("intent", {})
    updates = candidate.get("recipe_intent_updates") or {}
    for key, value in updates.items():
        if value not in (None, "", {}, []):
            intent[key] = value
    params = candidate.get("params") or {}
    for key in ("target_count", "num_images", "training_epochs", "model_family", "multiplier", "variant", "colormap", "scene_prompt", "width", "height"):
        if params.get(key) not in (None, "", {}, []):
            intent[key] = params[key]
    count = safe_int(intent.get("target_count") or intent.get("num_images"))
    if count is not None:
        capped = max(1, min(count, int(pilot_sample_count)))
        intent["_pilot_original_target_count"] = count
        intent["target_count"] = capped
        if intent.get("task") == "background_generation" or "num_images" in intent:
            intent["num_images"] = capped
    epochs = safe_int(intent.get("training_epochs"))
    if epochs is not None:
        intent["_pilot_original_training_epochs"] = epochs
        intent["training_epochs"] = max(1, min(epochs, 3))
    effective["candidate_pilot"] = {
        "source": CANDIDATE_PILOT_VERSION,
        "plan_id": candidate.get("plan_id"),
        "skill": candidate.get("skill"),
        "recipe_task": candidate.get("recipe_task"),
        "pilot_sample_count": int(pilot_sample_count),
    }
    return effective


def candidate_augmentation_plan_stub(candidate: dict[str, Any], source_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": source_plan.get("schema_version"),
        "selected_plan_id": candidate.get("plan_id"),
        "selected_skill": candidate.get("skill"),
        "selected_recipe_task": candidate.get("recipe_task"),
        "ranked_plans": [candidate],
        "source_selected_plan_id": source_plan.get("selected_plan_id"),
    }


def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> float:
        value = float(row.get("planner_score") or 0.0)
        if row.get("execution_status") == "succeeded":
            value += 0.2
        if row.get("status") == "failed":
            value -= 0.4
        if row.get("evidence_level") is not None:
            value += 0.1 * float(row.get("evidence_level") or 0)
        return value

    ranked = sorted(rows, key=score, reverse=True)
    return [
        {
            "rank": index,
            "plan_id": row.get("plan_id"),
            "skill": row.get("skill"),
            "score": round(score(row), 3),
            "status": row.get("status"),
            "execution_status": row.get("execution_status"),
            "evidence_level": row.get("evidence_level"),
            "candidate_dir": row.get("candidate_dir"),
        }
        for index, row in enumerate(ranked, start=1)
    ]


def render_candidate_pilot_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Candidate Recipe Pilot",
        "",
        f"- Source run: `{report.get('source_run_dir')}`",
        f"- Candidates: {report.get('candidate_count')}",
        f"- Execute: `{report.get('execute')}`",
        f"- Dry run: `{report.get('dry_run')}`",
        "",
        "## Candidates",
        "",
    ]
    for row in report.get("candidates") or []:
        lines.extend(
            [
                f"### {row.get('rank')}. {row.get('plan_id')}",
                "",
                f"- Skill: `{row.get('skill')}`",
                f"- Recipe task: `{row.get('recipe_task')}`",
                f"- Planner score: {row.get('planner_score')}",
                f"- Status: `{row.get('status')}`",
                f"- Recipe: `{row.get('recipe_path')}`",
                f"- Execution: `{row.get('execution_status')}`",
                f"- Evidence: `{row.get('evidence_level')}` `{row.get('evidence_label')}`",
                "",
            ]
        )
    lines.append("## Ranking")
    lines.append("")
    for row in report.get("ranking") or []:
        lines.append(f"- {row.get('rank')}. `{row.get('skill')}` score={row.get('score')} status={row.get('status')}")
    lines.append("")
    return "\n".join(lines)


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_optional_mapping(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {}
    return load_mapping(value)

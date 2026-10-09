from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.agent.benefit_evidence import build_benefit_evidence_report
from saga.core.config import load_mapping, save_json, save_text
from saga.core.recipe import SagaRecipe
from saga.executor.recipe_executor import execute_recipe
from saga.observer.parameter_repair import build_parameter_repair_plan


AUTO_REPAIR_VERSION = "saga_bounded_auto_repair_v1"


def run_bounded_auto_repair(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    max_trials: int = 3,
    trial_index: int = 0,
    execute_revised: bool = False,
    dry_run: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else run_path / "auto_repair"
    out.mkdir(parents=True, exist_ok=True)
    recipe_path = run_path / "recipe.yaml"
    execution_path = run_path / "execution" / "recipe_execution.json"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Missing recipe: {recipe_path}")
    recipe = SagaRecipe.from_path(recipe_path)
    execution = load_optional_mapping(execution_path)
    before_evidence = build_benefit_evidence_report(run_dir=run_path, output_dir=out / "before", execution_report=execution)
    repair_plan = build_parameter_repair_plan(
        recipe=recipe,
        evaluation=None,
        output_dir=out / "parameter_repair",
        max_trials=max_trials,
        trial_index=trial_index,
        dry_run=False,
        execution_report=execution,
    )
    revised_recipe = (repair_plan.get("artifacts") or {}).get("revised_recipe_yaml")
    should_execute = bool(execute_revised and revised_recipe and repair_plan.get("status") in {"planned", "dry_run"})
    revised_execution = None
    after_evidence = None
    if should_execute:
        revised_execution = execute_recipe(
            recipe_path=revised_recipe,
            output_dir=out / "revised_execution",
            dry_run=dry_run,
            stop_on_error=not continue_on_error,
        )
        candidate_state = {
            "request": (load_optional_mapping(run_path / "agent_state.json") or {}).get("request"),
            "task": revised_execution.get("task"),
            "dry_run": dry_run,
            "execution_status": revised_execution.get("status"),
        }
        save_json(out / "agent_state.json", candidate_state)
        after_evidence = build_benefit_evidence_report(
            run_dir=out,
            output_dir=out / "after",
            execution_report=revised_execution,
            state=candidate_state,
        )
    status = repair_status(repair_plan=repair_plan, revised_execution=revised_execution, execute_revised=execute_revised)
    report = {
        "schema_version": AUTO_REPAIR_VERSION,
        "status": status,
        "source_run_dir": run_path.as_posix(),
        "output_dir": out.as_posix(),
        "bounded": True,
        "max_trials": int(max_trials),
        "trial_index": int(trial_index),
        "execute_revised": bool(execute_revised),
        "dry_run": bool(dry_run),
        "before_evidence": summarize_evidence(before_evidence),
        "repair_plan": {
            "status": repair_plan.get("status"),
            "patch_count": repair_plan.get("patch_count"),
            "revised_recipe_yaml": revised_recipe,
            "trigger_count": repair_plan.get("trigger_count"),
        },
        "revised_execution": {
            "status": revised_execution.get("status") if revised_execution else None,
            "report_path": (out / "revised_execution" / "recipe_execution.json").as_posix() if revised_execution else None,
        },
        "after_evidence": summarize_evidence(after_evidence or {}),
        "accepted": accept_repair(before_evidence, after_evidence, revised_execution),
        "notes": [
            "Auto repair is bounded and whitelist-based; it mutates only parameters allowed by ParameterRepairPlan.",
            "By default this command writes repair artifacts without real expensive reruns.",
            "A revised run is accepted only when execution does not fail and evidence level does not regress.",
        ],
    }
    save_json(out / "auto_repair_report.json", report)
    save_text(out / "auto_repair_report.md", render_auto_repair_markdown(report))
    return report


def repair_status(
    *,
    repair_plan: dict[str, Any],
    revised_execution: dict[str, Any] | None,
    execute_revised: bool,
) -> str:
    if repair_plan.get("status") in {"blocked", "unsupported"}:
        return str(repair_plan.get("status"))
    if not repair_plan.get("patch_count"):
        return "not_needed"
    if not execute_revised:
        return "planned"
    if revised_execution is None:
        return "planned_not_executed"
    if revised_execution.get("status") == "failed":
        return "revised_failed"
    return "revised_executed"


def accept_repair(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    execution: dict[str, Any] | None,
) -> bool:
    if not after or not execution:
        return False
    if execution.get("status") == "failed":
        return False
    return int(after.get("evidence_level") or 0) >= int(before.get("evidence_level") or 0)


def summarize_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "evidence_level": report.get("evidence_level"),
        "evidence_label": report.get("evidence_label"),
        "downstream_claim_allowed": report.get("downstream_claim_allowed"),
        "recommended_next_action": report.get("recommended_next_action"),
    }


def render_auto_repair_markdown(report: dict[str, Any]) -> str:
    repair = report.get("repair_plan") or {}
    lines = [
        "# SAGA Bounded Auto Repair",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Source run: `{report.get('source_run_dir')}`",
        f"- Trial: {report.get('trial_index')} / {report.get('max_trials')}",
        f"- Patch count: {repair.get('patch_count')}",
        f"- Revised recipe: `{repair.get('revised_recipe_yaml')}`",
        f"- Execute revised: `{report.get('execute_revised')}`",
        f"- Revised execution: `{(report.get('revised_execution') or {}).get('status')}`",
        f"- Accepted: `{report.get('accepted')}`",
        "",
        "## Evidence",
        "",
        f"- Before: `{report.get('before_evidence')}`",
        f"- After: `{report.get('after_evidence')}`",
        "",
    ]
    return "\n".join(lines)


def load_optional_mapping(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {}
    return load_mapping(value)

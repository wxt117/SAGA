from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.agent.skill_registry import get_skill_registry
from saga.core.config import save_json, save_text


PLAN_VERIFICATION_VERSION = "saga_plan_verification_v1"


def verify_plan(
    proposal: dict[str, Any],
    guardrail: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    registry = get_skill_registry()
    pipeline = proposal.get("pipeline") or []
    issues = list(guardrail.get("issues") or [])
    skill_reports = []
    executable_step_count = 0
    planned_step_count = 0
    unsupported_step_count = 0
    gpu_required = False
    high_cost_steps = []

    for step in pipeline:
        skill_name = str(step.get("skill") or "")
        spec = registry.get(skill_name)
        if spec is None:
            unsupported_step_count += 1
            skill_reports.append(
                {
                    "step_id": step.get("id"),
                    "skill": skill_name,
                    "registry_status": "unsupported",
                    "execution_status": "blocked",
                    "issues": ["skill_not_in_registry"],
                }
            )
            continue
        if spec.status == "executable":
            executable_step_count += 1
        elif spec.status == "planned":
            planned_step_count += 1
        if spec.cost.get("gpu_required"):
            gpu_required = True
        if spec.cost.get("runtime") == "high":
            high_cost_steps.append(step.get("id"))
        skill_reports.append(
            {
                "step_id": step.get("id"),
                "skill": skill_name,
                "registry_status": spec.status,
                "execution_status": step.get("execution_status"),
                "task_types": spec.task_types,
                "input_contract": spec.input_contract,
                "output_contract": spec.output_contract,
                "cost": spec.cost,
                "safety": spec.safety,
                "limitations": spec.limitations,
                "issues": [],
            }
        )

    blocking_count = int(guardrail.get("blocking_issue_count") or len([issue for issue in issues if issue.get("severity") == "high"]))
    requires_clarification = bool(guardrail.get("requires_clarification"))
    bridge_valid = bool(bridge_report.get("valid"))
    task = proposal.get("task")
    status = "passed"
    if blocking_count or unsupported_step_count:
        status = "failed"
    elif requires_clarification:
        status = "needs_clarification"
    elif planned_step_count:
        status = "warning"
    elif not bridge_valid and not proposal_can_run_without_dataset_format(proposal):
        status = "warning"

    report = {
        "schema_version": PLAN_VERIFICATION_VERSION,
        "status": status,
        "task": task,
        "planner_decision": guardrail.get("decision"),
        "execution_allowed": bool(guardrail.get("execution_allowed")),
        "requires_clarification": requires_clarification,
        "bridge_valid": bridge_valid,
        "profile_level_after_validation": bridge_report.get("profile_level_after_validation"),
        "step_summary": {
            "total": len(pipeline),
            "executable": executable_step_count,
            "planned": planned_step_count,
            "unsupported": unsupported_step_count,
            "gpu_required": gpu_required,
            "high_cost_steps": high_cost_steps,
        },
        "skill_reports": skill_reports,
        "issues": issues,
        "clarification_questions": guardrail.get("clarification_questions", []),
        "verdict": plan_verdict(status),
    }
    if output_dir:
        write_plan_verification(output_dir, report)
    return report


def proposal_can_run_without_dataset_format(proposal: dict[str, Any]) -> bool:
    task = str(proposal.get("task") or "").strip().lower()
    updates = proposal.get("recipe_intent_updates") or {}
    if task == "geodiff_sar_generation" and (updates.get("dataset_source") or updates.get("dataset_root")) and updates.get("model_file"):
        return True
    if task == "raysar_synthesis" and (
        updates.get("pov_scene") or updates.get("model_file") or updates.get("contributions_txt")
    ):
        return True
    if task == "raysar_synthesis":
        for step in proposal.get("pipeline") or []:
            params = step.get("params") if isinstance(step, dict) else {}
            if not isinstance(params, dict):
                continue
            if params.get("pov_scene") or params.get("model_file") or params.get("contributions_txt"):
                return True
            if step.get("skill") in {"ModelToPOVSceneCompilerSkill", "RaySARSynthesisSkill"}:
                return True
    return False


def plan_verdict(status: str) -> str:
    if status == "passed":
        return "Plan can be compiled into an executable recipe."
    if status == "needs_clarification":
        return "Plan is structurally usable, but real execution should wait for user clarification."
    if status == "warning":
        return "Plan can be inspected, but contains planned skills or validation warnings."
    return "Plan should fall back to the deterministic rule planner or be revised."


def write_plan_verification(output_dir: str | Path, report: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    save_json(path / "plan_verification.json", report)
    save_text(path / "plan_verification.md", render_plan_verification_markdown(report))


def render_plan_verification_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Plan Verification",
        "",
        f"- Status: {report.get('status')}",
        f"- Task: `{report.get('task')}`",
        f"- Planner decision: `{report.get('planner_decision')}`",
        f"- Execution allowed: {report.get('execution_allowed')}",
        f"- Requires clarification: {report.get('requires_clarification')}",
        f"- Bridge valid: {report.get('bridge_valid')}",
        f"- Verdict: {report.get('verdict')}",
        "",
        "## Step Summary",
        "",
        f"- Total: {report.get('step_summary', {}).get('total')}",
        f"- Executable: {report.get('step_summary', {}).get('executable')}",
        f"- Planned: {report.get('step_summary', {}).get('planned')}",
        f"- Unsupported: {report.get('step_summary', {}).get('unsupported')}",
        f"- GPU required: {report.get('step_summary', {}).get('gpu_required')}",
        "",
        "## Skills",
        "",
    ]
    for item in report.get("skill_reports", []):
        lines.append(
            f"- `{item.get('step_id')}` -> `{item.get('skill')}` "
            f"registry={item.get('registry_status')} runtime={item.get('cost', {}).get('runtime')}"
        )
    lines.extend(["", "## Issues", ""])
    if not report.get("issues"):
        lines.append("- None")
    else:
        for issue in report["issues"]:
            lines.append(f"- `{issue.get('name')}` ({issue.get('severity')}): `{issue}`")
    lines.extend(["", "## Clarification Questions", ""])
    if not report.get("clarification_questions"):
        lines.append("- None")
    else:
        for question in report["clarification_questions"]:
            lines.append(f"- {question}")
    lines.append("")
    return "\n".join(lines)

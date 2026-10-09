from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from saga.agent.runtime import run_agent
from saga.core.config import load_mapping, save_json, save_text


SMOKE_MATRIX_VERSION = "saga_smoke_matrix_v1"


def run_smoke_matrix(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    case_ids: list[str] | None = None,
    run_real: bool = False,
    memory_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_mapping(config_path)
    matrix = config.get("smoke_matrix", config)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    selected_ids = set(case_ids or [])
    cases = []
    started = time.time()
    save_json(
        output_path / "smoke_matrix_status.json",
        {
            "schema_version": SMOKE_MATRIX_VERSION,
            "status": "running",
            "config_path": Path(config_path).expanduser().resolve().as_posix(),
            "output_dir": output_path.as_posix(),
            "run_real": bool(run_real),
            "started_at": wall_time(),
            "completed_case_count": 0,
            "cases": [],
        },
    )
    for raw_case in matrix.get("cases") or []:
        if not isinstance(raw_case, dict):
            continue
        case_id = str(raw_case.get("id") or "").strip()
        if selected_ids and case_id not in selected_ids:
            continue
        save_json(
            output_path / "smoke_matrix_status.json",
            {
                "schema_version": SMOKE_MATRIX_VERSION,
                "status": "running",
                "config_path": Path(config_path).expanduser().resolve().as_posix(),
                "output_dir": output_path.as_posix(),
                "run_real": bool(run_real),
                "started_at": wall_time(started),
                "updated_at": wall_time(),
                "current_case": case_id,
                "completed_case_count": len(cases),
                "cases": cases,
            },
        )
        case_result = run_smoke_case(
            raw_case,
            output_path=output_path,
            matrix=matrix,
            run_real=run_real,
            memory_dir=memory_dir or matrix.get("memory_dir") or output_path / "memory",
        )
        cases.append(case_result)
        save_json(
            output_path / "smoke_matrix_status.json",
            {
                "schema_version": SMOKE_MATRIX_VERSION,
                "status": "running",
                "config_path": Path(config_path).expanduser().resolve().as_posix(),
                "output_dir": output_path.as_posix(),
                "run_real": bool(run_real),
                "started_at": wall_time(started),
                "updated_at": wall_time(),
                "current_case": None,
                "completed_case_count": len(cases),
                "cases": cases,
            },
        )
    status = "passed" if cases and all(case.get("passed") for case in cases) else "failed"
    report = {
        "schema_version": SMOKE_MATRIX_VERSION,
        "status": status,
        "config_path": Path(config_path).expanduser().resolve().as_posix(),
        "output_dir": output_path.as_posix(),
        "case_count": len(cases),
        "passed_count": len([case for case in cases if case.get("passed")]),
        "elapsed_seconds": round(time.time() - started, 3),
        "run_real": bool(run_real),
        "cases": cases,
    }
    save_json(output_path / "smoke_matrix_report.json", report)
    save_text(output_path / "smoke_matrix_report.md", render_smoke_matrix_markdown(report))
    save_json(
        output_path / "smoke_matrix_status.json",
        {
            **report,
            "updated_at": wall_time(),
            "completed_at": wall_time(),
        },
    )
    return report


def run_smoke_case(
    raw_case: dict[str, Any],
    *,
    output_path: Path,
    matrix: dict[str, Any],
    run_real: bool,
    memory_dir: str | Path,
) -> dict[str, Any]:
    case_id = str(raw_case.get("id") or "case")
    case_dir = output_path / safe_id(case_id)
    expected = raw_case.get("expect") or {}
    allow_real = bool(raw_case.get("allow_real", False))
    dry_run = not (run_real and allow_real)
    started = time.time()
    case_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        case_dir / "case_status.json",
        {
            "id": case_id,
            "status": "running",
            "dry_run": dry_run,
            "run_real_requested": bool(run_real),
            "allow_real": allow_real,
            "started_at": wall_time(started),
            "updated_at": wall_time(),
            "request": str(raw_case.get("request", "")),
            "run_dir": case_dir.as_posix(),
        },
    )
    try:
        state = run_agent(
            request=str(raw_case["request"]),
            output_dir=case_dir,
            dataset_root=raw_case.get("dataset_root"),
            saga_config_path=raw_case.get("saga_config") or matrix.get("saga_config") or "configs/saga.yaml",
            memory_dir=memory_dir,
            use_memory=bool(raw_case.get("use_memory", matrix.get("use_memory", False))),
            dry_run=dry_run,
            sample_limit=int(raw_case.get("sample_limit", matrix.get("sample_limit", 24))),
            image_probe_limit=int(raw_case.get("image_probe_limit", matrix.get("image_probe_limit", 24))),
            bridge_sample_limit=int(raw_case.get("bridge_sample_limit", matrix.get("bridge_sample_limit", 64))),
            execute=bool(raw_case.get("execute", matrix.get("execute", True))),
        )
        checks = check_smoke_expectations(state=state, expected=expected, case_dir=case_dir)
        passed = all(item.get("passed") for item in checks)
        result = {
            "id": case_id,
            "passed": passed,
            "status": "passed" if passed else "failed",
            "dry_run": dry_run,
            "run_dir": case_dir.as_posix(),
            "elapsed_seconds": round(time.time() - started, 3),
            "state": {
                "task": state.get("task"),
                "selected_skill": state.get("augmentation_selected_skill"),
                "selected_recipe_task": state.get("augmentation_selected_recipe_task"),
                "bridge_valid": state.get("bridge_valid"),
                "plan_critic_status": state.get("plan_critic_status"),
                "execution_status": state.get("execution_status"),
                "recipe_path": state.get("recipe_path"),
            },
            "checks": checks,
        }
        save_json(case_dir / "case_status.json", {**result, "updated_at": wall_time(), "completed_at": wall_time()})
        return result
    except Exception as exc:
        result = {
            "id": case_id,
            "passed": False,
            "status": "error",
            "dry_run": dry_run,
            "run_dir": case_dir.as_posix(),
            "elapsed_seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": [],
        }
        save_json(case_dir / "case_status.json", {**result, "updated_at": wall_time(), "completed_at": wall_time()})
        return result


def check_smoke_expectations(state: dict[str, Any], expected: dict[str, Any], case_dir: Path) -> list[dict[str, Any]]:
    checks = []
    checks.append(check_equal("task", state.get("task"), expected.get("task")))
    checks.append(check_equal("selected_skill", state.get("augmentation_selected_skill"), expected.get("selected_skill")))
    checks.append(check_equal("selected_recipe_task", state.get("augmentation_selected_recipe_task"), expected.get("selected_recipe_task")))
    checks.append(check_file("recipe_exists", state.get("recipe_path")))
    execution_path = state.get("execution_report_path")
    if execution_path:
        checks.append(check_file("execution_report_exists", execution_path))
        execution = load_mapping(execution_path)
        expected_execution_status = expected.get("execution_status")
        if expected_execution_status is None:
            checks.append(
                {
                    "name": "execution_status_ok",
                    "passed": execution.get("status") in {"succeeded", "warning"},
                    "expected": "succeeded|warning",
                    "actual": execution.get("status"),
                }
            )
        else:
            checks.append(check_equal("execution_status", execution.get("status"), expected_execution_status))
        validation = execution.get("skill_report_validation") or {}
        checks.append(
            {
                "name": "skill_run_report_validation",
                "passed": bool(validation.get("valid", False)),
                "expected": True,
                "actual": validation.get("valid"),
                "details": validation.get("severity_counts"),
            }
        )
    if expected.get("plan_critic_not_blocked", True):
        checks.append(
            {
                "name": "plan_critic_not_blocked",
                "passed": state.get("plan_critic_status") != "blocked",
                "expected": "not blocked",
                "actual": state.get("plan_critic_status"),
            }
        )
    for key, value in (expected.get("recipe_inputs") or {}).items():
        recipe = load_mapping(state.get("recipe_path")) if state.get("recipe_path") else {}
        actual = (recipe.get("inputs") or {}).get(key)
        checks.append(check_equal(f"recipe_inputs.{key}", actual, value))
    return checks


def check_equal(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    if expected is None:
        return {"name": name, "passed": True, "expected": "not asserted", "actual": actual}
    return {"name": name, "passed": actual == expected, "expected": expected, "actual": actual}


def check_file(name: str, path: str | Path | None) -> dict[str, Any]:
    exists = bool(path and Path(path).expanduser().exists())
    return {"name": name, "passed": exists, "expected": "existing file", "actual": str(path) if path else None}


def safe_id(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def wall_time(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp or time.time()))


def render_smoke_matrix_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Smoke Matrix",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Cases: {report.get('passed_count')}/{report.get('case_count')} passed",
        f"- Run real: {report.get('run_real')}",
        f"- Output: `{report.get('output_dir')}`",
        "",
    ]
    for case in report.get("cases") or []:
        lines.extend(
            [
                f"## {case.get('id')}",
                "",
                f"- Status: `{case.get('status')}`",
                f"- Dry run: {case.get('dry_run')}",
                f"- Run dir: `{case.get('run_dir')}`",
                f"- State: `{case.get('state')}`",
                "",
                "Checks:",
            ]
        )
        for check in case.get("checks") or []:
            mark = "PASS" if check.get("passed") else "FAIL"
            lines.append(f"- {mark} `{check.get('name')}` expected=`{check.get('expected')}` actual=`{check.get('actual')}`")
        if case.get("error"):
            lines.append(f"- Error: `{case.get('error')}`")
        lines.append("")
    return "\n".join(lines)

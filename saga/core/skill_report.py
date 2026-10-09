from __future__ import annotations

from typing import Any


SKILL_RUN_REPORT_VERSION = "saga_skill_run_report_v1"
SKILL_RUN_REPORT_VALIDATION_VERSION = "saga_skill_run_report_validation_v1"

VALID_STATUSES = {
    "succeeded",
    "failed",
    "blocked",
    "warning",
    "triggered",
    "not_triggered",
    "dry_run",
    "skipped",
    "unknown",
}


INPUT_KEYS = {
    "baseline_dataset",
    "augmented_dataset",
    "val_dataset",
    "content_root",
    "content",
    "style",
    "dataset_root",
    "input_dir",
    "reference_dir",
    "generated_dir",
    "input_from_step",
    "augmented_from_step",
    "source_dir",
    "target_count",
    "multiplier",
    "training_epochs",
    "model_family",
    "variant",
    "epochs",
    "sample_limit",
    "prompt",
    "scene_prompt",
    "lora_weights",
    "controlnet_weights",
    "controlnet_init_weights",
    "model_file",
    "pov_scene",
    "pov_scene_from_step",
    "parameters_file",
    "simulation_parameters",
    "contributions_txt",
    "adapted_povray",
    "width",
    "height",
    "postprocess",
    "render",
    "extract_real_gefm",
    "render_gefm",
    "sar_intersection",
    "incidence_angle_deg",
    "depression_angle_deg",
    "azimuth_deg",
    "pitch_deg",
    "roll_deg",
    "target_extent_m",
    "scale_factor",
    "azimuth_values",
    "azimuth_sweep",
    "target_azimuths",
    "depressions",
    "input_up_axis",
    "pov_height_axis",
    "range_distance_m",
    "sensor_plane_m",
    "scene_center",
    "colormap",
    "preserve_tree",
    "num_images",
    "split",
    "top_k",
    "selection_seed",
    "seed",
    "cns",
    "steps",
    "scale",
    "cuda",
    "cpu_threads",
    "extra_prompt",
    "clear_cache_each_image",
    "show_top",
}

OUTPUT_KEYS = {
    "output_dir",
    "run_dir",
    "generated_output_dir",
    "report_dir",
    "selected_count",
    "image_count",
    "expected_count",
    "existing_image_count",
    "generated_image_count",
    "manifest_count",
    "caption_sidecar_count",
    "trigger_count",
    "pair_count",
    "caption_profile",
    "manifest",
    "map_count",
    "view_count",
    "preview_image",
    "contributions_txt",
    "compiled_scene",
}

NESTED_REPORT_KEYS = {
    "skill_report",
    "quality_report",
    "distribution_report",
    "sar_artifact_report",
    "leakage_report",
    "duplicate_report",
    "repair_policy",
    "export_report",
    "evaluator_report",
}


def attach_skill_run_report(
    result: dict[str, Any],
    *,
    step_id: str,
    skill: str,
    dry_run: bool,
    elapsed_seconds: float | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result["skill_run_report"] = build_skill_run_report(
        result=result,
        step_id=step_id,
        skill=skill,
        dry_run=dry_run,
        elapsed_seconds=elapsed_seconds,
        params=params or {},
    )
    return result


def validate_skill_run_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the normalized per-step SkillRunReport contract.

    This validator is intentionally structural. It does not decide whether a
    skill result is good enough for science; observer/evaluator skills handle
    that. Its job is to make every executor step auditable and comparable for
    tests, reports, and policy memory.
    """

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return validation_report(valid=False, issues=[issue("high", "missing_skill_run_report", "SkillRunReport is missing or not an object.")], warnings=[])

    required_keys = {
        "schema_version",
        "step_id",
        "skill",
        "status",
        "dry_run",
        "inputs",
        "outputs",
        "metrics",
        "artifacts",
        "issues",
        "warnings",
        "observer",
        "provenance",
    }
    for key in sorted(required_keys):
        if key not in report:
            issues.append(issue("high", "missing_required_key", f"SkillRunReport is missing `{key}`.", field=key))

    if report.get("schema_version") != SKILL_RUN_REPORT_VERSION:
        issues.append(
            issue(
                "high",
                "invalid_schema_version",
                "SkillRunReport schema_version is not supported.",
                expected=SKILL_RUN_REPORT_VERSION,
                actual=report.get("schema_version"),
            )
        )
    for key in ("step_id", "skill", "status"):
        value = report.get(key)
        if not isinstance(value, str) or not value.strip():
            issues.append(issue("high", "invalid_string_field", f"`{key}` must be a non-empty string.", field=key))
    status = str(report.get("status") or "unknown")
    if status not in VALID_STATUSES:
        warnings.append(issue("medium", "unknown_status", "SkillRunReport status is not in the known status set.", status=status))
    if not isinstance(report.get("dry_run"), bool):
        issues.append(issue("high", "invalid_dry_run", "`dry_run` must be a boolean."))
    for key in ("inputs", "outputs", "metrics", "artifacts", "observer", "provenance"):
        if not isinstance(report.get(key), dict):
            issues.append(issue("high", "invalid_mapping_field", f"`{key}` must be an object.", field=key))
    for key in ("issues", "warnings"):
        if not isinstance(report.get(key), list):
            issues.append(issue("high", "invalid_list_field", f"`{key}` must be a list.", field=key))

    observer = report.get("observer") if isinstance(report.get("observer"), dict) else {}
    if "trigger_count" in observer:
        try:
            trigger_count = int(observer.get("trigger_count") or 0)
        except (TypeError, ValueError):
            issues.append(issue("medium", "invalid_trigger_count", "`observer.trigger_count` must be an integer."))
        else:
            triggers = observer.get("triggers")
            if isinstance(triggers, list) and trigger_count != len(triggers):
                warnings.append(
                    issue(
                        "low",
                        "trigger_count_mismatch",
                        "`observer.trigger_count` does not match the number of stored triggers.",
                        trigger_count=trigger_count,
                        trigger_list_count=len(triggers),
                    )
                )

    provenance = report.get("provenance") if isinstance(report.get("provenance"), dict) else {}
    if not isinstance(provenance.get("raw_result_keys", []), list):
        warnings.append(issue("low", "raw_result_keys_not_list", "`provenance.raw_result_keys` should be a list."))
    return validation_report(valid=not any(item.get("severity") == "high" for item in issues), issues=issues, warnings=warnings)


def validate_step_result_skill_report(result: dict[str, Any]) -> dict[str, Any]:
    validation = validate_skill_run_report(result.get("skill_run_report"))
    expected_step = result.get("step_id")
    expected_skill = result.get("skill")
    report = result.get("skill_run_report") if isinstance(result.get("skill_run_report"), dict) else {}
    extra_issues = []
    if expected_step and report.get("step_id") != expected_step:
        extra_issues.append(
            issue(
                "high",
                "step_id_mismatch",
                "SkillRunReport step_id does not match executor result.",
                expected=expected_step,
                actual=report.get("step_id"),
            )
        )
    if expected_skill and report.get("skill") != expected_skill:
        extra_issues.append(
            issue(
                "high",
                "skill_mismatch",
                "SkillRunReport skill does not match executor result.",
                expected=expected_skill,
                actual=report.get("skill"),
            )
        )
    if extra_issues:
        validation["issues"].extend(extra_issues)
        validation["issue_count"] = len(validation["issues"])
        validation["valid"] = False
    return validation


def summarize_skill_report_validations(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    high = 0
    medium = 0
    low = 0
    for result in step_results:
        validation = result.get("skill_run_report_validation")
        if not isinstance(validation, dict):
            validation = validate_step_result_skill_report(result)
        severities = count_validation_severities(validation)
        high += severities.get("high", 0)
        medium += severities.get("medium", 0)
        low += severities.get("low", 0)
        rows.append(
            {
                "step_id": result.get("step_id"),
                "skill": result.get("skill"),
                "valid": bool(validation.get("valid")),
                "issue_count": validation.get("issue_count", 0),
                "warning_count": validation.get("warning_count", 0),
                "severity_counts": severities,
            }
        )
    return {
        "schema_version": SKILL_RUN_REPORT_VALIDATION_VERSION,
        "valid": high == 0,
        "step_count": len(step_results),
        "invalid_step_count": len([row for row in rows if not row["valid"]]),
        "severity_counts": {"high": high, "medium": medium, "low": low},
        "steps": rows,
    }


def build_skill_run_report(
    *,
    result: dict[str, Any],
    step_id: str,
    skill: str,
    dry_run: bool,
    elapsed_seconds: float | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    triggers = normalize_list(result.get("triggers"))
    issues = normalize_list(result.get("issues"))
    warnings = normalize_list(result.get("warnings"))
    if triggers:
        issues.extend({"type": "observer_trigger", "detail": trigger} for trigger in triggers[:20])

    return {
        "schema_version": SKILL_RUN_REPORT_VERSION,
        "step_id": step_id,
        "skill": skill,
        "status": str(result.get("status") or "unknown"),
        "dry_run": bool(dry_run),
        "message": result.get("message"),
        "elapsed_seconds": elapsed_seconds,
        "inputs": collect_inputs(result=result, params=params),
        "outputs": collect_outputs(result),
        "metrics": collect_metrics(result),
        "artifacts": collect_artifacts(result),
        "issues": issues,
        "warnings": warnings,
        "observer": {
            "trigger_count": len(triggers),
            "triggers": triggers[:20],
        },
        "provenance": {
            "param_keys": sorted(str(key) for key in params.keys()),
            "raw_result_keys": sorted(str(key) for key in result.keys() if key != "skill_run_report"),
            "nested_reports": collect_nested_report_summaries(result),
        },
    }


def collect_inputs(result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for key in sorted(INPUT_KEYS):
        value = result.get(key, params.get(key))
        if is_present(value):
            inputs[key] = value
    for key in ("filters", "exclude_filters", "config", "dataset_config", "baseline_dataset_config"):
        value = params.get(key)
        if is_present(value):
            inputs[key] = value
    return inputs


def collect_outputs(result: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for key in sorted(OUTPUT_KEYS):
        value = result.get(key)
        if is_present(value):
            outputs[key] = value
    return outputs


def collect_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    for key in ("skill_report", "quality_report", "distribution_report", "sar_artifact_report", "export_report", "evaluator_report"):
        nested = result.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("metrics"), dict) and nested["metrics"]:
            return nested["metrics"]
    summary = result.get("summary")
    return summary if isinstance(summary, dict) else {}


def collect_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        return {key: value for key, value in artifacts.items() if is_present(value)}
    return {}


def collect_nested_report_summaries(result: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for key in sorted(NESTED_REPORT_KEYS):
        report = result.get(key)
        if not isinstance(report, dict):
            continue
        summaries[key] = {
            "schema_version": report.get("schema_version"),
            "status": report.get("status"),
            "artifact_keys": sorted((report.get("artifacts") or {}).keys()) if isinstance(report.get("artifacts"), dict) else [],
            "metric_keys": sorted((report.get("metrics") or {}).keys()) if isinstance(report.get("metrics"), dict) else [],
        }
    return summaries


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def is_present(value: Any) -> bool:
    return value not in (None, "", {}, [])


def validation_report(*, valid: bool, issues: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SKILL_RUN_REPORT_VALIDATION_VERSION,
        "valid": bool(valid),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


def issue(severity: str, name: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"severity": severity, "name": name, "message": message}
    payload.update(extra)
    return payload


def count_validation_severities(validation: dict[str, Any]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for item in list(validation.get("issues") or []) + list(validation.get("warnings") or []):
        severity = str(item.get("severity") or "low")
        counts[severity] = counts.get(severity, 0) + 1
    return counts

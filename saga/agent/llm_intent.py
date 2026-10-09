from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from saga.agent.json_utils import loads_json_object
from saga.agent.llm_client import LLMConfig, OpenAICompatibleClient
from saga.agent.llm_planner import (
    is_safe_filter_name,
    is_safe_path_like,
    is_scalar,
    parse_exclude_filter_value,
    safe_positive_int,
)
from saga.core.config import save_json, save_text


LLM_INTENT_VERSION = "saga_llm_intent_proposal_v1"
LLM_INTENT_GUARDRAIL_VERSION = "saga_llm_intent_guardrail_v1"

SUPPORTED_TASKS = {
    "style_transfer",
    "diffusion_lora_generation",
    "geodiff_sar_generation",
    "gaussian_splatting_completion",
    "gan_generation",
    "traditional_augmentation",
    "pseudocolor_transform",
    "background_generation",
    "target_background_composition",
    "raysar_synthesis",
    "classification_evaluation",
    "classification",
    "object_detection",
    "segmentation",
    "augmentation_or_generation",
    "unknown",
}
MODEL_FAMILIES = {"flux", "sd3", "sdxl"}


def run_llm_intent_recognizer(
    text: str,
    rule_intent_spec: dict[str, Any],
    rule_format_hints: list[dict[str, Any]],
    llm_config: LLMConfig,
    output_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    client = OpenAICompatibleClient(llm_config)
    prompt = build_intent_prompt(
        text=text,
        rule_intent_spec=rule_intent_spec,
        rule_format_hints=rule_format_hints,
        dataset_root=dataset_root,
    )
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are SAGA's intent recognizer. Return only JSON. "
                    "You may propose intent fields and format hints, but deterministic "
                    "guardrails decide what is trusted."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    proposal = normalize_intent_proposal(loads_json_object(content))
    guardrail = validate_intent_proposal(proposal, rule_intent_spec=rule_intent_spec)
    effective_intent_spec = apply_intent_proposal(
        rule_intent_spec=rule_intent_spec,
        proposal=proposal,
        guardrail=guardrail,
    )
    merged_format_hints = merge_format_hints(rule_format_hints, proposal.get("format_hints", []))
    result = {
        "schema_version": LLM_INTENT_VERSION,
        "mode": "llm_intent_proposal_rule_guardrail",
        "proposal": proposal,
        "guardrail": guardrail,
        "effective_intent_spec": effective_intent_spec,
        "merged_format_hints": merged_format_hints,
    }
    if output_dir:
        write_intent_artifacts(output_dir, result)
    return result


def build_intent_failure_result(
    rule_intent_spec: dict[str, Any],
    rule_format_hints: list[dict[str, Any]],
    error: Exception,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    proposal = {
        "schema_version": LLM_INTENT_VERSION,
        "source": "llm_failed",
        "intent_updates": {},
        "format_hints": [],
        "confidence": 0.0,
        "unresolved": [],
        "clarification_questions": [],
        "rationale": f"{type(error).__name__}: {error}",
    }
    guardrail = {
        "schema_version": LLM_INTENT_GUARDRAIL_VERSION,
        "valid": False,
        "decision": "fallback_to_rule_intent",
        "issues": [
            {
                "severity": "high",
                "name": "llm_intent_failed",
                "error_type": type(error).__name__,
                "message": str(error)[:2000],
            }
        ],
        "applied_update_count": 0,
        "ignored_update_count": 0,
    }
    result = {
        "schema_version": LLM_INTENT_VERSION,
        "mode": "llm_intent_proposal_rule_guardrail",
        "proposal": proposal,
        "guardrail": guardrail,
        "effective_intent_spec": rule_intent_spec,
        "merged_format_hints": rule_format_hints,
    }
    if output_dir:
        write_intent_artifacts(output_dir, result)
    return result


def build_intent_prompt(
    text: str,
    rule_intent_spec: dict[str, Any],
    rule_format_hints: list[dict[str, Any]],
    dataset_root: str | Path | None,
) -> str:
    payload = {
        "user_request": text,
        "provided_dataset_root": str(dataset_root) if dataset_root else None,
        "rule_intent_spec": rule_intent_spec,
        "rule_format_hints": rule_format_hints,
        "contract": {
            "role": [
                "extract task, sources, filters, model family, target count, goals",
                "extract format hints from the user's natural language when present",
                "do not execute tools",
                "do not claim dataset facts that require file scanning",
            ],
            "output_schema": {
                "intent_updates": {
                    "task": "optional supported task such as diffusion_lora_generation, gan_generation, traditional_augmentation, pseudocolor_transform, background_generation, target_background_composition, style_transfer, raysar_synthesis, or classification_evaluation",
                    "style_source": "optional path",
                    "content_source": "optional path",
                    "dataset_source": "optional path",
                    "pov_scene": "optional RaySAR-compatible .pov path for raysar_synthesis",
                    "model_file": "optional OBJ/STL/PLY/GLB/point-cloud path for RaySAR model-to-POV compilation",
                    "parameters_file": "optional RaySAR parameters.txt/yaml/json path",
                    "contributions_txt": "optional existing Contributions.txt path for postprocess-only RaySAR",
                    "baseline_dataset": "optional path for downstream classification evaluation",
                    "augmented_dataset": "optional path for downstream classification evaluation",
                    "model_family": "optional flux/sd3/sdxl",
                    "target_count": "optional positive integer",
                    "training_epochs": "optional positive integer for LoRA training epochs",
                    "width": "optional RaySAR/render image width in pixels",
                    "height": "optional RaySAR/render image height in pixels",
                    "scene_prompt": "optional natural-language scene/background prompt for background_generation",
                    "target_source": "optional target image directory for target_background_composition",
                    "background_source": "optional background image directory for target_background_composition",
                    "mask_source": "optional mask directory for target_background_composition",
                    "blend_mode": "optional feather/laplacian/poisson",
                    "raysar_geometry": {
                        "incidence_angle_deg": "optional number",
                        "depression_angle_deg": "optional number",
                        "azimuth_deg": "optional number",
                        "target_extent_m": "optional number",
                        "sensor_plane_m": "optional number",
                        "range_distance_m": "optional number",
                        "input_up_axis": "optional x/y/z",
                        "azimuth_sweep": {"start": "number", "stop": "number", "step": "number"},
                        "azimuth_values": ["optional explicit azimuth numbers"],
                    },
                    "filters": {"field": "scalar"},
                    "exclude_filters": {"field": ["scalar values to exclude"]},
                    "goals": ["string"],
                },
                "format_hints": [
                    {
                        "type": "filename_anchor_fields",
                        "source": "llm_intent",
                        "source_text": "original phrase",
                        "rule": {
                            "anchor": "number or string",
                            "fields_after_anchor": [
                                {"raw": "下视角", "normalized": "depression_angle_deg"}
                            ],
                        },
                        "validation_required": True,
                    },
                    {
                        "type": "filename_suffix_field",
                        "source": "llm_intent",
                        "source_text": "original phrase",
                        "rule": {
                            "field": {"raw": "极化方式", "normalized": "polarization"},
                            "position": "last_underscore_token_before_extension",
                        },
                        "validation_required": True,
                    }
                ],
                "confidence": "0..1",
                "unresolved": ["string"],
                "clarification_questions": ["string"],
                "rationale": "short explanation",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_intent_proposal(raw: dict[str, Any]) -> dict[str, Any]:
    body = raw.get("intent_proposal", raw)
    updates = body.get("intent_updates") or body.get("intent") or {}
    if not isinstance(updates, dict):
        updates = {}
    return {
        "schema_version": LLM_INTENT_VERSION,
        "source": body.get("source", "llm"),
        "intent_updates": updates,
        "format_hints": [hint for hint in body.get("format_hints", []) if isinstance(hint, dict)]
        if isinstance(body.get("format_hints", []), list)
        else [],
        "confidence": clamp_float(body.get("confidence"), default=0.5),
        "unresolved": normalize_string_list(body.get("unresolved")),
        "clarification_questions": normalize_string_list(body.get("clarification_questions")),
        "rationale": str(body.get("rationale") or ""),
    }


def validate_intent_proposal(proposal: dict[str, Any], rule_intent_spec: dict[str, Any]) -> dict[str, Any]:
    issues = []
    updates = proposal.get("intent_updates") or {}
    task = str(updates.get("task") or "").strip()
    if task and task not in SUPPORTED_TASKS:
        issues.append({"severity": "high", "name": "unsupported_task", "value": task})
    for key in (
        "style_source",
        "content_source",
        "dataset_source",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
        "baseline_dataset",
        "augmented_dataset",
        "target_source",
        "background_source",
        "mask_source",
    ):
        value = updates.get(key)
        if value and not is_safe_path_like(str(value)):
            issues.append({"severity": "high", "name": "unsafe_path_update", "field": key, "value": str(value)[:200]})
    model_family = str(updates.get("model_family") or "").strip().lower()
    if model_family and model_family not in MODEL_FAMILIES:
        issues.append({"severity": "medium", "name": "unsupported_model_family", "value": model_family})
    blend_mode = str(updates.get("blend_mode") or "").strip().lower()
    if blend_mode and blend_mode not in {"feather", "laplacian", "poisson", "hard"}:
        issues.append({"severity": "medium", "name": "unsupported_blend_mode", "value": blend_mode})
    if updates.get("target_count") is not None and safe_positive_int(updates.get("target_count"), upper=10000) is None:
        issues.append({"severity": "medium", "name": "invalid_target_count", "value": updates.get("target_count")})
    if updates.get("training_epochs") is not None and safe_positive_int(updates.get("training_epochs"), upper=10000) is None:
        issues.append({"severity": "medium", "name": "invalid_training_epochs", "value": updates.get("training_epochs")})
    for key in ("width", "height"):
        if updates.get(key) is not None and safe_positive_int(updates.get(key), upper=16384) is None:
            issues.append({"severity": "medium", "name": f"invalid_{key}", "value": updates.get(key)})
    raysar_geometry = updates.get("raysar_geometry") or {}
    if raysar_geometry and not isinstance(raysar_geometry, dict):
        issues.append({"severity": "high", "name": "invalid_raysar_geometry"})
    elif isinstance(raysar_geometry, dict):
        issues.extend(validate_raysar_geometry_update(raysar_geometry))
    filters = updates.get("filters") or {}
    if filters and not isinstance(filters, dict):
        issues.append({"severity": "high", "name": "invalid_filters"})
    elif isinstance(filters, dict):
        for key, value in filters.items():
            if not is_safe_filter_name(str(key)) or not is_scalar(value):
                issues.append({"severity": "high", "name": "unsafe_filter", "field": str(key)})
    exclude_filters = updates.get("exclude_filters") or {}
    if exclude_filters and not isinstance(exclude_filters, dict):
        issues.append({"severity": "high", "name": "invalid_exclude_filters"})
    elif isinstance(exclude_filters, dict):
        for key, value in exclude_filters.items():
            if not is_safe_filter_name(str(key)) or not is_safe_exclude_filter_value(value):
                issues.append({"severity": "high", "name": "unsafe_exclude_filter", "field": str(key)})

    blocking = [issue for issue in issues if issue.get("severity") == "high"]
    return {
        "schema_version": LLM_INTENT_GUARDRAIL_VERSION,
        "valid": not blocking,
        "decision": "accept_with_guardrails" if not blocking else "fallback_to_rule_intent",
        "rule_task": rule_intent_spec.get("intent", {}).get("task"),
        "issues": issues,
        "clarification_questions": proposal.get("clarification_questions", []),
    }


def apply_intent_proposal(
    rule_intent_spec: dict[str, Any],
    proposal: dict[str, Any],
    guardrail: dict[str, Any],
) -> dict[str, Any]:
    effective = deepcopy(rule_intent_spec)
    intent = effective.setdefault("intent", {})
    updates = proposal.get("intent_updates") or {}
    applied = []
    ignored = []
    if not guardrail.get("valid"):
        effective["llm_intent_notes"] = {
            "decision": guardrail.get("decision"),
            "applied_updates": applied,
            "ignored_updates": [{"field": "*", "reason": "guardrail_invalid"}],
            "clarification_questions": guardrail.get("clarification_questions", []),
        }
        return effective

    task = str(updates.get("task") or "").strip()
    current_task = intent.get("task")
    if task:
        if current_task in {None, "", "unknown", "augmentation_or_generation"} or current_task == task:
            intent["task"] = task
            applied.append({"field": "task", "value": task})
        else:
            ignored.append({"field": "task", "value": task, "reason": f"kept_rule_task:{current_task}"})

    for key in (
        "style_source",
        "content_source",
        "dataset_source",
        "pov_scene",
        "model_file",
        "parameters_file",
        "contributions_txt",
        "baseline_dataset",
        "augmented_dataset",
        "target_source",
        "background_source",
        "mask_source",
    ):
        value = updates.get(key)
        if not value:
            continue
        if intent.get(key):
            ignored.append({"field": key, "value": value, "reason": "rule_intent_already_has_value"})
        else:
            intent[key] = str(value)
            applied.append({"field": key, "value": str(value)})

    blend_mode = str(updates.get("blend_mode") or "").strip().lower()
    if blend_mode:
        if blend_mode in {"feather", "laplacian", "poisson", "hard"}:
            intent["blend_mode"] = blend_mode
            applied.append({"field": "blend_mode", "value": blend_mode})
        else:
            ignored.append({"field": "blend_mode", "value": blend_mode, "reason": "unsupported_blend_mode"})

    model_family = str(updates.get("model_family") or "").strip().lower()
    if model_family in MODEL_FAMILIES:
        intent["model_family"] = model_family
        applied.append({"field": "model_family", "value": model_family})

    if updates.get("target_count") is not None:
        target_count = safe_positive_int(updates.get("target_count"), upper=10000)
        if target_count is not None:
            intent["target_count"] = target_count
            applied.append({"field": "target_count", "value": target_count})

    if updates.get("scene_prompt") not in (None, ""):
        prompt = str(updates.get("scene_prompt")).strip()
        if len(prompt) <= 500:
            intent["scene_prompt"] = prompt
            applied.append({"field": "scene_prompt", "value": prompt})
        else:
            ignored.append({"field": "scene_prompt", "value": prompt[:120], "reason": "too_long"})

    if updates.get("training_epochs") is not None:
        training_epochs = safe_positive_int(updates.get("training_epochs"), upper=10000)
        if training_epochs is not None:
            intent["training_epochs"] = training_epochs
            applied.append({"field": "training_epochs", "value": training_epochs})

    for key in ("width", "height"):
        if updates.get(key) is None:
            continue
        size = safe_positive_int(updates.get(key), upper=16384)
        if size is not None:
            intent[key] = size
            applied.append({"field": key, "value": size})
        else:
            ignored.append({"field": key, "value": updates.get(key), "reason": "not_positive_render_size"})

    raysar_geometry = updates.get("raysar_geometry") if isinstance(updates.get("raysar_geometry"), dict) else {}
    if raysar_geometry:
        merged_geometry = dict(intent.get("raysar_geometry") or {})
        for key, value in sanitize_raysar_geometry_update(raysar_geometry).items():
            merged_geometry[key] = value
            applied.append({"field": f"raysar_geometry.{key}", "value": value})
        intent["raysar_geometry"] = merged_geometry

    filters = dict(intent.get("filters") or {})
    exclude_filters = normalize_exclude_filters(intent.get("exclude_filters") or {})
    filter_updates = updates.get("filters") if isinstance(updates.get("filters"), dict) else {}
    for key, value in filter_updates.items():
        exclude_value = parse_exclude_filter_value(value)
        if exclude_value is not None and is_safe_filter_name(str(key)):
            add_exclude_filter_value(exclude_filters, str(key), exclude_value)
            applied.append({"field": f"exclude_filters.{key}", "value": exclude_value})
        elif is_safe_filter_name(str(key)) and is_scalar(value):
            filters[str(key)] = value
            applied.append({"field": f"filters.{key}", "value": value})
    intent["filters"] = filters

    exclude_updates = updates.get("exclude_filters") if isinstance(updates.get("exclude_filters"), dict) else {}
    for key, value in exclude_updates.items():
        if not is_safe_filter_name(str(key)):
            ignored.append({"field": f"exclude_filters.{key}", "reason": "unsafe_filter_name"})
            continue
        values = normalize_exclude_values(value)
        if values:
            for item in values:
                add_exclude_filter_value(exclude_filters, str(key), item)
            applied.append({"field": f"exclude_filters.{key}", "value": values})
        else:
            ignored.append({"field": f"exclude_filters.{key}", "reason": "empty_or_unsafe_exclude_filter"})
    intent["exclude_filters"] = exclude_filters

    goals = []
    for goal in list(intent.get("goals") or []) + normalize_string_list(updates.get("goals")):
        if goal and goal not in goals:
            goals.append(goal)
    if goals:
        intent["goals"] = goals

    effective["llm_intent_notes"] = {
        "decision": guardrail.get("decision"),
        "confidence": proposal.get("confidence"),
        "applied_updates": applied,
        "ignored_updates": ignored,
        "unresolved": proposal.get("unresolved", []),
        "clarification_questions": guardrail.get("clarification_questions", []),
        "rationale": proposal.get("rationale"),
    }
    return effective


def merge_format_hints(rule_hints: list[dict[str, Any]], llm_hints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for hint in list(rule_hints or []) + list(llm_hints or []):
        key = json.dumps(hint.get("rule", hint), ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hint)
    return merged


def write_intent_artifacts(output_dir: str | Path, result: dict[str, Any]) -> None:
    path = Path(output_dir).expanduser().resolve()
    save_json(path / "intent_llm_proposal.json", result["proposal"])
    save_json(path / "intent_guardrail.json", result["guardrail"])
    save_json(path / "intent_effective_spec.json", result["effective_intent_spec"])
    save_json(path / "intent_merged_format_hints.json", result["merged_format_hints"])
    save_text(path / "intent_llm_report.md", render_intent_report(result))


def render_intent_report(result: dict[str, Any]) -> str:
    proposal = result.get("proposal", {})
    guardrail = result.get("guardrail", {})
    notes = result.get("effective_intent_spec", {}).get("llm_intent_notes", {})
    lines = [
        "# SAGA LLM Intent Recognizer",
        "",
        f"- Mode: `{result.get('mode')}`",
        f"- Decision: `{guardrail.get('decision')}`",
        f"- Valid: {guardrail.get('valid')}",
        f"- Confidence: {proposal.get('confidence')}",
        f"- Format hints: {len(result.get('merged_format_hints') or [])}",
        "",
        "## Applied Updates",
        "",
    ]
    for item in notes.get("applied_updates") or []:
        lines.append(f"- `{item.get('field')}` = `{item.get('value')}`")
    if not notes.get("applied_updates"):
        lines.append("- None")
    lines.extend(["", "## Ignored Updates", ""])
    for item in notes.get("ignored_updates") or []:
        lines.append(f"- `{item.get('field')}`: {item.get('reason')}")
    if not notes.get("ignored_updates"):
        lines.append("- None")
    lines.extend(["", "## Clarification Questions", ""])
    for question in guardrail.get("clarification_questions") or []:
        lines.append(f"- {question}")
    if not guardrail.get("clarification_questions"):
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def is_safe_exclude_filter_value(value: Any) -> bool:
    if is_scalar(value):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(is_scalar(item) for item in value)
    return False


def normalize_exclude_filters(value: dict[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    if not isinstance(value, dict):
        return normalized
    for key, raw in value.items():
        if not is_safe_filter_name(str(key)):
            continue
        values = normalize_exclude_values(raw)
        if values:
            normalized[str(key)] = values
    return normalized


def normalize_exclude_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]
    values = []
    for item in raw_values:
        if not is_scalar(item):
            continue
        parsed = parse_exclude_filter_value(item)
        value_to_add = parsed if parsed is not None else item
        if value_to_add is not None and str(value_to_add) != "":
            values.append(value_to_add)
    return sorted({str(item): item for item in values}.values(), key=lambda item: str(item))


def add_exclude_filter_value(filters: dict[str, list[Any]], key: str, value: Any) -> None:
    existing = filters.setdefault(key, [])
    existing.append(value)
    filters[key] = sorted({str(item): item for item in existing}.values(), key=lambda item: str(item))


def validate_raysar_geometry_update(value: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    sanitized = sanitize_raysar_geometry_update(value)
    for key in value.keys():
        if key not in sanitized:
            issues.append({"severity": "medium", "name": "invalid_raysar_geometry_field", "field": str(key)})
    return issues


def sanitize_raysar_geometry_update(value: dict[str, Any]) -> dict[str, Any]:
    allowed_numeric = {
        "incidence_angle_deg": (1.0, 89.0),
        "depression_angle_deg": (1.0, 89.0),
        "azimuth_deg": (-360.0, 360.0),
        "target_extent_m": (0.001, 100000.0),
        "sensor_plane_m": (0.001, 1000000.0),
        "range_distance_m": (0.001, 10000000.0),
        "pitch_deg": (-360.0, 360.0),
        "roll_deg": (-360.0, 360.0),
        "scale_factor": (0.000001, 1000000.0),
    }
    sanitized: dict[str, Any] = {}
    for key, bounds in allowed_numeric.items():
        if value.get(key) is None:
            continue
        try:
            number = float(value[key])
        except (TypeError, ValueError):
            continue
        if bounds[0] <= number <= bounds[1]:
            sanitized[key] = number
    axis = str(value.get("input_up_axis") or "").lower()
    if axis in {"x", "y", "z"}:
        sanitized["input_up_axis"] = axis
    axis = str(value.get("pov_height_axis") or "").lower()
    if axis in {"x", "y", "z"}:
        sanitized["pov_height_axis"] = axis
    sweep = value.get("azimuth_sweep")
    if isinstance(sweep, dict):
        try:
            start = float(sweep["start"])
            stop = float(sweep["stop"])
            step = float(sweep["step"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if -3600 <= start <= 3600 and -3600 <= stop <= 3600 and 0 < abs(step) <= 360:
                sanitized["azimuth_sweep"] = {"start": start, "stop": stop, "step": step}
    values = value.get("azimuth_values")
    if isinstance(values, list):
        cleaned = []
        for item in values[:360]:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if -3600 <= number <= 3600:
                cleaned.append(number)
        if cleaned:
            sanitized["azimuth_values"] = cleaned
    return sanitized


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 3)

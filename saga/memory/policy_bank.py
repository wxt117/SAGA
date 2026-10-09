from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.core.jsonl import read_jsonl, write_jsonl


MEMORY_VERSION = "saga_policy_memory_v1"


def record_agent_run_memory(
    run_dir: str | Path,
    memory_dir: str | Path = "runs/memory",
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    memory_path = Path(memory_dir).expanduser().resolve()
    memory_path.mkdir(parents=True, exist_ok=True)

    entry = build_memory_entry(run_path)
    index_path = memory_path / "policy_memory.jsonl"
    entries = read_jsonl(index_path) if index_path.exists() else []
    entries = [item for item in entries if item.get("run_id") != entry["run_id"]]
    entries.append(entry)
    entries = sorted(entries, key=lambda item: item.get("created_at", 0))

    learning_summary = build_policy_learning_summary(entries)
    write_jsonl(index_path, entries)
    save_json(
        memory_path / "policy_memory_latest.json",
        {"schema_version": MEMORY_VERSION, "entries": entries[-50:], "policy_learning": learning_summary},
    )
    save_json(memory_path / "policy_learning.json", learning_summary)
    save_text(memory_path / "policy_memory.md", render_memory_markdown(entries[-30:]))
    save_text(memory_path / "policy_learning.md", render_policy_learning_markdown(learning_summary))
    return {
        "schema_version": MEMORY_VERSION,
        "status": "recorded",
        "memory_dir": memory_path.as_posix(),
        "index_path": index_path.as_posix(),
        "policy_learning_path": (memory_path / "policy_learning.json").as_posix(),
        "entry": entry,
        "entry_count": len(entries),
    }


def list_memory_entries(
    memory_dir: str | Path = "runs/memory",
    limit: int = 20,
    task: str | None = None,
    dataset_root: str | None = None,
) -> list[dict[str, Any]]:
    index_path = Path(memory_dir).expanduser().resolve() / "policy_memory.jsonl"
    if not index_path.exists():
        return []
    entries = read_jsonl(index_path)
    if task:
        entries = [entry for entry in entries if entry.get("task") == task]
    if dataset_root:
        needle = Path(dataset_root).expanduser().resolve().as_posix()
        entries = [entry for entry in entries if entry.get("dataset_root") == needle]
    return list(reversed(entries[-limit:]))


def retrieve_memory_context(
    raw_profile: dict[str, Any],
    task: str | None = None,
    dataset_root: str | None = None,
    memory_dir: str | Path = "runs/memory",
    limit: int = 5,
) -> dict[str, Any]:
    memory_path = Path(memory_dir).expanduser().resolve()
    entries = list_memory_entries(memory_dir=memory_path, limit=200)
    scored = []
    current_signature = dataset_signature(raw_profile)
    current_root = Path(dataset_root).expanduser().resolve().as_posix() if dataset_root else None
    for entry in entries:
        score, reasons = memory_similarity(
            current_signature=current_signature,
            entry=entry,
            task=task,
            dataset_root=current_root,
        )
        if score <= 0:
            continue
        scored.append(
            {
                "score": score,
                "reasons": reasons,
                "run_id": entry.get("run_id"),
                "run_dir": entry.get("run_dir"),
                "task": entry.get("task"),
                "recipe_id": entry.get("recipe_id"),
                "planner": entry.get("planner"),
                "bridge": entry.get("bridge"),
                "benefit": entry.get("benefit"),
                "execution": entry.get("execution"),
                "selected_skills": entry.get("selected_skills", []),
                "selected_skill_params": entry.get("selected_skill_params", {}),
                "quality": entry.get("quality"),
                "distribution": entry.get("distribution"),
                "sar_artifacts": entry.get("sar_artifacts"),
                "leakage": entry.get("leakage"),
                "duplicates": entry.get("duplicates"),
                "classification_evaluation": entry.get("classification_evaluation"),
                "benefit_evidence": entry.get("benefit_evidence"),
                "outcomes": entry.get("outcomes", {}),
                "repair": entry.get("repair"),
                "export": entry.get("export"),
                "reuse_notes": entry.get("reuse_notes", []),
            }
        )
    scored = sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]
    return {
        "schema_version": "saga_memory_retrieval_v1",
        "task": task,
        "dataset_root": current_root,
        "memory_dir": memory_path.as_posix(),
        "match_count": len(scored),
        "matches": scored,
        "policy_summary": build_policy_summary(scored),
        "global_policy_learning": load_global_policy_learning(memory_path),
    }


def build_policy_summary(matches: list[dict[str, Any]]) -> dict[str, Any]:
    skill_stats: dict[str, dict[str, Any]] = {}
    skill_param_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for item in matches:
        weight = min(float(item.get("score") or 0) / 100.0, 1.0)
        execution = item.get("execution") or {}
        quality = item.get("quality") or {}
        sar_artifacts = item.get("sar_artifacts") or {}
        benefit_evidence = item.get("benefit_evidence") or {}
        classification = (item.get("outcomes") or {}).get("classification") or {}
        for skill in item.get("selected_skills") or []:
            stats = skill_stats.setdefault(
                skill,
                {
                    "runs": 0,
                    "weighted_runs": 0.0,
                    "succeeded": 0,
                    "quality_trigger_runs": 0,
                    "sar_artifact_trigger_runs": 0,
                    "lightweight_passed_runs": 0,
                    "evidence_level_sum": 0.0,
                    "max_evidence_level": 0,
                    "downstream_claim_allowed_runs": 0,
                    "classification_improved_runs": 0,
                    "classification_regressed_runs": 0,
                },
            )
            stats["runs"] += 1
            stats["weighted_runs"] = round(float(stats["weighted_runs"]) + weight, 3)
            if execution.get("status") in {"succeeded", "dry_run"}:
                stats["succeeded"] += 1
            if int(quality.get("trigger_count") or 0) > 0:
                stats["quality_trigger_runs"] += 1
            if int(sar_artifacts.get("trigger_count") or 0) > 0:
                stats["sar_artifact_trigger_runs"] += 1
            evidence_level = safe_int(benefit_evidence.get("evidence_level")) or 0
            stats["evidence_level_sum"] = round(float(stats["evidence_level_sum"]) + weight * evidence_level, 3)
            stats["max_evidence_level"] = max(int(stats["max_evidence_level"]), evidence_level)
            if evidence_level >= 3:
                stats["lightweight_passed_runs"] += 1
            if benefit_evidence.get("downstream_claim_allowed"):
                stats["downstream_claim_allowed_runs"] += 1
            if classification.get("verdict") == "improved":
                stats["classification_improved_runs"] += 1
            elif classification.get("verdict") == "regressed":
                stats["classification_regressed_runs"] += 1
            add_param_policy_observation(
                skill_param_stats=skill_param_stats,
                skill=str(skill),
                params=(item.get("selected_skill_params") or {}).get(str(skill)) or [],
                classification=classification,
                execution=execution,
                quality=quality,
                sar_artifacts=sar_artifacts,
                weight=weight,
            )
    return {
        "schema_version": "saga_memory_policy_summary_v1",
        "skill_stats": skill_stats,
        "skill_param_stats": finalize_param_policy_summary(skill_param_stats),
    }


def load_global_policy_learning(memory_path: Path) -> dict[str, Any]:
    policy_path = memory_path / "policy_learning.json"
    if not policy_path.exists():
        return {}
    try:
        return load_mapping(policy_path)
    except Exception:
        return {}


def add_param_policy_observation(
    *,
    skill_param_stats: dict[str, dict[str, dict[str, Any]]],
    skill: str,
    params: list[dict[str, Any]],
    classification: dict[str, Any],
    execution: dict[str, Any],
    quality: dict[str, Any],
    sar_artifacts: dict[str, Any],
    weight: float,
) -> None:
    if skill not in {
        "TraditionalAugmentationSkill",
        "DiffusionLoRAGenerationSkill",
        "GeoDiffSARSkill",
        "GANImageToImageSkill",
        "StyleTransferSkill",
        "PseudocolorSkill",
        "BackgroundGenerationSkill",
        "RaySARSynthesisSkill",
        "RaySARSweepSynthesisSkill",
        "ClassificationEvaluationSkill",
    }:
        return
    for raw_params in params:
        if not isinstance(raw_params, dict) or not raw_params:
            continue
        compact = compact_policy_params(raw_params)
        if not compact:
            continue
        signature = param_signature(compact)
        stats = skill_param_stats.setdefault(skill, {}).setdefault(
            signature,
            {
                "params": compact,
                "runs": 0,
                "weighted_runs": 0.0,
                "succeeded": 0,
                "classification_improved_runs": 0,
                "classification_regressed_runs": 0,
                "classification_deltas": [],
                "quality_trigger_runs": 0,
                "sar_artifact_trigger_runs": 0,
            },
        )
        stats["runs"] += 1
        stats["weighted_runs"] = round(float(stats["weighted_runs"]) + float(weight), 3)
        if execution.get("status") in {"succeeded", "dry_run"}:
            stats["succeeded"] += 1
        if classification.get("verdict") == "improved":
            stats["classification_improved_runs"] += 1
        elif classification.get("verdict") == "regressed":
            stats["classification_regressed_runs"] += 1
        delta = safe_float(classification.get("delta"))
        if delta is not None:
            stats["classification_deltas"].append(delta)
        if int(quality.get("trigger_count") or 0) > 0:
            stats["quality_trigger_runs"] += 1
        if int(sar_artifacts.get("trigger_count") or 0) > 0:
            stats["sar_artifact_trigger_runs"] += 1


def compact_policy_params(params: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "target_count",
        "training_epochs",
        "model_family",
        "multiplier",
        "num_images",
        "epochs",
        "variant",
        "colormap",
        "width",
        "height",
        "cns",
        "steps",
        "scale",
        "model_file",
        "azimuth_values",
        "azimuth_sweep",
        "target_azimuths",
        "depressions",
        "depression_angle_deg",
        "width",
        "height",
        "raysar_geometry",
        "filters",
        "exclude_filters",
    )
    compact = {}
    for key in allowed:
        value = params.get(key)
        if value not in (None, "", {}, []):
            compact[key] = value
    return compact


def param_signature(params: dict[str, Any]) -> str:
    import json

    payload = json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def finalize_param_policy_summary(
    skill_param_stats: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    finalized: dict[str, list[dict[str, Any]]] = {}
    for skill, by_signature in skill_param_stats.items():
        rows = []
        for signature, stats in by_signature.items():
            deltas = [value for value in stats.get("classification_deltas", []) if safe_float(value) is not None]
            avg_delta = round(sum(deltas) / len(deltas), 6) if deltas else None
            score = (
                0.25 * float(stats.get("weighted_runs") or 0)
                + 0.15 * int(stats.get("succeeded") or 0)
                + 0.35 * int(stats.get("classification_improved_runs") or 0)
                - 0.4 * int(stats.get("classification_regressed_runs") or 0)
                - 0.12 * int(stats.get("quality_trigger_runs") or 0)
                - 0.12 * int(stats.get("sar_artifact_trigger_runs") or 0)
            )
            if avg_delta is not None:
                score += max(-0.35, min(0.35, avg_delta))
            row = dict(stats)
            row["signature"] = signature
            row["avg_classification_delta"] = avg_delta
            row["policy_score"] = round(score, 3)
            rows.append(row)
        finalized[skill] = sorted(rows, key=lambda item: item.get("policy_score", 0.0), reverse=True)[:5]
    return finalized


def memory_similarity(
    current_signature: dict[str, Any],
    entry: dict[str, Any],
    task: str | None,
    dataset_root: str | None,
) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    if task and entry.get("task") == task:
        score += 30
        reasons.append("same_task")
    if dataset_root and entry.get("dataset_root") == dataset_root:
        score += 25
        reasons.append("same_dataset_root")

    old_signature = entry.get("dataset_signature") or {}
    current_suffixes = current_signature.get("image_suffix_counts") or {}
    old_suffixes = old_signature.get("image_suffix_counts") or {}
    if current_suffixes and old_suffixes and set(current_suffixes.keys()) == set(old_suffixes.keys()):
        score += 10
        reasons.append("same_image_suffixes")

    current_sizes = current_signature.get("image_size_counts") or {}
    old_sizes = old_signature.get("image_size_counts") or {}
    if current_sizes and old_sizes and set(current_sizes.keys()) & set(old_sizes.keys()):
        score += 10
        reasons.append("overlapping_image_sizes")

    current_patterns = {
        item.get("pattern")
        for item in current_signature.get("path_group_patterns", [])
        if item.get("pattern")
    }
    old_patterns = {
        item.get("pattern")
        for item in old_signature.get("path_group_patterns", [])
        if item.get("pattern")
    }
    if current_patterns and old_patterns and current_patterns & old_patterns:
        score += 15
        reasons.append("overlapping_path_patterns")

    if entry.get("execution", {}).get("status") == "succeeded":
        score += 5
        reasons.append("previous_execution_succeeded")
    if entry.get("quality", {}).get("trigger_count", 0) == 0:
        score += 5
        reasons.append("no_previous_quality_triggers")
    evidence_level = safe_int((entry.get("benefit_evidence") or {}).get("evidence_level")) or 0
    if evidence_level >= 3:
        score += min(10, evidence_level * 2)
        reasons.append("previous_benefit_evidence")
    return score, reasons


def build_memory_entry(run_path: Path) -> dict[str, Any]:
    state = load_optional_mapping(run_path / "agent_state.json")
    raw_profile = load_optional_mapping(run_path / "raw_profile.json")
    bridge = load_optional_mapping(run_path / "format_bridge.json")
    execution = load_optional_mapping(run_path / "execution" / "recipe_execution.json")
    export_report = load_optional_mapping(run_path / "augmented_dataset" / "export_report.json")
    intent = load_optional_mapping(run_path / "effective_intent_spec.json") or load_optional_mapping(run_path / "intent_spec.json")
    planner_guardrail = load_optional_mapping(run_path / "planner_guardrail.json")
    plan_verification = load_optional_mapping(run_path / "plan_verification.json")
    dataset_need_profile = load_optional_mapping(run_path / "dataset_need_profile.json")
    skill_utility = load_optional_mapping(run_path / "skill_utility.json")
    plan_utility = load_optional_mapping(run_path / "plan_utility_report.json")
    benefit_evidence = load_optional_mapping(run_path / "benefit_evidence.json")

    steps = execution.get("steps") or []
    selected_step = find_step(steps, "select_content")
    quality_step = find_step(steps, "evaluate_outputs")
    distribution_step = find_step(steps, "evaluate_distribution")
    sar_artifact_step = find_step(steps, "evaluate_sar_artifacts")
    classification_step = find_step(steps, "evaluate_classification") or find_step(steps, "classification_evaluation")
    leakage_step = find_step(steps, "leakage_check")
    duplicate_step = find_step(steps, "duplicate_check")
    repair_step = find_step(steps, "repair_policy")
    export_step = find_step(steps, "export_dataset")
    quality_report = quality_step.get("quality_report") or {}
    distribution_report = distribution_step.get("distribution_report") or {}
    sar_artifact_report = sar_artifact_step.get("sar_artifact_report") or {}
    classification_report = classification_step.get("skill_report") or {}
    evaluator_report = classification_report.get("evaluator_report") or classification_step.get("evaluator_report") or {}
    leakage_report = leakage_step.get("leakage_report") or {}
    duplicate_report = duplicate_step.get("duplicate_report") or {}
    repair_policy = repair_step.get("repair_policy") or {}
    selected_skills = sorted(
        {
            str(step.get("skill"))
            for step in steps
            if step.get("skill") and step.get("status") not in {"skipped", "failed"}
        }
    )
    selected_skill_params = summarize_selected_skill_params(steps)
    classification_outcome = extract_classification_outcome(
        classification_report=classification_report,
        classification_step=classification_step,
        evaluator_report=evaluator_report,
    )
    classification_outcome = apply_data_gates_to_classification_outcome(
        classification_outcome=classification_outcome,
        leakage_report=leakage_report,
        leakage_step=leakage_step,
        duplicate_report=duplicate_report,
        duplicate_step=duplicate_step,
    )

    recipe_id = state.get("recipe_id") or execution.get("recipe_id") or "unknown_recipe"
    dataset_root = state.get("dataset_root") or raw_profile.get("root")
    run_id = stable_run_id(run_path=run_path, recipe_id=str(recipe_id), dataset_root=str(dataset_root))
    entry = {
        "schema_version": MEMORY_VERSION,
        "run_id": run_id,
        "created_at": round(time.time(), 3),
        "run_dir": run_path.as_posix(),
        "request": state.get("request") or intent.get("raw_text"),
        "dataset_root": dataset_root,
        "dataset_signature": dataset_signature(raw_profile),
        "task": (intent.get("intent") or {}).get("task") or execution.get("task"),
        "recipe_id": recipe_id,
        "recipe_path": state.get("recipe_path"),
        "selected_skills": selected_skills,
        "selected_skill_params": selected_skill_params,
        "planner": {
            "mode": state.get("planner_mode"),
            "decision": state.get("planner_decision") or planner_guardrail.get("decision"),
            "execution_allowed": state.get("planner_execution_allowed")
            if state.get("planner_execution_allowed") is not None
            else planner_guardrail.get("execution_allowed"),
            "requires_clarification": state.get("planner_requires_clarification")
            if state.get("planner_requires_clarification") is not None
            else planner_guardrail.get("requires_clarification"),
            "question_count": state.get("planner_question_count")
            if state.get("planner_question_count") is not None
            else len(planner_guardrail.get("clarification_questions") or []),
            "verification_status": state.get("plan_verification_status") or plan_verification.get("status"),
        },
        "bridge": {
            "valid": bridge.get("valid"),
            "profile_level_after_validation": bridge.get("profile_level_after_validation"),
            "required_fields": bridge.get("required_fields"),
            "field_coverage": (bridge.get("validation") or {}).get("field_coverage"),
        },
        "benefit": {
            "top_needs": dataset_need_profile.get("top_needs", []),
            "top_skill_recommendations": [
                {
                    "skill": item.get("skill"),
                    "status": item.get("status"),
                    "utility_score": item.get("utility_score"),
                    "verdict": item.get("verdict"),
                }
                for item in (skill_utility.get("top_recommendations") or [])[:8]
            ],
            "selected_plan_utility": plan_utility.get("selected_utility_score"),
        },
        "benefit_evidence": {
            "evidence_level": benefit_evidence.get("evidence_level"),
            "evidence_label": benefit_evidence.get("evidence_label"),
            "downstream_claim_allowed": benefit_evidence.get("downstream_claim_allowed"),
            "lightweight_claim_allowed": benefit_evidence.get("lightweight_claim_allowed"),
            "recommended_next_action": benefit_evidence.get("recommended_next_action"),
            "risks": benefit_evidence.get("risks", [])[:5],
        },
        "execution": {
            "status": execution.get("status") or state.get("execution_status"),
            "dry_run": execution.get("dry_run", state.get("dry_run")),
            "step_count": len(steps),
            "selected_count": selected_step.get("selected_count"),
        },
        "quality": {
            "status": quality_report.get("status") or quality_step.get("status"),
            "image_count": quality_report.get("image_count") or quality_step.get("image_count"),
            "trigger_count": len(quality_report.get("triggers") or quality_step.get("triggers") or []),
            "triggers": quality_report.get("triggers") or quality_step.get("triggers") or [],
        },
        "distribution": {
            "status": distribution_report.get("status") or distribution_step.get("status"),
            "trigger_count": len(distribution_report.get("triggers") or distribution_step.get("triggers") or []),
            "metrics": distribution_report.get("metrics") or distribution_step.get("metrics") or {},
        },
        "sar_artifacts": {
            "status": sar_artifact_report.get("status") or sar_artifact_step.get("status"),
            "trigger_count": len(sar_artifact_report.get("triggers") or sar_artifact_step.get("triggers") or []),
            "summary": sar_artifact_report.get("summary") or {},
        },
        "leakage": {
            "status": leakage_report.get("status") or leakage_step.get("status"),
            "trigger_count": safe_int(leakage_report.get("trigger_count") or leakage_step.get("trigger_count")) or 0,
            "overlap_count": leakage_overlap_count(leakage_report),
            "overlaps": leakage_report.get("overlaps", [])[:5],
        },
        "duplicates": {
            "status": duplicate_report.get("status") or duplicate_step.get("status"),
            "pair_count": safe_int(duplicate_report.get("pair_count") or duplicate_step.get("pair_count")) or 0,
            "hamming_threshold": duplicate_report.get("hamming_threshold"),
        },
        "classification_evaluation": {
            "status": classification_report.get("status") or classification_step.get("status"),
            "metrics": classification_report.get("metrics") or classification_step.get("metrics") or {},
            "evaluator_report": evaluator_report,
            "data_gate_verdict": classification_outcome.get("verdict")
            if classification_outcome.get("verdict") == "invalidated_by_data_gate"
            else "not_invalidated",
        },
        "outcomes": {
            "classification": classification_outcome,
            "quality": {
                "status": quality_report.get("status") or quality_step.get("status"),
                "trigger_count": len(quality_report.get("triggers") or quality_step.get("triggers") or []),
            },
            "distribution": {
                "status": distribution_report.get("status") or distribution_step.get("status"),
                "trigger_count": len(distribution_report.get("triggers") or distribution_step.get("triggers") or []),
            },
            "sar_artifacts": {
                "status": sar_artifact_report.get("status") or sar_artifact_step.get("status"),
                "trigger_count": len(sar_artifact_report.get("triggers") or sar_artifact_step.get("triggers") or []),
            },
            "leakage": {
                "status": leakage_report.get("status") or leakage_step.get("status"),
                "trigger_count": safe_int(leakage_report.get("trigger_count") or leakage_step.get("trigger_count")) or 0,
            },
            "duplicates": {
                "status": duplicate_report.get("status") or duplicate_step.get("status"),
                "pair_count": safe_int(duplicate_report.get("pair_count") or duplicate_step.get("pair_count")) or 0,
            },
        },
        "repair": {
            "status": repair_policy.get("status") or repair_step.get("status"),
            "suggested_action_count": len(repair_policy.get("suggested_actions") or []),
        },
        "export": {
            "status": export_report.get("status") or export_step.get("status"),
            "generated_image_count": export_report.get("generated_image_count") or export_step.get("generated_image_count"),
            "manifest_count": export_report.get("manifest_count") or export_step.get("manifest_count"),
            "output_dir": export_report.get("output_dir") or export_step.get("output_dir"),
        },
        "reuse_notes": build_reuse_notes(
            bridge=bridge,
            execution=execution,
            quality=quality_report,
            distribution=distribution_report,
            sar_artifacts=sar_artifact_report,
            leakage=leakage_report,
            duplicates=duplicate_report,
            classification=classification_report,
            export_report=export_report,
        ),
    }
    return entry


def summarize_selected_skill_params(steps: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for step in steps:
        skill = step.get("skill")
        if not skill or step.get("status") in {"skipped", "failed"}:
            continue
        params = step.get("skill_run_report", {}).get("inputs") or {}
        if not params:
            params = step.get("params") or {}
        compact = {}
        for key in (
            "target_count",
            "training_epochs",
            "epochs",
            "model_family",
            "filters",
            "exclude_filters",
            "multiplier",
            "num_images",
            "variant",
            "colormap",
            "width",
            "height",
            "cns",
            "steps",
            "scale",
            "config",
            "dataset_config",
            "baseline_dataset_config",
        ):
            if params.get(key) not in (None, "", {}, []):
                compact[key] = params.get(key)
        summary.setdefault(str(skill), []).append(compact)
    return summary


def extract_classification_outcome(
    classification_report: dict[str, Any],
    classification_step: dict[str, Any],
    evaluator_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluator_report = evaluator_report or {}
    if evaluator_report:
        return {
            "available": evaluator_report.get("baseline") is not None and evaluator_report.get("augmented") is not None,
            "status": evaluator_report.get("status"),
            "primary_metric": evaluator_report.get("primary_metric"),
            "baseline": safe_float(evaluator_report.get("baseline")),
            "augmented": safe_float(evaluator_report.get("augmented")),
            "delta": safe_float(evaluator_report.get("delta")),
            "verdict": evaluator_report.get("verdict") or "unknown",
            "confidence": evaluator_report.get("confidence"),
            "leakage_risk": evaluator_report.get("leakage_risk"),
        }
    metrics = classification_report.get("metrics") or classification_step.get("metrics") or {}
    primary = metrics.get("primary") if isinstance(metrics, dict) else {}
    if not isinstance(primary, dict) or not primary:
        return {
            "available": False,
            "status": classification_report.get("status") or classification_step.get("status"),
            "reason": "primary_metric_unavailable",
        }
    metric = primary.get("metric") or primary.get("name") or "primary"
    baseline = safe_float(primary.get("baseline"))
    augmented = safe_float(primary.get("augmented"))
    delta = safe_float(primary.get("delta"))
    if delta is None and baseline is not None and augmented is not None:
        delta = augmented - baseline
    verdict = primary.get("verdict")
    if not verdict and delta is not None:
        if delta > 0:
            verdict = "improved"
        elif delta < 0:
            verdict = "regressed"
        else:
            verdict = "unchanged"
    return {
        "available": baseline is not None and augmented is not None,
        "status": classification_report.get("status") or classification_step.get("status"),
        "primary_metric": metric,
        "baseline": baseline,
        "augmented": augmented,
        "delta": delta,
        "verdict": verdict or "unknown",
    }


def apply_data_gates_to_classification_outcome(
    *,
    classification_outcome: dict[str, Any],
    leakage_report: dict[str, Any],
    leakage_step: dict[str, Any],
    duplicate_report: dict[str, Any],
    duplicate_step: dict[str, Any],
) -> dict[str, Any]:
    leakage_triggers = safe_int(leakage_report.get("trigger_count") or leakage_step.get("trigger_count")) or 0
    duplicate_pairs = safe_int(duplicate_report.get("pair_count") or duplicate_step.get("pair_count")) or 0
    reasons = []
    if leakage_triggers > 0:
        reasons.append("leakage_check_triggered")
    if duplicate_pairs > 0:
        reasons.append("duplicate_near_duplicate_check_triggered")
    if not reasons:
        return classification_outcome
    gated = dict(classification_outcome)
    gated["raw_outcome_before_data_gates"] = classification_outcome
    gated["available"] = False
    gated["verdict"] = "invalidated_by_data_gate"
    gated["reason"] = ",".join(reasons)
    gated["leakage_risk"] = "high" if leakage_triggers > 0 else gated.get("leakage_risk", "medium")
    gated["duplicate_risk"] = "high" if duplicate_pairs > 0 else "low"
    gated["data_gates"] = {
        "leakage_trigger_count": leakage_triggers,
        "duplicate_pair_count": duplicate_pairs,
        "leakage_status": leakage_report.get("status") or leakage_step.get("status"),
        "duplicate_status": duplicate_report.get("status") or duplicate_step.get("status"),
    }
    return gated


def leakage_overlap_count(leakage_report: dict[str, Any]) -> int:
    total = 0
    for item in leakage_report.get("overlaps") or []:
        total += safe_int(item.get("hash_overlap_count")) or 0
        total += safe_int(item.get("stem_overlap_count")) or 0
    return total


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def dataset_signature(raw_profile: dict[str, Any]) -> dict[str, Any]:
    images = raw_profile.get("images") or {}
    filesystem = raw_profile.get("filesystem") or {}
    path_groups = raw_profile.get("path_groups") or []
    group_patterns = []
    for group in path_groups[:5]:
        group_patterns.append(
            {
                "pattern": group.get("pattern"),
                "count": group.get("count"),
                "examples": group.get("examples", [])[:3],
            }
        )
    return {
        "total_files": filesystem.get("total_files"),
        "total_images": images.get("total_images"),
        "image_suffix_counts": images.get("suffix_counts"),
        "image_mode_counts": (images.get("probe_summary") or {}).get("mode_counts"),
        "image_size_counts": (images.get("probe_summary") or {}).get("size_counts"),
        "path_group_patterns": group_patterns,
    }


def build_reuse_notes(
    bridge: dict[str, Any],
    execution: dict[str, Any],
    quality: dict[str, Any],
    distribution: dict[str, Any] | None,
    sar_artifacts: dict[str, Any] | None,
    leakage: dict[str, Any] | None,
    duplicates: dict[str, Any] | None,
    classification: dict[str, Any] | None,
    export_report: dict[str, Any],
) -> list[str]:
    notes = []
    if bridge.get("valid"):
        notes.append("Validated DatasetFormatSpec can be reused for similar path patterns.")
    if execution.get("status") == "succeeded":
        notes.append("Recipe DAG completed without failed steps.")
    if quality.get("status") == "passed":
        notes.append("Observer gate passed for generated outputs.")
    elif quality.get("status") in {"warning", "failed"}:
        notes.append("Observer gate raised triggers; reuse recipe only after inspecting repair policy.")
    if (distribution or {}).get("status") == "passed":
        notes.append("Distribution proxy metrics passed for sampled generated outputs.")
    elif (distribution or {}).get("triggers"):
        notes.append("Distribution proxy metrics raised triggers; compare against downstream task metrics before reuse.")
    if (sar_artifacts or {}).get("status") == "passed":
        notes.append("SAR artifact observer passed for sampled outputs.")
    elif (sar_artifacts or {}).get("triggers"):
        notes.append("SAR artifact observer raised structure/artifact triggers; inspect worst samples before reuse.")
    if safe_int((leakage or {}).get("trigger_count")):
        notes.append("Leakage gate triggered; do not reuse classification benefit as valid policy evidence.")
    if safe_int((duplicates or {}).get("pair_count")):
        notes.append("Duplicate/near-duplicate gate triggered; downstream benefit should be treated as invalid until cleaned.")
    if (classification or {}).get("metrics"):
        notes.append("Downstream classification metrics are available for policy comparison.")
    if export_report.get("status") == "succeeded":
        notes.append("Exported candidate augmented dataset is available.")
    return notes or ["Recorded for future comparison; no success claim is made."]


def find_step(steps: list[dict[str, Any]], step_id: str) -> dict[str, Any]:
    for step in steps:
        if step.get("step_id") == step_id:
            return step
    return {}


def load_optional_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_mapping(path)


def stable_run_id(run_path: Path, recipe_id: str, dataset_root: str) -> str:
    raw = f"{run_path.as_posix()}|{recipe_id}|{dataset_root}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_policy_learning_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_skill: dict[str, dict[str, Any]] = {}
    for entry in entries:
        benefit = entry.get("benefit_evidence") or {}
        classification = ((entry.get("outcomes") or {}).get("classification") or {})
        evidence_level = safe_int(benefit.get("evidence_level")) or 0
        for skill in entry.get("selected_skills") or []:
            row = by_skill.setdefault(
                str(skill),
                {
                    "runs": 0,
                    "succeeded": 0,
                    "evidence_level_sum": 0,
                    "max_evidence_level": 0,
                    "lightweight_passed_runs": 0,
                    "downstream_claim_allowed_runs": 0,
                    "classification_improved_runs": 0,
                    "classification_regressed_runs": 0,
                    "quality_trigger_runs": 0,
                    "sar_artifact_trigger_runs": 0,
                    "duplicate_risk_runs": 0,
                    "leakage_risk_runs": 0,
                },
            )
            row["runs"] += 1
            if (entry.get("execution") or {}).get("status") == "succeeded":
                row["succeeded"] += 1
            row["evidence_level_sum"] += evidence_level
            row["max_evidence_level"] = max(row["max_evidence_level"], evidence_level)
            if evidence_level >= 3:
                row["lightweight_passed_runs"] += 1
            if benefit.get("downstream_claim_allowed"):
                row["downstream_claim_allowed_runs"] += 1
            if classification.get("verdict") == "improved":
                row["classification_improved_runs"] += 1
            elif classification.get("verdict") == "regressed":
                row["classification_regressed_runs"] += 1
            if int((entry.get("quality") or {}).get("trigger_count") or 0) > 0:
                row["quality_trigger_runs"] += 1
            if int((entry.get("sar_artifacts") or {}).get("trigger_count") or 0) > 0:
                row["sar_artifact_trigger_runs"] += 1
            if int((entry.get("duplicates") or {}).get("pair_count") or 0) > 0:
                row["duplicate_risk_runs"] += 1
            if int((entry.get("leakage") or {}).get("trigger_count") or 0) > 0:
                row["leakage_risk_runs"] += 1
    rows = []
    for skill, row in by_skill.items():
        runs = max(1, int(row.get("runs") or 0))
        avg_level = float(row.get("evidence_level_sum") or 0) / runs
        policy_score = (
            0.2 * row.get("succeeded", 0)
            + 0.35 * row.get("lightweight_passed_runs", 0)
            + 0.75 * row.get("downstream_claim_allowed_runs", 0)
            + 0.4 * row.get("classification_improved_runs", 0)
            - 0.45 * row.get("classification_regressed_runs", 0)
            - 0.2 * row.get("quality_trigger_runs", 0)
            - 0.2 * row.get("sar_artifact_trigger_runs", 0)
            - 0.35 * row.get("duplicate_risk_runs", 0)
            - 0.45 * row.get("leakage_risk_runs", 0)
            + 0.1 * avg_level
        )
        item = dict(row)
        item["skill"] = skill
        item["avg_evidence_level"] = round(avg_level, 3)
        item["policy_score"] = round(policy_score, 3)
        item["recommendation"] = policy_recommendation(item)
        rows.append(item)
    rows = sorted(rows, key=lambda item: item.get("policy_score", 0), reverse=True)
    global_policy_summary = build_policy_summary(
        [
            {
                "score": 100,
                "selected_skills": entry.get("selected_skills", []),
                "selected_skill_params": entry.get("selected_skill_params", {}),
                "execution": entry.get("execution", {}),
                "quality": entry.get("quality", {}),
                "sar_artifacts": entry.get("sar_artifacts", {}),
                "benefit_evidence": entry.get("benefit_evidence", {}),
                "outcomes": entry.get("outcomes", {}),
            }
            for entry in entries
        ]
    )
    return {
        "schema_version": "saga_policy_learning_summary_v1",
        "entry_count": len(entries),
        "skill_count": len(rows),
        "ranked_skills": rows,
        "skill_priors": build_skill_priors(rows),
        "recommended_params_by_skill": global_policy_summary.get("skill_param_stats", {}),
        "notes": [
            "Policy learning summarizes observed evidence levels, lightweight probes, downstream claim gates, and failure risks.",
            "It is a planner prior, not a statistical guarantee of future task improvement.",
        ],
    }


def build_skill_priors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priors: dict[str, Any] = {}
    for row in rows:
        score = safe_float(row.get("policy_score")) or 0.0
        adjustment = max(-0.18, min(0.18, score / 12.0))
        priors[str(row.get("skill"))] = {
            "policy_score": round(score, 3),
            "planner_adjustment": round(adjustment, 3),
            "runs": row.get("runs"),
            "recommendation": row.get("recommendation"),
            "avg_evidence_level": row.get("avg_evidence_level"),
            "max_evidence_level": row.get("max_evidence_level"),
            "risk_counts": {
                "quality_trigger_runs": row.get("quality_trigger_runs", 0),
                "sar_artifact_trigger_runs": row.get("sar_artifact_trigger_runs", 0),
                "duplicate_risk_runs": row.get("duplicate_risk_runs", 0),
                "leakage_risk_runs": row.get("leakage_risk_runs", 0),
            },
        }
    return priors


def policy_recommendation(row: dict[str, Any]) -> str:
    if row.get("downstream_claim_allowed_runs", 0) > 0:
        return "strong_prior_when_dataset_is_similar"
    if row.get("lightweight_passed_runs", 0) > 0 and row.get("quality_trigger_runs", 0) == 0:
        return "promising_lightweight_prior"
    if row.get("duplicate_risk_runs", 0) or row.get("leakage_risk_runs", 0):
        return "reuse_only_after_data_gate_review"
    if row.get("quality_trigger_runs", 0) or row.get("sar_artifact_trigger_runs", 0):
        return "requires_repair_before_reuse"
    return "insufficient_positive_evidence"


def render_policy_learning_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Policy Learning",
        "",
        f"- Entries: {report.get('entry_count')}",
        f"- Skills: {report.get('skill_count')}",
        "",
        "## Ranked Skills",
        "",
    ]
    for item in report.get("ranked_skills") or []:
        lines.extend(
            [
                f"### {item.get('skill')}",
                "",
                f"- Policy score: {item.get('policy_score')}",
                f"- Runs: {item.get('runs')}",
                f"- Avg evidence level: {item.get('avg_evidence_level')}",
                f"- Max evidence level: {item.get('max_evidence_level')}",
                f"- Lightweight passed runs: {item.get('lightweight_passed_runs')}",
                f"- Downstream claim allowed runs: {item.get('downstream_claim_allowed_runs')}",
                f"- Recommendation: `{item.get('recommendation')}`",
                "",
            ]
        )
    return "\n".join(lines)


def render_memory_markdown(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# SAGA Policy Memory",
        "",
        f"- Entries shown: {len(entries)}",
        "",
    ]
    for entry in reversed(entries):
        lines.extend(
            [
                f"## {entry.get('recipe_id')}",
                "",
                f"- Run: `{entry.get('run_dir')}`",
                f"- Task: {entry.get('task')}",
                f"- Dataset: `{entry.get('dataset_root')}`",
                f"- Selected skills: `{entry.get('selected_skills', [])}`",
                f"- Selected skill params: `{entry.get('selected_skill_params', {})}`",
                f"- Bridge valid: {entry.get('bridge', {}).get('valid')}",
                f"- Planner: {entry.get('planner', {}).get('mode')} "
                f"({entry.get('planner', {}).get('decision')})",
                f"- Execution: {entry.get('execution', {}).get('status')}",
                f"- Benefit evidence: level {entry.get('benefit_evidence', {}).get('evidence_level')} "
                f"({entry.get('benefit_evidence', {}).get('evidence_label')})",
                f"- Quality: {entry.get('quality', {}).get('status')} "
                f"({entry.get('quality', {}).get('trigger_count')} triggers)",
                f"- Distribution: {entry.get('distribution', {}).get('status')} "
                f"({entry.get('distribution', {}).get('trigger_count')} triggers)",
                f"- SAR artifacts: {entry.get('sar_artifacts', {}).get('status')} "
                f"({entry.get('sar_artifacts', {}).get('trigger_count')} triggers)",
                f"- Leakage: {entry.get('leakage', {}).get('status')} "
                f"({entry.get('leakage', {}).get('trigger_count')} triggers)",
                f"- Duplicates: {entry.get('duplicates', {}).get('status')} "
                f"({entry.get('duplicates', {}).get('pair_count')} pairs)",
                f"- Classification eval: {entry.get('classification_evaluation', {}).get('status')}",
                f"- Classification outcome: `{entry.get('outcomes', {}).get('classification', {})}`",
                f"- Export: {entry.get('export', {}).get('status')} "
                f"({entry.get('export', {}).get('manifest_count')} rows)",
                "",
            ]
        )
    return "\n".join(lines)

from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.skills.dataset_balancing.skill import run_dataset_balancing_skill
from saga.skills.metadata_caption.skill import run_metadata_caption_skill
from saga.skills.sar_preprocess.skill import run_sar_preprocess_skill


PLANNING_EVIDENCE_VERSION = "saga_planning_evidence_v1"


def build_planning_evidence(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    validated_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    dry_run: bool = True,
) -> dict[str, Any]:
    """Build deterministic evidence before choosing an augmentation recipe.

    This module intentionally runs only cheap, auditable skills. It gives the
    planner evidence about preprocessing readiness, captionability, and class /
    metadata deficits before the planner selects LoRA, GAN, traditional
    augmentation, or another generator.
    """

    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    evidence_dir = output / "planning_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = validated_dataset_config(validated_profile)

    reports: dict[str, Any] = {}
    reports["sar_preprocess"] = safe_evidence_call(
        "SARPreprocessSkill",
        lambda: run_sar_preprocess_skill(
            input_dir=root,
            output_dir=evidence_dir / "sar_preprocess",
            dry_run=dry_run,
        ),
    )
    reports["metadata_caption"] = safe_evidence_call(
        "MetadataCaptionSkill",
        lambda: run_metadata_caption_skill(
            dataset_root=root,
            output_dir=evidence_dir / "metadata_caption",
            dataset_config=dataset_config,
            mode="manifest_only",
            dry_run=dry_run,
        ),
    )
    reports["dataset_balance"] = safe_evidence_call(
        "DatasetBalancingSkill",
        lambda: run_dataset_balancing_skill(
            dataset_root=root,
            output_dir=evidence_dir / "dataset_balance",
            dataset_config=dataset_config,
            fields=balance_fields(validated_profile=validated_profile, intent_spec=intent_spec),
            dry_run=True,
        ),
    )

    evidence = {
        "schema_version": PLANNING_EVIDENCE_VERSION,
        "dataset_root": root.as_posix(),
        "dataset_config": dataset_config,
        "dry_run": dry_run,
        "evidence_dir": evidence_dir.as_posix(),
        "reports": reports,
        "summary": summarize_planning_evidence(reports),
        "notes": [
            "This evidence is deterministic and cheap; it does not execute expensive augmentation.",
            "LLM planner may use the evidence, but recipes still compile through deterministic guardrails.",
            "DatasetBalancingSkill produces deficits; generator skills decide how to realize them.",
        ],
    }
    save_json(evidence_dir / "planning_evidence.json", evidence)
    save_text(evidence_dir / "planning_evidence.md", render_planning_evidence_markdown(evidence))
    return evidence


def validated_dataset_config(validated_profile: dict[str, Any]) -> str | None:
    value = validated_profile.get("dataset_config")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve().as_posix() if path.exists() else None


def balance_fields(*, validated_profile: dict[str, Any], intent_spec: dict[str, Any]) -> list[str]:
    fields = ["class"]
    intent = intent_spec.get("intent") or {}
    goals = set(intent.get("goals") or [])
    coverage = validated_profile.get("field_coverage") or {}
    requested = set((intent.get("filters") or {}).keys()) | set((intent.get("exclude_filters") or {}).keys())
    if "polarization_conditioned_generation" in goals or "polarization" in requested or field_ready(coverage, "polarization"):
        fields.append("polarization")
    if any(goal in goals for goal in ("complete_sparse_azimuth", "azimuth_completion")) or field_ready(coverage, "azimuth_deg"):
        fields.append("azimuth_deg")
    for field in ("band", "resolution_m", "depression_angle_deg", "incidence_angle_deg"):
        if field in requested or field_ready(coverage, field):
            fields.append(field)
    return sorted(dict.fromkeys(fields), key=fields.index)


def field_ready(coverage: dict[str, Any], field: str, threshold: float = 0.8) -> bool:
    try:
        return float(coverage.get(field, 0.0) or 0.0) >= threshold
    except (TypeError, ValueError):
        return False


def safe_evidence_call(skill: str, fn: Any) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {
            "skill": skill,
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
        }


def summarize_planning_evidence(reports: dict[str, Any]) -> dict[str, Any]:
    preprocess = reports.get("sar_preprocess") or {}
    captions = reports.get("metadata_caption") or {}
    balance = reports.get("dataset_balance") or {}
    caption_planned = int(captions.get("planned_count") or 0)
    sample_count = int(balance.get("sample_count") or caption_planned or preprocess.get("planned_count") or 0)
    total_deficit = int(balance.get("total_recommended_additions") or 0)
    top_deficits = balance.get("top_deficits") or []
    return {
        "preprocess": {
            "status": preprocess.get("status"),
            "planned_count": preprocess.get("planned_count"),
            "mode": (preprocess.get("config") or {}).get("mode"),
            "recommended_before_generation": bool(preprocess.get("planned_count")),
        },
        "metadata_caption": {
            "status": captions.get("status"),
            "planned_count": caption_planned,
            "caption_ready_ratio": round(caption_planned / max(sample_count, 1), 6),
            "usable_for_lora": caption_planned > 0,
        },
        "dataset_balance": {
            "status": balance.get("status"),
            "sample_count": sample_count,
            "fields": balance.get("fields") or [],
            "total_recommended_additions": total_deficit,
            "top_deficits": top_deficits[:8],
            "has_actionable_deficits": total_deficit > 0,
        },
    }


def compact_planning_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    summary = evidence.get("summary") or {}
    balance = summary.get("dataset_balance") or {}
    return {
        "schema_version": evidence.get("schema_version"),
        "dataset_config": evidence.get("dataset_config"),
        "preprocess": summary.get("preprocess") or {},
        "metadata_caption": summary.get("metadata_caption") or {},
        "dataset_balance": {
            "status": balance.get("status"),
            "sample_count": balance.get("sample_count"),
            "fields": balance.get("fields") or [],
            "total_recommended_additions": balance.get("total_recommended_additions"),
            "has_actionable_deficits": balance.get("has_actionable_deficits"),
            "top_deficits": (balance.get("top_deficits") or [])[:5],
        },
    }


def render_planning_evidence_markdown(evidence: dict[str, Any]) -> str:
    summary = evidence.get("summary") or {}
    preprocess = summary.get("preprocess") or {}
    captions = summary.get("metadata_caption") or {}
    balance = summary.get("dataset_balance") or {}
    lines = [
        "# SAGA Planning Evidence",
        "",
        f"- Dataset: `{evidence.get('dataset_root')}`",
        f"- Dataset config: `{evidence.get('dataset_config')}`",
        f"- Evidence dir: `{evidence.get('evidence_dir')}`",
        "",
        "## SAR Preprocess",
        "",
        f"- Status: `{preprocess.get('status')}`",
        f"- Planned count: {preprocess.get('planned_count')}",
        f"- Mode: `{preprocess.get('mode')}`",
        "",
        "## Metadata Caption",
        "",
        f"- Status: `{captions.get('status')}`",
        f"- Planned captions: {captions.get('planned_count')}",
        f"- Usable for LoRA: {captions.get('usable_for_lora')}",
        "",
        "## Dataset Balance",
        "",
        f"- Status: `{balance.get('status')}`",
        f"- Fields: `{balance.get('fields')}`",
        f"- Total recommended additions: {balance.get('total_recommended_additions')}",
        "",
    ]
    top_deficits = balance.get("top_deficits") or []
    if top_deficits:
        lines.extend(["### Top Deficits", ""])
        for row in top_deficits[:8]:
            lines.append(
                f"- bin={row.get('bin')} current={row.get('current_count')} "
                f"target={row.get('target_count')} add={row.get('recommended_additions')}"
            )
        lines.append("")
    return "\n".join(lines)

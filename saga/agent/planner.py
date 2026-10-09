from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from saga.agent.llm_client import LLMConfig, OpenAICompatibleClient
from saga.core.config import load_mapping, save_json, save_text


def rule_based_plan(diagnosis: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, Any]:
    request = request or {}
    goals = request.get("goals") or infer_goals(diagnosis)
    selected_skills = ["DatasetDiagnosisSkill", "TargetBankBuilderSkill", "LabelValidationSkill"]
    pipeline: list[dict[str, Any]] = [
        {"step_id": "build_target_bank", "skill": "TargetBankBuilderSkill", "params": {"mode": "whole_image"}},
    ]
    coverage = diagnosis.get("metadata_coverage", {})
    if coverage.get("azimuth_deg", 0) > 0.5:
        selected_skills.append("GaussianSplattingCompletionSkill")
        selected_skills.append("GeoDiffSARSkill")
        pipeline.append(
            {
                "step_id": "complete_sparse_azimuth_candidates",
                "skill": "GaussianSplattingCompletionSkill",
                "params": {
                    "status": "adapter_required",
                    "source": "planner_recommendation",
                },
            }
        )
        pipeline.append(
            {
                "step_id": "complete_sparse_azimuth_quality",
                "skill": "GeoDiffSARSkill",
                "params": {
                    "status": "adapter_required",
                    "source": "planner_recommendation",
                },
            }
        )
    if coverage.get("band", 0) > 0.2 or coverage.get("polarization", 0) > 0.2:
        selected_skills.append("StyleTransferSkill")
        pipeline.append(
            {
                "step_id": "style_transfer",
                "skill": "StyleTransferSkill",
                "params": {
                    "adapter": "style_transfer_general",
                    "status": "available_as_skill_wrapper",
                    "command": "saga style-transfer",
                    "config": "configs/skills/style_transfer.yaml",
                },
            }
        )
    selected_skills.append("TraditionalAugSkill")
    pipeline.extend(
        [
            {"step_id": "traditional_aug", "skill": "TraditionalAugSkill", "params": {"status": "todo"}},
            {"step_id": "validate_labels", "skill": "LabelValidationSkill", "params": {"required_fields": ["class"]}},
            {"step_id": "export_dataset", "skill": "DatasetExportSkill", "params": {"output_format": "saga_pair"}},
        ]
    )
    return {
        "recipe_id": request.get("request_id", f"{diagnosis.get('dataset_id', 'dataset')}_recipe_v1"),
        "planner": "rule_based",
        "goals": goals,
        "selected_skills": dedupe(selected_skills),
        "pipeline": pipeline,
        "filters": [
            {"name": "LabelValidationSkill"},
            {"name": "DataQualityEvaluationSkill", "status": "planned"},
        ],
        "warnings": collect_warnings(diagnosis),
        "rationale": build_rationale(diagnosis),
    }


def llm_plan(
    diagnosis: dict[str, Any],
    request: dict[str, Any],
    llm_config: LLMConfig,
) -> dict[str, Any]:
    client = OpenAICompatibleClient(llm_config)
    prompt = build_planner_prompt(diagnosis=diagnosis, request=request)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are SAGA Planner, a SAR data augmentation planning agent. "
                    "Return only valid JSON. Do not invent unavailable metadata. "
                    "Prefer structured, executable recipes."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        plan = json.loads(content)
    except json.JSONDecodeError:
        plan = {"planner": "llm", "raw_response": content, "parse_error": True}
    plan.setdefault("planner", f"llm:{llm_config.provider}:{llm_config.model}")
    return plan


def run_plan(
    diagnosis_path: str | Path,
    output_path: str | Path,
    request_path: str | Path | None = None,
    llm_config_path: str | Path | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    diagnosis = load_mapping(diagnosis_path)
    request = load_mapping(request_path) if request_path else {}
    if use_llm:
        llm_raw = load_mapping(llm_config_path) if llm_config_path else {}
        plan = llm_plan(diagnosis=diagnosis, request=request, llm_config=LLMConfig.from_mapping(llm_raw))
    else:
        plan = rule_based_plan(diagnosis=diagnosis, request=request)
    output = Path(output_path)
    if output.suffix.lower() in {".yaml", ".yml"}:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(plan, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        save_json(output, plan)
    return plan


def infer_goals(diagnosis: dict[str, Any]) -> list[str]:
    goals = ["improve_classification"]
    issue_types = {issue.get("type") for issue in diagnosis.get("issues", [])}
    if "sparse_azimuth" in issue_types or diagnosis.get("metadata_coverage", {}).get("azimuth_deg", 0) > 0.5:
        goals.append("complete_sparse_azimuth")
    if diagnosis.get("metadata_coverage", {}).get("band", 0) > 0.2:
        goals.append("adapt_to_new_band_or_payload")
    return goals


def collect_warnings(diagnosis: dict[str, Any]) -> list[str]:
    warnings = []
    for issue in diagnosis.get("issues", []):
        if issue.get("severity") in {"high", "medium"}:
            warnings.append(f"{issue.get('type')}: {issue.get('message')} Count: {issue.get('count')}")
    return warnings


def build_rationale(diagnosis: dict[str, Any]) -> str:
    coverage = diagnosis.get("metadata_coverage", {})
    parts = [
        f"Dataset contains {diagnosis.get('summary', {}).get('total_images', 0)} images.",
        f"Azimuth coverage is {coverage.get('azimuth_deg', 0):.2%}.",
        f"Band coverage is {coverage.get('band', 0):.2%}.",
        f"Polarization coverage is {coverage.get('polarization', 0):.2%}.",
    ]
    return " ".join(parts)


def build_planner_prompt(diagnosis: dict[str, Any], request: dict[str, Any]) -> str:
    available_skills = [
        "DatasetDiagnosisSkill",
        "TargetBankBuilderSkill",
        "TraditionalAugSkill",
        "GeoDiffSARSkill",
        "GaussianSplattingCompletionSkill",
        "StyleTransferSkill",
        "LabelValidationSkill",
        "DatasetExportSkill",
    ]
    compact_diagnosis = {
        "dataset_id": diagnosis.get("dataset_id"),
        "summary": diagnosis.get("summary"),
        "metadata_coverage": diagnosis.get("metadata_coverage"),
        "azimuth_analysis": diagnosis.get("azimuth_analysis"),
        "issues": diagnosis.get("issues"),
        "recommended_skill_candidates": diagnosis.get("recommended_skill_candidates"),
    }
    return json.dumps(
        {
            "request": request,
            "diagnosis": compact_diagnosis,
            "available_skills": available_skills,
            "required_output_schema": {
                "recipe_id": "string",
                "planner": "string",
                "goals": ["string"],
                "selected_skills": ["string"],
                "pipeline": [{"step_id": "string", "skill": "string", "params": {}}],
                "filters": [{"name": "string"}],
                "warnings": ["string"],
                "rationale": "string",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result

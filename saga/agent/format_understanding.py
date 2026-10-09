from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.agent.json_utils import loads_json_object
from saga.agent.llm_client import LLMConfig, OpenAICompatibleClient
from saga.core.config import load_mapping, save_json


def understand_format_with_llm(
    inspection_path: str | Path,
    output_path: str | Path,
    llm_config: LLMConfig,
    user_description: str = "",
    prompt_group_limit: int = 30,
    prompt_sample_limit: int = 40,
    txt_preview_limit: int = 8,
    use_response_format: bool = True,
) -> dict[str, Any]:
    inspection = load_mapping(inspection_path)
    prompt = build_format_understanding_prompt(
        inspection=inspection,
        user_description=user_description,
        group_limit=prompt_group_limit,
        sample_limit=prompt_sample_limit,
        txt_preview_limit=txt_preview_limit,
    )
    client = OpenAICompatibleClient(llm_config)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You analyze arbitrary SAR dataset layouts for SAGA ingestion. "
                    "Return compact JSON only. Do not write regex. "
                    "Use path templates with normalized field placeholders."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"} if use_response_format else None,
    )
    if not content.strip():
        raise RuntimeError("LLM returned an empty format understanding response.")
    data = normalize_understanding(loads_json_object(content))
    save_json(output_path, data)
    return data


def build_format_understanding_prompt(
    inspection: dict[str, Any],
    user_description: str = "",
    group_limit: int = 30,
    sample_limit: int = 40,
    txt_preview_limit: int = 8,
) -> str:
    compact = {
        "total_images": inspection.get("total_images"),
        "suffix_counts": inspection.get("suffix_counts"),
        "depth_counts": inspection.get("depth_counts"),
        "top_parent_dirs": inspection.get("top_parent_dirs"),
        "path_groups": compact_path_groups(inspection.get("path_groups", []), group_limit),
        "sample_relative_paths": [
            item.get("relative_path") for item in inspection.get("samples", [])[:sample_limit]
        ],
        "txt_previews": [
            {
                "relative_path": item.get("relative_path"),
                "txt_preview": item.get("txt_preview"),
            }
            for item in inspection.get("samples", [])[:txt_preview_limit]
            if item.get("txt_preview")
        ],
    }
    return (
        "SAGA needs to ingest user SAR datasets without forcing a fixed input layout. "
        "Your job is semantic understanding only. Do not emit Python regex. "
        "Summarize the visible layout as path templates with placeholders.\n\n"
        "Normalized placeholders may include:\n"
        "{class}, {azimuth_deg}, {incidence_angle_deg}, {elevation_angle_deg}, "
        "{polarization}, {band}, {resolution_m}, {target_name}, {target_attr}, "
        "{sample_index}, {frame_index}, {sea_state}, {processing}, {ext}.\n\n"
        "Rules:\n"
        "- Use templates that match visible path structure, for example "
        "'{class}/az-{azimuth_deg}-{polarization}.{ext}'.\n"
        "- Put fixed known values in constants only when stated by the user or obvious from a path component.\n"
        "- Do not invent azimuth, band, sensor, or resolution if not visible or stated.\n"
        "- If a filename index might encode azimuth but the mapping is unknown, mark it unresolved.\n"
        "- Prefer several simple templates over one vague template.\n\n"
        f"User description:\n{user_description or '(none)'}\n\n"
        f"Inspection JSON:\n{compact}\n\n"
        "Return JSON with exactly this shape:\n"
        "{\n"
        '  "format_understanding": {\n'
        '    "name": "user_dataset_understanding",\n'
        '    "task": "classification",\n'
        '    "summary": "short summary",\n'
        '    "rules": [\n'
        "      {\n"
        '        "name": "short_rule_name",\n'
        '        "template": "path/template/with/{field}_placeholders.{ext}",\n'
        '        "examples": ["relative/path.png"],\n'
        '        "fields": ["class", "azimuth_deg"],\n'
        '        "constants": {},\n'
        '        "confidence": 0.8,\n'
        '        "notes": "short note"\n'
        "      }\n"
        "    ],\n"
        '    "fallback": {"class_from_parent": true},\n'
        '    "unresolved": [\n'
        '      {"field": "azimuth_deg", "examples": ["path"], "reason": "not visible"}\n'
        "    ]\n"
        "  }\n"
        "}\n"
    )


def compact_path_groups(groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    compacted = []
    for group in groups[:limit]:
        compacted.append(
            {
                "signature": group.get("signature"),
                "count": group.get("count"),
                "examples": group.get("examples", [])[:4],
                "parent_examples": group.get("parent_examples", [])[:4],
                "suffix_counts": group.get("suffix_counts"),
                "depth_counts": group.get("depth_counts"),
            }
        )
    return compacted


def normalize_understanding(data: dict[str, Any]) -> dict[str, Any]:
    if "format_understanding" not in data:
        data = {"format_understanding": data}
    understanding = data["format_understanding"]
    if not isinstance(understanding, dict):
        raise ValueError("format_understanding must be a JSON object.")
    understanding.setdefault("name", "user_dataset_understanding")
    understanding.setdefault("task", "classification")
    understanding.setdefault("rules", [])
    understanding.setdefault("fallback", {"class_from_parent": True})
    understanding.setdefault("unresolved", [])
    return {"format_understanding": understanding}

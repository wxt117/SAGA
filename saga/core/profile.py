from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RAW_PROFILE_VERSION = "saga_raw_dataset_profile_v1"
LLM_CONTEXT_VERSION = "saga_llm_context_profile_v1"
INTENT_SPEC_VERSION = "saga_intent_spec_v1"
FORMAT_HINT_VERSION = "saga_format_hint_v1"
DATASET_PROFILE_VERSION = "saga_dataset_profile_v1"


NORMALIZED_METADATA_FIELDS = [
    "class",
    "target_name",
    "target_attr",
    "sample_index",
    "frame_index",
    "azimuth_deg",
    "incidence_angle_deg",
    "depression_angle_deg",
    "elevation_angle_deg",
    "resolution_m",
    "band",
    "polarization",
    "sensor",
    "platform",
    "sea_state",
    "processing",
]


FIELD_ALIASES = {
    "类别": "class",
    "类名": "class",
    "class": "class",
    "目标": "target_name",
    "目标名": "target_name",
    "目标类型": "target_name",
    "target": "target_name",
    "方位角": "azimuth_deg",
    "方位": "azimuth_deg",
    "azimuth": "azimuth_deg",
    "azimuth_angle": "azimuth_deg",
    "azimuth_deg": "azimuth_deg",
    "入射角": "incidence_angle_deg",
    "incidence": "incidence_angle_deg",
    "incidence_angle": "incidence_angle_deg",
    "incidence_angle_deg": "incidence_angle_deg",
    "下视角": "depression_angle_deg",
    "俯视角": "depression_angle_deg",
    "depression": "depression_angle_deg",
    "depression_angle": "depression_angle_deg",
    "depression_angle_deg": "depression_angle_deg",
    "仰角": "elevation_angle_deg",
    "elevation": "elevation_angle_deg",
    "elevation_angle": "elevation_angle_deg",
    "elevation_angle_deg": "elevation_angle_deg",
    "分辨率": "resolution_m",
    "距离分辨率": "resolution_m",
    "resolution": "resolution_m",
    "resolution_m": "resolution_m",
    "波段": "band",
    "频段": "band",
    "band": "band",
    "极化": "polarization",
    "极化方式": "polarization",
    "polarization": "polarization",
    "pol": "polarization",
    "传感器": "sensor",
    "载荷": "sensor",
    "sensor": "sensor",
    "平台": "platform",
    "platform": "platform",
    "海况": "sea_state",
    "sea_state": "sea_state",
    "处理方式": "processing",
    "成像处理": "processing",
    "processing": "processing",
    "编号": "sample_index",
    "序号": "sample_index",
    "index": "sample_index",
    "sample_index": "sample_index",
}


@dataclass
class ClarificationQuestion:
    id: str
    question: str
    severity: str = "info"
    reason: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_field_name(value: str) -> str:
    key = value.strip().strip("：:，,。；; ").lower()
    return FIELD_ALIASES.get(key, key)


def confidence_label(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"

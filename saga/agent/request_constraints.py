from __future__ import annotations

from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text


REQUEST_CONSTRAINTS_VERSION = "saga_request_constraints_v1"


def infer_request_constraints(
    intent_spec: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    intent = intent_spec.get("intent") or {}
    text = str(intent_spec.get("raw_text") or "").lower()
    raw_text = str(intent_spec.get("raw_text") or "")
    constraints = {
        "schema_version": REQUEST_CONSTRAINTS_VERSION,
        "speed_priority": infer_speed_priority(text=raw_text, lowered=text),
        "quality_priority": infer_quality_priority(text=raw_text, lowered=text),
        "gpu_budget": infer_gpu_budget(text=raw_text, lowered=text),
        "time_budget": infer_time_budget(text=raw_text, lowered=text),
        "needs_downstream_evidence": needs_downstream_evidence(text=raw_text, lowered=text, intent=intent),
        "benefit_probe_mode": infer_benefit_probe_mode(text=raw_text, lowered=text, intent=intent),
        "lightweight_probe_metrics": infer_lightweight_probe_metrics(text=raw_text, lowered=text),
        "needs_repair_loop": True,
        "prefer_executable_skills": True,
        "target_count": intent.get("target_count"),
        "training_epochs": intent.get("training_epochs"),
        "model_family": intent.get("model_family"),
        "explicit_goals": intent.get("goals") or [],
        "notes": [
            "Constraints are inferred from user language and explicit intent fields.",
            "They are policy hints for planning; deterministic validators still decide executable safety.",
        ],
    }
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        save_json(path / "request_constraints.json", constraints)
        save_text(path / "request_constraints.md", render_request_constraints_markdown(constraints))
    return constraints


def infer_speed_priority(text: str, lowered: str) -> str:
    if any(keyword in text for keyword in ["最快", "速度最快", "快速", "轻量", "低显存", "马上", "很快"]) or any(
        keyword in lowered for keyword in ["fast", "quick", "lightweight", "low memory"]
    ):
        return "high"
    if any(keyword in text for keyword in ["不计时间", "时间无所谓", "没有时间要求"]) or any(
        keyword in lowered for keyword in ["no time limit", "quality over speed"]
    ):
        return "low"
    return "balanced"


def infer_quality_priority(text: str, lowered: str) -> str:
    if any(keyword in text for keyword in ["高质量", "效果好", "质量最好", "不计时间", "不计显存"]) or any(
        keyword in lowered for keyword in ["high quality", "best quality"]
    ):
        return "high"
    if any(keyword in text for keyword in ["简单", "基础", "快速"]) or "fast" in lowered:
        return "balanced"
    return "balanced"


def infer_gpu_budget(text: str, lowered: str) -> str:
    if any(keyword in text for keyword in ["低显存", "显存少", "低内存", "轻量"]) or "low memory" in lowered:
        return "limited"
    if any(keyword in text for keyword in ["显存多", "cuda", "gpu充足", "不计显存"]) or "gpu" in lowered:
        return "available"
    return "unknown"


def infer_time_budget(text: str, lowered: str) -> str:
    if any(keyword in text for keyword in ["不计时间", "没有时间要求", "多花点时间"]) or "no time limit" in lowered:
        return "relaxed"
    if any(keyword in text for keyword in ["最快", "快速", "马上", "很快"]) or any(keyword in lowered for keyword in ["fast", "quick"]):
        return "tight"
    return "unknown"


def needs_downstream_evidence(text: str, lowered: str, intent: dict[str, Any]) -> bool:
    if excludes_downstream_evidence(text=text, lowered=lowered):
        return False
    if intent.get("task") == "classification_evaluation":
        return True
    if any(
        keyword in text
        for keyword in [
            "下游",
            "准确率",
            "识别率",
            "分类评估",
            "分类收益",
            "分类效果",
            "训练分类",
            "分类模型",
            "下游分类",
        ]
    ):
        return True
    if any(
        keyword in lowered
        for keyword in [
            "accuracy",
            "classification evaluation",
            "classification benchmark",
            "downstream",
            "benchmark",
            "train classifier",
            "classifier accuracy",
        ]
    ):
        return True
    if "收益" in text and any(keyword in text for keyword in ["分类", "识别", "下游", "准确率", "模型"]):
        return True
    return False


def excludes_downstream_evidence(text: str, lowered: str) -> bool:
    return any(
        keyword in text
        for keyword in [
            "不训练分类",
            "不训练分类器",
            "不训练分类模型",
            "不需要训练分类",
            "不需要训练分类器",
            "不需要训练分类模型",
            "不要训练分类",
            "不要训练分类器",
            "不要训练分类模型",
            "不做分类评估",
            "不需要分类评估",
            "不要分类评估",
            "不做下游",
            "不需要下游",
            "不要下游",
            "不评估下游",
            "不看准确率",
        ]
    ) or any(
        keyword in lowered
        for keyword in [
            "do not train classifier",
            "don't train classifier",
            "no classifier training",
            "no classification evaluation",
            "without downstream",
            "no downstream",
        ]
    )


def infer_benefit_probe_mode(text: str, lowered: str, intent: dict[str, Any]) -> str:
    if needs_downstream_evidence(text=text, lowered=lowered, intent=intent):
        return "downstream_classification"
    return "lightweight_metrics"


def infer_lightweight_probe_metrics(text: str, lowered: str) -> list[str]:
    metrics = ["quality", "sar_artifacts", "duplicate_near_duplicate"]
    if any(keyword in text for keyword in ["分布", "FID", "指标", "效果", "对比"]) or any(
        keyword in lowered for keyword in ["fid", "mmd", "distribution", "metric"]
    ):
        metrics.append("distribution")
    elif "distribution" not in metrics:
        metrics.append("distribution")
    if any(keyword in text for keyword in ["泄漏", "重复", "训练集", "验证集"]) or any(keyword in lowered for keyword in ["leakage", "duplicate"]):
        metrics.append("leakage")
    return sorted(set(metrics))


def render_request_constraints_markdown(constraints: dict[str, Any]) -> str:
    lines = [
        "# SAGA Request Constraints",
        "",
        f"- Speed priority: `{constraints.get('speed_priority')}`",
        f"- Quality priority: `{constraints.get('quality_priority')}`",
        f"- GPU budget: `{constraints.get('gpu_budget')}`",
        f"- Time budget: `{constraints.get('time_budget')}`",
        f"- Needs downstream evidence: {constraints.get('needs_downstream_evidence')}",
        f"- Benefit probe mode: `{constraints.get('benefit_probe_mode')}`",
        f"- Lightweight probe metrics: `{constraints.get('lightweight_probe_metrics')}`",
        f"- Target count: {constraints.get('target_count')}",
        f"- Training epochs: {constraints.get('training_epochs')}",
        f"- Model family: `{constraints.get('model_family')}`",
        f"- Explicit goals: `{constraints.get('explicit_goals')}`",
        "",
    ]
    return "\n".join(lines)

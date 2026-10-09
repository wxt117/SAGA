from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.data.discovery import is_image
from saga.skills.diffusion_lora.skill import run_diffusion_lora_generation_skill
from saga.skills.style_transfer.skill import run_style_transfer_skill


NUMERIC_SUFFIX_RE = re.compile(
    r"_(?P<anchor>-?\d+(?:\.\d+)?)_"
    r"(?P<view>-?\d+(?:\.\d+)?)_"
    r"(?P<azimuth>-?\d+(?:\.\d+)?)_"
    r"(?P<resolution>-?\d+(?:\.\d+)?)_"
    r"(?P<band>[A-Za-z0-9]+)\.[^.]+$"
)


def run_agent_command(
    text: str,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    request = parse_agent_command(text)
    if request["skill"] == "DiffusionLoRAGenerationSkill":
        return run_diffusion_lora_agent_command(
            text=text,
            request=request,
            output_dir=output_dir,
            config_path=config_path,
            dry_run=dry_run,
        )
    if request["skill"] != "StyleTransferSkill":
        raise ValueError(f"Unsupported skill intent: {request['skill']}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    style_path = Path(request["style_path"]).expanduser().resolve()
    content_root = Path(request["content_root"]).expanduser().resolve()
    selected_dir = output_path / "content_selection"
    selected = select_content_files(
        content_root=content_root,
        selected_dir=selected_dir,
        view_value=request.get("view_value"),
        anchor_value=request.get("anchor_value"),
    )
    skill_output_dir = output_path / "style_transfer_output"
    skill_report = run_style_transfer_skill(
        content=selected_dir,
        style=style_path,
        output_dir=skill_output_dir,
        config_path=config_path,
        dry_run=dry_run,
    )

    report = {
        "status": skill_report.get("status"),
        "dry_run": dry_run,
        "input_text": text,
        "interpretation": request,
        "selection": {
            "content_root": content_root.as_posix(),
            "selected_dir": selected_dir.as_posix(),
            "selected_count": len(selected),
            "examples": [path.as_posix() for path in selected[:20]],
        },
        "skill_report_path": (skill_output_dir / "style_transfer_run.json").as_posix(),
        "skill_report": skill_report,
    }
    save_json(output_path / "agent_command.json", report)
    save_text(output_path / "agent_command.md", render_agent_command_markdown(report))
    return report


def parse_agent_command(text: str) -> dict[str, Any]:
    if is_lora_command(text):
        return parse_lora_command(text)
    style_path = parse_style_path(text)
    content_root = parse_content_root(text)
    view_value = parse_view_value(text)
    anchor_value = parse_anchor_value(text)
    return {
        "skill": "StyleTransferSkill",
        "style_path": style_path,
        "content_root": content_root,
        "style_role": "guidance_image_pool",
        "content_role": "images_to_transform",
        "view_field": "incidence_angle_deg",
        "view_value": view_value,
        "anchor_field": "param_1",
        "anchor_value": anchor_value,
        "naming_schema": {
            "suffix_order_after_anchor": [
                "view_or_incidence_angle_deg",
                "azimuth_deg",
                "resolution_m",
                "band",
            ]
        },
        "notes": [
            "No style-content azimuth matching is applied.",
            "When style is a directory, the wrapped style-transfer script chooses style images by its own seeded sampler.",
        ],
    }


def run_diffusion_lora_agent_command(
    text: str,
    request: dict[str, Any],
    output_dir: str | Path,
    config_path: str | Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    skill_output_dir = output_path / "diffusion_lora_output"
    skill_report = run_diffusion_lora_generation_skill(
        dataset_root=request["dataset_root"],
        output_dir=skill_output_dir,
        config_path=config_path,
        model_family=request.get("model_family"),
        target_count=request.get("target_count"),
        train=True,
        infer=True,
        dry_run=dry_run,
    )
    report = {
        "status": skill_report.get("status"),
        "dry_run": dry_run,
        "input_text": text,
        "interpretation": request,
        "skill_report_path": (skill_output_dir / "diffusion_lora_run.json").as_posix(),
        "skill_report": skill_report,
    }
    save_json(output_path / "agent_command.json", report)
    save_text(output_path / "agent_command.md", render_agent_command_markdown(report))
    return report


def is_lora_command(text: str) -> bool:
    lowered = text.lower()
    return "lora" in lowered or any(keyword in text for keyword in ["训练一个Flux", "训练一个SD3", "训练一个SDXL"])


def parse_lora_command(text: str) -> dict[str, Any]:
    dataset_root = parse_lora_dataset_root(text)
    model_family = parse_lora_model_family(text)
    target_count = parse_lora_target_count(text)
    return {
        "skill": "DiffusionLoRAGenerationSkill",
        "dataset_root": dataset_root,
        "model_family": model_family,
        "target_count": target_count,
        "notes": [
            "This routes to text-caption LoRA training and inference.",
            "ControlNet and GeoDiff-SAR are not invoked by this command.",
        ],
    }


def parse_lora_dataset_root(text: str) -> str:
    patterns = [
        r"用(?P<path>\S+)(?:训练|来训练|作为训练数据|作为数据集)",
        r"使用(?P<path>\S+)(?:训练|来训练|作为训练数据|作为数据集)",
        r"数据集(?:为|是|:|：)?\s*(?P<path>\S+)",
        r"输入数据集(?:为|是|:|：)?\s*(?P<path>\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_path_text(match.group("path"))
    raise ValueError("Could not parse LoRA dataset path. Expected phrase like: 用<path>训练一个Flux LoRA")


def parse_lora_model_family(text: str) -> str | None:
    lowered = text.lower()
    if "flux" in lowered:
        return "flux"
    if "sd3.5" in lowered or "sd35" in lowered or "sd3" in lowered:
        return "sd3"
    if "sdxl" in lowered:
        return "sdxl"
    return None


def parse_lora_target_count(text: str) -> int | None:
    match = re.search(r"(?:生成|出|增广|扩增)\s*(?P<count>\d+)\s*(?:张|个|幅)", text)
    return int(match.group("count")) if match else None


def parse_style_path(text: str) -> str:
    explicit = re.search(r"[，,]\s*(?P<path>.+?)作为风格图", text)
    if explicit:
        return clean_path_text(explicit.group("path"))
    match = re.search(r"把(?P<path>.+?)作为指导图", text)
    if not match:
        raise ValueError("Could not parse style/guidance path. Expected phrase like: 把<path>作为指导图 or <path>作为风格图")
    return clean_path_text(match.group("path"))


def parse_content_root(text: str) -> str:
    explicit = re.search(r"把(?P<path>.+?)的数据作为内容图", text)
    if explicit:
        return clean_path_text(explicit.group("path"))
    match = re.search(r"迁移到(?P<path>.+?)(?:的数据上|数据上|上)", text)
    if not match:
        raise ValueError("Could not parse target/content path. Expected phrase like: 迁移到<path>的数据上 or 把<path>的数据作为内容图")
    return clean_path_text(match.group("path"))


def parse_view_value(text: str) -> float | None:
    match = re.search(r"下视角为(?P<value>-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return parse_number(match.group("value"))


def parse_anchor_value(text: str) -> float | None:
    # The user phrase may contain a typo such as "850湖面的数字"; we only need
    # the anchor number before the described suffix fields.
    match = re.search(r"命名中(?P<value>-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return parse_number(match.group("value"))


def clean_path_text(value: str) -> str:
    return value.strip().strip("，,。 ")


def select_content_files(
    content_root: Path,
    selected_dir: Path,
    view_value: float | None,
    anchor_value: float | None,
) -> list[Path]:
    if not content_root.exists():
        raise FileNotFoundError(f"Content root does not exist: {content_root}")
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    select_all_images = view_value is None and anchor_value is None
    for path in sorted(content_root.rglob("*")):
        if not is_image(path):
            continue
        fields = parse_numeric_suffix(path.name)
        if select_all_images and not fields:
            staged = selected_dir / path.name
            staged.symlink_to(path.resolve())
            selected.append(path.resolve())
            continue
        if not fields:
            continue
        if anchor_value is not None and not same_number(fields["anchor"], anchor_value):
            continue
        if view_value is not None and not same_number(fields["view"], view_value):
            continue
        staged = selected_dir / path.name
        staged.symlink_to(path.resolve())
        selected.append(path.resolve())

    if not selected:
        raise ValueError(
            f"No content images matched view={view_value} anchor={anchor_value} under {content_root}"
        )
    return selected


def parse_numeric_suffix(filename: str) -> dict[str, Any] | None:
    match = NUMERIC_SUFFIX_RE.search(filename)
    if not match:
        return None
    return {
        "anchor": parse_number(match.group("anchor")),
        "view": parse_number(match.group("view")),
        "azimuth": parse_number(match.group("azimuth")),
        "resolution": parse_number(match.group("resolution")),
        "band": match.group("band"),
    }


def parse_number(value: str) -> float:
    number = float(value)
    return int(number) if number.is_integer() else number


def same_number(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 1e-6


def render_agent_command_markdown(report: dict[str, Any]) -> str:
    interpretation = report["interpretation"]
    if interpretation["skill"] == "DiffusionLoRAGenerationSkill":
        artifacts = report.get("skill_report", {}).get("artifacts", {})
        return "\n".join(
            [
                "# SAGA Agent Command",
                "",
                f"- Status: {report.get('status')}",
                f"- Dry run: {report.get('dry_run')}",
                f"- Skill: {interpretation['skill']}",
                f"- Dataset root: `{interpretation.get('dataset_root')}`",
                f"- Model family: `{interpretation.get('model_family')}`",
                f"- Target count: `{interpretation.get('target_count')}`",
                f"- Train command: `{artifacts.get('train_command')}`",
                f"- Inference commands: `{artifacts.get('inference_commands')}`",
                "",
            ]
        )
    selection = report["selection"]
    command = report.get("skill_report", {}).get("command") or []
    lines = [
        "# SAGA Agent Command",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Skill: {interpretation['skill']}",
        f"- Style path: `{interpretation['style_path']}`",
        f"- Content root: `{interpretation['content_root']}`",
        f"- View filter: `{interpretation.get('view_field')} = {interpretation.get('view_value')}`",
        f"- Anchor filter: `{interpretation.get('anchor_field')} = {interpretation.get('anchor_value')}`",
        f"- Selected content images: {selection['selected_count']}",
        "",
        "## Examples",
        "",
    ]
    for item in selection.get("examples", [])[:10]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Skill Command", "", "```bash", " ".join(command), "```", ""])
    return "\n".join(lines)

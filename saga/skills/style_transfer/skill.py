from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_mapping, save_json, save_text
from saga.models.adapters.style_transfer import StyleTransferAdapter, StyleTransferAdapterConfig


@dataclass
class StyleTransferSkillConfig:
    name: str = "StyleTransferSkill"
    description: str = "Cross-payload or cross-domain SAR feature/style transfer using myproject/stytransfer."
    adapter: StyleTransferAdapterConfig = field(default_factory=StyleTransferAdapterConfig)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "StyleTransferSkillConfig":
        raw = raw or {}
        body = raw.get("style_transfer", raw)
        adapter_raw = body.get("adapter", body) if isinstance(body, dict) else {}
        return cls(
            name=str(body.get("name", "StyleTransferSkill")) if isinstance(body, dict) else "StyleTransferSkill",
            description=str(
                body.get(
                    "description",
                    "Cross-payload or cross-domain SAR feature/style transfer using myproject/stytransfer.",
                )
            )
            if isinstance(body, dict)
            else "Cross-payload or cross-domain SAR feature/style transfer using myproject/stytransfer.",
            adapter=StyleTransferAdapterConfig.from_mapping(adapter_raw),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "StyleTransferSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


class StyleTransferSkill:
    def __init__(self, config: StyleTransferSkillConfig | None = None) -> None:
        self.config = config or StyleTransferSkillConfig()
        self.adapter = StyleTransferAdapter(self.config.adapter)

    def run(
        self,
        content: str | Path,
        style: str | Path,
        output_dir: str | Path,
        skill_overrides: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        override_report = apply_skill_overrides(self.config, skill_overrides or {})
        self.adapter = StyleTransferAdapter(self.config.adapter)
        content_path = Path(content).expanduser().resolve()
        style_path = Path(style).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()

        validate_input_path(content_path, "content")
        validate_input_path(style_path, "style")
        output_path.mkdir(parents=True, exist_ok=True)

        result = self.adapter.run(
            content=content_path,
            style=style_path,
            output_dir=output_path,
            dry_run=dry_run,
        )
        report = {
            "skill": self.config.name,
            "status": result.get("status"),
            "dry_run": dry_run,
            "inputs": {
                "content": content_path.as_posix(),
                "style": style_path.as_posix(),
            },
            "output_dir": output_path.as_posix(),
            "adapter_config": asdict(self.config.adapter),
            "override_report": override_report,
            "command": result.get("command", []),
            "returncode": result.get("returncode"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
        write_run_artifacts(output_path, report)
        return report


def run_style_transfer_skill(
    content: str | Path,
    style: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    skill_overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = StyleTransferSkillConfig.from_path(config_path)
    return StyleTransferSkill(config).run(
        content=content,
        style=style,
        output_dir=output_dir,
        skill_overrides=skill_overrides,
        dry_run=dry_run,
    )


def apply_skill_overrides(config: StyleTransferSkillConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    if not overrides:
        return {"applied": [], "ignored": []}
    allowed = {"steps", "lr", "content_weight", "image_size", "limit", "overwrite", "self_layers", "device"}
    applied = []
    ignored = []
    for key, value in overrides.items():
        if key not in allowed:
            ignored.append({"field": key, "reason": "not_whitelisted"})
            continue
        old_value = getattr(config.adapter, key)
        if key in {"steps", "image_size", "limit"}:
            new_value = int(value)
        elif key in {"lr", "content_weight"}:
            new_value = float(value)
        elif key == "overwrite":
            new_value = bool(value)
        else:
            new_value = str(value)
        setattr(config.adapter, key, new_value)
        applied.append({"field": key, "old_value": old_value, "new_value": new_value})
    return {"applied": applied, "ignored": ignored}


def validate_input_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"{label} path must be a file or directory: {path}")


def write_run_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    save_json(output_dir / "style_transfer_run.json", report)
    command = report.get("command") or []
    if command:
        save_text(output_dir / "style_transfer_command.sh", shell_join(command) + "\n")
    save_text(output_dir / "style_transfer_run.md", render_markdown(report))


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def render_markdown(report: dict[str, Any]) -> str:
    command = report.get("command") or []
    lines = [
        f"# {report['skill']} Run",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Content: `{report['inputs']['content']}`",
        f"- Style: `{report['inputs']['style']}`",
        f"- Output dir: `{report['output_dir']}`",
        "",
        "## Command",
        "",
        "```bash",
        shell_join(command) if command else "",
        "```",
        "",
    ]
    if report.get("returncode") is not None:
        lines.extend(
            [
                "## Result",
                "",
                f"- Return code: {report.get('returncode')}",
                f"- Stdout chars: {len(report.get('stdout') or '')}",
                f"- Stderr chars: {len(report.get('stderr') or '')}",
                "",
            ]
        )
    return "\n".join(lines)

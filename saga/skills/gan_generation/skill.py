from __future__ import annotations

import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text


GAN_GENERATION_VERSION = "saga_gan_generation_run_v1"


def run_gan_generation_skill(
    dataset_root: str | Path,
    output_dir: str | Path,
    project_dir: str | Path | None = None,
    variant: str = "gpu",
    target_count: int = 100,
    epochs: int | None = None,
    model_path: str | Path | None = None,
    train: bool = True,
    infer: bool = True,
    dry_run: bool = True,
) -> dict[str, Any]:
    started = time.time()
    dataset = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    project = resolve_project_dir(project_dir, variant)
    xml_path = output / "gan_parameters.xml"
    generated_dir = output / "generated_images"
    model_dir = output / "models"
    image_dir = output / "training_samples"
    commands = build_commands(project=project, xml_path=xml_path, train=train, infer=infer)
    write_gan_xml(
        template_path=project / "parameters.xml",
        output_path=xml_path,
        dataset=dataset,
        output=output,
        generated_dir=generated_dir,
        model_dir=model_dir,
        image_dir=image_dir,
        target_count=target_count,
        epochs=epochs,
        model_path=Path(model_path).expanduser().resolve() if model_path else None,
    )
    report = {
        "schema_version": GAN_GENERATION_VERSION,
        "skill": "GANImageToImageSkill",
        "status": "dry_run",
        "message": "GAN generation commands prepared.",
        "dry_run": dry_run,
        "dataset_root": dataset.as_posix(),
        "output_dir": output.as_posix(),
        "project_dir": project.as_posix(),
        "variant": variant,
        "target_count": int(target_count),
        "epochs": epochs,
        "train": bool(train),
        "infer": bool(infer),
        "generated_output_dir": generated_dir.as_posix(),
        "commands": commands,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "report_json": (output / "gan_generation_report.json").as_posix(),
            "report_md": (output / "gan_generation_report.md").as_posix(),
            "parameters_xml": xml_path.as_posix(),
        },
        "notes": [
            "This wrapper keeps myproject/dcgan and myproject/dcgan-cpu unchanged.",
            "The DCGAN scripts are suitable for simple target distributions and fast baselines, not high-fidelity controlled SAR synthesis.",
        ],
    }
    if not dry_run:
        results = []
        status = "succeeded"
        for command in commands:
            proc = subprocess.run(command, cwd=project.as_posix(), text=True, capture_output=True)
            results.append(
                {
                    "command": command,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-4000:],
                    "stderr_tail": proc.stderr[-4000:],
                }
            )
            if proc.returncode != 0:
                status = "failed"
                break
        report["status"] = status
        report["message"] = "GAN generation completed." if status == "succeeded" else "GAN generation failed."
        report["command_results"] = results
    save_json(output / "gan_generation_report.json", report)
    save_text(output / "gan_generation_report.md", render_report(report))
    return report


def resolve_project_dir(project_dir: str | Path | None, variant: str) -> Path:
    if project_dir:
        return Path(project_dir).expanduser().resolve()
    name = "dcgan-cpu" if variant.lower() == "cpu" else "dcgan"
    return (Path("myproject") / name).resolve()


def build_commands(project: Path, xml_path: Path, train: bool, infer: bool) -> list[list[str]]:
    commands = []
    relative_xml = xml_path.as_posix()
    if train:
        commands.append(["python", "dcgan-bigship.py", relative_xml])
    if infer:
        commands.append(["python", "dcgan-bigship-test.py", relative_xml])
    return commands


def write_gan_xml(
    *,
    template_path: Path,
    output_path: Path,
    dataset: Path,
    output: Path,
    generated_dir: Path,
    model_dir: Path,
    image_dir: Path,
    target_count: int,
    epochs: int | None,
    model_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(template_path)
    root = tree.getroot()
    training = root.find("training")
    testing = root.find("testing")
    evaluation = root.find("evaluation")
    if training is not None:
        set_text(training, "data_dir", dataset.as_posix())
        set_text(training, "output_dir", output.as_posix())
        set_text(training, "model_save_dir", model_dir.as_posix())
        set_text(training, "image_save_dir", image_dir.as_posix())
        if epochs is not None:
            set_text(training, "n_epochs", str(int(epochs)))
    if testing is not None:
        if model_path is not None:
            set_text(testing, "model_path", model_path.as_posix())
        else:
            set_text(testing, "model_path", (model_dir / "generator_latest.pth").as_posix())
        set_text(testing, "output_path", generated_dir.as_posix())
        set_text(testing, "num_images", str(int(target_count)))
    if evaluation is not None:
        set_text(evaluation, "reference_path", dataset.as_posix())
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def set_text(parent: ET.Element, name: str, value: str) -> None:
    node = parent.find(name)
    if node is None:
        node = ET.SubElement(parent, name)
    node.text = value


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SAGA GAN Generation Skill",
            "",
            f"- Status: `{report.get('status')}`",
            f"- Variant: `{report.get('variant')}`",
            f"- Project: `{report.get('project_dir')}`",
            f"- Dataset: `{report.get('dataset_root')}`",
            f"- Target count: {report.get('target_count')}",
            f"- Commands: `{report.get('commands')}`",
            "",
        ]
    )

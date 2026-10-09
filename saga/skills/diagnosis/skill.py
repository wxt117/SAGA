from __future__ import annotations

from pathlib import Path

from saga.core.config import save_json, save_text
from saga.core.jsonl import write_jsonl
from saga.core.protocol import DatasetConfig
from saga.data.loader import load_samples
from saga.skills.diagnosis.report import build_machine_report, render_markdown


def run_diagnosis(config: DatasetConfig, output_dir: str | Path, read_image_size: bool = True) -> dict:
    output_path = Path(output_dir)
    samples = load_samples(config, read_image_size=read_image_size)
    report = build_machine_report(dataset_name=config.name, samples=samples)
    save_json(output_path / "machine_report.json", report)
    write_jsonl(output_path / "samples.jsonl", (sample.to_dict() for sample in samples))
    save_text(output_path / "report.md", render_markdown(report))
    return report

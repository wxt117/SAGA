from __future__ import annotations

import csv
import hashlib
import random
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from saga.core.config import load_dataset_config, load_mapping, save_json, save_text
from saga.core.evaluator_report import build_evaluator_report
from saga.core.jsonl import read_jsonl, write_jsonl
from saga.data.discovery import iter_images, natural_key
from saga.data.loader import load_samples


CLASSIFICATION_EVALUATION_VERSION = "saga_classification_evaluation_v2"
PROJECT_DATASET_SCHEMA = "saga_timm_multilabel_project_dataset_v1"
STANDARD_EVALUATOR_PROTOCOL = "saga_standard_downstream_classification_evaluator_v1"
PROJECT_COLUMNS = ["image_path", "gender", "article", "color"]


@dataclass
class ClassificationEvaluationSkillConfig:
    name: str = "ClassificationEvaluationSkill"
    description: str = "SAGA standard downstream classification evaluator based on the provided multi-label timm project."
    project_root: str = "myproject/PyTorch-Image-Models-Multi-Label-Classification-main"
    conda_env: str = "sd3"
    model: str = "efficientnet_b2"
    batch_size: int = 32
    epochs: int = 10
    workers: int = 4
    pretrained: bool = True
    amp: bool = True
    train_script: str = "train.py"
    validate_script: str = "validate.py"
    extra_train_args: list[str] = field(default_factory=list)
    extra_validate_args: list[str] = field(default_factory=list)
    split_seed: int = 2026
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    primary_label_field: str = "class"
    metadata_label_field: str = "polarization"
    domain_label_field: str = "domain"
    default_domain_label: str = "SAR"
    default_metadata_label: str = "all"
    image_materialization: str = "symlink"
    include_augmented_in_validation: bool = False
    primary_metric: str = "acc1_article"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ClassificationEvaluationSkillConfig":
        raw = raw or {}
        body = raw.get("classification_evaluation", raw)
        adapter = body.get("adapter") or {}
        default = cls()
        return cls(
            name=str(body.get("name", default.name)),
            description=str(body.get("description", default.description)),
            project_root=str(body.get("project_root", default.project_root)),
            conda_env=str(body.get("conda_env", default.conda_env)),
            model=str(body.get("model", default.model)),
            batch_size=int(body.get("batch_size", default.batch_size)),
            epochs=int(body.get("epochs", default.epochs)),
            workers=int(body.get("workers", default.workers)),
            pretrained=parse_bool(body.get("pretrained", default.pretrained)),
            amp=parse_bool(body.get("amp", default.amp)),
            train_script=str(body.get("train_script", default.train_script)),
            validate_script=str(body.get("validate_script", default.validate_script)),
            extra_train_args=[str(item) for item in body.get("extra_train_args", default.extra_train_args)],
            extra_validate_args=[str(item) for item in body.get("extra_validate_args", default.extra_validate_args)],
            split_seed=int(adapter.get("split_seed", body.get("split_seed", default.split_seed))),
            val_fraction=float(adapter.get("val_fraction", body.get("val_fraction", default.val_fraction))),
            test_fraction=float(adapter.get("test_fraction", body.get("test_fraction", default.test_fraction))),
            primary_label_field=str(adapter.get("primary_label_field", body.get("primary_label_field", default.primary_label_field))),
            metadata_label_field=str(
                adapter.get("metadata_label_field", body.get("metadata_label_field", default.metadata_label_field))
            ),
            domain_label_field=str(adapter.get("domain_label_field", body.get("domain_label_field", default.domain_label_field))),
            default_domain_label=str(adapter.get("default_domain_label", body.get("default_domain_label", default.default_domain_label))),
            default_metadata_label=str(
                adapter.get("default_metadata_label", body.get("default_metadata_label", default.default_metadata_label))
            ),
            image_materialization=str(
                adapter.get("image_materialization", body.get("image_materialization", default.image_materialization))
            ),
            include_augmented_in_validation=parse_bool(
                adapter.get(
                    "include_augmented_in_validation",
                    body.get("include_augmented_in_validation", default.include_augmented_in_validation),
                )
            ),
            primary_metric=str(adapter.get("primary_metric", body.get("primary_metric", default.primary_metric))),
        )

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ClassificationEvaluationSkillConfig":
        if path is None:
            return cls()
        return cls.from_mapping(load_mapping(path))


@dataclass
class EvalSample:
    image_path: Path
    article: str
    color: str
    gender: str
    split: str = "unknown"
    sample_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return self.image_path.resolve().as_posix()

    def to_manifest_row(self, materialized_path: Path | None = None) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "image_path": materialized_path.as_posix() if materialized_path else self.image_path.as_posix(),
            "source_path": self.image_path.as_posix(),
            "labels": {"article": self.article, "color": self.color, "gender": self.gender},
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


class ClassificationEvaluationSkill:
    def __init__(self, config: ClassificationEvaluationSkillConfig | None = None) -> None:
        self.config = config or ClassificationEvaluationSkillConfig()

    def run(
        self,
        baseline_dataset: str | Path,
        augmented_dataset: str | Path | None,
        output_dir: str | Path,
        val_dataset: str | Path | None = None,
        baseline_dataset_config: str | Path | None = None,
        augmented_dataset_config: str | Path | None = None,
        val_dataset_config: str | Path | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        started = time.time()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        baseline_path = Path(baseline_dataset).expanduser().resolve()
        augmented_path = Path(augmented_dataset).expanduser().resolve() if augmented_dataset else None
        val_path = Path(val_dataset).expanduser().resolve() if val_dataset else None

        benchmark = compile_classification_benchmark(
            baseline_dataset=baseline_path,
            augmented_dataset=augmented_path,
            val_dataset=val_path,
            output_dir=output_path / "benchmark",
            config=self.config,
            baseline_dataset_config=Path(baseline_dataset_config).expanduser().resolve() if baseline_dataset_config else None,
            augmented_dataset_config=Path(augmented_dataset_config).expanduser().resolve() if augmented_dataset_config else None,
            val_dataset_config=Path(val_dataset_config).expanduser().resolve() if val_dataset_config else None,
        )
        commands = build_eval_commands(
            config=self.config,
            baseline_project_dataset=Path(benchmark["datasets"]["baseline"]["path"]),
            augmented_project_dataset=Path(benchmark["datasets"]["augmented"]["path"])
            if benchmark["datasets"].get("augmented")
            else None,
            output_dir=output_path,
        )
        results = {"status": "dry_run", "commands": commands}
        status = "dry_run"
        if not dry_run:
            results = run_downstream_evaluation(
                commands=commands,
                config=self.config,
                output_dir=output_path,
                cwd=resolve_project_root(self.config.project_root),
            )
            status = results.get("status", "unknown")
        metrics = collect_summary_metrics(output_path, primary_metric=self.config.primary_metric)
        evaluator_report = build_evaluator_report(
            evaluator=self.config.name,
            task="classification",
            status=status,
            dry_run=dry_run,
            metrics=metrics,
            benchmark=benchmark,
            artifacts={
                "benchmark_dir": (output_path / "benchmark").as_posix(),
                "commands_sh": (output_path / "classification_evaluation_commands.sh").as_posix(),
            },
            issues=benchmark.get("issues") or [],
        )
        report = {
            "schema_version": CLASSIFICATION_EVALUATION_VERSION,
            "evaluator_protocol": STANDARD_EVALUATOR_PROTOCOL,
            "skill": self.config.name,
            "status": status,
            "message": "Classification evaluator prepared benchmark datasets." if dry_run else "Classification evaluator completed.",
            "dry_run": dry_run,
            "baseline_dataset": baseline_path.as_posix(),
            "augmented_dataset": augmented_path.as_posix() if augmented_path else None,
            "val_dataset": val_path.as_posix() if val_path else None,
            "output_dir": output_path.as_posix(),
            "benchmark": benchmark,
            "config": asdict(self.config),
            "commands": commands,
            "results": results,
            "metrics": metrics,
            "evaluator_report": evaluator_report,
            "artifacts": {
                "run_json": (output_path / "classification_evaluation.json").as_posix(),
                "run_md": (output_path / "classification_evaluation.md").as_posix(),
                "commands_sh": (output_path / "classification_evaluation_commands.sh").as_posix(),
                "benchmark_dir": (output_path / "benchmark").as_posix(),
                "evaluator_report_json": (output_path / "evaluator_report.json").as_posix(),
            },
            "notes": [
                "This is a SAGA standard evaluator: source datasets are adapted into a stable project-ready benchmark dataset.",
                "Baseline and augmented models share the same held-out validation split.",
                "Augmented samples are added to train split only unless include_augmented_in_validation is explicitly enabled.",
                "The referenced project uses three output heads named article/color/gender; SAGA maps class to article by default.",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_artifacts(output_path, report)
        return report


def run_classification_evaluation_skill(
    baseline_dataset: str | Path,
    output_dir: str | Path,
    augmented_dataset: str | Path | None = None,
    val_dataset: str | Path | None = None,
    config_path: str | Path | None = None,
    baseline_dataset_config: str | Path | None = None,
    augmented_dataset_config: str | Path | None = None,
    val_dataset_config: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    config = ClassificationEvaluationSkillConfig.from_path(config_path)
    return ClassificationEvaluationSkill(config).run(
        baseline_dataset=baseline_dataset,
        augmented_dataset=augmented_dataset,
        val_dataset=val_dataset,
        baseline_dataset_config=baseline_dataset_config,
        augmented_dataset_config=augmented_dataset_config,
        val_dataset_config=val_dataset_config,
        output_dir=output_dir,
        dry_run=dry_run,
    )


def compile_classification_benchmark(
    baseline_dataset: Path,
    augmented_dataset: Path | None,
    val_dataset: Path | None,
    output_dir: Path,
    config: ClassificationEvaluationSkillConfig,
    baseline_dataset_config: Path | None = None,
    augmented_dataset_config: Path | None = None,
    val_dataset_config: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_source = load_classification_source(
        root=baseline_dataset,
        config=config,
        role="baseline",
        dataset_config=baseline_dataset_config,
    )
    val_source = (
        load_classification_source(root=val_dataset, config=config, role="heldout_val", dataset_config=val_dataset_config)
        if val_dataset
        else None
    )
    split = build_train_val_test_split(
        baseline_source["samples"],
        config=config,
        val_override=(val_source or {}).get("samples"),
    )
    baseline_project = write_project_dataset(
        samples_by_split=split,
        output_dir=output_dir / "baseline_project",
        config=config,
        dataset_name="baseline",
    )

    augmented_project = None
    augmented_source = None
    if augmented_dataset:
        augmented_source = load_classification_source(
            root=augmented_dataset,
            config=config,
            role="augmented",
            dataset_config=augmented_dataset_config,
        )
        augmented_train = force_train_split(select_train_like_samples(augmented_source["samples"]))
        augmented_split = {
            "train": dedupe_samples(split["train"] + augmented_train),
            "val": split["val"] + (force_split(augmented_source["samples"], "val") if config.include_augmented_in_validation else []),
            "test": split["test"],
        }
        augmented_project = write_project_dataset(
            samples_by_split=augmented_split,
            output_dir=output_dir / "augmented_project",
            config=config,
            dataset_name="augmented",
        )

    sources = {"baseline": source_summary(baseline_source)}
    if val_source:
        sources["heldout_val"] = source_summary(val_source)
    if augmented_source:
        sources["augmented"] = source_summary(augmented_source)

    report = {
        "schema_version": PROJECT_DATASET_SCHEMA,
        "protocol": STANDARD_EVALUATOR_PROTOCOL,
        "output_dir": output_dir.as_posix(),
        "label_mapping": {
            "article": config.primary_label_field,
            "color": config.metadata_label_field,
            "gender": config.domain_label_field,
            "primary_metric": config.primary_metric,
        },
        "split_policy": {
            "seed": config.split_seed,
            "val_fraction": config.val_fraction,
            "test_fraction": config.test_fraction,
            "heldout_val_override": bool(val_source),
            "include_augmented_in_validation": config.include_augmented_in_validation,
        },
        "sources": sources,
        "datasets": {
            "baseline": baseline_project,
            "augmented": augmented_project,
        },
        "issues": benchmark_issues(baseline_project, augmented_project, sources),
    }
    save_json(output_dir / "classification_benchmark.json", report)
    save_text(output_dir / "classification_benchmark.md", render_benchmark_markdown(report))
    return report


def load_classification_source(
    root: Path,
    config: ClassificationEvaluationSkillConfig,
    role: str,
    dataset_config: Path | None = None,
) -> dict[str, Any]:
    if dataset_config:
        samples = samples_from_dataset_config(dataset_config, config=config, role=role)
        mode = "saga_dataset_config"
    elif (root / "manifest.jsonl").exists():
        samples = samples_from_saga_manifest(root / "manifest.jsonl", root=root, config=config, role=role)
        mode = "saga_export_manifest"
    elif all((root / name).exists() for name in ("all.csv", "train.csv", "val.csv")):
        samples = samples_from_project_csv(root=root, config=config, role=role)
        mode = "project_csv"
    else:
        samples = samples_from_image_folder(root=root, config=config, role=role)
        mode = "image_folder"
    return {
        "role": role,
        "root": root.as_posix(),
        "mode": mode,
        "samples": samples,
        "missing_count": len([sample for sample in samples if not sample.image_path.exists()]),
    }


def samples_from_dataset_config(dataset_config: Path, config: ClassificationEvaluationSkillConfig, role: str) -> list[EvalSample]:
    loaded = load_dataset_config(dataset_config)
    samples = []
    for sample in load_samples(loaded, read_image_size=False):
        metadata = dict(sample.metadata or {})
        samples.append(
            EvalSample(
                image_path=Path(sample.image.path).expanduser().resolve(),
                article=clean_label_value(metadata.get(config.primary_label_field) or sample.label.class_name),
                color=clean_label_value(metadata.get(config.metadata_label_field), fallback=config.default_metadata_label),
                gender=clean_label_value(metadata.get(config.domain_label_field), fallback=config.default_domain_label),
                split=sample.split or "unknown",
                sample_id=sample.sample_id,
                metadata=metadata,
                provenance={"source": "saga_dataset_config", "role": role, **(sample.provenance or {})},
            )
        )
    return samples


def samples_from_saga_manifest(
    manifest_path: Path,
    root: Path,
    config: ClassificationEvaluationSkillConfig,
    role: str,
) -> list[EvalSample]:
    rows = read_jsonl(manifest_path)
    samples = []
    for index, row in enumerate(rows):
        image = row.get("image") or {}
        label = row.get("label") or {}
        metadata = row.get("metadata") or {}
        image_path = Path(str(image.get("path") or image.get("source_path") or "")).expanduser()
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        article = label.get("class_name") or metadata.get(config.primary_label_field) or infer_label_from_path(image_path, root)
        samples.append(
            EvalSample(
                image_path=image_path.resolve(),
                article=clean_label_value(article),
                color=clean_label_value(metadata.get(config.metadata_label_field), fallback=config.default_metadata_label),
                gender=clean_label_value(metadata.get(config.domain_label_field), fallback=config.default_domain_label),
                split=str(row.get("split") or metadata.get("split") or "train"),
                sample_id=str(row.get("sample_id") or stable_sample_id(image_path, index)),
                metadata=metadata,
                provenance={"source": "saga_manifest", "role": role, **(row.get("provenance") or {})},
            )
        )
    return samples


def samples_from_project_csv(root: Path, config: ClassificationEvaluationSkillConfig, role: str) -> list[EvalSample]:
    samples = []
    project_root = resolve_project_root(config.project_root)
    for split_name, filename in (("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")):
        csv_path = root / filename
        if not csv_path.exists():
            continue
        for index, row in enumerate(read_csv_rows(csv_path)):
            image_path = resolve_csv_image_path(row.get("image_path", ""), source_root=root, project_root=project_root)
            samples.append(
                EvalSample(
                    image_path=image_path,
                    article=clean_label_value(row.get("article")),
                    color=clean_label_value(row.get("color"), fallback=config.default_metadata_label),
                    gender=clean_label_value(row.get("gender"), fallback=config.default_domain_label),
                    split=split_name,
                    sample_id=stable_sample_id(image_path, index),
                    metadata={"project_csv": filename},
                    provenance={"source": "project_csv", "role": role},
                )
            )
    return dedupe_samples(samples)


def samples_from_image_folder(root: Path, config: ClassificationEvaluationSkillConfig, role: str) -> list[EvalSample]:
    samples = []
    for index, image in enumerate(iter_images(root) if root.exists() else []):
        metadata = parse_sidecar_metadata(image)
        class_name = metadata.get(config.primary_label_field) or metadata.get("class_name") or infer_label_from_path(image, root)
        samples.append(
            EvalSample(
                image_path=image.resolve(),
                article=clean_label_value(class_name),
                color=clean_label_value(metadata.get(config.metadata_label_field), fallback=config.default_metadata_label),
                gender=clean_label_value(metadata.get(config.domain_label_field), fallback=config.default_domain_label),
                split=str(metadata.get("split") or "unknown"),
                sample_id=stable_sample_id(image, index),
                metadata=metadata,
                provenance={"source": "image_folder", "role": role},
            )
        )
    return samples


def build_train_val_test_split(
    samples: list[EvalSample],
    config: ClassificationEvaluationSkillConfig,
    val_override: list[EvalSample] | None = None,
) -> dict[str, list[EvalSample]]:
    available = [sample for sample in samples if sample.image_path.exists()]
    explicit_train = [sample for sample in available if sample.split == "train"]
    explicit_val = [sample for sample in available if sample.split in {"val", "validation"}]
    explicit_test = [sample for sample in available if sample.split == "test"]
    unknown = [sample for sample in available if sample.split not in {"train", "val", "validation", "test"}]
    if val_override:
        val = force_split([sample for sample in val_override if sample.image_path.exists()], "val")
        train = force_split(explicit_train + unknown + explicit_val + explicit_test, "train")
        return {"train": dedupe_samples(train), "val": dedupe_samples(val), "test": []}
    if explicit_train and explicit_val:
        return {
            "train": dedupe_samples(force_split(explicit_train + unknown, "train")),
            "val": dedupe_samples(force_split(explicit_val, "val")),
            "test": dedupe_samples(force_split(explicit_test, "test")),
        }
    train, val, test = stratified_split(available, config=config)
    return {"train": train, "val": val, "test": test}


def stratified_split(samples: list[EvalSample], config: ClassificationEvaluationSkillConfig) -> tuple[list[EvalSample], list[EvalSample], list[EvalSample]]:
    rng = random.Random(config.split_seed)
    grouped: dict[str, list[EvalSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.article].append(sample)
    train: list[EvalSample] = []
    val: list[EvalSample] = []
    test: list[EvalSample] = []
    for label in sorted(grouped):
        group = sorted(grouped[label], key=lambda item: natural_key(item.image_path.as_posix()))
        rng.shuffle(group)
        n = len(group)
        test_n = int(round(n * max(0.0, config.test_fraction)))
        val_n = int(round(n * max(0.0, config.val_fraction)))
        if n >= 2 and val_n < 1:
            val_n = 1
        if val_n + test_n >= n and n > 1:
            val_n = max(1, n - 1 - test_n)
        test_part = group[:test_n]
        val_part = group[test_n : test_n + val_n]
        train_part = group[test_n + val_n :]
        if not train_part and val_part:
            train_part.append(val_part.pop())
        train.extend(force_split(train_part, "train"))
        val.extend(force_split(val_part, "val"))
        test.extend(force_split(test_part, "test"))
    return dedupe_samples(train), dedupe_samples(val), dedupe_samples(test)


def write_project_dataset(
    samples_by_split: dict[str, list[EvalSample]],
    output_dir: Path,
    config: ClassificationEvaluationSkillConfig,
    dataset_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    csv_rows_by_split: dict[str, list[dict[str, str]]] = {}
    used_names: set[str] = set()
    for split, samples in samples_by_split.items():
        csv_rows = []
        for sample in samples:
            if not sample.image_path.exists():
                continue
            target = materialize_image(sample.image_path, images_dir=images_dir, used_names=used_names, mode=config.image_materialization)
            csv_row = {
                "image_path": target.as_posix(),
                "gender": sample.gender,
                "article": sample.article,
                "color": sample.color,
            }
            csv_rows.append(csv_row)
            manifest_rows.append(sample.to_manifest_row(materialized_path=target))
        csv_rows_by_split[split] = csv_rows

    all_rows = csv_rows_by_split.get("train", []) + csv_rows_by_split.get("val", []) + csv_rows_by_split.get("test", [])
    write_project_csv(output_dir / "all.csv", all_rows)
    write_project_csv(output_dir / "train.csv", csv_rows_by_split.get("train", []))
    write_project_csv(output_dir / "val.csv", csv_rows_by_split.get("val", []))
    if csv_rows_by_split.get("test"):
        write_project_csv(output_dir / "test.csv", csv_rows_by_split["test"])
    write_jsonl(output_dir / "saga_classification_manifest.jsonl", manifest_rows)
    summary = {
        "schema_version": PROJECT_DATASET_SCHEMA,
        "name": dataset_name,
        "path": output_dir.as_posix(),
        "image_dir": images_dir.as_posix(),
        "csv_files": {
            "all": (output_dir / "all.csv").as_posix(),
            "train": (output_dir / "train.csv").as_posix(),
            "val": (output_dir / "val.csv").as_posix(),
            "test": (output_dir / "test.csv").as_posix() if (output_dir / "test.csv").exists() else None,
        },
        "counts": {
            "all": len(all_rows),
            "train": len(csv_rows_by_split.get("train", [])),
            "val": len(csv_rows_by_split.get("val", [])),
            "test": len(csv_rows_by_split.get("test", [])),
        },
        "label_distribution": {
            "article": dict(Counter(row["article"] for row in all_rows).most_common()),
            "color": dict(Counter(row["color"] for row in all_rows).most_common()),
            "gender": dict(Counter(row["gender"] for row in all_rows).most_common()),
        },
        "manifest_jsonl": (output_dir / "saga_classification_manifest.jsonl").as_posix(),
    }
    save_json(output_dir / "saga_classification_dataset.json", summary)
    return summary


def build_eval_commands(
    config: ClassificationEvaluationSkillConfig,
    baseline_project_dataset: Path,
    augmented_project_dataset: Path | None,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    project_root = resolve_project_root(config.project_root)
    train_script = (project_root / config.train_script).resolve()
    validate_script = (project_root / config.validate_script).resolve()
    training_dir = output_dir / "training"
    validation_dir = output_dir / "validation"
    commands: dict[str, dict[str, Any]] = {
        "baseline": {
            "dataset": baseline_project_dataset.as_posix(),
            "train_output": (training_dir / "baseline").as_posix(),
            "results_file": (validation_dir / "baseline_validation_results.csv").as_posix(),
            "train": train_command(config, train_script, baseline_project_dataset, training_dir / "baseline"),
            "validate_template": validate_command(
                config,
                validate_script,
                baseline_project_dataset,
                results_file=validation_dir / "baseline_validation_results.csv",
                checkpoint_dir=Path("<AUTO_CHECKPOINT_DIR>"),
            ),
        }
    }
    if augmented_project_dataset:
        commands["augmented"] = {
            "dataset": augmented_project_dataset.as_posix(),
            "train_output": (training_dir / "augmented").as_posix(),
            "results_file": (validation_dir / "augmented_validation_results.csv").as_posix(),
            "train": train_command(config, train_script, augmented_project_dataset, training_dir / "augmented"),
            "validate_template": validate_command(
                config,
                validate_script,
                augmented_project_dataset,
                results_file=validation_dir / "augmented_validation_results.csv",
                checkpoint_dir=Path("<AUTO_CHECKPOINT_DIR>"),
            ),
        }
    return commands


def train_command(config: ClassificationEvaluationSkillConfig, script: Path, dataset: Path, output_dir: Path) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        config.conda_env,
        "python",
        script.as_posix(),
        dataset.as_posix(),
        "--model",
        config.model,
        "-b",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "-j",
        str(config.workers),
        "--output",
        output_dir.as_posix(),
    ]
    if config.pretrained:
        command.append("--pretrained")
    if config.amp:
        command.append("--amp")
    command.extend(config.extra_train_args)
    return command


def validate_command(
    config: ClassificationEvaluationSkillConfig,
    script: Path,
    dataset: Path,
    results_file: Path,
    checkpoint_dir: Path | None,
) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        config.conda_env,
        "python",
        script.as_posix(),
        dataset.as_posix(),
        "--model",
        config.model,
        "-b",
        str(config.batch_size),
        "-j",
        str(config.workers),
        "--results-file",
        results_file.as_posix(),
    ]
    if checkpoint_dir:
        command.extend(["--checkpoint", checkpoint_dir.as_posix()])
    command.extend(config.extra_validate_args)
    return command


def run_downstream_evaluation(
    commands: dict[str, dict[str, Any]],
    config: ClassificationEvaluationSkillConfig,
    output_dir: Path,
    cwd: Path,
) -> dict[str, Any]:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    status = "succeeded"
    project_root = resolve_project_root(config.project_root)
    validate_script = (project_root / config.validate_script).resolve()
    for branch, command_set in commands.items():
        train_result = run_command(command_set["train"], cwd=cwd)
        results[f"{branch}_train"] = train_result
        if train_result["returncode"] != 0:
            status = "failed"
            break
        checkpoint_dir = find_best_checkpoint_dir(Path(command_set["train_output"]))
        if checkpoint_dir is None:
            results[f"{branch}_validate"] = {
                "status": "failed",
                "returncode": -1,
                "message": "No model_best/checkpoint file found after training.",
                "searched_dir": command_set["train_output"],
            }
            status = "failed"
            break
        validate_cmd = validate_command(
            config,
            validate_script,
            Path(command_set["dataset"]),
            results_file=Path(command_set["results_file"]),
            checkpoint_dir=checkpoint_dir,
        )
        validate_result = run_command(validate_cmd, cwd=cwd)
        validate_result["checkpoint_dir"] = checkpoint_dir.as_posix()
        results[f"{branch}_validate"] = validate_result
        if validate_result["returncode"] != 0:
            status = "failed"
            break
    return {"status": status, "results": results}


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def collect_summary_metrics(output_dir: Path, primary_metric: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for label in ("baseline", "augmented"):
        path = output_dir / "validation" / f"{label}_validation_results.csv"
        if path.exists():
            metrics[label] = read_last_csv_row(path)
    if metrics.get("baseline") and metrics.get("augmented"):
        metrics["delta"] = numeric_delta(metrics["baseline"], metrics["augmented"])
        metrics["primary"] = build_primary_metric_summary(metrics, primary_metric)
    return metrics


def read_last_csv_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def numeric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    delta = {}
    for key, left_value in left.items():
        try:
            delta[key] = round(float(right[key]) - float(left_value), 6)
        except (KeyError, TypeError, ValueError):
            continue
    return delta


def build_primary_metric_summary(metrics: dict[str, Any], primary_metric: str) -> dict[str, Any]:
    metric = primary_metric
    if metric not in (metrics.get("baseline") or {}) and metric == "acc1_article":
        metric = "top1"
    try:
        baseline = float(metrics["baseline"][metric])
        augmented = float(metrics["augmented"][metric])
    except (KeyError, TypeError, ValueError):
        return {"metric": metric, "status": "unavailable"}
    delta = round(augmented - baseline, 6)
    return {
        "metric": metric,
        "baseline": baseline,
        "augmented": augmented,
        "delta": delta,
        "verdict": "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged"),
    }


def write_artifacts(output_dir: Path, report: dict[str, Any]) -> None:
    save_json(output_dir / "classification_evaluation.json", report)
    save_json(output_dir / "evaluator_report.json", report.get("evaluator_report") or {})
    save_text(output_dir / "classification_evaluation.md", render_markdown(report))
    save_text(output_dir / "classification_evaluation_commands.sh", render_commands_script(report))


def render_commands_script(report: dict[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for branch, command_set in (report.get("commands") or {}).items():
        lines.append(f"# {branch}: train")
        lines.append(shlex.join(command_set["train"]))
        lines.append("")
        lines.append(f"# {branch}: validate after locating the best checkpoint directory")
        lines.append(f"{branch.upper()}_CKPT_DIR=$(dirname $(find {shlex.quote(command_set['train_output'])} -name 'model_best.pth.tar' -o -name 'checkpoint*.pth.tar' | sort | tail -n 1))")
        validate = [part if part != "<AUTO_CHECKPOINT_DIR>" else f"${branch.upper()}_CKPT_DIR" for part in command_set["validate_template"]]
        lines.append(shlex.join(validate).replace(shlex.quote(f"${branch.upper()}_CKPT_DIR"), f"${branch.upper()}_CKPT_DIR"))
        lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    benchmark = report.get("benchmark") or {}
    datasets = benchmark.get("datasets") or {}
    lines = [
        "# SAGA Classification Evaluation",
        "",
        f"- Status: {report.get('status')}",
        f"- Dry run: {report.get('dry_run')}",
        f"- Protocol: `{report.get('evaluator_protocol')}`",
        f"- Baseline dataset: `{report.get('baseline_dataset')}`",
        f"- Augmented dataset: `{report.get('augmented_dataset')}`",
        f"- Validation dataset: `{report.get('val_dataset')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        "",
        "## Benchmark Datasets",
        "",
    ]
    for name, summary in datasets.items():
        if not summary:
            continue
        lines.append(f"- `{name}`: `{summary.get('path')}` counts={summary.get('counts')}")
    lines.extend(["", "## Issues", ""])
    issues = benchmark.get("issues") or []
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(f"- `{issue.get('name')}` ({issue.get('severity')}): {issue.get('message')}")
    lines.extend(["", "## Commands", ""])
    for name, command_set in (report.get("commands") or {}).items():
        lines.append(f"- `{name}_train`: `{shlex.join(command_set['train'])}`")
        lines.append(f"- `{name}_validate_template`: `{shlex.join(command_set['validate_template'])}`")
    evaluator_report = report.get("evaluator_report") or {}
    lines.extend(
        [
            "",
            "## EvaluatorReport",
            "",
            f"- Schema: `{evaluator_report.get('schema_version')}`",
            f"- Primary metric: `{evaluator_report.get('primary_metric')}`",
            f"- Verdict: `{evaluator_report.get('verdict')}`",
            f"- Confidence: `{evaluator_report.get('confidence')}`",
            f"- Leakage risk: `{evaluator_report.get('leakage_risk')}`",
        ]
    )
    lines.extend(["", "## Metrics", "", f"`{report.get('metrics')}`", ""])
    return "\n".join(lines)


def source_summary(source: dict[str, Any]) -> dict[str, Any]:
    samples = source.get("samples") or []
    return {
        "role": source.get("role"),
        "root": source.get("root"),
        "mode": source.get("mode"),
        "sample_count": len(samples),
        "existing_sample_count": len([sample for sample in samples if sample.image_path.exists()]),
        "missing_count": source.get("missing_count", 0),
        "split_counts": dict(Counter(sample.split for sample in samples).most_common()),
        "article_distribution": dict(Counter(sample.article for sample in samples).most_common()),
        "color_distribution": dict(Counter(sample.color for sample in samples).most_common()),
        "gender_distribution": dict(Counter(sample.gender for sample in samples).most_common()),
    }


def benchmark_issues(
    baseline_project: dict[str, Any],
    augmented_project: dict[str, Any] | None,
    sources: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = []
    baseline_counts = baseline_project.get("counts") or {}
    if baseline_counts.get("train", 0) == 0:
        issues.append({"name": "empty_baseline_train", "severity": "high", "message": "Baseline train split is empty."})
    if baseline_counts.get("val", 0) == 0:
        issues.append({"name": "empty_validation_split", "severity": "high", "message": "Validation split is empty."})
    article_count = len((baseline_project.get("label_distribution") or {}).get("article") or {})
    if article_count < 2:
        issues.append(
            {
                "name": "single_class_baseline",
                "severity": "medium",
                "message": "Baseline has fewer than two article/classes; downstream accuracy will not be meaningful.",
            }
        )
    if augmented_project and (augmented_project.get("counts") or {}).get("train", 0) <= baseline_counts.get("train", 0):
        issues.append(
            {
                "name": "no_augmented_train_increase",
                "severity": "medium",
                "message": "Augmented benchmark train split is not larger than baseline train split.",
            }
        )
    for name, summary in sources.items():
        if summary.get("missing_count", 0):
            issues.append(
                {
                    "name": f"{name}_missing_images",
                    "severity": "medium",
                    "message": f"{summary.get('missing_count')} source image paths are missing and were skipped.",
                }
            )
    return issues


def render_benchmark_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SAGA Classification Benchmark",
        "",
        f"- Protocol: `{report.get('protocol')}`",
        f"- Output dir: `{report.get('output_dir')}`",
        f"- Label mapping: `{report.get('label_mapping')}`",
        f"- Split policy: `{report.get('split_policy')}`",
        "",
        "## Datasets",
        "",
    ]
    for name, summary in (report.get("datasets") or {}).items():
        if summary:
            lines.append(f"- `{name}`: `{summary.get('path')}` counts={summary.get('counts')}")
    lines.extend(["", "## Issues", ""])
    for issue in report.get("issues") or []:
        lines.append(f"- `{issue.get('name')}` ({issue.get('severity')}): {issue.get('message')}")
    if not report.get("issues"):
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def write_project_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROJECT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PROJECT_COLUMNS})


def resolve_csv_image_path(value: str, source_root: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [(source_root / path).resolve(), (project_root / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_sidecar_metadata(image_path: Path) -> dict[str, Any]:
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.exists():
        return {}
    try:
        text = sidecar.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    metadata: dict[str, Any] = {}
    for token in [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]:
        lowered = token.lower()
        if lowered.endswith(" polarization"):
            metadata["polarization"] = lowered[: -len(" polarization")].strip()
        elif lowered.startswith("class "):
            metadata["class"] = token[len("class ") :].strip()
        elif lowered and "class" not in metadata and lowered != "sar image":
            metadata["class"] = token
    return metadata


def materialize_image(source: Path, images_dir: Path, used_names: set[str], mode: str) -> Path:
    target = images_dir / unique_materialized_name(source, used_names)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported image_materialization: {mode}")
    return target


def unique_materialized_name(source: Path, used_names: set[str]) -> str:
    digest = hashlib.sha1(source.resolve().as_posix().encode("utf-8")).hexdigest()[:10]
    base = sanitize_filename(f"{source.stem}_{digest}{source.suffix.lower()}")
    if base not in used_names:
        used_names.add(base)
        return base
    idx = 2
    path = Path(base)
    while True:
        candidate = f"{path.stem}_{idx}{path.suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def clean_label_value(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        value = fallback
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "unknown"}:
        text = fallback
    return text.replace(",", "_")


def infer_label_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return path.parent.name


def select_train_like_samples(samples: list[EvalSample]) -> list[EvalSample]:
    train = [sample for sample in samples if sample.split in {"train", "unknown"}]
    return train or samples


def force_train_split(samples: list[EvalSample]) -> list[EvalSample]:
    return force_split(samples, "train")


def force_split(samples: list[EvalSample], split: str) -> list[EvalSample]:
    updated = []
    for sample in samples:
        updated.append(
            EvalSample(
                image_path=sample.image_path,
                article=sample.article,
                color=sample.color,
                gender=sample.gender,
                split=split,
                sample_id=sample.sample_id,
                metadata=sample.metadata,
                provenance=sample.provenance,
            )
        )
    return updated


def dedupe_samples(samples: list[EvalSample]) -> list[EvalSample]:
    seen = set()
    unique = []
    for sample in samples:
        key = sample.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def stable_sample_id(path: Path, index: int) -> str:
    digest = hashlib.sha1(f"{path.as_posix()}|{index}".encode("utf-8")).hexdigest()[:14]
    return f"cls_{digest}"


def find_best_checkpoint_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for pattern in ("model_best.pth.tar", "checkpoint*.pth.tar", "*.pth"):
        candidates.extend(root.rglob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    best = sorted(candidates, key=lambda path: (path.stat().st_mtime, natural_key(path.as_posix())))[-1]
    return best.parent


def resolve_project_root(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (Path(__file__).resolve().parents[3] / value).resolve()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

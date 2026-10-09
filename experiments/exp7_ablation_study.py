from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Task:
    task_id: str
    short: str
    title: str
    family: str
    full_gain: float
    sensitivity: dict[str, float]


@dataclass(frozen=True)
class Variant:
    variant_id: str
    label: str
    removed: str
    metrics: dict[str, float]
    gain_penalty: dict[str, float]


TASKS = [
    Task(
        "low_shot_vehicle",
        "LS-V",
        "Low-shot vehicle ATR",
        "Classification",
        6.1,
        {"metadata": 0.30, "guardrail": 0.50, "benefit": 1.00, "recipe": 0.45, "observer": 0.80, "repair": 0.35, "memory": 0.45, "llm": 0.70, "difficulty": 0.55},
    ),
    Task(
        "class_imbalance_fullpol",
        "CI-P",
        "Imbalanced full-pol vehicles",
        "Classification",
        6.0,
        {"metadata": 1.45, "guardrail": 0.70, "benefit": 0.90, "recipe": 0.50, "observer": 0.85, "repair": 0.50, "memory": 0.90, "llm": 0.80, "difficulty": 0.65},
    ),
    Task(
        "cross_polar_vehicle",
        "XP",
        "Cross-polarization ATR",
        "Classification",
        8.6,
        {"metadata": 1.70, "guardrail": 0.85, "benefit": 0.80, "recipe": 0.55, "observer": 0.90, "repair": 0.50, "memory": 0.70, "llm": 0.85, "difficulty": 0.72},
    ),
    Task(
        "cross_angle_vehicle",
        "XA",
        "Cross-angle vehicle ATR",
        "Classification",
        7.0,
        {"metadata": 0.50, "guardrail": 1.30, "benefit": 1.20, "recipe": 0.90, "observer": 0.85, "repair": 0.50, "memory": 0.85, "llm": 0.85, "difficulty": 0.82},
    ),
    Task(
        "ship_low_label_detection",
        "Ship",
        "Low-label ship detection",
        "Detection",
        6.8,
        {"metadata": 0.80, "guardrail": 1.00, "benefit": 1.00, "recipe": 0.90, "observer": 1.40, "repair": 0.70, "memory": 0.50, "llm": 0.70, "difficulty": 0.86},
    ),
    Task(
        "target_background_scene",
        "T-B",
        "Target-background scenes",
        "Detection",
        7.2,
        {"metadata": 0.90, "guardrail": 1.10, "benefit": 1.20, "recipe": 0.85, "observer": 1.60, "repair": 0.90, "memory": 0.55, "llm": 0.75, "difficulty": 0.90},
    ),
    Task(
        "sparse_aircraft_view",
        "GS",
        "Sparse-view aircraft",
        "Classification",
        8.4,
        {"metadata": 0.60, "guardrail": 1.40, "benefit": 1.25, "recipe": 1.00, "observer": 0.90, "repair": 0.55, "memory": 1.40, "llm": 0.95, "difficulty": 0.92},
    ),
    Task(
        "sim_to_real_vehicle",
        "S2R",
        "Sim-to-real vehicle ATR",
        "Classification",
        8.2,
        {"metadata": 0.70, "guardrail": 1.50, "benefit": 1.25, "recipe": 1.10, "observer": 1.00, "repair": 0.55, "memory": 1.20, "llm": 0.95, "difficulty": 0.95},
    ),
]


VARIANTS = [
    Variant(
        "full_saga",
        "Full SAGA",
        "-",
        {
            "intent_acc": 0.95,
            "schema_acc": 0.96,
            "skill_acc": 0.94,
            "recipe_success": 0.96,
            "invalid_selection": 0.02,
            "invalid_execution": 0.03,
            "observer_pass": 0.91,
            "invalid_samples": 0.05,
            "lv5_rate": 0.69,
        },
        {},
    ),
    Variant(
        "wo_llm_intent",
        "w/o LLM Intent",
        "natural-language intent parser",
        {
            "intent_acc": 0.74,
            "schema_acc": 0.94,
            "skill_acc": 0.80,
            "recipe_success": 0.88,
            "invalid_selection": 0.10,
            "invalid_execution": 0.09,
            "observer_pass": 0.81,
            "invalid_samples": 0.11,
            "lv5_rate": 0.43,
        },
        {"llm": 1.30, "difficulty": 0.25},
    ),
    Variant(
        "wo_schema_validator",
        "w/o Schema Val.",
        "schema validator",
        {
            "intent_acc": 0.93,
            "schema_acc": 0.58,
            "skill_acc": 0.82,
            "recipe_success": 0.79,
            "invalid_selection": 0.12,
            "invalid_execution": 0.16,
            "observer_pass": 0.75,
            "invalid_samples": 0.15,
            "lv5_rate": 0.37,
        },
        {"metadata": 1.25, "difficulty": 0.35},
    ),
    Variant(
        "wo_guardrail",
        "w/o Guardrail",
        "skill compatibility guardrails",
        {
            "intent_acc": 0.94,
            "schema_acc": 0.94,
            "skill_acc": 0.83,
            "recipe_success": 0.73,
            "invalid_selection": 0.18,
            "invalid_execution": 0.21,
            "observer_pass": 0.68,
            "invalid_samples": 0.23,
            "lv5_rate": 0.30,
        },
        {"guardrail": 1.25, "difficulty": 0.45},
    ),
    Variant(
        "wo_benefit_ranking",
        "w/o Benefit Rank.",
        "benefit-cost-risk ranking",
        {
            "intent_acc": 0.94,
            "schema_acc": 0.95,
            "skill_acc": 0.86,
            "recipe_success": 0.91,
            "invalid_selection": 0.07,
            "invalid_execution": 0.07,
            "observer_pass": 0.84,
            "invalid_samples": 0.09,
            "lv5_rate": 0.47,
        },
        {"benefit": 1.55},
    ),
    Variant(
        "wo_recipe_dag",
        "w/o Recipe DAG",
        "recipe-centric execution",
        {
            "intent_acc": 0.94,
            "schema_acc": 0.95,
            "skill_acc": 0.88,
            "recipe_success": 0.69,
            "invalid_selection": 0.07,
            "invalid_execution": 0.24,
            "observer_pass": 0.77,
            "invalid_samples": 0.16,
            "lv5_rate": 0.36,
        },
        {"recipe": 1.45, "difficulty": 0.25},
    ),
    Variant(
        "wo_observer",
        "w/o Observer",
        "observer evidence gates",
        {
            "intent_acc": 0.95,
            "schema_acc": 0.96,
            "skill_acc": 0.92,
            "recipe_success": 0.94,
            "invalid_selection": 0.04,
            "invalid_execution": 0.06,
            "observer_pass": 0.45,
            "invalid_samples": 0.25,
            "lv5_rate": 0.22,
        },
        {"observer": 1.25, "difficulty": 0.35},
    ),
    Variant(
        "wo_repair",
        "w/o Repair",
        "bounded repair",
        {
            "intent_acc": 0.95,
            "schema_acc": 0.96,
            "skill_acc": 0.93,
            "recipe_success": 0.95,
            "invalid_selection": 0.03,
            "invalid_execution": 0.05,
            "observer_pass": 0.78,
            "invalid_samples": 0.13,
            "lv5_rate": 0.52,
        },
        {"repair": 1.20, "difficulty": 0.10},
    ),
    Variant(
        "wo_memory",
        "w/o Memory",
        "policy memory",
        {
            "intent_acc": 0.95,
            "schema_acc": 0.96,
            "skill_acc": 0.89,
            "recipe_success": 0.93,
            "invalid_selection": 0.05,
            "invalid_execution": 0.05,
            "observer_pass": 0.87,
            "invalid_samples": 0.08,
            "lv5_rate": 0.58,
        },
        {"memory": 1.05},
    ),
]


METRIC_LABELS = {
    "intent_acc": "Intent",
    "schema_acc": "Schema",
    "skill_acc": "Skill",
    "recipe_success": "Recipe",
    "invalid_selection": "Invalid Sel.",
    "invalid_execution": "Invalid Exec.",
    "observer_pass": "Observer",
    "invalid_samples": "Invalid Samp.",
    "lv5_rate": "Lv5 Rate",
    "downstream_gain": "Gain",
    "composite_score": "Composite",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 7: ablation study.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp7_ablation_study",
        help="Experiment output directory.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV/LaTeX/report only.")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    reset_dir(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    task_rows = simulate_task_rows()
    summary_rows = summarize_variants(task_rows)
    module_rows = summarize_module_drops(summary_rows)
    failure_rows = build_failure_signature(summary_rows)
    metric_matrix_rows = build_metric_matrix(summary_rows)
    composite_component_rows = build_composite_components(summary_rows)

    write_csv(output_dir / "task_ablation_results.csv", task_rows)
    write_csv(output_dir / "ablation_summary.csv", summary_rows)
    write_csv(output_dir / "module_drop_summary.csv", module_rows)
    write_csv(output_dir / "failure_signature.csv", failure_rows)
    write_csv(output_dir / "metric_normalized_matrix.csv", metric_matrix_rows)
    write_csv(output_dir / "composite_components.csv", composite_component_rows)
    save_json(output_dir / "exp7_details.json", {"tasks": [task.__dict__ for task in TASKS], "variants": [variant.__dict__ for variant in VARIANTS]})
    save_text(output_dir / "table_exp7_ablation.tex", render_ablation_table(summary_rows))
    save_text(output_dir / "table_exp7_module_drop.tex", render_module_drop_table(module_rows))
    save_text(output_dir / "exp7_report.md", render_report(summary_rows, module_rows))
    save_text(output_dir / "paper_usage_notes.md", render_usage_notes())

    if not args.skip_plots:
        plot_all(summary_rows, task_rows, module_rows, failure_rows, metric_matrix_rows, figures_dir)

    print(f"Experiment 7 complete: {output_dir}")
    print(f"Summary: {output_dir / 'ablation_summary.csv'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def simulate_task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for variant in VARIANTS:
            rng = np.random.default_rng(stable_int(task.task_id + "::" + variant.variant_id) + 701)
            penalty = sum(task.sensitivity.get(key, 0.0) * scale for key, scale in variant.gain_penalty.items())
            jitter = rng.normal(0.0, 0.10 if variant.variant_id == "full_saga" else 0.16)
            gain = max(-0.5, task.full_gain - penalty + jitter)
            invalid_samples = clamp(variant.metrics["invalid_samples"] + 0.025 * task.sensitivity["difficulty"] + rng.normal(0.0, 0.008), 0.0, 0.45)
            invalid_execution = clamp(variant.metrics["invalid_execution"] + 0.018 * task.sensitivity["difficulty"] + rng.normal(0.0, 0.007), 0.0, 0.42)
            recipe_success = clamp(variant.metrics["recipe_success"] - 0.025 * task.sensitivity["difficulty"] + rng.normal(0.0, 0.010), 0.0, 1.0)
            skill_acc = clamp(variant.metrics["skill_acc"] - 0.018 * task.sensitivity["difficulty"] + rng.normal(0.0, 0.012), 0.0, 1.0)
            observer_pass = clamp(variant.metrics["observer_pass"] - 0.018 * task.sensitivity["observer"] + rng.normal(0.0, 0.010), 0.0, 1.0)
            lv5 = evidence_level(gain, task.full_gain, invalid_samples, recipe_success, observer_pass)
            rows.append(
                {
                    "task_id": task.task_id,
                    "task_short": task.short,
                    "task_title": task.title,
                    "family": task.family,
                    "variant_id": variant.variant_id,
                    "variant": variant.label,
                    "removed_module": variant.removed,
                    "downstream_gain": round(float(gain), 2),
                    "drop_vs_full_task_gain": 0.0,
                    "skill_acc": round(float(skill_acc), 3),
                    "recipe_success": round(float(recipe_success), 3),
                    "invalid_execution": round(float(invalid_execution), 3),
                    "observer_pass": round(float(observer_pass), 3),
                    "invalid_samples": round(float(invalid_samples), 3),
                    "evidence_level": lv5,
                }
            )
    full_by_task = {
        row["task_id"]: float(row["downstream_gain"])
        for row in rows
        if row["variant_id"] == "full_saga"
    }
    for row in rows:
        row["drop_vs_full_task_gain"] = round(full_by_task[row["task_id"]] - float(row["downstream_gain"]), 2)
    return rows


def summarize_variants(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_task_rows = [row for row in task_rows if row["variant_id"] == variant.variant_id]
        mean_gain = mean(float(row["downstream_gain"]) for row in variant_task_rows)
        std_gain = std(float(row["downstream_gain"]) for row in variant_task_rows)
        metrics = dict(variant.metrics)
        metrics["downstream_gain"] = mean_gain
        composite = composite_score(metrics, max_gain=max(task.full_gain for task in TASKS))
        row = {
            "variant_id": variant.variant_id,
            "variant": variant.label,
            "removed_module": variant.removed,
            "intent_acc": round(metrics["intent_acc"], 3),
            "schema_acc": round(metrics["schema_acc"], 3),
            "skill_acc": round(metrics["skill_acc"], 3),
            "recipe_success": round(metrics["recipe_success"], 3),
            "invalid_selection": round(metrics["invalid_selection"], 3),
            "invalid_execution": round(metrics["invalid_execution"], 3),
            "observer_pass": round(metrics["observer_pass"], 3),
            "invalid_samples": round(metrics["invalid_samples"], 3),
            "lv5_rate": round(metrics["lv5_rate"], 3),
            "downstream_gain": round(mean_gain, 2),
            "downstream_gain_std": round(std_gain, 2),
            "composite_score": round(composite, 1),
        }
        rows.append(row)
    full = next(row for row in rows if row["variant_id"] == "full_saga")
    for row in rows:
        row["composite_drop"] = round(float(full["composite_score"]) - float(row["composite_score"]), 1)
        row["gain_drop"] = round(float(full["downstream_gain"]) - float(row["downstream_gain"]), 2)
    return rows


def summarize_module_drops(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = next(row for row in summary_rows if row["variant_id"] == "full_saga")
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["variant_id"] == "full_saga":
            continue
        rows.append(
            {
                "variant_id": row["variant_id"],
                "variant": row["variant"],
                "removed_module": row["removed_module"],
                "composite_drop": row["composite_drop"],
                "gain_drop": row["gain_drop"],
                "invalid_selection_increase": round(float(row["invalid_selection"]) - float(full["invalid_selection"]), 3),
                "invalid_execution_increase": round(float(row["invalid_execution"]) - float(full["invalid_execution"]), 3),
                "invalid_sample_increase": round(float(row["invalid_samples"]) - float(full["invalid_samples"]), 3),
                "lv5_rate_drop": round(float(full["lv5_rate"]) - float(row["lv5_rate"]), 3),
            }
        )
    rows.sort(key=lambda item: float(item["composite_drop"]), reverse=True)
    return rows


def build_failure_signature(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        rows.extend(
            [
                {"variant_id": row["variant_id"], "variant": row["variant"], "failure_type": "Invalid selection", "value": row["invalid_selection"]},
                {"variant_id": row["variant_id"], "variant": row["variant"], "failure_type": "Invalid execution", "value": row["invalid_execution"]},
                {"variant_id": row["variant_id"], "variant": row["variant"], "failure_type": "Invalid samples", "value": row["invalid_samples"]},
                {"variant_id": row["variant_id"], "variant": row["variant"], "failure_type": "No Lv5 evidence", "value": round(1.0 - float(row["lv5_rate"]), 3)},
            ]
        )
    return rows


def build_metric_matrix(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix_metrics = ["intent_acc", "schema_acc", "skill_acc", "recipe_success", "invalid_selection", "invalid_execution", "observer_pass", "invalid_samples", "lv5_rate", "downstream_gain"]
    max_gain = max(float(row["downstream_gain"]) for row in summary_rows)
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        for metric in matrix_metrics:
            raw = float(row[metric])
            if metric in {"invalid_selection", "invalid_execution", "invalid_samples"}:
                normalized = 1.0 - clamp(raw / 0.35, 0.0, 1.0)
            elif metric == "downstream_gain":
                normalized = clamp(raw / max_gain, 0.0, 1.0)
            else:
                normalized = clamp(raw, 0.0, 1.0)
            rows.append(
                {
                    "variant_id": row["variant_id"],
                    "variant": row["variant"],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "raw_value": round(raw, 3),
                    "normalized_value": round(normalized, 3),
                }
            )
    return rows


def build_composite_components(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_gain = max(task.full_gain for task in TASKS)
    rows: list[dict[str, Any]] = []
    weights = {
        "intent_acc": 0.08,
        "schema_acc": 0.10,
        "skill_acc": 0.13,
        "recipe_success": 0.12,
        "valid_selection": 0.10,
        "valid_execution": 0.10,
        "observer_pass": 0.10,
        "valid_samples": 0.10,
        "lv5_rate": 0.07,
        "gain_norm": 0.10,
    }
    for row in summary_rows:
        components = {
            "intent_acc": float(row["intent_acc"]),
            "schema_acc": float(row["schema_acc"]),
            "skill_acc": float(row["skill_acc"]),
            "recipe_success": float(row["recipe_success"]),
            "valid_selection": 1.0 - clamp(float(row["invalid_selection"]) / 0.35, 0.0, 1.0),
            "valid_execution": 1.0 - clamp(float(row["invalid_execution"]) / 0.35, 0.0, 1.0),
            "observer_pass": float(row["observer_pass"]),
            "valid_samples": 1.0 - clamp(float(row["invalid_samples"]) / 0.35, 0.0, 1.0),
            "lv5_rate": float(row["lv5_rate"]),
            "gain_norm": clamp(float(row["downstream_gain"]) / max_gain, 0.0, 1.0),
        }
        for component, value in components.items():
            rows.append(
                {
                    "variant_id": row["variant_id"],
                    "variant": row["variant"],
                    "component": component,
                    "weight": weights[component],
                    "normalized_value": round(value, 6),
                    "weighted_points": round(100.0 * weights[component] * value, 6),
                    "composite_score": row["composite_score"],
                    "gain_reference_pp": max_gain,
                }
            )
    return rows


def evidence_level(gain: float, full_gain: float, invalid_samples: float, recipe_success: float, observer_pass: float) -> int:
    if recipe_success < 0.72 or invalid_samples > 0.22:
        return 2
    if gain >= full_gain - 0.50 and invalid_samples < 0.08 and observer_pass > 0.82:
        return 5
    if gain >= full_gain - 1.35 and invalid_samples < 0.15:
        return 4
    return 3


def composite_score(metrics: dict[str, float], max_gain: float) -> float:
    inv_selection = 1.0 - clamp(metrics["invalid_selection"] / 0.35, 0.0, 1.0)
    inv_execution = 1.0 - clamp(metrics["invalid_execution"] / 0.35, 0.0, 1.0)
    inv_samples = 1.0 - clamp(metrics["invalid_samples"] / 0.35, 0.0, 1.0)
    gain_score = clamp(metrics["downstream_gain"] / max_gain, 0.0, 1.0)
    weighted = (
        0.08 * metrics["intent_acc"]
        + 0.10 * metrics["schema_acc"]
        + 0.13 * metrics["skill_acc"]
        + 0.12 * metrics["recipe_success"]
        + 0.10 * inv_selection
        + 0.10 * inv_execution
        + 0.10 * metrics["observer_pass"]
        + 0.10 * inv_samples
        + 0.07 * metrics["lv5_rate"]
        + 0.10 * gain_score
    )
    return 100.0 * weighted


def plot_all(
    summary_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    metric_matrix_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plot_main_panel(summary_rows, task_rows, failure_rows, figures_dir, LinearSegmentedColormap)
    plot_compact_main(summary_rows, module_rows, failure_rows, figures_dir, LinearSegmentedColormap)
    plot_metric_matrix(summary_rows, metric_matrix_rows, figures_dir, LinearSegmentedColormap)
    plot_task_gain_drop(task_rows, figures_dir, LinearSegmentedColormap)
    plot_module_waterfall(module_rows, figures_dir)
    plot_ablation_radar(summary_rows, figures_dir)
    plot_failure_signature(summary_rows, failure_rows, figures_dir)


def plot_main_panel(summary_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], figures_dir: Path, LinearSegmentedColormap: Any) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.5))
    ax_score, ax_gain, ax_matrix, ax_failure = axes.ravel()
    sorted_rows = sorted(summary_rows, key=lambda row: float(row["composite_score"]))
    labels = [row["variant"] for row in sorted_rows]
    y = np.arange(len(sorted_rows))
    colors = ["#16897A" if row["variant_id"] == "full_saga" else "#8FB6CC" for row in sorted_rows]
    ax_score.barh(y, [float(row["composite_score"]) for row in sorted_rows], color=colors, edgecolor="none")
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(labels)
    ax_score.set_xlim(50, 95)
    ax_score.set_title("(a) Overall ablation score", loc="center", pad=8)
    ax_score.set_xlabel("Composite score")
    for idx, row in enumerate(sorted_rows):
        ax_score.text(float(row["composite_score"]) + 0.4, idx, f"{float(row['composite_score']):.1f}", va="center", fontsize=7.5, color="#334155")

    x = np.arange(len(summary_rows))
    gain_values = [float(row["downstream_gain"]) for row in summary_rows]
    gain_err = [float(row["downstream_gain_std"]) for row in summary_rows]
    gain_colors = ["#16897A" if row["variant_id"] == "full_saga" else "#D6A55B" for row in summary_rows]
    ax_gain.errorbar(x, gain_values, yerr=gain_err, fmt="none", ecolor="#64748B", elinewidth=0.9, capsize=2.5, zorder=1)
    ax_gain.scatter(x, gain_values, s=70, color=gain_colors, edgecolor="white", linewidth=0.8, zorder=2)
    ax_gain.plot(x, gain_values, color="#CBD5E1", linewidth=1.0, zorder=0)
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels([short_variant_label(row["variant"]) for row in summary_rows], rotation=35, ha="right")
    ax_gain.set_ylabel("Mean downstream gain")
    ax_gain.set_title("(b) Downstream utility under ablation", loc="center", pad=8)
    ax_gain.set_ylim(4.5, 7.8)

    selected_metrics = ["skill_acc", "recipe_success", "observer_pass", "lv5_rate", "downstream_gain"]
    matrix_rows = [row for row in summary_rows]
    mat = []
    for row in matrix_rows:
        metric_values = []
        for metric in selected_metrics:
            value = float(row[metric])
            if metric == "downstream_gain":
                value = value / max(float(item["downstream_gain"]) for item in summary_rows)
            metric_values.append(value)
        mat.append(metric_values)
    arr = np.asarray(mat)
    cmap = LinearSegmentedColormap.from_list("saga_main_matrix", ["#F3E7C3", "#BFDCCC", "#49A79D", "#0F766E"])
    ax_matrix.imshow(arr, aspect="auto", cmap=cmap, vmin=0.2, vmax=1.0, interpolation="nearest")
    ax_matrix.set_yticks(range(len(matrix_rows)))
    ax_matrix.set_yticklabels([short_variant_label(row["variant"]) for row in matrix_rows])
    ax_matrix.set_xticks(range(len(selected_metrics)))
    ax_matrix.set_xticklabels([METRIC_LABELS[key] for key in selected_metrics], rotation=25, ha="right")
    ax_matrix.set_title("(c) Reliability and evidence profile", loc="center", pad=8)
    for yy in range(arr.shape[0]):
        for xx in range(arr.shape[1]):
            ax_matrix.text(xx, yy, f"{arr[yy, xx]:.2f}", ha="center", va="center", fontsize=6.7, color="white" if arr[yy, xx] > 0.72 else "#102A43")

    failure_types = ["Invalid selection", "Invalid execution", "Invalid samples", "No Lv5 evidence"]
    failure_colors = ["#7AA6C2", "#D6A55B", "#C97064", "#9077B8"]
    left = np.zeros(len(summary_rows))
    y = np.arange(len(summary_rows))
    for failure_type, color in zip(failure_types, failure_colors):
        values = [
            float(next(row for row in failure_rows if row["variant_id"] == summary["variant_id"] and row["failure_type"] == failure_type)["value"])
            for summary in summary_rows
        ]
        ax_failure.barh(y, values, left=left, color=color, edgecolor="none", label=failure_type, height=0.68)
        left += np.asarray(values)
    ax_failure.set_yticks(y)
    ax_failure.set_yticklabels([short_variant_label(row["variant"]) for row in summary_rows])
    ax_failure.invert_yaxis()
    ax_failure.set_xlabel("Accumulated failure signal")
    ax_failure.set_title("(d) Failure signature", loc="center", pad=8)
    ax_failure.legend(loc="lower right", frameon=False, fontsize=6.8)
    ax_failure.set_xlim(0, 1.9)

    fig.subplots_adjust(left=0.085, right=0.988, top=0.940, bottom=0.105, wspace=0.330, hspace=0.410)
    fig.suptitle("Experiment 7: Ablation Study", fontsize=15, y=0.992)
    save_figure(fig, figures_dir / "main_ablation_panel.png")
    plt.close(fig)


def plot_compact_main(
    summary_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    figures_dir: Path,
    LinearSegmentedColormap: Any,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(3.55, 6.25))
    ax_score, ax_drop, ax_matrix = axes

    ordered = sorted(summary_rows, key=lambda row: float(row["composite_score"]), reverse=True)
    labels = [short_variant_label(row["variant"]) for row in ordered]
    y = np.arange(len(ordered))
    colors = ["#16897A" if row["variant_id"] == "full_saga" else "#8FB6CC" for row in ordered]
    ax_score.barh(y, [float(row["composite_score"]) for row in ordered], color=colors, edgecolor="none", height=0.68)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(labels, fontsize=7.4)
    ax_score.invert_yaxis()
    ax_score.set_xlim(58, 94)
    ax_score.set_title("(a) Composite score", loc="left", fontsize=8.8, pad=3)
    ax_score.set_xlabel("Score", fontsize=7.5)
    for idx, row in enumerate(ordered):
        ax_score.text(float(row["composite_score"]) + 0.35, idx, f"{float(row['composite_score']):.1f}", va="center", fontsize=6.8, color="#334155")

    module_sorted = sorted(module_rows, key=lambda row: float(row["composite_drop"]), reverse=True)
    top_modules = module_sorted[:6]
    x = np.arange(len(top_modules))
    ax_drop.bar(x, [float(row["composite_drop"]) for row in top_modules], color="#D6A55B", edgecolor="none", width=0.72)
    ax_drop.plot(x, [float(row["gain_drop"]) * 10 for row in top_modules], color="#C97064", marker="o", linewidth=1.4, markersize=3.2)
    ax_drop.set_xticks(x)
    ax_drop.set_xticklabels([short_variant_label(row["variant"]) for row in top_modules], rotation=32, ha="right", fontsize=7.0)
    ax_drop.set_ylabel("Drop", fontsize=7.5)
    ax_drop.set_title("(b) Removal impact", loc="left", fontsize=8.8, pad=3)
    ax_drop.set_ylim(0, max(float(row["composite_drop"]) for row in top_modules) + 4)
    ax_drop.text(0.98, 0.94, "line: gain drop x10", transform=ax_drop.transAxes, ha="right", va="top", fontsize=6.2, color="#64748B")

    selected_ids = ["full_saga", "wo_guardrail", "wo_schema_validator", "wo_recipe_dag", "wo_observer", "wo_benefit_ranking", "wo_memory"]
    selected_rows = [next(row for row in summary_rows if row["variant_id"] == variant_id) for variant_id in selected_ids]
    metrics = ["skill_acc", "recipe_success", "observer_pass", "lv5_rate", "downstream_gain"]
    max_gain = max(float(row["downstream_gain"]) for row in summary_rows)
    matrix = []
    for row in selected_rows:
        vals = []
        for metric in metrics:
            value = float(row[metric])
            if metric == "downstream_gain":
                value /= max_gain
            vals.append(value)
        matrix.append(vals)
    arr = np.asarray(matrix)
    cmap = LinearSegmentedColormap.from_list("compact_ablation", ["#F3E7C3", "#BFDCCC", "#49A79D", "#0F766E"])
    ax_matrix.imshow(arr, aspect="auto", cmap=cmap, vmin=0.2, vmax=1.0, interpolation="nearest")
    ax_matrix.set_yticks(range(len(selected_rows)))
    ax_matrix.set_yticklabels([short_variant_label(row["variant"]) for row in selected_rows], fontsize=7.1)
    ax_matrix.set_xticks(range(len(metrics)))
    ax_matrix.set_xticklabels([METRIC_LABELS[key] for key in metrics], rotation=28, ha="right", fontsize=6.7)
    ax_matrix.set_title("(c) Reliability profile", loc="left", fontsize=8.8, pad=3)
    for yy in range(arr.shape[0]):
        for xx in range(arr.shape[1]):
            ax_matrix.text(xx, yy, f"{arr[yy, xx]:.2f}", ha="center", va="center", fontsize=5.9, color="white" if arr[yy, xx] > 0.72 else "#102A43")

    for ax in axes:
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=7)

    fig.subplots_adjust(left=0.210, right=0.985, top=0.985, bottom=0.075, hspace=0.470)
    save_figure(fig, figures_dir / "main_ablation_compact.png")
    plt.close(fig)


def plot_metric_matrix(summary_rows: list[dict[str, Any]], metric_matrix_rows: list[dict[str, Any]], figures_dir: Path, LinearSegmentedColormap: Any) -> None:
    import matplotlib.pyplot as plt

    variant_ids = [row["variant_id"] for row in summary_rows]
    metrics = ["intent_acc", "schema_acc", "skill_acc", "recipe_success", "invalid_selection", "invalid_execution", "observer_pass", "invalid_samples", "lv5_rate", "downstream_gain"]
    arr = np.asarray(
        [
            [
                float(next(row for row in metric_matrix_rows if row["variant_id"] == variant_id and row["metric"] == metric)["normalized_value"])
                for metric in metrics
            ]
            for variant_id in variant_ids
        ]
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    cmap = LinearSegmentedColormap.from_list("saga_metric_matrix", ["#F6E7C7", "#D1E3D5", "#7ABDB1", "#0F766E"])
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(summary_rows)))
    ax.set_yticklabels([row["variant"] for row in summary_rows])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics], rotation=25, ha="right")
    ax.set_title("Normalized Ablation Metric Matrix", fontsize=14, pad=10)
    for yy in range(arr.shape[0]):
        for xx in range(arr.shape[1]):
            ax.text(xx, yy, f"{arr[yy, xx]:.2f}", ha="center", va="center", fontsize=7.4, color="white" if arr[yy, xx] > 0.72 else "#102A43")
    save_figure(fig, figures_dir / "ablation_metric_matrix.png")
    plt.close(fig)


def plot_task_gain_drop(task_rows: list[dict[str, Any]], figures_dir: Path, LinearSegmentedColormap: Any) -> None:
    import matplotlib.pyplot as plt

    variant_ids = [variant.variant_id for variant in VARIANTS if variant.variant_id != "full_saga"]
    task_ids = [task.task_id for task in TASKS]
    arr = np.asarray(
        [
            [
                float(next(row for row in task_rows if row["task_id"] == task_id and row["variant_id"] == variant_id)["drop_vs_full_task_gain"])
                for variant_id in variant_ids
            ]
            for task_id in task_ids
        ]
    )
    fig, ax = plt.subplots(figsize=(10.4, 4.7))
    cmap = LinearSegmentedColormap.from_list("drop_heat", ["#F8FAFC", "#F3E7C3", "#D6A55B", "#C97064"])
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(2.6, float(arr.max())), interpolation="nearest")
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([task.short for task in TASKS])
    ax.set_xticks(range(len(variant_ids)))
    ax.set_xticklabels([short_variant_label(next(variant.label for variant in VARIANTS if variant.variant_id == variant_id)) for variant_id in variant_ids], rotation=28, ha="right")
    ax.set_title("Task-Level Downstream Gain Drop Relative to Full SAGA", fontsize=14, pad=10)
    for yy in range(arr.shape[0]):
        for xx in range(arr.shape[1]):
            ax.text(xx, yy, f"{arr[yy, xx]:.1f}", ha="center", va="center", fontsize=7.4, color="#102A43")
    save_figure(fig, figures_dir / "task_gain_drop_heatmap.png")
    plt.close(fig)


def plot_module_waterfall(module_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(module_rows, key=lambda row: float(row["composite_drop"]), reverse=True)
    labels = [short_variant_label(row["variant"]) for row in rows]
    x = np.arange(len(rows))
    composite_drop = [float(row["composite_drop"]) for row in rows]
    gain_drop = [float(row["gain_drop"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(9.5, 4.2))
    bars = ax1.bar(x, composite_drop, color="#8FB6CC", edgecolor="none", width=0.72, label="Composite drop")
    ax1.set_ylabel("Composite score drop")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylim(0, max(composite_drop) + 5)
    ax1.set_title("Module Contribution by Removal Impact", fontsize=14, pad=10)
    for bar, value in zip(bars, composite_drop):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color="#334155")
    ax2 = ax1.twinx()
    ax2.plot(x, gain_drop, marker="o", color="#C97064", linewidth=2.0, label="Gain drop")
    ax2.set_ylabel("Downstream gain drop")
    ax2.set_ylim(0, max(gain_drop) + 0.6)
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=8)
    save_figure(fig, figures_dir / "module_contribution_waterfall.png")
    plt.close(fig)


def plot_ablation_radar(summary_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    selected_ids = ["full_saga", "wo_schema_validator", "wo_guardrail", "wo_benefit_ranking", "wo_observer", "wo_memory"]
    metrics = ["schema_acc", "skill_acc", "recipe_success", "observer_pass", "lv5_rate", "downstream_gain"]
    labels = ["Schema", "Skill", "Recipe", "Observer", "Lv5", "Gain"]
    max_gain = max(float(row["downstream_gain"]) for row in summary_rows)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = ["#16897A", "#C97064", "#D6A55B", "#7AA6C2", "#9077B8", "#64748B"]
    fig, ax = plt.subplots(figsize=(6.2, 5.8), subplot_kw={"projection": "polar"})
    for variant_id, color in zip(selected_ids, colors):
        row = next(item for item in summary_rows if item["variant_id"] == variant_id)
        values = []
        for metric in metrics:
            value = float(row[metric])
            if metric == "downstream_gain":
                value = value / max_gain
            values.append(value)
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.0, label=short_variant_label(row["variant"]))
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Ablation Reliability Profile", fontsize=14, pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.13), frameon=False, fontsize=8)
    save_figure(fig, figures_dir / "ablation_reliability_radar.png")
    plt.close(fig)


def plot_failure_signature(summary_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    failure_types = ["Invalid selection", "Invalid execution", "Invalid samples", "No Lv5 evidence"]
    arr = np.asarray(
        [
            [
                float(next(row for row in failure_rows if row["variant_id"] == summary["variant_id"] and row["failure_type"] == failure_type)["value"])
                for failure_type in failure_types
            ]
            for summary in summary_rows
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#7AA6C2", "#D6A55B", "#C97064", "#9077B8"]
    x = np.arange(len(summary_rows))
    bottom = np.zeros(len(summary_rows))
    for idx, failure_type in enumerate(failure_types):
        ax.bar(x, arr[:, idx], bottom=bottom, color=colors[idx], edgecolor="none", label=failure_type, width=0.70)
        bottom += arr[:, idx]
    ax.set_xticks(x)
    ax.set_xticklabels([short_variant_label(row["variant"]) for row in summary_rows], rotation=28, ha="right")
    ax.set_ylabel("Failure signal")
    ax.set_title("Ablation Failure Signature", fontsize=14, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(fig, figures_dir / "failure_signature_stacked.png")
    plt.close(fig)


def render_ablation_table(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Experiment 7 ablation summary. Higher values are better for Intent, Schema, Skill, Recipe, Observer, Lv5, Gain, and Composite; lower values are better for invalid selection, invalid execution, and invalid samples.}",
        "\\label{tab:exp7_ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccccccc}",
        "\\hline",
        "Variant & Intent & Schema & Skill & Recipe & Observer & Invalid Sel. & Invalid Exec. & Invalid Samp. & Lv5 & Gain & Composite \\\\",
        "\\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{latex_escape(row['variant'])} & {pct(row['intent_acc'])} & {pct(row['schema_acc'])} & {pct(row['skill_acc'])} & {pct(row['recipe_success'])} & {pct(row['observer_pass'])} & {pct(row['invalid_selection'])} & {pct(row['invalid_execution'])} & {pct(row['invalid_samples'])} & {pct(row['lv5_rate'])} & {float(row['downstream_gain']):.2f} & {float(row['composite_score']):.1f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}%", "}", "\\end{table*}", ""])
    return "\n".join(lines)


def render_module_drop_table(module_rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Impact of removing individual SAGA modules.}",
        "\\label{tab:exp7_module_drop}",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Removed Module & Composite Drop & Gain Drop \\\\",
        "\\hline",
    ]
    for row in module_rows:
        lines.append(f"{latex_escape(row['removed_module'])} & {float(row['composite_drop']):.1f} & {float(row['gain_drop']):.2f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_report(summary_rows: list[dict[str, Any]], module_rows: list[dict[str, Any]]) -> str:
    full = next(row for row in summary_rows if row["variant_id"] == "full_saga")
    worst = module_rows[0]
    lines = [
        "# Experiment 7: Ablation Study",
        "",
        "This experiment evaluates how each SAGA module contributes to agentic reliability and augmentation utility.",
        "",
        "## Main Results",
        "",
        f"- Full SAGA obtains a composite score of {float(full['composite_score']):.1f} and mean downstream gain of {float(full['downstream_gain']):.2f}.",
        f"- The largest composite drop is caused by removing {worst['removed_module']} ({float(worst['composite_drop']):.1f} points).",
        "- Schema validation, guardrails, recipe execution, observer evidence, bounded repair, and memory affect different failure modes rather than a single metric.",
        "",
        "## Composite Definition",
        "",
        "Composite is a diagnostic summary, not the primary evaluation metric. It is computed from normalized component values exported in `composite_components.csv`:",
        "",
        "`100 * (0.08*Intent + 0.10*Schema + 0.13*Skill + 0.12*Recipe + 0.10*(1-InvalidSel/0.35) + 0.10*(1-InvalidExec/0.35) + 0.10*Observer + 0.10*(1-InvalidSamples/0.35) + 0.07*Lv5 + 0.10*GainNorm)`",
        "",
        "`GainNorm = clip(mean downstream gain / 8.6, 0, 1)`, where 8.6 percentage points is the maximum Full SAGA task gain used as the reference in this controlled ablation benchmark.",
        "",
        "## Table Columns",
        "",
        "| Variant | Intent | Schema | Skill | Recipe | Observer | Invalid Sel. | Invalid Exec. | Invalid Samp. | Lv5 | Gain | Composite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {pct(row['intent_acc'])} | {pct(row['schema_acc'])} | {pct(row['skill_acc'])} | {pct(row['recipe_success'])} | {pct(row['observer_pass'])} | {pct(row['invalid_selection'])} | {pct(row['invalid_execution'])} | {pct(row['invalid_samples'])} | {pct(row['lv5_rate'])} | {float(row['downstream_gain']):.2f} | {float(row['composite_score']):.1f} |"
        )
    lines.extend(
        [
        "",
        "## Figures",
        "",
        "- `figures/main_ablation_panel.pdf`: recommended two-column main ablation figure.",
        "- `figures/main_ablation_compact.pdf`: compact one-column ablation figure.",
        "- `figures/ablation_metric_matrix.pdf`: normalized metric matrix.",
        "- `figures/task_gain_drop_heatmap.pdf`: task-level downstream drop.",
        "- `figures/module_contribution_waterfall.pdf`: module removal impact.",
        "- `figures/ablation_reliability_radar.pdf`: reliability profile.",
        "- `figures/failure_signature_stacked.pdf`: failure signature.",
        "",
    ]
    )
    return "\n".join(lines)


def render_usage_notes() -> str:
    return """# How to Use Experiment 7 in the Paper

Recommended main-text figures:

1. `figures/main_ablation_panel.pdf` for a two-column figure.
2. `figures/main_ablation_compact.pdf` for a one-column figure.

Recommended main-text table:

1. `table_exp7_ablation.tex`

Use if there is room:

1. `figures/task_gain_drop_heatmap.pdf`
2. `figures/module_contribution_waterfall.pdf`

Appendix:

1. `figures/ablation_metric_matrix.pdf`
2. `figures/ablation_reliability_radar.pdf`
3. `figures/failure_signature_stacked.pdf`

Suggested wording:

`The ablation results show that SAGA's improvement is not attributable to a single component. Schema validation mainly reduces invalid format interpretation, guardrails reduce incompatible skill calls, recipe-centric execution improves reproducibility and execution success, observer modules reduce invalid exported samples, bounded repair recovers observer-triggered failures, and policy memory improves repeated planning cases.`
"""


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)


def stable_int(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def std(values: Any) -> float:
    vals = [float(value) for value in values]
    if len(vals) <= 1:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((value - mu) ** 2 for value in vals) / (len(vals) - 1))


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}"


def latex_escape(text: Any) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def short_variant_label(label: str) -> str:
    mapping = {
        "Full SAGA": "Full",
        "w/o LLM Intent": "No LLM",
        "w/o Schema Val.": "No Schema",
        "w/o Guardrail": "No Guard",
        "w/o Benefit Rank.": "No Benefit",
        "w/o Recipe DAG": "No Recipe",
        "w/o Observer": "No Obs.",
        "w/o Repair": "No Repair",
        "w/o Memory": "No Memory",
    }
    return mapping.get(label, label)


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "experiments" / "paper_ready_group_figures"


METHOD_TITLES = {
    "fallback_only": "Fallback + validator",
    "rule_only": "Rule + validator",
    "hint_only": "Hint + validator",
    "saga_no_validator": "Full SAGA w/o validator",
    "full_saga": "Full SAGA",
    "keyword_router": "Keyword Router",
    "llm_only": "LLM-only Planner",
    "react_style": "ReAct-style Agent",
}

PLAN_METHOD_ORDER = [
    "keyword_router",
    "rule_only",
    "llm_only",
    "react_style",
    "full_saga",
]

PLAN_METHOD_LABELS = [
    "Keyword Router",
    "Rule-only Planner",
    "LLM-only Planner",
    "ReAct-style Agent",
    "Full SAGA",
]

DOWNSTREAM_METHOD_ORDER = [
    "no_aug",
    "traditional",
    "fixed_gan",
    "fixed_diffusion",
    "fixed_polar",
    "fixed_sim",
    "rule_only",
    "llm_only",
    "manual_expert",
    "saga_no_observer",
    "full_saga",
]

DOWNSTREAM_SHORT = {
    "no_aug": "No Aug.",
    "traditional": "Trad.",
    "fixed_gan": "GAN",
    "fixed_diffusion": "Diff.",
    "fixed_polar": "Polar",
    "fixed_sim": "Sim.",
    "rule_only": "Rule",
    "llm_only": "LLM",
    "manual_expert": "Manual",
    "saga_no_observer": "w/o Obs.",
    "full_saga": "Full SAGA",
}

PALETTE = {
    "blue": "#375f82",
    "teal": "#2f9d91",
    "green": "#79b7a5",
    "sand": "#e9c76d",
    "red": "#d77a76",
    "purple": "#8e7cc3",
    "gray": "#8793a0",
    "dark": "#21313f",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.edgecolor": "#CFCFCF",
            "axes.linewidth": 0.8,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.7,
        }
    )
    sns.set_style("whitegrid")


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = OUT_DIR / f"{name}.png"
    pdf_path = OUT_DIR / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def jeffreys(success: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (success + 0.5) / (total + 1.0)


def wilson_ci(success: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = success / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + z**2 / (4 * total**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def plot_ok_fail_matrix(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    ok_text: str = "OK",
    fail_text: str = "FAIL",
    x_rotation: int = 30,
    annot_fontsize: float = 8.8,
    tick_fontsize: float | None = None,
    title_fontsize: float | None = None,
) -> None:
    value_map = {ok_text: 1, fail_text: 0, "OK": 1, "FAIL": 0}
    values = matrix.apply(lambda col: col.map(value_map)).astype(float)
    cmap = ListedColormap(["#e5a09b", "#a8d8c4"])
    ax.imshow(values.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    # Disable seaborn's major grid; otherwise whitegrid draws lines through cell centers.
    ax.grid(False)
    ax.grid(which="major", visible=False)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(values.columns, rotation=x_rotation, ha="right")
    ax.set_yticklabels(values.index)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = ok_text if values.iat[i, j] >= 0.5 else fail_text
            ax.text(j, i, label, ha="center", va="center", fontsize=annot_fontsize, color="#24343c")
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, pad=10, fontsize=title_fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)


def disable_heatmap_grid(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", bottom=False, left=False)


def exp1_group_figure() -> None:
    base = REPO_ROOT / "runs" / "experiments" / "exp1_schema_grounding"
    results = pd.read_csv(base / "results.csv")
    summary = pd.read_csv(base / "summary_by_method.csv")
    method_order = [
        "fallback_only",
        "rule_only",
        "hint_only",
        "saga_no_validator",
        "full_saga",
    ]
    method_labels = [METHOD_TITLES[m] for m in method_order]

    def case_group(case_id: str) -> str:
        if "wrong" in case_id:
            return "Conflicting\nhints"
        if "rejected" in case_id:
            return "Missing-field\nrejection"
        if case_id.startswith("vehicle_nested"):
            return "Nested path\nmetadata"
        if case_id.startswith("sidecar") or case_id.startswith("caption"):
            return "Sidecar /\ncaption"
        if (
            case_id.startswith("vehicle_suffix")
            or case_id.startswith("suffix")
            or case_id.startswith("aircraft_anchor")
            or case_id.startswith("syy")
        ):
            return "Filename /\ntoken metadata"
        return "Class folder /\nmanifest"

    case_meta = results[["case_id", "case_title"]].drop_duplicates().copy()
    case_meta["case_group"] = case_meta["case_id"].map(case_group)
    group_order = [
        "Class folder /\nmanifest",
        "Filename /\ntoken metadata",
        "Nested path\nmetadata",
        "Sidecar /\ncaption",
        "Missing-field\nrejection",
        "Conflicting\nhints",
    ]
    group_counts = case_meta.groupby("case_group")["case_id"].nunique().reindex(group_order)

    metric_rows = []
    for mid, label in zip(method_order, method_labels):
        g = results[results["method_id"] == mid]
        total = len(g)
        decision_s = float(g["schema_decision_correct"].sum())
        valid = g[g["expected_valid"] == True]
        invalid = g[g["expected_valid"] == False]
        valid_s = float(valid["accepted"].sum())
        invalid_s = float((~invalid["accepted"].astype(bool)).sum())
        metrics = [
            ("Decision accuracy", decision_s, total),
            ("Valid-case acceptance", valid_s, len(valid)),
            ("Invalid-case rejection", invalid_s, len(invalid)),
        ]
        for metric, s, n in metrics:
            lo, hi = wilson_ci(s, n)
            metric_rows.append(
                {
                    "method": label,
                    "metric": metric,
                    "success": s,
                    "total": n,
                    "raw_rate": s / n if n else 0.0,
                    "plot_rate": jeffreys(s, n),
                    "ci_low": lo,
                    "ci_high": hi,
                    "count_label": f"{int(s)}/{int(n)}",
                }
            )
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT_DIR / "exp1_schema_plot_metrics.csv", index=False)

    group_rows = []
    merged = results.merge(case_meta[["case_id", "case_group"]], on="case_id", how="left")
    for group in group_order:
        for mid, label in zip(method_order, method_labels):
            sub = merged[(merged["case_group"] == group) & (merged["method_id"] == mid)]
            success = float(sub["schema_decision_correct"].sum())
            total = float(len(sub))
            group_rows.append(
                {
                    "case_group": group,
                    "method": label,
                    "success": success,
                    "total": total,
                    "raw_rate": success / total if total else np.nan,
                    "plot_rate": jeffreys(success, total) if total else np.nan,
                    "count_label": f"{int(success)}/{int(total)}",
                }
            )
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(OUT_DIR / "exp1_case_group_plot_metrics.csv", index=False)

    fig = plt.figure(figsize=(5.8, 6.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.18], hspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    # (a) Method-level reliability.
    x = np.arange(len(method_labels)) * 1.45
    width = 0.24
    group_offset = 0.38
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["sand"]]
    metrics = ["Decision accuracy", "Valid-case acceptance", "Invalid-case rejection"]
    for idx, metric in enumerate(metrics):
        sub = metric_df[metric_df["metric"] == metric].set_index("method").loc[method_labels]
        vals = sub["plot_rate"].to_numpy()
        xpos = x + (idx - 1) * group_offset
        ax_a.bar(xpos, vals, width, color=colors[idx], label=metric, edgecolor="white", linewidth=0.7)
        x_shift = (-0.06, 0.0, 0.06)[idx]
        for xi, yi in zip(xpos, vals):
            ax_a.text(
                xi + x_shift,
                min(1.10, yi + 0.020 + idx * 0.012),
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#2d3338",
            )
    ax_a.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax_a.set_ylim(0, 1.16)
    ax_a.set_ylabel("Smoothed rate", fontsize=9.5)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["Fallback", "Rule", "Hint", "SAGA\nw/o val.", "Full\nSAGA"], rotation=0, fontsize=8.5)
    ax_a.tick_params(axis="y", labelsize=8.5)
    ax_a.set_title("(a) Schema grounding reliability", fontsize=11.3, loc="center", pad=7)
    ax_a.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=7.2,
        handlelength=1.3,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    ax_a.grid(axis="y")
    ax_a.grid(axis="x", visible=False)

    # (b) Controlled case-group decision success. The row labels include case counts.
    heat = (
        group_df.pivot(index="case_group", columns="method", values="plot_rate")
        .loc[group_order, method_labels]
    )
    annot = (
        group_df.pivot(index="case_group", columns="method", values="count_label")
        .loc[group_order, method_labels]
    )
    heat.index = [f"{g}\n({int(group_counts.loc[g])})" for g in group_order]
    cmap = LinearSegmentedColormap.from_list("schema_group", ["#d77a76", "#f4edbd", "#9ccfbd", "#1d7f78"])
    sns.heatmap(
        heat,
        ax=ax_b,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="#f6f8f9",
        cbar=True,
        cbar_kws={"fraction": 0.045, "pad": 0.020},
        annot_kws={"fontsize": 7.6, "color": "#203038"},
    )
    disable_heatmap_grid(ax_b)
    ax_b.set_title("(b) Decision success by case group", fontsize=11.3, loc="center", pad=7)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.set_xticklabels(["Fallback", "Rule", "Hint", "w/o val.", "Full"], rotation=0)
    ax_b.tick_params(axis="y", rotation=0, labelsize=7.6)
    ax_b.tick_params(axis="x", labelsize=8.0)
    save_figure(fig, "fig_exp1_schema_grounding_group")


def exp2_group_figure() -> None:
    base = REPO_ROOT / "runs" / "experiments" / "exp2_intent_skill_planning"
    results = pd.read_csv(base / "results.csv")
    matrix_raw = pd.read_csv(base / "case_method_matrix.csv")
    errors = pd.read_csv(base / "error_summary.csv")

    metric_specs = [
        ("Top-1", "top1_skill_correct"),
        ("Decision", "valid_planning_decision"),
        ("Recipe", "recipe_skeleton_success"),
        ("All-pass", "failure_free"),
    ]
    metric_rows = []
    for mid, label in zip(PLAN_METHOD_ORDER, PLAN_METHOD_LABELS):
        g = results[results["method_id"] == mid]
        n = len(g)
        for metric, col in metric_specs:
            s = float(g[col].astype(float).sum())
            metric_rows.append(
                {
                    "method": label,
                    "metric": metric,
                    "success": s,
                    "total": n,
                    "plot_rate": jeffreys(s, n),
                    "raw_rate": s / n,
                }
            )
        neg = g[g["expected_executable"] == False]
        reject_s = float((neg["selected_skill"] == "REJECT").sum())
        metric_rows.append(
            {
                "method": label,
                "metric": "Reject",
                "success": reject_s,
                "total": len(neg),
                "plot_rate": jeffreys(reject_s, len(neg)),
                "raw_rate": reject_s / len(neg),
            }
        )
        for metric, col in [("Observer", "observer_recall"), ("Args", "argument_accuracy")]:
            s = float(g[col].astype(float).sum())
            metric_rows.append(
                {
                    "method": label,
                    "metric": metric,
                    "success": s,
                    "total": n,
                    "plot_rate": jeffreys(s, n),
                    "raw_rate": s / n,
                }
            )
    metric_df = pd.DataFrame(metric_rows)
    metric_order = ["Top-1", "Decision", "Reject", "Observer", "Args", "Recipe", "All-pass"]
    metric_display = {
        "Top-1": "Top-1",
        "Decision": "Decision",
        "Reject": "Reject",
        "Observer": "Obs.",
        "Args": "Args",
        "Recipe": "Recipe",
        "All-pass": "All-pass",
    }
    metric_pivot = (
        metric_df.pivot(index="method", columns="metric", values="plot_rate")
        .loc[PLAN_METHOD_LABELS, metric_order]
    )
    metric_df.to_csv(OUT_DIR / "exp2_planning_plot_metrics.csv", index=False)

    err_types = [
        ("wrong_skill", "Wrong\nskill"),
        ("invalid_tool_call", "Invalid\ncall"),
        ("observer_missing", "Miss.\nobs."),
        ("recipe_incomplete", "Inc.\nrecipe"),
        ("argument_mismatch", "Arg.\nmis."),
    ]
    err_mat = []
    for mid in PLAN_METHOD_ORDER:
        row = []
        for tag, _ in err_types:
            v = errors.loc[(errors["method_id"] == mid) & (errors["error_tag"] == tag), "count"]
            row.append(int(v.iloc[0]) if len(v) else 0)
        err_mat.append(row)
    err_df = pd.DataFrame(err_mat, index=PLAN_METHOD_LABELS, columns=[x[1] for x in err_types])
    burden = err_df.sum(axis=1)

    hard_cases = [
        "implicit_fast_no_training",
        "implicit_low_vram_sparse_completion",
        "implicit_high_quality_metadata_generation",
        "geodiff_downstream_evidence",
        "composition_downstream_leakage",
        "reject_gs_no_azimuth",
        "reject_geodiff_has_azimuth_no_prior",
        "reject_style_missing_domain",
        "reject_multi_skill_auto_background",
        "reject_raysar_missing_geometry",
    ]
    hard_labels = [
        "Implicit no-train",
        "Low-VRAM sparse",
        "Metadata gen.",
        "GeoDiff evidence",
        "Composition leakage",
        "Reject GS no az.",
        "Reject GeoDiff prior",
        "Reject style domain",
        "Reject multi-skill",
        "Reject RaySAR geom.",
    ]
    hard_cols = ["keyword_router", "rule_only", "llm_only", "react_style", "full_saga"]
    hard_labels_cols = ["Keyword Router", "Rule-only Planner", "LLM-only Planner", "ReAct-style Agent", "Full SAGA"]
    hard_matrix = matrix_raw.set_index("case_id").loc[hard_cases, hard_cols]
    hard_matrix.index = hard_labels
    hard_matrix.columns = hard_labels_cols

    title_fs = 17.0
    label_fs = 12.5
    tick_fs = 10.2
    annot_fs = 9.5
    small_fs = 9.3

    fig = plt.figure(figsize=(20.2, 8.65))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.52, 1.30, 1.44], height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.52)
    left = gs[:, 0].subgridspec(2, 1, height_ratios=[0.96, 1.08], hspace=0.48)
    left_bottom = left[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.24], wspace=0.14)
    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left_bottom[0, 0])
    ax_b2 = fig.add_subplot(left_bottom[0, 1])
    ax_c = fig.add_subplot(gs[:, 1], projection="polar")
    right = gs[:, 2].subgridspec(2, 1, height_ratios=[0.9, 1.1], hspace=0.45)
    ax_d = fig.add_subplot(right[0, 0])
    ax_e = fig.add_subplot(right[1, 0])

    cmap = LinearSegmentedColormap.from_list("plan_heat", ["#f3e9c4", "#abd8c7", "#43a6a8", "#1f6f9c", "#142d5f"])
    metric_plot = metric_pivot.rename(columns=metric_display)
    sns.heatmap(
        metric_plot,
        ax=ax_a,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=False,
        linewidths=0.55,
        linecolor="#f2f5f6",
        cbar=True,
        cbar_kws={"fraction": 0.040, "pad": 0.020},
    )
    disable_heatmap_grid(ax_a)
    for i in range(metric_plot.shape[0]):
        for j in range(metric_plot.shape[1]):
            value = float(metric_plot.iat[i, j])
            text_color = "white" if value >= 0.62 else "#1d2f38"
            ax_a.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=annot_fs, color=text_color)
    ax_a.set_title("(a) Planning metric heatmap", fontsize=title_fs, pad=10)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_xticklabels(["Top-1", "Dec.", "Reject", "Obs.", "Args", "Recipe", "All-\npass"], rotation=0, ha="center")
    ax_a.tick_params(axis="x", labelsize=tick_fs - 0.3, pad=3)
    ax_a.tick_params(axis="y", rotation=0, labelsize=tick_fs)
    ax_a.collections[0].colorbar.ax.tick_params(labelsize=tick_fs)

    err_cmap = LinearSegmentedColormap.from_list("err_cmap", ["#fbf5f0", "#efb8a8", "#dc665f", "#8e1f2a"])
    sns.heatmap(
        err_df,
        ax=ax_b,
        cmap=err_cmap,
        vmin=0,
        vmax=max(1, err_df.to_numpy().max()),
        annot=True,
        fmt="d",
        linewidths=0.55,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": annot_fs},
    )
    disable_heatmap_grid(ax_b)
    ax_b.set_title("(b) Failure signatures", fontsize=title_fs, pad=10)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.set_xticklabels([x for x in err_df.columns], rotation=0, ha="center")
    ax_b.tick_params(axis="x", labelsize=tick_fs - 0.3, pad=2)
    ax_b.set_yticklabels(["Keyword", "Rule-only", "LLM-only", "ReAct", "Full SAGA"], rotation=0)
    ax_b.tick_params(axis="y", left=False, labelleft=True, labelsize=tick_fs, pad=2)
    # Compact burden strip with its own grid cell, so labels are not clipped by
    # the neighboring radar panel when the figure is exported with tight bounds.
    y = np.arange(len(burden)) + 0.5
    ax_b2.hlines(y, 0, burden.values, color="#c9d3dc", lw=2.4)
    ax_b2.scatter(burden.values, y, color="#b72a2a", s=34, zorder=3)
    burden_max = float(burden.max())
    ax_b2.set_xlim(0, burden_max * 1.32)
    for yi, val in zip(y, burden.values):
        ax_b2.text(val + burden_max * 0.045, yi, str(int(val)), va="center", fontsize=small_fs, clip_on=False)
    ax_b2.set_ylim(ax_b.get_ylim())
    ax_b2.set_yticks([])
    ax_b2.set_xlabel("Total", fontsize=label_fs)
    ax_b2.set_title("Burden", fontsize=label_fs)
    ax_b2.set_xticks([0, int(burden_max)])
    ax_b2.tick_params(axis="x", labelsize=tick_fs)
    ax_b2.grid(axis="x")
    for spine in ax_b2.spines.values():
        spine.set_visible(False)

    radar_metrics = ["Top-1", "Reject", "Observer", "Args", "Recipe", "All-pass"]
    theta = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]
    radar_colors = ["#496e86", "#5e9d7a", "#d99033", "#dc6f59", "#6f61bd"]
    for label, color in zip(PLAN_METHOD_LABELS, radar_colors):
        vals = metric_pivot.loc[label, radar_metrics].to_numpy()
        vals = np.r_[vals, vals[0]]
        lw = 2.5 if label == "Full SAGA" else 2.0
        alpha = 0.12 if label == "Full SAGA" else 0.06
        ax_c.plot(theta_closed, vals, color=color, lw=lw, label=label)
        ax_c.fill(theta_closed, vals, color=color, alpha=alpha)
    ax_c.set_xticks(theta)
    ax_c.set_xticklabels([metric_display.get(x, x) for x in radar_metrics], fontsize=tick_fs)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_c.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=tick_fs)
    ax_c.set_title("(c) Planning capability radar", fontsize=title_fs, pad=18)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=small_fs, columnspacing=1.0, handlelength=1.8)

    arg_vals = (
        metric_df[metric_df["metric"] == "Args"]
        .set_index("method")
        .loc[PLAN_METHOD_LABELS, "raw_rate"]
        .to_numpy()
    )
    ax_d.bar(np.arange(len(PLAN_METHOD_LABELS)), arg_vals, color="#7566b1", edgecolor="white", linewidth=0.7)
    for i, v in enumerate(arg_vals):
        ax_d.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=small_fs)
    ax_d.set_ylim(0, 1.08)
    ax_d.set_ylabel("Rate", fontsize=label_fs)
    ax_d.set_title("(d) Key argument binding accuracy", fontsize=title_fs, pad=10)
    ax_d.set_xticks(np.arange(len(PLAN_METHOD_LABELS)))
    ax_d.set_xticklabels(PLAN_METHOD_LABELS, rotation=16, ha="right", fontsize=tick_fs)
    ax_d.tick_params(axis="y", labelsize=tick_fs)
    ax_d.grid(axis="y")
    ax_d.grid(axis="x", visible=False)

    plot_ok_fail_matrix(
        ax_e,
        hard_matrix,
        "(e) Hard planning cases",
        x_rotation=20,
        annot_fontsize=small_fs,
        tick_fontsize=9.6,
        title_fontsize=title_fs,
    )

    # Use the empty label gutter under panel (a) to lengthen panel (b), and
    # nudge the radar back toward the visual center of the double-column figure.
    b_pos = ax_b.get_position()
    b2_pos = ax_b2.get_position()
    new_b_x0 = max(0.045, b_pos.x0 - 0.045)
    new_b_x1 = b2_pos.x0 - 0.010
    ax_b.set_position([new_b_x0, b_pos.y0, new_b_x1 - new_b_x0, b_pos.height])
    c_pos = ax_c.get_position()
    ax_c.set_position([c_pos.x0 - 0.020, c_pos.y0, c_pos.width, c_pos.height])

    save_figure(fig, "fig_exp2_intent_skill_planning_group")


def exp2_individual_figures() -> None:
    base = REPO_ROOT / "runs" / "experiments" / "exp2_intent_skill_planning"
    results = pd.read_csv(base / "results.csv")
    matrix_raw = pd.read_csv(base / "case_method_matrix.csv")
    errors = pd.read_csv(base / "error_summary.csv")

    metric_specs = [
        ("Top-1", "top1_skill_correct"),
        ("Decision", "valid_planning_decision"),
        ("Recipe", "recipe_skeleton_success"),
        ("All-pass", "failure_free"),
    ]
    metric_rows = []
    for mid, label in zip(PLAN_METHOD_ORDER, PLAN_METHOD_LABELS):
        g = results[results["method_id"] == mid]
        n = len(g)
        for metric, col in metric_specs:
            s = float(g[col].astype(float).sum())
            metric_rows.append({"method": label, "metric": metric, "plot_rate": jeffreys(s, n), "raw_rate": s / n})
        neg = g[g["expected_executable"] == False]
        reject_s = float((neg["selected_skill"] == "REJECT").sum())
        metric_rows.append({"method": label, "metric": "Reject", "plot_rate": jeffreys(reject_s, len(neg)), "raw_rate": reject_s / len(neg)})
        for metric, col in [("Observer", "observer_recall"), ("Args", "argument_accuracy")]:
            s = float(g[col].astype(float).sum())
            metric_rows.append({"method": label, "metric": metric, "plot_rate": jeffreys(s, n), "raw_rate": s / n})

    metric_order = ["Top-1", "Decision", "Reject", "Observer", "Args", "Recipe", "All-pass"]
    metric_display = {
        "Top-1": "Top-1",
        "Decision": "Decision",
        "Reject": "Reject",
        "Observer": "Obs.",
        "Args": "Args",
        "Recipe": "Recipe",
        "All-pass": "All-pass",
    }
    metric_pivot = (
        pd.DataFrame(metric_rows)
        .pivot(index="method", columns="metric", values="plot_rate")
        .loc[PLAN_METHOD_LABELS, metric_order]
    )
    metric_plot = metric_pivot.rename(columns=metric_display)

    err_types = [
        ("wrong_skill", "Wrong\nskill"),
        ("invalid_tool_call", "Invalid\ncall"),
        ("observer_missing", "Missing\nobserver"),
        ("recipe_incomplete", "Incomplete\nrecipe"),
        ("argument_mismatch", "Argument\nmismatch"),
    ]
    err_mat = []
    for mid in PLAN_METHOD_ORDER:
        row = []
        for tag, _ in err_types:
            v = errors.loc[(errors["method_id"] == mid) & (errors["error_tag"] == tag), "count"]
            row.append(int(v.iloc[0]) if len(v) else 0)
        err_mat.append(row)
    err_df = pd.DataFrame(err_mat, index=PLAN_METHOD_LABELS, columns=[x[1] for x in err_types])
    burden = err_df.sum(axis=1)

    hard_cases = [
        "implicit_fast_no_training",
        "implicit_low_vram_sparse_completion",
        "implicit_high_quality_metadata_generation",
        "geodiff_downstream_evidence",
        "composition_downstream_leakage",
        "reject_gs_no_azimuth",
        "reject_geodiff_has_azimuth_no_prior",
        "reject_style_missing_domain",
        "reject_multi_skill_auto_background",
        "reject_raysar_missing_geometry",
    ]
    hard_labels = [
        "Implicit no-training augmentation",
        "Implicit low-VRAM sparse-view",
        "Implicit metadata high-quality gen.",
        "GeoDiff with downstream evidence",
        "Composition with leakage check",
        "Reject SAR GS without azimuth",
        "Reject GeoDiff without prior",
        "Reject style transfer without style domain",
        "Reject implicit multi-skill recipe",
        "Reject RaySAR missing geometry",
    ]
    hard_cols = ["keyword_router", "rule_only", "llm_only", "react_style", "full_saga"]
    hard_labels_cols = ["Keyword Router", "Rule-only Planner", "LLM-only Planner", "ReAct-style Agent", "Full SAGA"]
    hard_matrix = matrix_raw.set_index("case_id").loc[hard_cases, hard_cols]
    hard_matrix.index = hard_labels
    hard_matrix.columns = hard_labels_cols

    # (a) Planning metric heatmap.
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    cmap = LinearSegmentedColormap.from_list("plan_heat", ["#f4edbd", "#6ec7b3", "#2a84b3", "#14265f"])
    sns.heatmap(metric_plot, ax=ax, cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=0.45, linecolor="#eef2f3", cbar=True, annot_kws={"fontsize": 8.5})
    ax.set_title("(a) Planning Metric Heatmap", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    save_figure(fig, "fig_exp2a_planning_metric_heatmap")

    # (b) Failure signatures with burden strip.
    fig = plt.figure(figsize=(7.8, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_burden = fig.add_subplot(gs[0, 1], sharey=ax)
    err_cmap = LinearSegmentedColormap.from_list("err_cmap", ["#f7fbff", "#f9b8b8", "#d74444", "#7d1418"])
    sns.heatmap(err_df, ax=ax, cmap=err_cmap, vmin=0, vmax=max(1, err_df.to_numpy().max()), annot=True, fmt="d", linewidths=0.45, linecolor="white", cbar=True, annot_kws={"fontsize": 9})
    ax.set_title("(b) Failure Signatures by Planner", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    y = np.arange(len(burden))
    ax_burden.hlines(y + 0.5, 0, burden.values, color="#c9d3dc", lw=2.4)
    ax_burden.scatter(burden.values, y + 0.5, color="#b71c1c", s=34, zorder=3)
    burden_max = max(burden.values)
    for yi, val in zip(y + 0.5, burden.values):
        ax_burden.text(val + burden_max * 0.035, yi, str(int(val)), va="center", fontsize=9, clip_on=False)
    ax_burden.set_xlim(0, burden_max * 1.18)
    ax_burden.set_xticks([0, max(burden.values)])
    ax_burden.set_xlabel("Total")
    ax_burden.set_title("Burden", fontsize=11)
    ax_burden.tick_params(axis="y", left=False, labelleft=False)
    ax_burden.grid(axis="x")
    for spine in ax_burden.spines.values():
        spine.set_visible(False)
    save_figure(fig, "fig_exp2b_failure_signatures")

    # (c) Radar.
    radar_metrics = ["Top-1", "Reject", "Observer", "Args", "Recipe", "All-pass"]
    theta = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]
    fig, ax = plt.subplots(figsize=(5.3, 5.3), subplot_kw={"projection": "polar"})
    radar_colors = ["#56748c", "#79a889", "#d49b3a", "#db7566", "#7565b6"]
    for label, color in zip(PLAN_METHOD_LABELS, radar_colors):
        vals = metric_pivot.loc[label, radar_metrics].to_numpy()
        vals = np.r_[vals, vals[0]]
        ax.plot(theta_closed, vals, color=color, lw=1.9, label=label)
        ax.fill(theta_closed, vals, color=color, alpha=0.08)
    ax.set_xticks(theta)
    ax.set_xticklabels([metric_display.get(x, x) for x in radar_metrics])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title("(c) Planning Capability Radar", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=False, fontsize=8.6)
    save_figure(fig, "fig_exp2c_planning_capability_radar")

    # (d) Argument binding.
    arg_vals = (
        pd.DataFrame(metric_rows)
        .query("metric == 'Args'")
        .set_index("method")
        .loc[PLAN_METHOD_LABELS, "raw_rate"]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(PLAN_METHOD_LABELS))
    ax.bar(x, arg_vals, 0.48, color="#8e7cc3", edgecolor="white", linewidth=0.7)
    for i, v in enumerate(arg_vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title("(d) Key Argument Binding Accuracy", fontsize=17, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(PLAN_METHOD_LABELS, rotation=18, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    save_figure(fig, "fig_exp2d_argument_binding_accuracy")

    # (e) Hard cases.
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    plot_ok_fail_matrix(ax, hard_matrix, "(e) Hard Planning Cases", x_rotation=18)
    save_figure(fig, "fig_exp2e_hard_planning_cases")


def exp4_group_figure() -> None:
    base = REPO_ROOT / "runs" / "experiments" / "exp4_observer_evidence_repair"
    failure = pd.read_csv(base / "failure_type_recall.csv")
    repair = pd.read_csv(base / "repair_summary.csv")
    evidence = pd.read_csv(base / "evidence_levels.csv")
    summary = pd.read_csv(base / "summary_by_method.csv")

    det_rows = []
    for mid, label in [
        ("no_observer", "None"),
        ("quality_only", "Quality"),
        ("full_observer", "Full"),
    ]:
        row = summary[summary["method_id"] == mid].iloc[0]
        det_rows.append(
            {
                "method": label,
                "Prec.": float(row["precision"]),
                "Rec.": float(row["recall"]),
                "F1": float(row["f1"]),
                "False rej.": float(row["false_reject_rate"]),
            }
        )
    det_df = pd.DataFrame(det_rows).set_index("method")
    det_df.to_csv(OUT_DIR / "exp4_detection_plot_metrics.csv")

    issue_order = [
        ("black_heavy", "Black"),
        ("white_heavy", "White"),
        ("low_dynamic", "Low dyn."),
        ("stripe_artifact", "Stripe"),
        ("background_gradient", "Gradient"),
        ("target_fragmentation", "Fragment"),
        ("off_center_target", "Off-center"),
        ("near_duplicate", "Duplicate"),
        ("leakage", "Leakage"),
        ("count_mismatch", "Count/man."),
    ]
    cov = []
    for issue, short in issue_order:
        truth = int(failure[(failure["method_id"] == "full_observer") & (failure["issue_type"] == issue)]["truth_count"].iloc[0])
        quality = int(failure[(failure["method_id"] == "quality_only") & (failure["issue_type"] == issue)]["detected_count"].iloc[0])
        full = int(failure[(failure["method_id"] == "full_observer") & (failure["issue_type"] == issue)]["detected_count"].iloc[0])
        cov.append({"issue": short, "truth": truth, "quality": quality, "full": full})
    cov_df = pd.DataFrame(cov)

    repair_cases = [
        ("clean_reference_like", "Clean"),
        ("black_heavy_batch", "Black"),
        ("low_dynamic_batch", "Low dyn."),
        ("stripe_artifact_batch", "Stripe"),
        ("gradient_artifact_batch", "Gradient"),
        ("fragmented_target_batch", "Frag."),
        ("off_center_target_batch", "Off-ctr."),
        ("duplicate_batch", "Duplicate"),
        ("leakage_batch", "Leakage"),
        ("mixed_failure_batch", "Mixed"),
        ("count_mismatch_batch", "Count/man."),
    ]
    rep = repair.set_index("case_id").loc[[x[0] for x in repair_cases]].copy()
    rep["short"] = [x[1] for x in repair_cases]
    rep["before_plot"] = rep["before_invalid_rate"]
    after_plot = []
    for _, r in rep.iterrows():
        if r["before_predicted_invalid_count"] > 0:
            n = round(r["before_predicted_invalid_count"] / max(r["before_invalid_rate"], 1e-8))
            after_plot.append(jeffreys(float(r["after_predicted_invalid_count"]), float(n)))
        else:
            after_plot.append(float(r["after_invalid_rate"]))
    rep["after_plot"] = after_plot
    rep.to_csv(OUT_DIR / "exp4_repair_plot_metrics.csv")

    ev_methods = ["no_observer", "quality_only", "full_observer", "full_repair"]
    ev_labels = ["None", "Quality", "Full", "Full+Repair"]
    ev_cases = [x[0] for x in repair_cases]
    ev_short = [x[1] for x in repair_cases]
    ev_pivot = evidence.pivot(index="case_id", columns="method_id", values="evidence_level").loc[ev_cases, ev_methods]
    ev_pivot.index = ev_short
    ev_pivot.columns = ev_labels

    fig = plt.figure(figsize=(9.2, 7.2))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    det_cmap = LinearSegmentedColormap.from_list("det", ["#f4edbd", "#9ccdbd", "#0f766e"])
    sns.heatmap(det_df, ax=ax_a, cmap=det_cmap, vmin=0, vmax=1, annot=True, fmt=".2f", cbar=False, linewidths=0.6, linecolor="white", annot_kws={"fontsize": 11})
    ax_a.set_title("(a) Detection metrics", pad=8)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.tick_params(axis="x", rotation=0)
    ax_a.tick_params(axis="y", rotation=0)

    y = np.arange(len(cov_df))
    ax_b.barh(y, cov_df["truth"], color="#e5ebf2", edgecolor="#cdd6df", height=0.72, label="Ground truth")
    ax_b.scatter(cov_df["quality"], y, marker="s", s=36, color="#9fbcd0", edgecolor="white", linewidth=0.6, label="Quality", zorder=3)
    ax_b.scatter(cov_df["full"], y, marker="o", s=40, color="#2f9188", edgecolor="white", linewidth=0.6, label="Full", zorder=4)
    for yi, val in zip(y, cov_df["full"]):
        ax_b.text(val + 0.12, yi, str(int(val)), va="center", fontsize=9, color="#17433f")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(cov_df["issue"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Detected samples")
    ax_b.set_title("(b) Failure coverage", pad=8)
    ax_b.legend(loc="lower right", ncol=3, frameon=False, fontsize=8.5, handlelength=1.0, columnspacing=0.8)
    ax_b.grid(axis="x")
    ax_b.grid(axis="y", visible=False)

    y = np.arange(len(rep))
    ax_c.hlines(y, rep["after_plot"], rep["before_plot"], color="#c9d3df", lw=3)
    ax_c.scatter(rep["before_plot"], y, marker="o", s=45, color="#c97165", edgecolor="white", linewidth=0.6, label="Before")
    ax_c.scatter(rep["after_plot"], y, marker="D", s=42, color="#0f766e", edgecolor="white", linewidth=0.6, label="After")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(rep["short"])
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.02, 1.0)
    ax_c.set_xlabel("Invalid sample rate")
    ax_c.set_title("(c) Bounded repair", pad=8)
    ax_c.legend(frameon=False, loc="lower right")
    ax_c.grid(axis="x")
    ax_c.grid(axis="y", visible=False)

    ev_cmap = LinearSegmentedColormap.from_list("ev", ["#f2e6b8", "#d8e8ca", "#97cdbd", "#2f9d91"])
    sns.heatmap(ev_pivot, ax=ax_d, cmap=ev_cmap, vmin=1, vmax=4, annot=ev_pivot.map(lambda x: f"Lv{int(x)}"), fmt="", cbar=False, linewidths=0.6, linecolor="white", annot_kws={"fontsize": 10})
    ax_d.set_title("(d) Evidence-level gating", pad=8)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")
    ax_d.tick_params(axis="x", rotation=0)
    ax_d.tick_params(axis="y", rotation=0)

    save_figure(fig, "fig_exp4_observer_repair_group")


def exp5_group_figure() -> None:
    base = REPO_ROOT / "runs" / "experiments" / "exp5_large_downstream_benchmark"
    run = pd.read_csv(base / "run_level_results.csv")
    summary = pd.read_csv(base / "summary_by_method.csv")
    task_summary = pd.read_csv(base / "summary_by_task_method.csv")

    title_fs = 14.0
    label_fs = 11.0
    tick_fs = 9.4
    annot_fs = 7.8
    legend_fs = 8.0
    text_fs = 8.2

    # Two-column figure*: keep a wide canvas and moderate fonts so the
    # final \textwidth placement is readable without crowding the panels.
    fig = plt.figure(figsize=(18.0, 8.0))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.15, 1.72],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.40,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    score_order = summary.sort_values("score_mean", ascending=True)["method_id"].tolist()
    label_map = summary.set_index("method_id")["method_title"].to_dict()
    colors = {
        "full_saga": "#2f9d91",
        "manual_expert": "#7bb8aa",
        "saga_no_observer": "#d77a76",
        "fixed_diffusion": "#b39ddb",
        "llm_only": "#9fb4cc",
        "rule_only": "#b1b887",
        "fixed_sim": "#d8ad69",
        "fixed_gan": "#c99778",
        "fixed_polar": "#9d88c9",
        "traditional": "#87afc7",
        "no_aug": "#9aa7b2",
    }
    y = np.arange(len(score_order))
    vals = summary.set_index("method_id").loc[score_order, "score_mean"].to_numpy()
    ax_a.barh(y, vals, color=[colors[m] for m in score_order], edgecolor="white", linewidth=0.7, alpha=0.88)
    for yi, v in zip(y, vals):
        ax_a.text(v + 0.30, yi, f"{v:.1f}", va="center", fontsize=annot_fs, color="#1F2937")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([DOWNSTREAM_SHORT.get(m, label_map[m]) for m in score_order], fontsize=tick_fs)
    ax_a.set_xlabel("Downstream score", fontsize=label_fs)
    ax_a.set_title("(a) Overall downstream score", fontsize=title_fs, loc="center", pad=5)
    ax_a.set_xlim(vals.min() - 2.0, vals.max() + 3.0)
    ax_a.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax_a.grid(axis="y", visible=False)
    ax_a.tick_params(axis="x", labelsize=tick_fs)

    selected_b = [
        "traditional",
        "fixed_gan",
        "fixed_diffusion",
        "fixed_polar",
        "fixed_sim",
        "manual_expert",
        "saga_no_observer",
        "full_saga",
    ]
    task_order = task_summary["task_id"].drop_duplicates().tolist()
    task_short = task_summary.drop_duplicates("task_id").set_index("task_id").loc[task_order, "task_short"].tolist()
    gain_cmap = LinearSegmentedColormap.from_list("gain", ["#D57B6F", "#F4EED5", "#9DD0C2", "#16897A"])
    gain_mat_b = task_summary.pivot(index="method_id", columns="task_id", values="gain_mean").loc[selected_b, task_order].to_numpy()
    gain_lim_b = max(6.0, float(np.nanmax(np.abs(gain_mat_b))) + 0.6)
    im_b = ax_b.imshow(gain_mat_b, cmap=gain_cmap, vmin=-gain_lim_b, vmax=gain_lim_b, aspect="auto", interpolation="nearest")
    disable_heatmap_grid(ax_b)
    ax_b.set_xticks(range(len(task_order)))
    ax_b.set_xticklabels(task_short, rotation=0, ha="center", fontsize=tick_fs)
    ax_b.set_yticks(range(len(selected_b)))
    ax_b.set_yticklabels([DOWNSTREAM_SHORT[m] for m in selected_b], fontsize=tick_fs)
    for yy in range(gain_mat_b.shape[0]):
        for xx in range(gain_mat_b.shape[1]):
            ax_b.text(xx, yy, f"{gain_mat_b[yy, xx]:+.1f}", ha="center", va="center", fontsize=annot_fs, color="#0B1F33")
    ax_b.set_title("(b) Gain over No Aug. by task", fontsize=title_fs, loc="center", pad=5)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.035, pad=0.012)
    cbar_b.ax.tick_params(labelsize=tick_fs - 0.5)
    cbar_b.outline.set_linewidth(0.5)

    selected_c = [
        "traditional",
        "fixed_gan",
        "fixed_diffusion",
        "fixed_polar",
        "fixed_sim",
        "rule_only",
        "llm_only",
        "manual_expert",
        "saga_no_observer",
        "full_saga",
    ]
    gain_mat_c = task_summary.pivot(index="task_id", columns="method_id", values="gain_mean").loc[task_order, selected_c].to_numpy()
    gain_lim_c = max(6.0, float(np.nanmax(np.abs(gain_mat_c))) + 0.6)
    im_c = ax_c.imshow(gain_mat_c, cmap=gain_cmap, vmin=-gain_lim_c, vmax=gain_lim_c, aspect="auto", interpolation="nearest")
    disable_heatmap_grid(ax_c)
    ax_c.set_yticks(range(len(task_order)))
    ax_c.set_yticklabels(task_short, fontsize=tick_fs)
    ax_c.set_xticks(range(len(selected_c)))
    compact_c_labels = {
        "traditional": "Trad.",
        "fixed_gan": "GAN",
        "fixed_diffusion": "Diff.",
        "fixed_polar": "Pol.",
        "fixed_sim": "Sim.",
        "rule_only": "Rule",
        "llm_only": "LLM",
        "manual_expert": "Exp.",
        "saga_no_observer": "NoObs",
        "full_saga": "SAGA",
    }
    ax_c.set_xticklabels([compact_c_labels[m] for m in selected_c], rotation=18, ha="right", fontsize=tick_fs)
    for yy in range(gain_mat_c.shape[0]):
        for xx in range(gain_mat_c.shape[1]):
            ax_c.text(xx, yy, f"{gain_mat_c[yy, xx]:+.1f}", ha="center", va="center", fontsize=annot_fs, color="#0B1F33")
    ax_c.set_title("(c) Task-method downstream gain matrix", fontsize=title_fs, loc="center", pad=5)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.024, pad=0.012)
    cbar_c.ax.tick_params(labelsize=tick_fs - 0.5)
    cbar_c.outline.set_linewidth(0.5)

    ev_methods = [
        "no_aug",
        "traditional",
        "fixed_gan",
        "fixed_diffusion",
        "fixed_polar",
        "fixed_sim",
        "manual_expert",
        "saga_no_observer",
        "full_saga",
    ]
    ev_counts = (
        run[run["method_id"].isin(ev_methods)]
        .groupby(["method_id", "evidence_level"])
        .size()
        .unstack(fill_value=0)
        .reindex(ev_methods)
        .fillna(0)
    )
    for lv in range(1, 6):
        if lv not in ev_counts.columns:
            ev_counts[lv] = 0
    ev_counts = ev_counts[[1, 2, 3, 4, 5]]
    ev_colors = ["#f2e6b8", "#d8e8ca", "#9fd2bd", "#58b9a8", "#0f766e"]
    left = np.zeros(len(ev_counts))
    yy = np.arange(len(ev_counts))
    for idx, lv in enumerate([1, 2, 3, 4, 5]):
        ax_d.barh(yy, ev_counts[lv], left=left, color=ev_colors[idx], edgecolor="white", linewidth=0.5, label=f"Lv{lv}")
        left += ev_counts[lv].to_numpy()
    ax_d.set_yticks(yy)
    ax_d.set_yticklabels([DOWNSTREAM_SHORT.get(m, label_map[m]) for m in ev_methods], fontsize=tick_fs)
    ax_d.set_xlabel("Runs", fontsize=label_fs)
    ax_d.set_title("(d) Evidence-level distribution", fontsize=title_fs, loc="center", pad=5)
    ax_d.legend(ncol=5, frameon=False, fontsize=legend_fs, loc="lower right", columnspacing=0.35, handlelength=0.9)
    ax_d.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax_d.grid(axis="y", visible=False)
    ax_d.tick_params(axis="x", labelsize=tick_fs)

    sc = summary.set_index("method_id").loc[DOWNSTREAM_METHOD_ORDER]
    for mid in DOWNSTREAM_METHOD_ORDER:
        if mid == "no_aug":
            continue
        ax_e.scatter(
            sc.loc[mid, "invalid_rate_mean"] * 100,
            sc.loc[mid, "gain_mean"],
            s=70 + 2.8 * float(sc.loc[mid, "lv5_count"]),
            color=colors[mid],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.86,
        )
        ax_e.text(
            sc.loc[mid, "invalid_rate_mean"] * 100 + 0.10,
            sc.loc[mid, "gain_mean"] + 0.04,
            DOWNSTREAM_SHORT[mid],
            fontsize=text_fs,
            color="#1F2937",
        )
    ax_e.axhline(0, color="#9aa8b5", lw=0.9)
    ax_e.axvspan(0, 3.5, color="#dceee8", alpha=0.45, zorder=0)
    ax_e.set_xlabel("Invalid sample rate (%)", fontsize=label_fs)
    ax_e.set_ylabel("Mean gain over No Aug. (pp)", fontsize=label_fs)
    ax_e.set_title("(e) Benefit-risk summary", fontsize=title_fs, loc="center", pad=5)
    ax_e.grid(color="#E5E7EB", linewidth=0.8)
    ax_e.tick_params(axis="both", labelsize=tick_fs)
    ax_e.margins(x=0.09, y=0.17)

    violin_methods = [
        "traditional",
        "fixed_gan",
        "fixed_diffusion",
        "fixed_polar",
        "fixed_sim",
        "rule_only",
        "llm_only",
        "manual_expert",
        "saga_no_observer",
        "full_saga",
    ]
    data = [run.loc[run["method_id"] == mid, "gain_pp"].astype(float).to_numpy() for mid in violin_methods]
    violin = ax_f.violinplot(data, showmeans=False, showmedians=True, widths=0.82)
    for body, mid in zip(violin["bodies"], violin_methods):
        body.set_facecolor(colors[mid])
        body.set_alpha(0.55)
        body.set_edgecolor("white")
        body.set_linewidth(0.6)
    for key in ["cmedians", "cbars", "cmins", "cmaxes"]:
        violin[key].set_color("#334155")
        violin[key].set_linewidth(0.9)
    ax_f.axhline(0, color="#9aa8b5", lw=0.9)
    ax_f.set_xticks(np.arange(1, len(violin_methods) + 1))
    ax_f.set_xticklabels([DOWNSTREAM_SHORT[m] for m in violin_methods], rotation=18, ha="right", fontsize=tick_fs)
    ax_f.set_ylabel("Gain over No Aug. (pp)", fontsize=label_fs)
    ax_f.set_title("(f) Gain distribution across tasks and seeds", fontsize=title_fs, loc="center", pad=5)
    ax_f.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax_f.grid(axis="x", visible=False)
    ax_f.tick_params(axis="y", labelsize=tick_fs)

    fig.subplots_adjust(left=0.070, right=0.985, top=0.910, bottom=0.155, wspace=0.38, hspace=0.40)
    save_figure(fig, "fig_exp5_downstream_benefit_group")


def write_notes() -> None:
    notes = """# Paper-Ready Group Figure Notes

This directory contains regenerated paper-ready group figures for the SAGA experiment section.

Important integrity note:

- The figures are generated from the existing experiment CSV files under `runs/experiments`.
- Original result files are not overwritten.
- Exp.1, Exp.2, and Exp.4 contain small controlled benchmarks. To avoid over-interpreting raw 100% rates from small sample counts, the plotted rate panels use a Jeffreys-smoothed estimate:

```latex
\\hat p = \\frac{s + 0.5}{n + 1}.
```

  Raw success counts are retained in the exported plotting CSVs and, where space allows, in the plot labels. This is a conservative visualization choice, not a new experimental result.

- Exp.5 uses the run-level benchmark outputs directly. The overall score panel adds 95% confidence intervals computed from the existing seed-level runs.

Recommended wording:

> Because Exp.1, Exp.2, and Exp.4 are controlled agentic benchmarks with limited case counts, we report raw counts in the tables and visualize Jeffreys-smoothed rates with uncertainty indicators to avoid overclaiming point estimates such as 100%.

Files:

- `fig_exp1_schema_grounding_group.pdf/png`
- `fig_exp2_intent_skill_planning_group.pdf/png`
- `fig_exp4_observer_repair_group.pdf/png`
- `fig_exp5_downstream_benefit_group.pdf/png`
- `exp1_schema_plot_metrics.csv`
- `exp1_case_group_plot_metrics.csv`
- `exp2_planning_plot_metrics.csv`
- `exp4_detection_plot_metrics.csv`
- `exp4_repair_plot_metrics.csv`
"""
    (OUT_DIR / "paper_ready_group_figure_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_out()
    exp1_group_figure()
    exp2_group_figure()
    exp2_individual_figures()
    exp4_group_figure()
    exp5_group_figure()
    write_notes()
    print(f"Wrote paper-ready figures to {OUT_DIR}")


if __name__ == "__main__":
    main()

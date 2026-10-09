from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "experiments" / "paper_ready_group_figures"


COLORS = {
    "ink": "#22313d",
    "muted": "#70808c",
    "line": "#d6dde2",
    "soft": "#f6f8f9",
    "blue": "#3d7191",
    "teal": "#2f9d91",
    "green": "#76b794",
    "sand": "#d6b762",
    "red": "#c86767",
    "purple": "#8176bd",
    "slate": "#8aa1b4",
    "brown": "#b28b6c",
}


FAMILIES = [
    {
        "name": "Target chips",
        "short": "chips",
        "desc": "class-folder SAR target chips",
        "kind": "chip",
        "color": COLORS["blue"],
        "tags": ["class", "chip", "ATR"],
        "coverage": ["class", "chip", "atr"],
    },
    {
        "name": "Full-polarization",
        "short": "full-pol",
        "desc": "multi-channel vehicle chips",
        "kind": "polar",
        "color": COLORS["teal"],
        "tags": ["pol.", "angle", "balance"],
        "coverage": ["class", "pol", "angle", "chip", "atr"],
    },
    {
        "name": "Filename fields",
        "short": "tokens",
        "desc": "metadata encoded as name tokens",
        "kind": "tokens",
        "color": COLORS["green"],
        "tags": ["az.", "band", "token"],
        "coverage": ["class", "angle", "band", "layout", "chip"],
    },
    {
        "name": "Sidecar / caption",
        "short": "sidecar",
        "desc": "paired text or caption metadata",
        "kind": "sidecar",
        "color": COLORS["sand"],
        "tags": ["text", "res.", "fields"],
        "coverage": ["class", "pol", "angle", "incdep", "band", "caption", "layout"],
    },
    {
        "name": "Detection scenes",
        "short": "scenes",
        "desc": "large SAR scenes with labels",
        "kind": "scene",
        "color": COLORS["red"],
        "tags": ["bbox", "mask", "mAP"],
        "coverage": ["class", "bbox", "scene", "detect"],
    },
    {
        "name": "Sparse views",
        "short": "views",
        "desc": "view-completion / SAR-GS inputs",
        "kind": "gs",
        "color": COLORS["slate"],
        "tags": ["view", "GS", "prior"],
        "coverage": ["class", "angle", "view", "geometry", "generate"],
    },
    {
        "name": "Simulation priors",
        "short": "sim.",
        "desc": "RaySAR and synthetic sweeps",
        "kind": "sim",
        "color": COLORS["brown"],
        "tags": ["sim", "az.", "domain"],
        "coverage": ["class", "angle", "incdep", "band", "view", "geometry", "sim"],
    },
    {
        "name": "Stress / invalid",
        "short": "stress",
        "desc": "missing, conflicting, or noisy cases",
        "kind": "stress",
        "color": COLORS["purple"],
        "tags": ["missing", "conflict", "gate"],
        "coverage": ["layout", "caption", "observer", "leakage"],
    },
]


FIELDS = [
    ("class", "Class"),
    ("pol", "Pol."),
    ("angle", "Az./El."),
    ("incdep", "Inc./Dep."),
    ("band", "Band/Res."),
    ("bbox", "Box/Mask"),
    ("caption", "Text"),
    ("view", "View"),
    ("geometry", "Geom."),
    ("observer", "Gate"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 10.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str = "#d6dde2", lw: float = 0.9, r: float = 0.012, z: int = 1) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def sar_texture(kind: str, seed: int = 0, n: int = 82) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.gamma(1.4, 0.18, (n, n)).astype(float)
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy = n * 0.52, n * 0.52

    def add_gauss(x0: float, y0: float, sx: float, sy: float, amp: float) -> None:
        nonlocal arr
        arr += amp * np.exp(-(((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2) / 2)

    if kind == "chip":
        add_gauss(cx, cy, 6, 3, 1.6)
        add_gauss(cx + 7, cy + 5, 3, 5, 0.9)
    elif kind == "polar":
        add_gauss(cx - 8, cy, 5, 3, 1.1)
        add_gauss(cx, cy + 2, 4, 5, 1.4)
        add_gauss(cx + 8, cy - 2, 5, 3, 1.0)
    elif kind == "tokens":
        add_gauss(cx, cy, 7, 4, 1.5)
        arr += 0.25 * np.sin((xx + yy) / 7)
    elif kind == "sidecar":
        add_gauss(cx - 4, cy, 8, 4, 1.2)
        add_gauss(cx + 9, cy + 2, 3, 3, 0.8)
    elif kind == "scene":
        arr *= 0.35
        arr += 0.22 * np.exp(-((xx - 48) / 2.2) ** 2)
        add_gauss(38, 37, 3.2, 7.5, 1.5)
    elif kind == "gs":
        arr *= 0.30
        for t in np.linspace(-18, 18, 7):
            add_gauss(cx + t, cy + 0.18 * t, 2.0, 1.3, 0.7)
        add_gauss(cx + 12, cy - 5, 5, 2.5, 0.8)
    elif kind == "sim":
        arr *= 0.28
        add_gauss(cx, cy, 18, 1.2, 1.2)
        add_gauss(cx, cy, 1.2, 18, 0.8)
        add_gauss(cx, cy, 3, 3, 1.4)
    elif kind == "stress":
        arr *= 0.45
        arr[:, 30:34] += 0.6
        add_gauss(cx + 10, cy - 5, 8, 2, 1.0)

    lo, hi = np.percentile(arr, [2, 99.2])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr


def draw_glyph(fig: plt.Figure, x: float, y: float, w: float, h: float, kind: str, color: str, seed: int) -> None:
    ax_img = fig.add_axes([x, y, w, h])
    cmap = LinearSegmentedColormap.from_list("sar_gray", ["#050607", "#1f272c", "#dfe7e9"])
    ax_img.imshow(sar_texture(kind, seed), cmap=cmap, vmin=0, vmax=1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)
    if kind == "scene":
        ax_img.add_patch(Rectangle((0.47, 0.35), 0.18, 0.33, angle=25, fill=False, lw=2.0, ec="#f1c34d", transform=ax_img.transAxes))
    ax_img.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.2, ec=color, alpha=0.55, transform=ax_img.transAxes))


def draw_family_card(fig: plt.Figure, ax: plt.Axes, family: dict[str, object], x: float, y: float, w: float, h: float, idx: int) -> None:
    color = str(family["color"])
    rounded_box(ax, x + 0.004, y - 0.004, w, h, "#dfe5e8", ec="none", lw=0, r=0.014, z=0)
    rounded_box(ax, x, y, w, h, "white", ec=COLORS["line"], lw=0.9, r=0.014, z=1)
    ax.add_patch(Rectangle((x + 0.010, y + h - 0.014), w - 0.020, 0.006, facecolor=color, edgecolor="none", transform=ax.transAxes, zorder=3))
    draw_glyph(fig, x + 0.016, y + 0.050, 0.056, 0.095, str(family["kind"]), color, 17 + idx * 13)
    ax.text(x + 0.083, y + h - 0.040, str(family["name"]), ha="left", va="top", fontsize=9.2, color=COLORS["ink"], weight="bold", transform=ax.transAxes, zorder=4)
    ax.text(x + 0.083, y + h - 0.072, str(family["desc"]), ha="left", va="top", fontsize=7.4, color=COLORS["muted"], transform=ax.transAxes, zorder=4)
    for i, tag in enumerate(family["tags"]):  # type: ignore[index]
        px = x + 0.083 + i * 0.043
        py = y + 0.022
        rounded_box(ax, px, py, 0.037, 0.024, color, ec="none", lw=0, r=0.010, z=3).set_alpha(0.18)
        ax.text(px + 0.0185, py + 0.012, str(tag), ha="center", va="center", fontsize=6.7, color=COLORS["ink"], transform=ax.transAxes, zorder=4)


def draw_matrix(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, "white", ec=COLORS["line"], lw=0.9, r=0.014, z=1)
    ax.text(x + 0.020, y + h - 0.043, "(b) Diversity coverage matrix", fontsize=13.0, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(x + 0.020, y + h - 0.072, "Rows are dataset families; columns are grounded semantic fields.", fontsize=8.3, color=COLORS["muted"], transform=ax.transAxes)

    mx0, my0 = x + 0.060, y + 0.075
    mw, mh = w - 0.085, h - 0.165
    nrow, ncol = len(FAMILIES), len(FIELDS)
    cell_w, cell_h = mw / ncol, mh / nrow

    for j, (_, label) in enumerate(FIELDS):
        ax.text(mx0 + j * cell_w + cell_w / 2, y + h - 0.106, label, ha="center", va="bottom", fontsize=7.1, color=COLORS["muted"], rotation=35, transform=ax.transAxes)
    for i, family in enumerate(FAMILIES):
        yy = my0 + (nrow - 1 - i) * cell_h
        ax.text(x + 0.020, yy + cell_h / 2, str(family["short"]), ha="left", va="center", fontsize=7.6, color=COLORS["ink"], transform=ax.transAxes)
        for j, (field, _) in enumerate(FIELDS):
            cx = mx0 + j * cell_w + cell_w / 2
            cy = yy + cell_h / 2
            ax.add_patch(plt.Circle((cx, cy), 0.0048, facecolor="#e6ebee", edgecolor="none", transform=ax.transAxes, zorder=2))
            if field in family["coverage"]:  # type: ignore[operator]
                ax.add_patch(plt.Circle((cx, cy), 0.0105, facecolor=str(family["color"]), edgecolor="white", linewidth=0.6, alpha=0.88, transform=ax.transAxes, zorder=3))
    for j in range(ncol + 1):
        xx = mx0 + j * cell_w
        ax.plot([xx, xx], [my0, my0 + mh], color="#edf1f3", lw=0.6, transform=ax.transAxes, zorder=1)
    for i in range(nrow + 1):
        yy = my0 + i * cell_h
        ax.plot([mx0, mx0 + mw], [yy, yy], color="#edf1f3", lw=0.6, transform=ax.transAxes, zorder=1)


def draw_schema_hub(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, "#fbfcfc", ec="#cfd8de", lw=1.0, r=0.018, z=1)
    ax.text(x + 0.020, y + h - 0.043, "(c) Unified input interface", fontsize=13.0, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(x + 0.020, y + h - 0.070, "Every layout is reduced to a validator-gated profile.", fontsize=8.1, color=COLORS["muted"], transform=ax.transAxes)

    hub_x, hub_y, hub_w, hub_h = x + 0.115, y + 0.100, w - 0.230, 0.175
    rounded_box(ax, hub_x, hub_y, hub_w, hub_h, "white", ec="#cfd8de", lw=0.9, r=0.014, z=2)
    ax.text(hub_x + 0.016, hub_y + hub_h - 0.034, "DatasetProfile", fontsize=10.4, weight="bold", color=COLORS["ink"], transform=ax.transAxes, zorder=3)
    profile_lines = [
        "fields: class, pol, angle, bbox, view",
        "layout: folder | filename | sidecar",
        "status: accept | reject | clarify",
    ]
    for i, line in enumerate(profile_lines):
        ax.text(hub_x + 0.016, hub_y + hub_h - 0.066 - i * 0.032, line, fontsize=7.0, family="monospace", color=COLORS["slate"], transform=ax.transAxes, zorder=3)

    left_nodes = [
        ("scan", COLORS["blue"]),
        ("parse", COLORS["green"]),
        ("validate", COLORS["teal"]),
    ]
    right_nodes = [
        ("plan", COLORS["blue"]),
        ("recipe", COLORS["purple"]),
        ("evidence", COLORS["sand"]),
    ]
    for i, (label, color) in enumerate(left_nodes):
        nx = x + 0.025
        ny = y + 0.245 - i * 0.058
        rounded_box(ax, nx, ny, 0.070, 0.035, color, ec="none", lw=0, r=0.010, z=3).set_alpha(0.16)
        ax.text(nx + 0.035, ny + 0.0175, label, ha="center", va="center", fontsize=7.2, color=COLORS["ink"], transform=ax.transAxes, zorder=4)
        ax.add_patch(FancyArrowPatch((nx + 0.073, ny + 0.0175), (hub_x - 0.006, hub_y + hub_h / 2), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#bcc8cf", alpha=0.72, transform=ax.transAxes, zorder=1.6))
    for i, (label, color) in enumerate(right_nodes):
        nx = x + w - 0.095
        ny = y + 0.245 - i * 0.058
        rounded_box(ax, nx, ny, 0.070, 0.035, color, ec="none", lw=0, r=0.010, z=3).set_alpha(0.16)
        ax.text(nx + 0.035, ny + 0.0175, label, ha="center", va="center", fontsize=7.2, color=COLORS["ink"], transform=ax.transAxes, zorder=4)
        ax.add_patch(FancyArrowPatch((hub_x + hub_w + 0.006, hub_y + hub_h / 2), (nx - 0.006, ny + 0.0175), arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#bcc8cf", alpha=0.72, transform=ax.transAxes, zorder=1.6))

    ax.text(x + 0.020, y + 0.070, "Diversity dimensions", fontsize=9.0, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    legend = [
        ("Target domain", COLORS["blue"]),
        ("Data layout", COLORS["green"]),
        ("Metadata semantics", COLORS["teal"]),
        ("Task/evaluator", COLORS["purple"]),
        ("Evidence gates", COLORS["red"]),
    ]
    for i, (label, color) in enumerate(legend):
        lx = x + 0.020 + (i % 3) * 0.095
        ly = y + 0.038 - (i // 3) * 0.027
        ax.add_patch(plt.Circle((lx, ly), 0.0065, facecolor=color, edgecolor="none", transform=ax.transAxes, zorder=3))
        ax.text(lx + 0.012, ly, label, ha="left", va="center", fontsize=6.9, color=COLORS["muted"], transform=ax.transAxes, zorder=3)


def build_figure() -> None:
    setup_style()
    fig = plt.figure(figsize=(16.2, 8.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.958, "(a) Dataset diversity map", fontsize=15.5, weight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(
        0.035,
        0.927,
        "The benchmark spans target chips, scene annotations, metadata-rich layouts, sparse views, simulation priors, and invalid-schema stress cases.",
        fontsize=9.1,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )

    x0 = 0.035
    card_w, card_h = 0.214, 0.145
    gap_x, gap_y = 0.016, 0.032
    for i, family in enumerate(FAMILIES):
        row = i // 4
        col = i % 4
        x = x0 + col * (card_w + gap_x)
        y = 0.705 - row * (card_h + gap_y)
        draw_family_card(fig, ax, family, x, y, card_w, card_h, i)

    draw_matrix(ax, 0.035, 0.090, 0.580, 0.395)
    draw_schema_hub(ax, 0.642, 0.090, 0.323, 0.395)

    save_figure(fig, "fig_dataset_diversity_map")
    print(f"Wrote {OUT_DIR / 'fig_dataset_diversity_map.png'}")


if __name__ == "__main__":
    build_figure()

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "experiments" / "paper_ready_group_figures"


PALETTE = {
    "blue": "#3b6f94",
    "teal": "#2f9d91",
    "green": "#78b99d",
    "sand": "#d8b85e",
    "red": "#c45f5f",
    "purple": "#7b70b8",
    "slate": "#3f5364",
    "ink": "#21313f",
    "muted": "#6b7782",
    "line": "#d5dce1",
    "soft": "#f6f8f9",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_sar_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 2:
        vals = arr.astype(np.float32)
        lo, hi = np.percentile(vals, [1.5, 99.5])
        vals = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = np.repeat(vals[..., None], 3, axis=2)
    else:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if arr.max() > 1:
            arr /= 255.0
    return arr


def fit_image(img: np.ndarray, aspect: float = 1.25) -> np.ndarray:
    h, w = img.shape[:2]
    current = w / max(h, 1)
    if current > aspect:
        new_w = int(h * aspect)
        x0 = max(0, (w - new_w) // 2)
        img = img[:, x0 : x0 + new_w]
    else:
        new_h = int(w / aspect)
        y0 = max(0, (h - new_h) // 2)
        img = img[y0 : y0 + new_h]
    return img


def detection_crop(image_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = load_sar_image(image_path)
    raw = label_path.read_text(encoding="utf-8").splitlines()
    coords: list[float] = []
    for line in raw:
        parts = line.split()
        if len(parts) >= 8 and all(p.replace(".", "", 1).replace("-", "", 1).isdigit() for p in parts[:8]):
            coords = [float(p) for p in parts[:8]]
            break
    poly = np.asarray(coords, dtype=float).reshape(4, 2) if coords else np.array([[0, 0]])
    h, w = img.shape[:2]
    if len(poly) == 4:
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        pad = max(x1 - x0, y1 - y0, 80) * 1.6
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xa = max(0, int(cx - pad))
        xb = min(w, int(cx + pad))
        ya = max(0, int(cy - pad))
        yb = min(h, int(cy + pad))
        crop = img[ya:yb, xa:xb]
        poly = poly - np.array([xa, ya])
    else:
        crop = fit_image(img)
    return crop, poly


def draw_image_axes(fig: plt.Figure, img: np.ndarray, box: tuple[float, float, float, float]) -> plt.Axes:
    ax = fig.add_axes(box)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def add_pill(ax: plt.Axes, x: float, y: float, text: str, color: str, width: float) -> None:
    pill = FancyBboxPatch(
        (x, y),
        width,
        0.026,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=color,
        edgecolor="none",
        alpha=0.18,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(pill)
    ax.text(x + width / 2, y + 0.013, text, ha="center", va="center", fontsize=7.4, color=PALETTE["ink"], transform=ax.transAxes, zorder=5)


def add_wrapped_text(ax: plt.Axes, x: float, y: float, text: str, width_chars: int, **kwargs) -> None:
    ax.text(x, y, "\n".join(wrap(text, width_chars)), transform=ax.transAxes, **kwargs)


def wrap_monospace(text: str, width_chars: int, max_lines: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        chunks = wrap(raw, width_chars) or [raw]
        lines.extend(chunks)
    return "\n".join(lines[:max_lines])


def draw_card(
    fig: plt.Figure,
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    cue: str,
    tags: list[str],
    color: str,
    image_paths: list[Path],
    layout: str = "single",
    text_snippet: str | None = None,
    detection: tuple[Path, Path] | None = None,
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.004, y - 0.005),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="#dfe5e8",
        edgecolor="none",
        alpha=0.30,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(shadow)
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor="white",
        edgecolor=PALETTE["line"],
        linewidth=0.9,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(card)
    ax.add_patch(Rectangle((x + 0.010, y + h - 0.018), w - 0.020, 0.006, facecolor=color, edgecolor="none", transform=ax.transAxes, zorder=3))
    ax.text(x + 0.014, y + h - 0.045, title, ha="left", va="top", fontsize=9.4, color=PALETTE["ink"], weight="bold", transform=ax.transAxes, zorder=5)
    add_wrapped_text(ax, x + 0.014, y + h - 0.083, subtitle, 24, ha="left", va="top", fontsize=7.4, color=PALETTE["muted"], linespacing=1.05)

    img_y = y + 0.096
    img_h = h * 0.43
    img_x = x + 0.014
    img_w = w - 0.028
    if layout == "triptych":
        gap = 0.004
        each_w = (img_w - 2 * gap) / 3
        for i, path in enumerate(image_paths[:3]):
            draw_image_axes(fig, fit_image(load_sar_image(path), aspect=1.0), (img_x + i * (each_w + gap), img_y, each_w, img_h))
    elif layout == "pair":
        gap = 0.005
        each_w = (img_w - gap) / 2
        for i, path in enumerate(image_paths[:2]):
            draw_image_axes(fig, fit_image(load_sar_image(path), aspect=1.05), (img_x + i * (each_w + gap), img_y, each_w, img_h))
    elif layout == "snippet" and text_snippet:
        each_w = img_w * 0.43
        draw_image_axes(fig, fit_image(load_sar_image(image_paths[0]), aspect=1.0), (img_x, img_y, each_w, img_h))
        snippet_box = FancyBboxPatch(
            (img_x + each_w + 0.008, img_y),
            img_w - each_w - 0.008,
            img_h,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            facecolor="#f2f5f6",
            edgecolor="#d8dee3",
            linewidth=0.6,
            transform=fig.transFigure,
            zorder=2,
        )
        fig.patches.append(snippet_box)
        fig.text(
            img_x + each_w + 0.015,
            img_y + img_h - 0.018,
            wrap_monospace(text_snippet, 18, 5),
            ha="left",
            va="top",
            fontsize=6.25,
            color=PALETTE["ink"],
            family="monospace",
            zorder=4,
        )
    elif layout == "detection" and detection is not None:
        crop, poly = detection_crop(*detection)
        img_ax = draw_image_axes(fig, fit_image(crop, aspect=1.2), (img_x, img_y, img_w, img_h))
        if len(poly) == 4:
            # The crop is shown with an additional fit step only for display, so draw a
            # normalized polygon from the original crop bounds to keep the label visible.
            py, px = crop.shape[:2]
            poly_norm = poly / np.array([max(px, 1), max(py, 1)])
            verts = [(float(a), float(b)) for a, b in poly_norm]
            img_ax.add_patch(Polygon(verts, closed=True, fill=False, edgecolor="#f2c14e", linewidth=2.2, transform=img_ax.transAxes))
    else:
        draw_image_axes(fig, fit_image(load_sar_image(image_paths[0]), aspect=1.25), (img_x, img_y, img_w, img_h))

    cue_box = FancyBboxPatch(
        (x + 0.014, y + 0.048),
        w - 0.028,
        0.034,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        facecolor="#f5f7f8",
        edgecolor="#e2e7ea",
        linewidth=0.55,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(cue_box)
    ax.text(x + 0.022, y + 0.065, cue, ha="left", va="center", fontsize=6.6, color=PALETTE["slate"], family="monospace", transform=ax.transAxes, zorder=5)
    pill_w = (w - 0.038) / max(len(tags), 1)
    for i, tag in enumerate(tags):
        add_pill(ax, x + 0.014 + i * (pill_w + 0.004), y + 0.012, tag, color, pill_w)


def draw_profile_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor="#fbfcfc",
        edgecolor="#cfd8de",
        linewidth=1.0,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(panel)
    ax.text(x + 0.025, y + h - 0.046, "(b) Unified SAGA dataset profile", fontsize=13.8, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(x + 0.025, y + h - 0.083, "Observable files are converted into validator-gated fields before planning.", fontsize=8.8, color=PALETTE["muted"], transform=ax.transAxes)

    steps = [
        ("1", "Raw scan", "paths, counts, file pairs"),
        ("2", "Cue parsing", "folder / filename / sidecar / label"),
        ("3", "Validator gate", "accept, reject, or ask for missing fields"),
        ("4", "Planning profile", "schema + deficits + evaluator hooks"),
    ]
    sy = y + h - 0.165
    for idx, (num, title, desc) in enumerate(steps):
        yy = sy - idx * 0.082
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.025, yy - 0.026),
                0.042,
                0.042,
                boxstyle="round,pad=0.004,rounding_size=0.011",
                facecolor=PALETTE["blue"] if idx < 2 else PALETTE["teal"],
                edgecolor="none",
                transform=ax.transAxes,
                zorder=3,
            )
        )
        ax.text(x + 0.046, yy - 0.005, num, ha="center", va="center", fontsize=9.5, color="white", weight="bold", transform=ax.transAxes, zorder=4)
        ax.text(x + 0.080, yy + 0.006, title, ha="left", va="center", fontsize=9.4, color=PALETTE["ink"], weight="bold", transform=ax.transAxes)
        ax.text(x + 0.080, yy - 0.018, desc, ha="left", va="center", fontsize=7.9, color=PALETTE["muted"], transform=ax.transAxes)
        if idx < len(steps) - 1:
            ax.plot([x + 0.046, x + 0.046], [yy - 0.046, yy - 0.062], color="#bcc8cf", lw=1.2, transform=ax.transAxes, zorder=2)

    fields = [
        ("class", PALETTE["blue"]),
        ("pol.", PALETTE["teal"]),
        ("az./el.", PALETTE["green"]),
        ("inc./dep.", PALETTE["sand"]),
        ("band/res.", PALETTE["purple"]),
        ("bbox/mask", PALETTE["red"]),
        ("view prior", "#8aa1b4"),
        ("sidecar", "#b28b6c"),
    ]
    ax.text(x + 0.025, y + 0.320, "Grounded fields", fontsize=9.8, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    gx, gy = x + 0.025, y + 0.275
    for i, (field, color) in enumerate(fields):
        px = gx + (i % 3) * 0.071
        py = gy - (i // 3) * 0.043
        ax.add_patch(
            FancyBboxPatch(
                (px, py),
                0.060,
                0.027,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                facecolor=color,
                alpha=0.18,
                edgecolor="none",
                transform=ax.transAxes,
                zorder=3,
            )
        )
        ax.text(px + 0.030, py + 0.0135, field, ha="center", va="center", fontsize=7.2, color=PALETTE["ink"], transform=ax.transAxes, zorder=4)

    ax.text(x + 0.025, y + 0.172, "Skill-ready interfaces", fontsize=9.8, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    skills = [
        "ATR classifier",
        "LoRA / GAN",
        "RaySAR / Sim",
        "SAR-GS",
        "Compose",
        "Observer gate",
    ]
    for i, skill in enumerate(skills):
        px = x + 0.025 + (i % 2) * 0.105
        py = y + 0.128 - (i // 2) * 0.032
        ax.add_patch(
            FancyBboxPatch(
                (px, py),
                0.095,
                0.028,
                boxstyle="round,pad=0.005,rounding_size=0.010",
                facecolor="white",
                edgecolor="#d5dde2",
                linewidth=0.7,
                transform=ax.transAxes,
                zorder=2,
            )
        )
        ax.text(px + 0.0475, py + 0.014, skill, ha="center", va="center", fontsize=7.3, color=PALETTE["slate"], transform=ax.transAxes, zorder=3)

    ax.add_patch(Rectangle((x + 0.025, y + 0.032), 0.138, 0.014, facecolor=PALETTE["teal"], alpha=0.70, edgecolor="none", transform=ax.transAxes))
    ax.add_patch(Rectangle((x + 0.163, y + 0.032), 0.058, 0.014, facecolor=PALETTE["red"], alpha=0.55, edgecolor="none", transform=ax.transAxes))
    ax.text(x + 0.025, y + 0.061, "Exp.1 schema stress set", fontsize=8.5, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(x + 0.025, y + 0.014, "35 cases: 20 valid accepted / 15 invalid rejected", fontsize=7.5, color=PALETTE["muted"], transform=ax.transAxes)


def build_dataset_showcase() -> None:
    setup_style()
    fig = plt.figure(figsize=(16.2, 8.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.958, "(a) Heterogeneous SAR inputs", fontsize=15.0, weight="bold", color=PALETTE["ink"], transform=ax.transAxes)
    ax.text(0.035, 0.928, "SAGA accepts different file layouts, metadata carriers, scene labels, and geometry sources.", fontsize=9.2, color=PALETTE["muted"], transform=ax.transAxes)

    cards = [
        {
            "title": "Class-folder chips",
            "subtitle": "semantic label comes from parent folder",
            "cue": "ship/ship_0000.png",
            "tags": ["class", "chip"],
            "color": PALETTE["blue"],
            "paths": [
                REPO_ROOT / "runs/experiments/exp1_schema_grounding/benchmark_data/class_folder_multi/ship/ship_0000.png",
                REPO_ROOT / "runs/experiments/exp1_schema_grounding/benchmark_data/class_folder_multi/vehicle/vehicle_0000.png",
            ],
            "layout": "pair",
        },
        {
            "title": "Filename metadata",
            "subtitle": "azimuth and polarization encoded in names",
            "cue": "D7_0_{hh,hv,vv}.png",
            "tags": ["azimuth", "pol."],
            "color": PALETTE["teal"],
            "paths": [
                REPO_ROOT / "exampledataset/车辆数据-全极化/ZJGC-X/D7_0_hh.png",
                REPO_ROOT / "exampledataset/车辆数据-全极化/ZJGC-X/D7_0_hv.png",
                REPO_ROOT / "exampledataset/车辆数据-全极化/ZJGC-X/D7_0_vv.png",
            ],
            "layout": "triptych",
        },
        {
            "title": "Nested path fields",
            "subtitle": "class and band in folders, angles in files",
            "cue": "bmp2/Ku/inci-20-azim-40-AHH.png",
            "tags": ["path", "angle"],
            "color": PALETTE["green"],
            "paths": [
                REPO_ROOT / "runs/experiments/exp1_schema_grounding/benchmark_data/vehicle_nested_inci/bmp2（步兵战车)/Ku/inci-20-azim-40.0000-AHH.png"
            ],
            "layout": "single",
        },
        {
            "title": "Sidecar key-values",
            "subtitle": "image and metadata are paired by stem",
            "cue": "BTR70_sidecar_000.{png,txt}",
            "tags": ["sidecar", "band"],
            "color": PALETTE["sand"],
            "paths": [
                REPO_ROOT / "runs/experiments/exp1_schema_grounding/benchmark_data/sidecar_kv_metadata/BTR70/BTR70_sidecar_000.png"
            ],
            "layout": "snippet",
            "snippet": "class: BTR70\nazimuth: 0\nincidence: 20\nband: X\npol: HH",
        },
        {
            "title": "Caption sidecars",
            "subtitle": "free text is parsed into constrained fields",
            "cue": "caption_ship_000.{png,txt}",
            "tags": ["caption", "res."],
            "color": PALETTE["purple"],
            "paths": [
                REPO_ROOT / "runs/experiments/exp1_schema_grounding/benchmark_data/caption_sidecar_metadata/ship/caption_ship_000.png"
            ],
            "layout": "snippet",
            "snippet": "SAR ship target, azimuth 0,\nincidence 25, X-band,\nHH polarization, 0.5 m",
        },
        {
            "title": "Scene annotations",
            "subtitle": "large SAR scenes with detection labels",
            "cue": "images/L135.png + labels/L135.txt",
            "tags": ["bbox", "scene"],
            "color": PALETTE["red"],
            "paths": [REPO_ROOT / "exampledataset/SRSDD-V1.0/train/images/L135.png"],
            "layout": "detection",
            "detection": (
                REPO_ROOT / "exampledataset/SRSDD-V1.0/train/images/L135.png",
                REPO_ROOT / "exampledataset/SRSDD-V1.0/train/labels/L135.txt",
            ),
        },
        {
            "title": "Sparse-view SAR-GS",
            "subtitle": "geometry-aware rendered views are profiled",
            "cue": "fine_gt.png -> fine_render_final.png",
            "tags": ["view", "GS"],
            "color": "#8aa1b4",
            "paths": [
                REPO_ROOT / "myproject/SAR GS V1/output/result/20260414_130723_lowpass_loss-0.054788/fine_gt.png",
                REPO_ROOT / "myproject/SAR GS V1/output/result/20260414_130723_lowpass_loss-0.054788/fine_render_final.png",
            ],
            "layout": "pair",
        },
        {
            "title": "Simulation views",
            "subtitle": "full-azimuth synthetic views and priors",
            "cue": "raytracing/747/{20,96,172}.png",
            "tags": ["sim", "azimuth"],
            "color": "#b28b6c",
            "paths": [
                REPO_ROOT / "myproject/geodiff_components/raytracing/747/20.png",
                REPO_ROOT / "myproject/geodiff_components/raytracing/747/96.png",
                REPO_ROOT / "myproject/geodiff_components/raytracing/747/172.png",
            ],
            "layout": "triptych",
        },
    ]

    grid_x0, grid_y0 = 0.035, 0.110
    card_w, card_h = 0.151, 0.365
    gap_x, gap_y = 0.018, 0.055
    for i, card in enumerate(cards):
        row = 1 - (i // 4)
        col = i % 4
        x = grid_x0 + col * (card_w + gap_x)
        y = grid_y0 + row * (card_h + gap_y)
        draw_card(
            fig,
            ax,
            x,
            y,
            card_w,
            card_h,
            card["title"],
            card["subtitle"],
            card["cue"],
            card["tags"],
            card["color"],
            card["paths"],
            layout=card["layout"],
            text_snippet=card.get("snippet"),
            detection=card.get("detection"),
        )

    arrow = FancyArrowPatch(
        (0.705, 0.500),
        (0.740, 0.500),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        color="#9fb0bc",
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(arrow)
    ax.text(0.675, 0.526, "ground", fontsize=8.4, color=PALETTE["muted"], transform=ax.transAxes)
    ax.text(0.676, 0.506, "validate", fontsize=8.4, color=PALETTE["muted"], transform=ax.transAxes)
    ax.text(0.681, 0.486, "route", fontsize=8.4, color=PALETTE["muted"], transform=ax.transAxes)

    draw_profile_panel(ax, 0.735, 0.125, 0.240, 0.785)

    save_figure(fig, "fig_dataset_diversity_showcase")
    print(f"Wrote {OUT_DIR / 'fig_dataset_diversity_showcase.png'}")


if __name__ == "__main__":
    build_dataset_showcase()

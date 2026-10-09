from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Circle, Wedge


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "experiments" / "paper_ready_group_figures"
FIG_NAME = "fig_dataset_taxonomy_radial"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DatasetRecord:
    group: str
    label: str
    count: int
    rel_path: str


FALLBACK_RECORDS = [
    DatasetRecord("Aircraft", "B52-full-pol", 7200, "provided_inventory/aircraft/B52-full-pol"),
    DatasetRecord("Aircraft", "C130-full-pol", 7200, "provided_inventory/aircraft/C130-full-pol"),
    DatasetRecord("Aircraft", "C17-full-pol", 7200, "provided_inventory/aircraft/C17-full-pol"),
    DatasetRecord("Aircraft", "E2-full-pol", 7200, "provided_inventory/aircraft/E2-full-pol"),
    DatasetRecord("Aircraft", "F35-full-pol", 7200, "provided_inventory/aircraft/F35-full-pol"),
    DatasetRecord("Aircraft", "KC135-full-pol", 7200, "provided_inventory/aircraft/KC135-full-pol"),
    DatasetRecord("Aircraft", "AT-402-1", 3457, "provided_inventory/aircraft/AT-402-1"),
    DatasetRecord("Aircraft", "AT-402-2", 3457, "provided_inventory/aircraft/AT-402-2"),
    DatasetRecord("Aircraft", "Cessna-208", 3457, "provided_inventory/aircraft/Cessna-208"),
    DatasetRecord("Aircraft", "Big-Brown-Bear", 3457, "provided_inventory/aircraft/Big-Brown-Bear"),
    DatasetRecord("Aircraft", "F-15-Eagle", 1031, "provided_inventory/aircraft/F-15-Eagle"),
    DatasetRecord("Aircraft", "C-130-Hercules", 740, "provided_inventory/aircraft/C-130-Hercules"),
    DatasetRecord("Aircraft", "B52-Stratofortress", 645, "provided_inventory/aircraft/B52-Stratofortress"),
    DatasetRecord("Aircraft", "A-10-Thunderbolt-II", 428, "provided_inventory/aircraft/A-10-Thunderbolt-II"),
    DatasetRecord("Aircraft", "F-16-Fighting-Falcon", 364, "provided_inventory/aircraft/F-16-Fighting-Falcon"),
    DatasetRecord("Aircraft", "E-2-Hawkeye", 226, "provided_inventory/aircraft/E-2-Hawkeye"),
    DatasetRecord("Aircraft", "KC135", 216, "provided_inventory/aircraft/KC135"),
    DatasetRecord("Aircraft", "B-1B-Lancer", 178, "provided_inventory/aircraft/B-1B-Lancer"),
    DatasetRecord("Aircraft", "F-4-Phantom-II", 173, "provided_inventory/aircraft/F-4-Phantom-II"),
    DatasetRecord("Aircraft", "UH-60-Black-Hawk", 95, "provided_inventory/aircraft/UH-60-Black-Hawk"),
    DatasetRecord("Aircraft", "C-47-Skytrain", 92, "provided_inventory/aircraft/C-47-Skytrain"),
    DatasetRecord("Aircraft", "EP-3-Aries", 81, "provided_inventory/aircraft/EP-3-Aries"),
    DatasetRecord("Aircraft", "FC-1-Xiaolong", 69, "provided_inventory/aircraft/FC-1-Xiaolong"),
    DatasetRecord("Aircraft", "C-17-Globemaster", 18, "provided_inventory/aircraft/C-17-Globemaster"),
    DatasetRecord("Vehicle", "BMP2-full-pol", 2880, "provided_inventory/vehicle/BMP2-full-pol"),
    DatasetRecord("Vehicle", "BTR60-full-pol", 2880, "provided_inventory/vehicle/BTR60-full-pol"),
    DatasetRecord("Vehicle", "D7-full-pol", 2880, "provided_inventory/vehicle/D7-full-pol"),
    DatasetRecord("Vehicle", "HMS-full-pol", 2880, "provided_inventory/vehicle/HMS-full-pol"),
    DatasetRecord("Vehicle", "M1A1-full-pol", 2880, "provided_inventory/vehicle/M1A1-full-pol"),
    DatasetRecord("Vehicle", "M551-full-pol", 2880, "provided_inventory/vehicle/M551-full-pol"),
    DatasetRecord("Vehicle", "Dongfeng-EQ6608LTV", 2535, "provided_inventory/vehicle/Dongfeng-EQ6608LTV"),
    DatasetRecord("Vehicle", "SDLG-ZL40F", 2500, "provided_inventory/vehicle/SDLG-ZL40F"),
    DatasetRecord("Vehicle", "Changan-CS75-Plus", 2469, "provided_inventory/vehicle/Changan-CS75-Plus"),
    DatasetRecord("Vehicle", "Great-Wall-Voleex-C50", 2426, "provided_inventory/vehicle/Great-Wall-Voleex-C50"),
    DatasetRecord("Vehicle", "ZXTK-full-pol", 2345, "provided_inventory/vehicle/ZXTK-full-pol"),
    DatasetRecord("Vehicle", "WAW-Aochi-Hongrui", 2299, "provided_inventory/vehicle/WAW-Aochi-Hongrui"),
    DatasetRecord("Vehicle", "JINBEI-SY5033XJH", 2204, "provided_inventory/vehicle/JINBEI-SY5033XJH"),
    DatasetRecord("Vehicle", "Dongfeng-Forthing-Lingzhi", 2196, "provided_inventory/vehicle/Dongfeng-Forthing-Lingzhi"),
    DatasetRecord("Vehicle", "Iveco-Proud-2009", 2125, "provided_inventory/vehicle/Iveco-Proud-2009"),
    DatasetRecord("Vehicle", "Buick-Excelle-GT", 2085, "provided_inventory/vehicle/Buick-Excelle-GT"),
    DatasetRecord("Vehicle", "Hawtai-EV160B", 2011, "provided_inventory/vehicle/Hawtai-EV160B"),
    DatasetRecord("Vehicle", "Great-Wall-poer", 2007, "provided_inventory/vehicle/Great-Wall-poer"),
    DatasetRecord("Vehicle", "Mitsubishi-Outlander-2003", 1957, "provided_inventory/vehicle/Mitsubishi-Outlander-2003"),
    DatasetRecord("Vehicle", "ZJYS-full-pol", 1955, "provided_inventory/vehicle/ZJYS-full-pol"),
    DatasetRecord("Vehicle", "JAC-Junling", 1934, "provided_inventory/vehicle/JAC-Junling"),
    DatasetRecord("Vehicle", "Changan-Starlight-4500", 1932, "provided_inventory/vehicle/Changan-Starlight-4500"),
    DatasetRecord("Vehicle", "Buick-GL8", 1914, "provided_inventory/vehicle/Buick-GL8"),
    DatasetRecord("Vehicle", "Hongqi-CA7180A3E", 1866, "provided_inventory/vehicle/Hongqi-CA7180A3E"),
    DatasetRecord("Vehicle", "Chery-qq3", 1838, "provided_inventory/vehicle/Chery-qq3"),
    DatasetRecord("Vehicle", "Yutong-ZK6120HY1", 1791, "provided_inventory/vehicle/Yutong-ZK6120HY1"),
    DatasetRecord("Vehicle", "Foton-BJ1045V9JB5-54", 1774, "provided_inventory/vehicle/Foton-BJ1045V9JB5-54"),
    DatasetRecord("Vehicle", "Yangzi-YZK6590XCA", 1761, "provided_inventory/vehicle/Yangzi-YZK6590XCA"),
    DatasetRecord("Vehicle", "MAXUS-V80", 1746, "provided_inventory/vehicle/MAXUS-V80"),
    DatasetRecord("Vehicle", "Dongfeng-Tianjin-DFH2200B", 1695, "provided_inventory/vehicle/Dongfeng-Tianjin-DFH2200B"),
    DatasetRecord("Vehicle", "Dongfeng-Tianjin-KR230", 1695, "provided_inventory/vehicle/Dongfeng-Tianjin-KR230"),
    DatasetRecord("Vehicle", "Changlin-8228-5", 1692, "provided_inventory/vehicle/Changlin-8228-5"),
    DatasetRecord("Vehicle", "CNHTC-HOWO", 1692, "provided_inventory/vehicle/CNHTC-HOWO"),
    DatasetRecord("Vehicle", "Hyundai-HLF25-II", 1688, "provided_inventory/vehicle/Hyundai-HLF25-II"),
    DatasetRecord("Vehicle", "Lincoln-MKC", 1686, "provided_inventory/vehicle/Lincoln-MKC"),
    DatasetRecord("Vehicle", "Wuling-Rongguang-V", 1668, "provided_inventory/vehicle/Wuling-Rongguang-V"),
    DatasetRecord("Vehicle", "FAW-Jiabao-T51", 1665, "provided_inventory/vehicle/FAW-Jiabao-T51"),
    DatasetRecord("Vehicle", "FAW-J6P", 1659, "provided_inventory/vehicle/FAW-J6P"),
    DatasetRecord("Vehicle", "Huanghai-N1", 1612, "provided_inventory/vehicle/Huanghai-N1"),
    DatasetRecord("Vehicle", "Hongqi-h5", 1602, "provided_inventory/vehicle/Hongqi-h5"),
    DatasetRecord("Vehicle", "WAW-Aochi-1800", 1598, "provided_inventory/vehicle/WAW-Aochi-1800"),
    DatasetRecord("Vehicle", "Chery-Arrizo-5", 1593, "provided_inventory/vehicle/Chery-Arrizo-5"),
    DatasetRecord("Vehicle", "SHACMAN-DeLong-X3000", 1580, "provided_inventory/vehicle/SHACMAN-DeLong-X3000"),
    DatasetRecord("Vehicle", "Jeep-Patriot", 1557, "provided_inventory/vehicle/Jeep-Patriot"),
    DatasetRecord("Vehicle", "Dongfeng-Duolika", 1488, "provided_inventory/vehicle/Dongfeng-Duolika"),
    DatasetRecord("Vehicle", "ZJGC-full-pol", 1370, "provided_inventory/vehicle/ZJGC-full-pol"),
    DatasetRecord("Vehicle", "ZJZC-full-pol", 1370, "provided_inventory/vehicle/ZJZC-full-pol"),
    DatasetRecord("Vehicle", "SHACMAN-DeLong-M3000", 1359, "provided_inventory/vehicle/SHACMAN-DeLong-M3000"),
    DatasetRecord("Vehicle", "Chevrolet-Blazer-1998", 1356, "provided_inventory/vehicle/Chevrolet-Blazer-1998"),
    DatasetRecord("Vehicle", "Changfeng-Cheetah-CFA6473C", 1267, "provided_inventory/vehicle/Changfeng-Cheetah-CFA6473C"),
    DatasetRecord("Vehicle", "BBZC-full-pol", 980, "provided_inventory/vehicle/BBZC-full-pol"),
    DatasetRecord("Vehicle", "2S1", 647, "provided_inventory/vehicle/2S1"),
    DatasetRecord("Vehicle", "ZSU-23-4", 647, "provided_inventory/vehicle/ZSU-23-4"),
    DatasetRecord("Vehicle", "T72-SN-132", 448, "provided_inventory/vehicle/T72-SN-132"),
    DatasetRecord("Vehicle", "BMP2-SN-9563", 447, "provided_inventory/vehicle/BMP2-SN-9563"),
    DatasetRecord("Vehicle", "BTR70-SN-C71", 417, "provided_inventory/vehicle/BTR70-SN-C71"),
    DatasetRecord("Vehicle", "QXTK-full-pol", 360, "provided_inventory/vehicle/QXTK-full-pol"),
    DatasetRecord("Vehicle", "M60", 352, "provided_inventory/vehicle/M60"),
    DatasetRecord("Vehicle", "D7", 299, "provided_inventory/vehicle/D7"),
    DatasetRecord("Vehicle", "T62", 299, "provided_inventory/vehicle/T62"),
    DatasetRecord("Vehicle", "ZIL131", 299, "provided_inventory/vehicle/ZIL131"),
    DatasetRecord("Vehicle", "BRDM-2", 298, "provided_inventory/vehicle/BRDM-2"),
    DatasetRecord("Vehicle", "M1", 258, "provided_inventory/vehicle/M1"),
    DatasetRecord("Vehicle", "M35", 258, "provided_inventory/vehicle/M35"),
    DatasetRecord("Vehicle", "BTR-60", 256, "provided_inventory/vehicle/BTR-60"),
    DatasetRecord("Vehicle", "M2", 256, "provided_inventory/vehicle/M2"),
    DatasetRecord("Vehicle", "M548", 256, "provided_inventory/vehicle/M548"),
    DatasetRecord("Ship", "Ticonderoga-full-pol", 3024, "provided_inventory/ship/Ticonderoga-full-pol"),
    DatasetRecord("Ship", "Izumo-full-pol", 2892, "provided_inventory/ship/Izumo-full-pol"),
    DatasetRecord("Ship", "Hatakaze", 2889, "provided_inventory/ship/Hatakaze"),
    DatasetRecord("Ship", "Hyuga-Ise", 2888, "provided_inventory/ship/Hyuga-Ise"),
    DatasetRecord("Ship", "Perry-full-pol", 2876, "provided_inventory/ship/Perry-full-pol"),
    DatasetRecord("Ship", "Arleigh-Burke-full-pol", 2876, "provided_inventory/ship/Arleigh-Burke-full-pol"),
    DatasetRecord("Ship", "Charles-de-Gaulle", 2748, "provided_inventory/ship/Charles-de-Gaulle"),
    DatasetRecord("Ship", "Ticonderoga", 2700, "provided_inventory/ship/Ticonderoga"),
    DatasetRecord("Ship", "Perry", 2655, "provided_inventory/ship/Perry"),
    DatasetRecord("Ship", "Arleigh-Burke", 2654, "provided_inventory/ship/Arleigh-Burke"),
    DatasetRecord("Ship", "Constellation", 2652, "provided_inventory/ship/Constellation"),
    DatasetRecord("Ship", "Izumo", 2651, "provided_inventory/ship/Izumo"),
    DatasetRecord("Ship", "Murasame", 2651, "provided_inventory/ship/Murasame"),
    DatasetRecord("Ship", "Ford-Kennedy", 2649, "provided_inventory/ship/Ford-Kennedy"),
    DatasetRecord("Ship", "Reagan", 2649, "provided_inventory/ship/Reagan"),
    DatasetRecord("Ship", "Zumwalt", 2648, "provided_inventory/ship/Zumwalt"),
]


GROUP_ORDER = ["Aircraft", "Ship", "Vehicle"]
GROUP_COLORS = {
    "Aircraft": ("#dceefa", "#3d789d", "#17486b"),
    "Ship": ("#ddf0ea", "#409d8d", "#17685e"),
    "Vehicle": ("#f7e2dc", "#c96d62", "#8f3f46"),
}
INK = "#24313b"
MUTED = "#6c7a86"
LINE = "#d6dee4"
SOFT = "#f5f8fa"
MIN_VISIBLE_COUNT = 500

DISPLAY_LABELS = {
    "B52-full-pol": "B52+pol",
    "C130-full-pol": "C130+pol",
    "C17-full-pol": "C17+pol",
    "E2-full-pol": "E2+pol",
    "F35-full-pol": "F35+pol",
    "KC135-full-pol": "KC135+pol",
    "Big-Brown-Bear": "Big Bear",
    "F-15-Eagle": "F-15",
    "C-130-Hercules": "C-130-H",
    "B52-Stratofortress": "B52-S",
    "A-10-Thunderbolt-II": "A-10",
    "F-16-Fighting-Falcon": "F-16",
    "E-2-Hawkeye": "E-2",
    "B-1B-Lancer": "B-1B",
    "F-4-Phantom-II": "F-4",
    "UH-60-Black-Hawk": "UH-60",
    "C-47-Skytrain": "C-47",
    "EP-3-Aries": "EP-3",
    "FC-1-Xiaolong": "FC-1",
    "C-17-Globemaster": "C-17-G",
    "BMP2-full-pol": "BMP2+pol",
    "BTR60-full-pol": "BTR60+pol",
    "D7-full-pol": "D7+pol",
    "HMS-full-pol": "HMS+pol",
    "M1A1-full-pol": "M1A1+pol",
    "M551-full-pol": "M551+pol",
    "Dongfeng-EQ6608LTV": "DF-EQ6608",
    "Changan-CS75-Plus": "CS75+",
    "Great-Wall-Voleex-C50": "GW-C50",
    "WAW-Aochi-Hongrui": "WAW-HR",
    "JINBEI-SY5033XJH": "JINBEI",
    "Dongfeng-Forthing-Lingzhi": "DF-Lingzhi",
    "Iveco-Proud-2009": "Iveco",
    "Buick-Excelle-GT": "Excelle",
    "Great-Wall-poer": "GW-poer",
    "Mitsubishi-Outlander-2003": "Outlander",
    "ZJYS-full-pol": "ZJYS+pol",
    "Changan-Starlight-4500": "Starlight",
    "Hongqi-CA7180A3E": "Hongqi-A3E",
    "Yutong-ZK6120HY1": "Yutong",
    "Foton-BJ1045V9JB5-54": "Foton",
    "Dongfeng-Tianjin-DFH2200B": "DFH2200B",
    "Dongfeng-Tianjin-KR230": "KR230",
    "Hyundai-HLF25-II": "HLF25-II",
    "Lincoln-MKC": "Lincoln",
    "Wuling-Rongguang-V": "Rongguang",
    "FAW-Jiabao-T51": "Jiabo-T51",
    "FAW-J6P": "FAW-J6P",
    "Huanghai-N1": "Huanghai",
    "Hongqi-h5": "Hongqi-h5",
    "WAW-Aochi-1800": "WAW-1800",
    "Chery-Arrizo-5": "Arrizo-5",
    "SHACMAN-DeLong-X3000": "X3000",
    "Jeep-Patriot": "Patriot",
    "Dongfeng-Duolika": "Duolika",
    "ZJGC-full-pol": "ZJGC+pol",
    "ZJZC-full-pol": "ZJZC+pol",
    "SHACMAN-DeLong-M3000": "M3000",
    "Chevrolet-Blazer-1998": "Blazer",
    "Changfeng-Cheetah-CFA6473C": "Cheetah",
    "BBZC-full-pol": "BBZC+pol",
    "ZSU-23-4": "ZSU-23",
    "QXTK-full-pol": "QXTK+pol",
    "Ticonderoga-full-pol": "Ticon+pol",
    "Izumo-full-pol": "Izumo+pol",
    "Hyuga-Ise": "Hyuga",
    "Perry-full-pol": "Perry+pol",
    "Arleigh-Burke-full-pol": "Burke+pol",
    "Charles-de-Gaulle": "Gaulle",
    "Ticonderoga": "Ticon",
    "Arleigh-Burke": "Burke",
    "Ford-Kennedy": "Ford",
}

LABEL_INDEXES = {
    "Aircraft": set(range(13)),
    "Ship": {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15},
    "Vehicle": {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52},
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 10.5,
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "axes.unicode_minus": False,
        }
    )


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_records() -> list[DatasetRecord]:
    return list(FALLBACK_RECORDS)


def write_counts_csv(records: list[DatasetRecord]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "dataset_taxonomy_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "label", "count", "shown_in_figure", "relative_path"])
        for rec in records:
            shown = rec.count >= MIN_VISIBLE_COUNT
            writer.writerow([rec.group, rec.label, rec.count, int(shown), rec.rel_path])


def lighten(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def make_group_colors(group: str, n: int) -> list[str]:
    lo, mid, hi = GROUP_COLORS[group]
    cmap = LinearSegmentedColormap.from_list(f"{group}_cmap", [lo, mid, hi])
    if n == 1:
        return [mid]
    return [cmap(0.22 + 0.68 * i / (n - 1)) for i in range(n)]


def display_label(label: str) -> str:
    if label in DISPLAY_LABELS:
        return DISPLAY_LABELS[label]
    label = label.replace("-full-pol", "+pol")
    if len(label) <= 15:
        return label
    parts = label.split("-")
    if len(parts) >= 2:
        short = "-".join(parts[:2])
        if len(short) <= 15:
            return short
    return label[:14] + "."


def select_labeled_records(group_records: dict[str, list[DatasetRecord]]) -> dict[DatasetRecord, int]:
    labeled: dict[DatasetRecord, int] = {}
    for group, rows in group_records.items():
        for idx, rec in enumerate(rows):
            if idx in LABEL_INDEXES[group]:
                labeled[rec] = idx
    return labeled


def polar_text_rotation(theta_deg: float) -> tuple[float, str]:
    rotation = theta_deg - 90
    ha = "left"
    if 90 < theta_deg < 270:
        rotation += 180
        ha = "right"
    return rotation, ha


def radial_text_rotation(theta_deg: float) -> tuple[float, str]:
    # Matplotlib text rotation is measured from the horizontal screen axis.
    # With zero at north and clockwise theta, a radial label uses 90 - theta.
    rotation = (90.0 - theta_deg + 180.0) % 360.0 - 180.0
    if rotation > 90.0:
        rotation -= 180.0
    elif rotation < -90.0:
        rotation += 180.0
    return rotation, "center"


def add_arc_label(
    ax: plt.Axes,
    start_deg: float,
    end_deg: float,
    radius: float,
    text: str,
    color: str,
    fontsize: float = 12.5,
) -> None:
    theta_deg = (start_deg + end_deg) / 2
    theta = np.deg2rad(theta_deg)
    rotation, ha = polar_text_rotation(theta_deg)
    ax.text(
        theta,
        radius,
        text,
        ha=ha,
        va="center",
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=fontsize,
        color=color,
        weight="bold",
        clip_on=False,
    )


def add_group_badge(
    ax: plt.Axes,
    start_deg: float,
    end_deg: float,
    radius: float,
    group: str,
    total: int,
) -> None:
    theta = np.deg2rad((start_deg + end_deg) / 2)
    ax.text(
        theta,
        radius,
        f"{group}  {total:,}",
        ha="center",
        va="center",
        rotation=0,
        fontsize=10.8,
        color=GROUP_COLORS[group][2],
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.26,rounding_size=0.12",
            "fc": "white",
            "ec": lighten(GROUP_COLORS[group][1], 0.54),
            "lw": 0.8,
            "alpha": 0.92,
        },
        clip_on=False,
        zorder=24,
    )


def draw_radial_taxonomy(records: list[DatasetRecord]) -> plt.Figure:
    all_group_records = {group: [r for r in records if r.group == group] for group in GROUP_ORDER}
    group_records = {
        group: [r for r in all_group_records[group] if r.count >= MIN_VISIBLE_COUNT]
        for group in GROUP_ORDER
    }
    for group in GROUP_ORDER:
        group_records[group] = sorted(group_records[group], key=lambda r: r.count, reverse=True)
    labeled_records = select_labeled_records(group_records)

    totals = {group: sum(r.count for r in all_group_records[group]) for group in GROUP_ORDER}
    visible_totals = {group: sum(r.count for r in group_records[group]) for group in GROUP_ORDER}
    hidden_counts = {group: len(all_group_records[group]) - len(group_records[group]) for group in GROUP_ORDER}
    grand_total = sum(totals.values())
    max_count = max(r.count for rows in group_records.values() for r in rows)

    # Fixed spans keep the three color sections legible while giving the dense vehicle sector more room.
    gap = 9.0
    sector_span = {"Aircraft": 92.0, "Ship": 76.0, "Vehicle": 165.0}

    inner = 2.20
    max_height = 3.05
    outer = inner + max_height
    label_radius = outer + 0.43

    fig = plt.figure(figsize=(12.4, 8.9), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.040, right=0.960, top=0.975, bottom=0.045)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, label_radius + 0.64)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # Subtle circular reference grid for sqrt-scaled counts.
    for level in [250, 1000, 3000, 5000]:
        r = inner + np.sqrt(level) / np.sqrt(max_count) * max_height
        theta = np.linspace(0, 2 * np.pi, 720)
        ax.plot(theta, np.full_like(theta, r), color=LINE, lw=0.85, ls=(0, (2.5, 4.0)), zorder=0)
        ax.text(
            np.deg2rad(354),
            r,
            f"{level:,}",
            ha="right",
            va="center",
            fontsize=8.4,
            color="#8996a0",
            bbox={"boxstyle": "round,pad=0.13", "fc": "white", "ec": "none", "alpha": 0.78},
            zorder=10,
        )

    theta_cursor = -151.0
    sector_ranges: dict[str, tuple[float, float]] = {}
    all_bar_positions: list[tuple[DatasetRecord, float, float, float]] = []

    for group in GROUP_ORDER:
        rows = group_records[group]
        start = theta_cursor
        end = theta_cursor + sector_span[group]
        sector_ranges[group] = (start, end)

        # Background sector band.
        bg = Wedge(
            (0.5, 0.5),
            0.497,
            90 - end,
            90 - start,
            width=0.110,
            transform=ax.transAxes,
            facecolor=lighten(GROUP_COLORS[group][1], 0.73),
            edgecolor="none",
            alpha=0.58,
            zorder=-2,
        )
        ax.add_patch(bg)

        n = len(rows)
        step = sector_span[group] / n
        width = np.deg2rad(step * 0.78)
        colors = make_group_colors(group, n)
        for i, rec in enumerate(rows):
            theta_deg = start + step * (i + 0.5)
            theta = np.deg2rad(theta_deg)
            height = np.sqrt(rec.count) / np.sqrt(max_count) * max_height
            edge_lw = 1.05 if n <= 24 else 0.42
            ax.bar(
                theta,
                height,
                width=width,
                bottom=inner,
                color=colors[i],
                edgecolor="white",
                linewidth=edge_lw,
                alpha=0.96,
                zorder=4,
            )
            # Small cap gives every bar a clean endpoint without black error-bar-like lines.
            cap_size = 12 if n <= 24 else 5
            ax.scatter([theta], [inner + height], s=cap_size, color=colors[i], edgecolor="white", linewidth=0.35, zorder=6)
            all_bar_positions.append((rec, theta_deg, theta, height))

        theta_cursor = end + gap

    # Outer labels after bars so text sits on top. Small labels intentionally use short names.
    for rec, theta_deg, theta, height in all_bar_positions:
        if rec not in labeled_records:
            continue
        rotation, ha = radial_text_rotation(theta_deg)
        idx = labeled_records[rec]
        radial_offset = 0.02
        if rec.group == "Vehicle":
            radial_offset = 0.01 + 0.075 * (idx % 2)
        elif rec.group == "Aircraft" and idx >= 10:
            radial_offset = 0.085
        ax.text(
            theta,
            label_radius + radial_offset,
            f"{display_label(rec.label)}\n{rec.count:,}",
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=7.25 if rec.group == "Vehicle" else 7.65,
            color=INK,
            linespacing=0.93,
            clip_on=False,
            zorder=12,
        )

    # Group separators. The colored background sectors and compact legend encode group identity.
    for group, (start, end) in sector_ranges.items():
        for deg in [start, end]:
            theta = np.deg2rad(deg)
            ax.plot([theta, theta], [inner - 0.05, outer + 0.24], color="white", lw=2.6, zorder=7)
            ax.plot([theta, theta], [inner - 0.05, outer + 0.24], color="#cfd9df", lw=0.75, zorder=8)

    # Inner circle and center annotation.
    center = Circle((0.5, 0.5), 0.232, transform=ax.transAxes, facecolor="white", edgecolor=LINE, linewidth=1.0, zorder=20)
    ax.add_patch(center)
    ax.text(
        0.5,
        0.548,
        "SAGA",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=25,
        weight="bold",
        color=INK,
        zorder=21,
    )
    ax.text(
        0.5,
        0.502,
        "SAR Dataset Taxonomy",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.6,
        color=MUTED,
        zorder=21,
    )
    ax.text(
        0.5,
        0.463,
        f"{grand_total:,} images",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15.4,
        weight="bold",
        color=INK,
        zorder=21,
    )
    totals_line = "   ".join(f"{group}: {totals[group]:,}" for group in GROUP_ORDER)
    ax.text(
        0.5,
        0.421,
        totals_line,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.55,
        color="#5f6d77",
        zorder=21,
    )
    ax.text(
        0.5,
        0.389,
        f"Displayed bars: >= {MIN_VISIBLE_COUNT:,} images, {sum(visible_totals.values()):,} shown",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        color="#8b98a2",
        zorder=21,
    )

    return fig


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    write_counts_csv(records)
    fig = draw_radial_taxonomy(records)
    fig.savefig(OUT_DIR / f"{FIG_NAME}.png", bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(OUT_DIR / f"{FIG_NAME}.pdf", bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(OUT_DIR / f"{FIG_NAME}.png")
    print(OUT_DIR / f"{FIG_NAME}.pdf")
    print(OUT_DIR / "dataset_taxonomy_counts.csv")


if __name__ == "__main__":
    main()

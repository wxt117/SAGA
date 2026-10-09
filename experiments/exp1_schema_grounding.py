from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saga.core.config import load_mapping, save_json, save_text
from saga.data.discovery import IMAGE_EXTS, natural_key
from saga.data.format_bridge import compile_and_validate_format
from saga.data.profile import profile_dataset


FieldMap = dict[str, Any]
GoldExtractor = Callable[[Path, str], FieldMap]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    root: str
    request: str
    required_fields: list[str]
    expected_valid: bool
    hints: list[dict[str, Any]] = field(default_factory=list)
    task: str = "classification"
    filters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    gold_extractor_name: str = "none"


@dataclass(frozen=True)
class Variant:
    method_id: str
    title: str
    use_hints: bool
    include_heuristics: bool
    validator_gate: bool
    description: str


VARIANTS = [
    Variant(
        method_id="fallback_only",
        title="Fallback + validator",
        use_hints=False,
        include_heuristics=False,
        validator_gate=True,
        description=(
            "Directory fallback and sidecar parsers only; no semantic filename rules. "
            "The same deterministic validator is still used to decide acceptance."
        ),
    ),
    Variant(
        method_id="rule_only",
        title="Rule + validator",
        use_hints=False,
        include_heuristics=True,
        validator_gate=True,
        description=(
            "Built-in deterministic filename heuristics without user semantic hints, "
            "followed by the same validator gate."
        ),
    ),
    Variant(
        method_id="hint_only",
        title="Hint + validator",
        use_hints=True,
        include_heuristics=False,
        validator_gate=True,
        description="User or LLM-like semantic hints, followed by deterministic validation.",
    ),
    Variant(
        method_id="saga_no_validator",
        title="Full SAGA w/o validator",
        use_hints=True,
        include_heuristics=True,
        validator_gate=False,
        description="Candidate schema is accepted without the validator gate.",
    ),
    Variant(
        method_id="full_saga",
        title="Full SAGA",
        use_hints=True,
        include_heuristics=True,
        validator_gate=True,
        description="Hints plus deterministic heuristics, accepted only after validation.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAGA Experiment 1: schema-grounded dataset profiling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "experiments" / "exp1_schema_grounding",
        help="Experiment output directory.",
    )
    parser.add_argument("--sample-limit", type=int, default=160, help="Validation sample limit per case.")
    parser.add_argument("--image-limit", type=int, default=80, help="Maximum images copied into each fixture.")
    parser.add_argument("--skip-plots", action="store_true", help="Write tables only.")
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    benchmark_root = output_dir / "benchmark_data"
    case_run_root = output_dir / "case_runs"
    figures_dir = output_dir / "figures"
    reset_dir(benchmark_root)
    reset_dir(case_run_root)
    reset_dir(figures_dir)

    cases = prepare_benchmark_data(benchmark_root=benchmark_root, image_limit=args.image_limit)
    results, details = run_experiment(
        cases=cases,
        output_dir=output_dir,
        case_run_root=case_run_root,
        sample_limit=args.sample_limit,
    )

    summary = summarize_results(results)
    write_csv(output_dir / "results.csv", results)
    write_csv(output_dir / "summary_by_method.csv", summary)
    write_csv(output_dir / "case_catalog.csv", [case_to_row(case) for case in cases])
    save_json(output_dir / "case_details.json", details)

    if not args.skip_plots:
        plot_all(results=results, cases=cases, summary=summary, figures_dir=figures_dir)

    latex = render_latex_summary_table(summary)
    save_text(output_dir / "table_exp1_summary.tex", latex)
    report = render_report(output_dir=output_dir, cases=cases, variants=VARIANTS, summary=summary, results=results)
    save_text(output_dir / "exp1_report.md", report)

    print(f"Experiment 1 complete: {output_dir}")
    print(f"Results: {output_dir / 'results.csv'}")
    print(f"Summary: {output_dir / 'summary_by_method.csv'}")
    print(f"Report: {output_dir / 'exp1_report.md'}")
    if not args.skip_plots:
        print(f"Figures: {figures_dir}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_benchmark_data(benchmark_root: Path, image_limit: int) -> list[BenchmarkCase]:
    roots: dict[str, Path] = {}
    roots["class_folder"] = build_class_folder_fixture(benchmark_root / "class_folder_multi", image_limit)
    roots["vehicle_suffix"] = build_vehicle_suffix_fixture(benchmark_root / "vehicle_suffix_pol", image_limit)
    roots["vehicle_nested"] = build_vehicle_nested_fixture(benchmark_root / "vehicle_nested_inci", image_limit)
    roots["aircraft_anchor"] = build_aircraft_anchor_fixture(benchmark_root / "aircraft_anchor_850", image_limit)
    roots["syy_elev"] = build_syy_fixture(benchmark_root / "syy_elev_azim_band", image_limit)
    roots["sidecar_kv"] = build_sidecar_kv_fixture(benchmark_root / "sidecar_kv_metadata", min(image_limit, 32))
    roots["caption_sidecar"] = build_caption_sidecar_fixture(
        benchmark_root / "caption_sidecar_metadata", min(image_limit, 32)
    )
    roots["suffix_pol"] = build_suffix_pol_fixture(benchmark_root / "suffix_pol_metadata", min(image_limit, 32))
    roots["ship_only"] = build_ship_only_fixture(benchmark_root / "ship_chips_only", min(image_limit, 48))
    roots["ship_plain"] = build_plain_ship_fixture(benchmark_root / "ship_plain_no_suffix", min(image_limit, 48))

    cases = [
        BenchmarkCase(
            case_id="class_folder_multi",
            title="Class-folder chips",
            root=roots["class_folder"].as_posix(),
            request="The dataset is organized by class folders. Only the class label is required.",
            required_fields=["class"],
            expected_valid=True,
            description="A basic target-chip layout where the only trusted semantic field is the parent class folder.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="vehicle_suffix_pol",
            title="Vehicle filename azimuth/polarization",
            root=roots["vehicle_suffix"].as_posix(),
            request="Filenames follow target_azimuth_polarization.png, e.g., D7_135_hv.png.",
            required_fields=["class", "azimuth_deg", "polarization"],
            expected_valid=True,
            hints=[
                anchor_hint(
                    anchor="D7",
                    fields=["azimuth_deg", "polarization"],
                    text="For D7 chips, the fields after the D7 token are azimuth_deg and polarization.",
                )
            ],
            description="Tests whether SAGA can promote filename tokens into validated metadata.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="vehicle_suffix_band_constant",
            title="Vehicle filename paraphrased hint",
            root=roots["vehicle_suffix"].as_posix(),
            request="The vehicle name is followed by azimuth and polarization tokens; these chips are X-band but the user only needs class, azimuth, and polarization.",
            required_fields=["class", "azimuth_deg", "polarization"],
            expected_valid=True,
            hints=[
                anchor_hint(
                    anchor="D7",
                    fields=["azimuth_deg", "polarization"],
                    text="For D7 chips, the fields after the D7 token are azimuth_deg and polarization.",
                )
            ],
            description="Tests a paraphrased filename-schema hint without requiring an extra constant field.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="suffix_pol_hint",
            title="Suffix polarization hint",
            root=roots["suffix_pol"].as_posix(),
            request="The filename suffix after the final underscore is polarization.",
            required_fields=["class", "polarization"],
            expected_valid=True,
            hints=[
                suffix_field_hint(
                    field="polarization",
                    text="The final underscore suffix in each filename is the polarization field.",
                )
            ],
            description="Tests a user/LLM-proposed suffix-field schema that is not covered by the built-in SAR heuristics.",
            gold_extractor_name="suffix_pol",
        ),
        BenchmarkCase(
            case_id="vehicle_nested_inci_azim_pol",
            title="Nested vehicle incidence/azimuth/polarization",
            root=roots["vehicle_nested"].as_posix(),
            request="Nested folders encode class and band; filenames encode incidence, azimuth, and polarization.",
            required_fields=["class", "band", "incidence_angle_deg", "azimuth_deg", "polarization"],
            expected_valid=True,
            description="Tests deterministic SAR-specific path heuristics on nested class/band folders.",
            gold_extractor_name="vehicle_nested",
        ),
        BenchmarkCase(
            case_id="aircraft_anchor_850",
            title="Aircraft anchor-token metadata",
            root=roots["aircraft_anchor"].as_posix(),
            request="In B747 filenames, fields after the 850 anchor are depression angle, azimuth, resolution, and band.",
            required_fields=["class", "depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
            expected_valid=True,
            hints=[
                anchor_hint(
                    anchor="850",
                    fields=["depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
                    text="The four tokens after 850 are depression_angle_deg, azimuth_deg, resolution_m, and band.",
                )
            ],
            description="Tests whether user semantic hints can override an otherwise ambiguous numeric token layout.",
            gold_extractor_name="aircraft_anchor",
        ),
        BenchmarkCase(
            case_id="syy_elev_azim_band",
            title="Elevation/azimuth/band filenames",
            root=roots["syy_elev"].as_posix(),
            request="Filenames contain elev, azim, and band tokens.",
            required_fields=["class", "elevation_angle_deg", "azimuth_deg", "band"],
            expected_valid=True,
            description="Tests class-folder plus explicit keyword metadata in filenames.",
            gold_extractor_name="syy_elev",
        ),
        BenchmarkCase(
            case_id="sidecar_kv_metadata",
            title="Key-value sidecar metadata",
            root=roots["sidecar_kv"].as_posix(),
            request="Each image has a txt sidecar with class, azimuth, incidence, band, and polarization.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg", "band", "polarization"],
            expected_valid=True,
            description="Tests deterministic sidecar parsing independent of filename semantics.",
            gold_extractor_name="sidecar_kv",
        ),
        BenchmarkCase(
            case_id="caption_sidecar_metadata",
            title="Caption sidecar metadata",
            root=roots["caption_sidecar"].as_posix(),
            request="Each txt caption describes a SAR ship target with azimuth, incidence, band, polarization, and resolution.",
            required_fields=[
                "class",
                "azimuth_deg",
                "incidence_angle_deg",
                "band",
                "polarization",
                "resolution_m",
            ],
            expected_valid=True,
            description="Tests lightweight caption-derived metadata extraction.",
            gold_extractor_name="caption_sidecar",
        ),
        BenchmarkCase(
            case_id="sidecar_missing_resolution_rejected",
            title="Sidecar missing requested resolution",
            root=roots["sidecar_kv"].as_posix(),
            request="Each txt sidecar has class, azimuth, incidence, band, and polarization, but the user also asks for resolution.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg", "band", "polarization", "resolution_m"],
            expected_valid=False,
            description="Tests rejection when sidecars are otherwise parseable but a requested metadata field is absent.",
            gold_extractor_name="sidecar_kv",
        ),
        BenchmarkCase(
            case_id="wrong_hint_rejected",
            title="Wrong semantic hint rejection",
            root=roots["vehicle_suffix"].as_posix(),
            request="Incorrect hint: filenames supposedly contain the 850 anchor and depression/resolution fields.",
            required_fields=["class", "depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
            expected_valid=False,
            hints=[
                anchor_hint(
                    anchor="850",
                    fields=["depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
                    text="Incorrect hint intentionally used for rejection testing.",
                )
            ],
            description="Tests whether the validator rejects unsupported user or LLM semantic proposals.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="wrong_suffix_hint_rejected",
            title="Wrong suffix hint rejection",
            root=roots["ship_plain"].as_posix(),
            request="Incorrect hint: the final underscore suffix is polarization, although the files are plain shipchip0001.png chips.",
            required_fields=["class", "polarization"],
            expected_valid=False,
            hints=[
                suffix_field_hint(
                    field="polarization",
                    text="Incorrect hint intentionally claims a polarization suffix.",
                )
            ],
            description="Tests whether unsupported suffix-field hints are rejected after validation.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="missing_azimuth_rejected",
            title="Missing requested azimuth",
            root=roots["ship_only"].as_posix(),
            request="The user asks to filter ship chips by azimuth, but the dataset has no azimuth metadata.",
            required_fields=["class", "azimuth_deg"],
            expected_valid=False,
            description="Tests clarification/rejection when the requested semantic field is not present.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="class_folder_azimuth_rejected",
            title="Class-folder with unsupported azimuth filter",
            root=roots["class_folder"].as_posix(),
            request="The user asks to filter class-folder chips by azimuth, but only class folders are available.",
            required_fields=["class", "azimuth_deg"],
            expected_valid=False,
            description="Tests rejection when a simple class-folder layout is over-specified by the request.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="syy_missing_polarization_rejected",
            title="Elevation dataset missing polarization",
            root=roots["syy_elev"].as_posix(),
            request="Filenames contain elev, azim, and band tokens, but the user also requests polarization.",
            required_fields=["class", "elevation_angle_deg", "azimuth_deg", "band", "polarization"],
            expected_valid=False,
            description="Tests rejection of a partially valid filename schema when a requested field is absent.",
            gold_extractor_name="syy_elev",
        ),
    ]
    cases.extend(additional_schema_stress_cases(roots))
    if len(cases) != 35:
        raise RuntimeError(f"Experiment 1 should contain 35 schema cases, got {len(cases)}")
    return cases


def additional_schema_stress_cases(roots: dict[str, Path]) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="class_folder_two_class_filter",
            title="Class-folder two-class selection",
            root=roots["class_folder"].as_posix(),
            request="The dataset uses ship and vehicle class folders; keep class labels and validate a two-class split.",
            required_fields=["class"],
            expected_valid=True,
            filters={"class": ["ship", "vehicle"]},
            description="Validates class-folder semantics under a class-filtered request.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="class_folder_resolution_rejected",
            title="Class-folder missing resolution",
            root=roots["class_folder"].as_posix(),
            request="Class folders are present, but the user also asks for per-image resolution metadata.",
            required_fields=["class", "resolution_m"],
            expected_valid=False,
            description="Rejects over-specified requests when only folder labels exist.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="vehicle_suffix_class_only",
            title="Vehicle suffix class-only fallback",
            root=roots["vehicle_suffix"].as_posix(),
            request="Use only the parent class folder from the vehicle filename dataset.",
            required_fields=["class"],
            expected_valid=True,
            description="Checks that richer filename layouts can still satisfy class-only requests.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="vehicle_suffix_band_request",
            title="Vehicle suffix constant band",
            root=roots["vehicle_suffix"].as_posix(),
            request="Filenames encode class, azimuth, and polarization; all samples are X-band.",
            required_fields=["class", "azimuth_deg", "polarization", "band"],
            expected_valid=True,
            hints=[
                anchor_hint(
                    anchor="D7",
                    fields=["azimuth_deg", "polarization"],
                    text="For D7 chips, the two tokens after the target token are azimuth_deg and polarization; band is a constant X-band dataset attribute.",
                )
            ],
            description="Tests a filename schema plus constant dataset-level band metadata.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="vehicle_suffix_missing_incidence_rejected",
            title="Vehicle suffix missing incidence",
            root=roots["vehicle_suffix"].as_posix(),
            request="Use class, azimuth, polarization, and incidence angle from the ZJGC-X filenames.",
            required_fields=["class", "azimuth_deg", "polarization", "incidence_angle_deg"],
            expected_valid=False,
            hints=[
                anchor_hint(
                    anchor="D7",
                    fields=["azimuth_deg", "polarization"],
                    text="For D7 chips, the fields after the D7 token are azimuth_deg and polarization.",
                )
            ],
            description="Rejects a partially valid filename schema when incidence is requested but absent.",
            gold_extractor_name="vehicle_suffix",
        ),
        BenchmarkCase(
            case_id="vehicle_nested_filter_hv",
            title="Nested vehicle HV filter",
            root=roots["vehicle_nested"].as_posix(),
            request="Nested vehicle folders encode band and filenames encode incidence, azimuth, and polarization; select HV chips.",
            required_fields=["class", "band", "incidence_angle_deg", "azimuth_deg", "polarization"],
            expected_valid=True,
            filters={"polarization": "HV"},
            description="Valid nested metadata with a polarization filter.",
            gold_extractor_name="vehicle_nested",
        ),
        BenchmarkCase(
            case_id="vehicle_nested_missing_resolution_rejected",
            title="Nested vehicle missing resolution",
            root=roots["vehicle_nested"].as_posix(),
            request="Nested vehicle folders encode band and filenames encode incidence/azimuth/polarization, but the request also needs resolution.",
            required_fields=["class", "band", "incidence_angle_deg", "azimuth_deg", "polarization", "resolution_m"],
            expected_valid=False,
            description="Rejects missing resolution in an otherwise parseable nested layout.",
            gold_extractor_name="vehicle_nested",
        ),
        BenchmarkCase(
            case_id="aircraft_anchor_angle_only",
            title="Aircraft anchor angle-only",
            root=roots["aircraft_anchor"].as_posix(),
            request="In B747 filenames, after token 850, parse depression angle and azimuth for angle-coverage analysis.",
            required_fields=["class", "depression_angle_deg", "azimuth_deg"],
            expected_valid=True,
            hints=[
                anchor_hint(
                    anchor="850",
                    fields=["depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
                    text="The tokens after 850 are depression_angle_deg, azimuth_deg, resolution_m, and band.",
                )
            ],
            description="Validates partial use of a richer anchor-token schema.",
            gold_extractor_name="aircraft_anchor",
        ),
        BenchmarkCase(
            case_id="aircraft_anchor_missing_polarization_rejected",
            title="Aircraft anchor missing polarization",
            root=roots["aircraft_anchor"].as_posix(),
            request="Parse B747 depression, azimuth, resolution, band, and polarization from filenames.",
            required_fields=["class", "depression_angle_deg", "azimuth_deg", "resolution_m", "band", "polarization"],
            expected_valid=False,
            hints=[
                anchor_hint(
                    anchor="850",
                    fields=["depression_angle_deg", "azimuth_deg", "resolution_m", "band"],
                    text="The four tokens after 850 are depression_angle_deg, azimuth_deg, resolution_m, and band.",
                )
            ],
            description="Rejects a request that adds polarization to an anchor-token layout that lacks it.",
            gold_extractor_name="aircraft_anchor",
        ),
        BenchmarkCase(
            case_id="syy_angle_filter",
            title="SYY angle-band filter",
            root=roots["syy_elev"].as_posix(),
            request="Use elev, azim, and band tokens and filter to X-band armored-car chips.",
            required_fields=["class", "elevation_angle_deg", "azimuth_deg", "band"],
            expected_valid=True,
            filters={"band": "X"},
            description="Valid keyword-token filename schema with a band filter.",
            gold_extractor_name="syy_elev",
        ),
        BenchmarkCase(
            case_id="syy_missing_incidence_rejected",
            title="SYY missing incidence",
            root=roots["syy_elev"].as_posix(),
            request="Filenames contain elev, azim, and band tokens, but the user asks for incidence as well.",
            required_fields=["class", "elevation_angle_deg", "azimuth_deg", "band", "incidence_angle_deg"],
            expected_valid=False,
            description="Rejects incidence requests when only elevation is encoded.",
            gold_extractor_name="syy_elev",
        ),
        BenchmarkCase(
            case_id="sidecar_kv_class_angle_only",
            title="Sidecar class-angle subset",
            root=roots["sidecar_kv"].as_posix(),
            request="Use same-stem txt sidecars, but only class, azimuth, and incidence are required.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg"],
            expected_valid=True,
            description="Validates subset extraction from key-value sidecars.",
            gold_extractor_name="sidecar_kv",
        ),
        BenchmarkCase(
            case_id="sidecar_kv_missing_depression_rejected",
            title="Sidecar missing depression",
            root=roots["sidecar_kv"].as_posix(),
            request="Sidecars have class, azimuth, incidence, band, and polarization; the user also requests depression angle.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg", "band", "polarization", "depression_angle_deg"],
            expected_valid=False,
            description="Rejects unsupported depression-angle requests for key-value sidecars.",
            gold_extractor_name="sidecar_kv",
        ),
        BenchmarkCase(
            case_id="caption_sidecar_no_resolution",
            title="Caption sidecar without resolution request",
            root=roots["caption_sidecar"].as_posix(),
            request="Captions describe ship target class, azimuth, incidence, band, and polarization.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg", "band", "polarization"],
            expected_valid=True,
            description="Validates caption metadata when resolution is not required.",
            gold_extractor_name="caption_sidecar",
        ),
        BenchmarkCase(
            case_id="caption_sidecar_missing_depression_rejected",
            title="Caption sidecar missing depression",
            root=roots["caption_sidecar"].as_posix(),
            request="Captions describe ships, but the user asks for depression angle in addition to caption metadata.",
            required_fields=["class", "azimuth_deg", "incidence_angle_deg", "band", "polarization", "depression_angle_deg"],
            expected_valid=False,
            description="Rejects a caption-derived schema when a non-captioned field is required.",
            gold_extractor_name="caption_sidecar",
        ),
        BenchmarkCase(
            case_id="suffix_pol_class_pol_only",
            title="Suffix polarization class-pol",
            root=roots["suffix_pol"].as_posix(),
            request="The final filename suffix is polarization; only class and polarization are required.",
            required_fields=["class", "polarization"],
            expected_valid=True,
            hints=[
                suffix_field_hint(
                    field="polarization",
                    text="The final underscore suffix in each filename is the polarization field.",
                )
            ],
            description="Valid suffix schema for class and polarization.",
            gold_extractor_name="suffix_pol",
        ),
        BenchmarkCase(
            case_id="suffix_pol_missing_azimuth_rejected",
            title="Suffix polarization missing azimuth",
            root=roots["suffix_pol"].as_posix(),
            request="The final filename suffix is polarization, and the user additionally asks for azimuth.",
            required_fields=["class", "polarization", "azimuth_deg"],
            expected_valid=False,
            hints=[
                suffix_field_hint(
                    field="polarization",
                    text="The final underscore suffix in each filename is the polarization field.",
                )
            ],
            description="Rejects azimuth requests for a suffix-only polarization layout.",
            gold_extractor_name="suffix_pol",
        ),
        BenchmarkCase(
            case_id="plain_ship_class_only",
            title="Plain ship class-only",
            root=roots["ship_plain"].as_posix(),
            request="Plain shipchip files sit under a ship folder; only class labels are required.",
            required_fields=["class"],
            expected_valid=True,
            description="Validates simple class extraction from plain filenames.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="plain_ship_pol_rejected",
            title="Plain ship missing polarization",
            root=roots["ship_plain"].as_posix(),
            request="Plain shipchip files are under a ship folder, but the user asks for polarization and band.",
            required_fields=["class", "polarization", "band"],
            expected_valid=False,
            description="Rejects semantic fields absent from plain chip names.",
            gold_extractor_name="parent_class",
        ),
        BenchmarkCase(
            case_id="ship_only_manifest_count_class",
            title="Ship-only class manifest",
            root=roots["ship_only"].as_posix(),
            request="Build a manifest for ship chips and preserve only the ship class label.",
            required_fields=["class"],
            expected_valid=True,
            description="Validates a chip-only export request with no metadata beyond class.",
            gold_extractor_name="parent_class",
        ),
    ]


def build_class_folder_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    copy_images(
        REPO_ROOT / "exampledataset" / "ship" / "cnt",
        dst / "ship",
        limit=max(4, image_limit // 2),
        rename_prefix="ship",
    )
    copy_images(
        REPO_ROOT / "exampledataset" / "车辆数据-全极化" / "ZJGC-X",
        dst / "vehicle",
        limit=max(4, image_limit // 2),
        rename_prefix="vehicle",
    )
    return dst


def build_vehicle_suffix_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    copy_images(
        REPO_ROOT / "exampledataset" / "车辆数据-全极化" / "ZJGC-X",
        dst / "ZJGC-X",
        limit=image_limit,
    )
    return dst


def build_vehicle_nested_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src = REPO_ROOT / "exampledataset" / "车辆数据-全极化" / "bmp2（步兵战车)"
    copy_images(src, dst / "bmp2（步兵战车)", limit=image_limit, preserve_relative_root=src)
    return dst


def build_aircraft_anchor_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    copy_images(REPO_ROOT / "exampledataset" / "B747", dst / "B747", limit=image_limit)
    return dst


def build_syy_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src = REPO_ROOT / "exampledataset" / "syy_armoredcars"
    copy_images(src, dst, limit=image_limit, preserve_relative_root=src)
    return dst


def build_sidecar_kv_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src_images = list_images(REPO_ROOT / "exampledataset" / "ship" / "cnt")[:image_limit]
    pols = ["HH", "HV", "VH", "VV"]
    for idx, src in enumerate(src_images):
        target = dst / "BTR70" / f"BTR70_sidecar_{idx:03d}{src.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        azimuth = (idx * 17) % 360
        incidence = [20, 30, 45, 60][idx % 4]
        pol = pols[idx % len(pols)]
        txt = "\n".join(
            [
                "class: BTR70",
                f"azimuth: {azimuth}",
                f"incidence: {incidence}",
                "band: X",
                f"pol: {pol}",
                "",
            ]
        )
        target.with_suffix(".txt").write_text(txt, encoding="utf-8")
    return dst


def build_caption_sidecar_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src_images = list_images(REPO_ROOT / "exampledataset" / "ship" / "cnt")[:image_limit]
    pols = ["HH", "HV", "VH", "VV"]
    for idx, src in enumerate(src_images):
        target = dst / "ship" / f"caption_ship_{idx:03d}{src.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        azimuth = (idx * 23) % 360
        incidence = [25, 35, 45, 55][idx % 4]
        pol = pols[idx % len(pols)]
        caption = (
            f"SAR ship target, azimuth {azimuth}, incidence {incidence}, "
            f"X-band, {pol} polarization, 0.5 m resolution."
        )
        target.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return dst


def build_suffix_pol_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src_images = list_images(REPO_ROOT / "exampledataset" / "ship" / "cnt")[:image_limit]
    pols = ["HH", "HV", "VH", "VV"]
    for idx, src in enumerate(src_images):
        pol = pols[idx % len(pols)]
        target = dst / "ship" / f"shipchip_{idx:03d}_{pol}{src.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dst


def build_ship_only_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    copy_images(
        REPO_ROOT / "exampledataset" / "ship" / "cnt",
        dst / "ship",
        limit=image_limit,
        rename_prefix="ship",
    )
    return dst


def build_plain_ship_fixture(dst: Path, image_limit: int) -> Path:
    reset_dir(dst)
    src_images = list_images(REPO_ROOT / "exampledataset" / "ship" / "cnt")[:image_limit]
    for idx, src in enumerate(src_images):
        target = dst / "ship" / f"shipchip{idx:04d}{src.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dst


def copy_images(
    src_root: Path,
    dst_root: Path,
    limit: int,
    preserve_relative_root: Path | None = None,
    rename_prefix: str | None = None,
) -> None:
    images = list_images(src_root)[:limit]
    for idx, src in enumerate(images):
        if preserve_relative_root is not None:
            rel = src.relative_to(preserve_relative_root)
            dst = dst_root / rel
        elif rename_prefix:
            dst = dst_root / f"{rename_prefix}_{idx:04d}{src.suffix.lower()}"
        else:
            dst = dst_root / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def list_images(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: natural_key(path.as_posix()),
    )


def anchor_hint(anchor: str, fields: list[str], text: str) -> dict[str, Any]:
    return {
        "type": "filename_anchor_fields",
        "source_text": text,
        "rule": {
            "anchor": anchor,
            "fields_after_anchor": [{"normalized": field, "raw": field} for field in fields],
        },
    }


def suffix_field_hint(field: str, text: str) -> dict[str, Any]:
    return {
        "type": "filename_suffix_field",
        "source_text": text,
        "rule": {
            "field": {"normalized": field, "raw": field},
        },
    }


def run_experiment(
    cases: list[BenchmarkCase],
    output_dir: Path,
    case_run_root: Path,
    sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "experiment": "exp1_schema_grounding",
        "cases": {},
        "variants": [variant.__dict__ for variant in VARIANTS],
    }

    for case in cases:
        profile_dir = case_run_root / case.case_id / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        intent_spec = {"intent": {"task": case.task, "filters": case.filters}}
        profile = profile_dataset(
            root=case.root,
            output_dir=profile_dir,
            request=case.request,
            intent_spec=intent_spec,
            format_hints=case.hints,
            sample_limit=sample_limit,
            image_probe_limit=sample_limit,
        )
        raw_profile = load_mapping(profile_dir / "raw_profile.json")
        details["cases"][case.case_id] = {
            "case": case_to_row(case),
            "profile_dir": profile_dir.as_posix(),
            "profile_summary": summarize_raw_profile(raw_profile),
            "clarification_questions": profile.get("clarification_questions", []),
            "methods": {},
        }

        for variant in VARIANTS:
            variant_dir = case_run_root / case.case_id / variant.method_id
            variant_dir.mkdir(parents=True, exist_ok=True)
            hints = case.hints if variant.use_hints else []
            bridge_report = compile_and_validate_format(
                root=case.root,
                output_dir=variant_dir,
                raw_profile=raw_profile,
                intent_spec=intent_spec,
                format_hints=hints,
                sample_limit=sample_limit,
                required_fields=case.required_fields,
                include_heuristics=variant.include_heuristics,
            )
            validation_valid = bool(bridge_report.get("valid"))
            accepted = validation_valid if variant.validator_gate else True
            field_coverage = bridge_report.get("validation", {}).get("field_coverage", {}) or {}
            required_coverage = average([as_float(field_coverage.get(field, 0.0)) for field in case.required_fields])
            metadata_accuracy = compute_metadata_accuracy(
                root=Path(case.root),
                preview=load_mapping(variant_dir / "format_validation.json").get("preview", []),
                required_fields=case.required_fields,
                extractor=gold_extractor(case.gold_extractor_name),
            )
            schema_decision_correct = accepted == case.expected_valid
            invalid_acceptance = (not case.expected_valid) and accepted
            row = {
                "case_id": case.case_id,
                "case_title": case.title,
                "method_id": variant.method_id,
                "method_title": variant.title,
                "validator_gate": variant.validator_gate,
                "expected_valid": case.expected_valid,
                "validation_valid": validation_valid,
                "accepted": accepted,
                "schema_decision_correct": schema_decision_correct,
                "invalid_acceptance": invalid_acceptance,
                "required_field_coverage": round(required_coverage, 4),
                "metadata_value_accuracy": metadata_accuracy,
                "missing_required_count": bridge_report.get("validation", {}).get("missing_required_count"),
                "validated_samples": bridge_report.get("validation", {}).get("validated_samples"),
                "total_samples_seen": bridge_report.get("validation", {}).get("total_samples_seen"),
                "compiled_rule_count": len(bridge_report.get("compiled_rules", [])),
                "format_rules": compact_counter_dict(bridge_report.get("validation", {}).get("format_rules", {})),
                "parse_status": compact_counter_dict(bridge_report.get("validation", {}).get("parse_status", {})),
                "profile_level_after_validation": bridge_report.get("profile_level_after_validation"),
                "output_dir": variant_dir.as_posix(),
            }
            for field_name, value in field_coverage.items():
                row[f"coverage_{field_name}"] = value
            results.append(row)
            details["cases"][case.case_id]["methods"][variant.method_id] = {
                "bridge_report_path": (variant_dir / "format_bridge.json").as_posix(),
                "validation_report_path": (variant_dir / "format_validation.json").as_posix(),
                "metrics": row,
            }

    save_json(output_dir / "raw_case_details.json", details)
    return results, details


def summarize_raw_profile(raw_profile: dict[str, Any]) -> dict[str, Any]:
    images = raw_profile.get("images", {})
    sidecars = raw_profile.get("sidecars", {})
    return {
        "total_images": images.get("total_images"),
        "suffix_counts": images.get("suffix_counts"),
        "path_group_count": len(raw_profile.get("path_groups", [])),
        "standard_format_candidates": raw_profile.get("standard_format_candidates", []),
        "same_stem_sidecar_counts": sidecars.get("same_stem_sidecar_counts"),
    }


def gold_extractor(name: str) -> GoldExtractor:
    return {
        "none": extract_none,
        "parent_class": extract_parent_class,
        "vehicle_suffix": extract_vehicle_suffix,
        "vehicle_nested": extract_vehicle_nested,
        "aircraft_anchor": extract_aircraft_anchor,
        "syy_elev": extract_syy_elev,
        "sidecar_kv": extract_sidecar_kv,
        "caption_sidecar": extract_caption_sidecar,
        "suffix_pol": extract_suffix_pol,
    }.get(name, extract_none)


def compute_metadata_accuracy(
    root: Path,
    preview: list[dict[str, Any]],
    required_fields: list[str],
    extractor: GoldExtractor,
) -> float | None:
    correct = 0
    total = 0
    for item in preview:
        relpath = str(item.get("path", ""))
        gold = extractor(root, relpath)
        for field in required_fields:
            if field not in gold:
                continue
            total += 1
            if values_equal(item.get("class") if field == "class" else item.get(field), gold[field]):
                correct += 1
    if total == 0:
        return None
    return round(correct / total, 4)


def extract_none(root: Path, relpath: str) -> FieldMap:
    return {}


def extract_parent_class(root: Path, relpath: str) -> FieldMap:
    path = Path(relpath)
    if len(path.parts) >= 2:
        return {"class": path.parts[-2]}
    return {}


def extract_vehicle_suffix(root: Path, relpath: str) -> FieldMap:
    match = re.search(r"^(?P<class>[^/]+)/(?P<target>[^/_]+)_(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<pol>[^.]+)\.", relpath)
    if not match:
        return extract_parent_class(root, relpath)
    return {
        "class": match.group("class"),
        "azimuth_deg": coerce_number(match.group("azimuth")),
        "polarization": match.group("pol"),
        "band": "X",
    }


def extract_vehicle_nested(root: Path, relpath: str) -> FieldMap:
    pattern = (
        r"^(?P<class>[^/]+)/(?P<band>[^/]+)/inci-(?P<incidence>-?\d+(?:\.\d+)?)-"
        r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)-(?P<pol>[^.]+)\."
    )
    match = re.search(pattern, relpath)
    if not match:
        return extract_parent_class(root, relpath)
    return {
        "class": match.group("class").split("（")[0].split("(")[0],
        "band": match.group("band"),
        "incidence_angle_deg": coerce_number(match.group("incidence")),
        "azimuth_deg": coerce_number(match.group("azimuth")),
        "polarization": match.group("pol"),
    }


def extract_aircraft_anchor(root: Path, relpath: str) -> FieldMap:
    match = re.search(
        r"^(?P<class>[^/]+)/.*?_850_(?P<depression>-?\d+(?:\.\d+)?)_"
        r"(?P<azimuth>-?\d+(?:\.\d+)?)_(?P<resolution>-?\d+(?:\.\d+)?)_(?P<band>[A-Za-z0-9]+)\.",
        relpath,
    )
    if not match:
        return extract_parent_class(root, relpath)
    return {
        "class": match.group("class"),
        "depression_angle_deg": coerce_number(match.group("depression")),
        "azimuth_deg": coerce_number(match.group("azimuth")),
        "resolution_m": coerce_number(match.group("resolution")),
        "band": match.group("band"),
    }


def extract_syy_elev(root: Path, relpath: str) -> FieldMap:
    match = re.search(
        r"^(?P<class>[^/]+)/.*?elev-(?P<elev>-?\d+(?:\.\d+)?)-"
        r"azim-(?P<azimuth>-?\d+(?:\.\d+)?)-(?P<band>[A-Za-z0-9]+)\.",
        relpath,
    )
    if not match:
        return extract_parent_class(root, relpath)
    return {
        "class": match.group("class"),
        "elevation_angle_deg": coerce_number(match.group("elev")),
        "azimuth_deg": coerce_number(match.group("azimuth")),
        "band": match.group("band"),
    }


def extract_sidecar_kv(root: Path, relpath: str) -> FieldMap:
    txt_path = (root / relpath).with_suffix(".txt")
    if not txt_path.exists():
        return {}
    values: FieldMap = {}
    aliases = {
        "azimuth": "azimuth_deg",
        "incidence": "incidence_angle_deg",
        "pol": "polarization",
    }
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key = aliases.get(key, key)
        values[key] = coerce_number(value)
    return values


def extract_caption_sidecar(root: Path, relpath: str) -> FieldMap:
    txt_path = (root / relpath).with_suffix(".txt")
    if not txt_path.exists():
        return {}
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    values: FieldMap = {}
    if "ship" in text.lower():
        values["class"] = "ship"
    patterns = {
        "azimuth_deg": r"azimuth\s+(-?\d+(?:\.\d+)?)",
        "incidence_angle_deg": r"incidence\s+(-?\d+(?:\.\d+)?)",
        "band": r"\b([A-Za-z0-9]+)-band\b",
        "polarization": r"\b(HH|HV|VH|VV|AHH|pauli)\b",
        "resolution_m": r"(-?\d+(?:\.\d+)?)\s*m\s+resolution",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            values[field] = coerce_number(match.group(1))
    return values


def extract_suffix_pol(root: Path, relpath: str) -> FieldMap:
    match = re.search(r"^(?P<class>[^/]+)/.*?_(?P<pol>HH|HV|VH|VV)\.[^.]+$", relpath, flags=re.IGNORECASE)
    if not match:
        return extract_parent_class(root, relpath)
    return {
        "class": match.group("class"),
        "polarization": match.group("pol"),
    }


def coerce_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def values_equal(actual: Any, expected: Any) -> bool:
    actual = coerce_number(actual)
    expected = coerce_number(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6)
    return str(actual).lower() == str(expected).lower()


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["method_id"]].append(row)

    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = grouped[variant.method_id]
        valid_rows = [row for row in rows if bool(row["expected_valid"])]
        invalid_rows = [row for row in rows if not bool(row["expected_valid"])]
        accepted_valid = [row for row in valid_rows if bool(row["accepted"])]
        rejected_invalid = [row for row in invalid_rows if not bool(row["accepted"])]
        metadata_values = [
            as_float(row.get("metadata_value_accuracy"))
            for row in rows
            if row.get("metadata_value_accuracy") not in {None, ""}
        ]
        summary.append(
            {
                "method_id": variant.method_id,
                "method_title": variant.title,
                "cases": len(rows),
                "schema_decision_accuracy": round(average([row["schema_decision_correct"] for row in rows]), 4),
                "valid_case_acceptance_rate": round(len(accepted_valid) / max(len(valid_rows), 1), 4),
                "invalid_case_rejection_rate": round(len(rejected_invalid) / max(len(invalid_rows), 1), 4),
                "invalid_acceptance_rate": round(
                    len([row for row in invalid_rows if bool(row["accepted"])]) / max(len(invalid_rows), 1),
                    4,
                ),
                "avg_required_field_coverage": round(
                    average([as_float(row.get("required_field_coverage", 0.0)) for row in rows]), 4
                ),
                "avg_metadata_value_accuracy": round(average(metadata_values), 4) if metadata_values else "",
                "avg_compiled_rule_count": round(
                    average([as_float(row.get("compiled_rule_count", 0.0)) for row in rows]), 2
                ),
            }
        )
    return summary


def plot_all(
    results: list[dict[str, Any]],
    cases: list[BenchmarkCase],
    summary: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        save_text(figures_dir / "PLOTS_SKIPPED.txt", f"matplotlib import failed: {exc}\n")
        return

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    plot_schema_summary(summary, figures_dir / "schema_success_by_method.png", plt)
    plot_invalid_acceptance(summary, figures_dir / "invalid_acceptance_by_method.png", plt)
    plot_case_method_matrix(results, cases, figures_dir / "case_method_matrix.png", plt)
    plot_full_saga_field_coverage(results, cases, figures_dir / "full_saga_field_coverage_heatmap.png", plt)
    plot_metadata_accuracy(summary, figures_dir / "metadata_accuracy_by_method.png", plt)
    plot_dataset_overview(results, cases, figures_dir / "profile_case_overview.png", plt)


def plot_schema_summary(summary: list[dict[str, Any]], out: Path, plt: Any) -> None:
    rows = ordered_summary_rows(summary)
    labels = method_axis_labels()
    decision = [as_float(row["schema_decision_accuracy"]) for row in rows]
    valid_accept = [as_float(row["valid_case_acceptance_rate"]) for row in rows]
    invalid_reject = [as_float(row["invalid_case_rejection_rate"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar([i - width for i in x], decision, width, label="Decision accuracy", color="#335C81")
    ax.bar(x, valid_accept, width, label="Valid-case acceptance", color="#2A9D8F")
    ax.bar([i + width for i in x], invalid_reject, width, label="Invalid-case rejection", color="#E9C46A")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=14)
    ax.set_title("Schema Grounding Reliability by Method", fontsize=18, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False, fontsize=12)
    annotate_bars(ax, fontsize=10)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_invalid_acceptance(summary: list[dict[str, Any]], out: Path, plt: Any) -> None:
    labels = [row["method_title"] for row in summary]
    values = [as_float(row["invalid_acceptance_rate"]) for row in summary]
    colors = ["#D1495B" if value > 0 else "#2A9D8F" for value in values]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Invalid schema acceptance rate")
    ax.set_title("Validator Gate Prevents Unsupported Schemas")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_case_method_matrix(results: list[dict[str, Any]], cases: list[BenchmarkCase], out: Path, plt: Any) -> None:
    from matplotlib.colors import ListedColormap

    case_ids = [case.case_id for case in cases]
    method_ids = [variant.method_id for variant in VARIANTS]
    method_titles = method_axis_labels()
    lookup = {(row["case_id"], row["method_id"]): row for row in results}
    matrix = [
        [1 if lookup[(case_id, method_id)]["schema_decision_correct"] else 0 for method_id in method_ids]
        for case_id in case_ids
    ]
    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    cmap = ListedColormap(["#E7A8A1", "#A9D8C4"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(method_titles)))
    ax.set_xticklabels(method_titles, rotation=25, ha="right", fontsize=17)
    ax.set_yticks(range(len(case_ids)))
    ax.set_yticklabels([case.title for case in cases], fontsize=17)
    ax.set_title("Case-Level Schema Decision Matrix", fontsize=24, pad=14)
    ax.grid(False)
    ax.set_xticks([idx - 0.5 for idx in range(1, len(method_titles))], minor=True)
    ax.set_yticks([idx - 0.5 for idx in range(1, len(case_ids))], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, "OK" if value else "FAIL", ha="center", va="center", fontsize=14, color="#1F2933")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_full_saga_field_coverage(results: list[dict[str, Any]], cases: list[BenchmarkCase], out: Path, plt: Any) -> None:
    rows = [row for row in results if row["method_id"] == "full_saga"]
    fields = sorted({field for case in cases for field in case.required_fields}, key=field_sort_key)
    matrix: list[list[float]] = []
    for case in cases:
        row = next(item for item in rows if item["case_id"] == case.case_id)
        matrix.append([as_float(row.get(f"coverage_{field}", 0.0)) if field in case.required_fields else math.nan for field in fields])

    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#EFEFEF")
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, rotation=30, ha="right")
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([case.title for case in cases])
    ax.set_title("Full SAGA Required-Field Coverage")
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            if math.isnan(value):
                label = "-"
            else:
                label = f"{value:.2f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_metadata_accuracy(summary: list[dict[str, Any]], out: Path, plt: Any) -> None:
    labels = [row["method_title"] for row in summary]
    values = [as_float(row["avg_metadata_value_accuracy"]) for row in summary]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, values, color="#5E548E")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average value accuracy")
    ax.set_title("Metadata Value Accuracy on Gold-Checkable Fields")
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax)
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def plot_dataset_overview(results: list[dict[str, Any]], cases: list[BenchmarkCase], out: Path, plt: Any) -> None:
    full_rows = [row for row in results if row["method_id"] == "full_saga"]
    by_case = {row["case_id"]: row for row in full_rows}
    labels = [case.title for case in cases]
    samples = [as_float(by_case[case.case_id].get("total_samples_seen", 0)) for case in cases]
    missing = [as_float(by_case[case.case_id].get("missing_required_count", 0)) for case in cases]
    x = list(range(len(cases)))
    fig, ax1 = plt.subplots(figsize=(10.8, 4.8))
    ax1.bar(x, samples, label="Samples", color="#457B9D")
    ax1.set_ylabel("Samples seen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, missing, label="Missing required fields", marker="o", color="#D1495B")
    ax2.set_ylabel("Missing required count")
    ax1.set_title("Benchmark Case Scale and Full-SAGA Missing-Field Signals")
    lines1, names1 = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, names1 + names2, loc="upper right")
    fig.tight_layout()
    save_figure(fig, out)
    plt.close(fig)


def save_figure(fig: Any, out: Path) -> None:
    fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)


def method_axis_labels() -> list[str]:
    return [variant.title for variant in VARIANTS]


def ordered_summary_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["method_id"]: row for row in summary}
    return [by_id[variant.method_id] for variant in VARIANTS]


def annotate_bars(ax: Any, fontsize: int = 8) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 2),
            textcoords="offset points",
        )


def render_latex_summary_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset schema grounding results in Experiment 1.}",
        r"\label{tab:exp1_schema_grounding}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Method & Dec. Acc. $\uparrow$ & Valid Acc. $\uparrow$ & Invalid Rej. $\uparrow$ & Field Cov. $\uparrow$ & Value Acc. $\uparrow$ \\",
        r"\hline",
    ]
    for row in summary:
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["method_title"])),
                    pct(row["schema_decision_accuracy"]),
                    pct(row["valid_case_acceptance_rate"]),
                    pct(row["invalid_case_rejection_rate"]),
                    pct(row["avg_required_field_coverage"]),
                    pct(row["avg_metadata_value_accuracy"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(
    output_dir: Path,
    cases: list[BenchmarkCase],
    variants: list[Variant],
    summary: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> str:
    best = max(summary, key=lambda row: as_float(row["schema_decision_accuracy"]))
    lines = [
        "# Experiment 1: Dataset Profiling and Schema Grounding",
        "",
        "This experiment evaluates the agent-side ability of SAGA to convert heterogeneous SAR datasets into validated dataset schemas.",
        "It does not evaluate image generation quality. The key question is whether the system can accept supported formats, reject unsupported semantic claims, and expose field coverage before downstream augmentation planning.",
        "",
        "## Benchmark Cases",
        "",
        "| Case | Expected | Required fields | Purpose |",
        "|---|---:|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case.title} | {'valid' if case.expected_valid else 'invalid'} | "
            f"`{', '.join(case.required_fields)}` | {case.description} |"
        )
    lines.extend(["", "## Compared Methods", ""])
    for variant in variants:
        lines.append(f"- **{variant.title}**: {variant.description}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Method | Decision Acc. | Valid Accept | Invalid Reject | Field Coverage | Value Acc. | Invalid Accept |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method_title']} | {pct(row['schema_decision_accuracy'])} | "
            f"{pct(row['valid_case_acceptance_rate'])} | {pct(row['invalid_case_rejection_rate'])} | "
            f"{pct(row['avg_required_field_coverage'])} | {pct(row['avg_metadata_value_accuracy'])} | "
            f"{pct(row['invalid_acceptance_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"Best schema-decision accuracy in this run: **{best['method_title']}** ({pct(best['schema_decision_accuracy'])}).",
            "",
            "## What This Demonstrates",
            "",
            "- **Format coverage**: valid class-folder, filename-token, anchor-token, keyword-token, key-value sidecar, and caption-sidecar cases are included.",
            "- **Validator value**: invalid cases deliberately contain unsupported semantic requests; methods without the validator can accept them despite missing required fields.",
            "- **Agent reliability rather than skill quality**: metrics are based on schema decisions, metadata coverage, exact metadata values where gold labels are available, and missing-field localization.",
            "- **Planner readiness**: a valid schema means the dataset can be promoted to the next planning stage; an invalid schema should trigger clarification or rejection instead of unsafe augmentation.",
            "",
            "## Output Artifacts",
            "",
            f"- `results.csv`: per-case, per-method metrics.",
            f"- `summary_by_method.csv`: aggregate method comparison.",
            f"- `case_details.json`: paths to profile, bridge, and validation reports.",
            f"- `table_exp1_summary.tex`: LaTeX-ready summary table.",
            f"- `figures/schema_success_by_method.png`: acceptance/rejection reliability.",
            f"- `figures/case_method_matrix.png`: case-level decision matrix.",
            f"- `figures/full_saga_field_coverage_heatmap.png`: field coverage for Full SAGA.",
            f"- `figures/invalid_acceptance_by_method.png`: unsupported schema acceptance risk.",
            f"- `figures/metadata_accuracy_by_method.png`: value accuracy on gold-checkable fields.",
            f"- `figures/profile_case_overview.png`: case scale and missing-field signals.",
            "",
            "## Case-Level Full SAGA Outcomes",
            "",
            "| Case | Accepted | Validated | Field coverage | Missing required | Profile level |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        row = next(item for item in results if item["case_id"] == case.case_id and item["method_id"] == "full_saga")
        lines.append(
            f"| {case.title} | {bool_word(row['accepted'])} | {bool_word(row['validation_valid'])} | "
            f"{pct(row['required_field_coverage'])} | {row['missing_required_count']} | "
            f"{row['profile_level_after_validation']} |"
        )
    lines.extend(
        [
            "",
            "## Paper-Ready Interpretation",
            "",
            "Experiment 1 supports the Schema-Grounded Dataset Profiling section. It shows that SAGA treats user or LLM semantic mappings as candidate schemas, not trusted facts. The validator promotes valid datasets to a planner-ready profile and blocks missing or inconsistent metadata before recipe generation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def case_to_row(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "title": case.title,
        "root": case.root,
        "expected_valid": case.expected_valid,
        "required_fields": ",".join(case.required_fields),
        "hint_count": len(case.hints),
        "description": case.description,
    }


def compact_counter_dict(value: dict[str, Any]) -> str:
    if not value:
        return ""
    return "; ".join(f"{key}:{val}" for key, val in value.items())


def average(values: list[Any]) -> float:
    nums = [as_float(value) for value in values if value not in {None, ""}]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def as_float(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def pct(value: Any) -> str:
    if value in {None, ""}:
        return "--"
    return f"{100.0 * as_float(value):.1f}"


def bool_word(value: Any) -> str:
    return "yes" if bool(value) else "no"


def field_sort_key(field: str) -> tuple[int, str]:
    order = {
        "class": 0,
        "azimuth_deg": 1,
        "incidence_angle_deg": 2,
        "depression_angle_deg": 3,
        "elevation_angle_deg": 4,
        "band": 5,
        "polarization": 6,
        "resolution_m": 7,
    }
    return (order.get(field, 99), field)


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


if __name__ == "__main__":
    main()

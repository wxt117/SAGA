from __future__ import annotations

import argparse
from pathlib import Path

from saga.agent.command_router import run_agent_command
from saga.agent.continuation import continue_agent_run
from saga.agent.auto_repair import run_bounded_auto_repair
from saga.agent.benefit_evidence import build_benefit_evidence_report
from saga.agent.candidate_pilot import build_candidate_recipe_pilot
from saga.agent.format_inference import infer_format_spec_with_llm
from saga.agent.format_understanding import understand_format_with_llm
from saga.agent.intent_recognizer import recognize_request
from saga.agent.llm_client import LLMConfig
from saga.agent.planner import run_plan
from saga.agent.plan_critic import critique_agent_plan
from saga.agent.runtime import run_agent
from saga.agent.run_analysis import compare_agent_runs, explain_agent_run
from saga.agent.smoke_matrix import run_smoke_matrix
from saga.core.config import load_dataset_config, load_llm_config
from saga.data.format_builder import build_format_spec
from saga.data.format_bridge import compile_and_validate_format_from_profile_dir
from saga.data.format_validation import validate_dataset_format
from saga.data.inspect import inspect_dataset_format
from saga.data.profile import profile_dataset
from saga.executor.recipe_executor import execute_recipe
from saga.exporter import export_augmented_dataset
from saga.memory import list_memory_entries, record_agent_run_memory
from saga.observer.parameter_repair import build_parameter_repair_plan
from saga.observer.distribution import evaluate_distribution
from saga.observer.quality import evaluate_output_directory
from saga.observer.repair_policy import build_repair_policy_report
from saga.observer.sar_artifacts import evaluate_sar_artifacts
from saga.core.recipe import SagaRecipe
from saga.skills.background_generation.skill import run_background_generation_skill
from saga.skills.diagnosis.skill import run_diagnosis
from saga.skills.diffusion_lora.skill import run_diffusion_lora_generation_skill
from saga.skills.classification_evaluation.skill import run_classification_evaluation_skill
from saga.skills.dataset_balancing.skill import run_dataset_balancing_skill
from saga.skills.evaluation_utils.skill import (
    run_duplicate_near_duplicate_skill,
    run_leakage_check_skill,
    run_metadata_slice_evaluation_skill,
)
from saga.skills.gan_generation.skill import run_gan_generation_skill
from saga.skills.geodiff_sar.skill import run_geodiff_sar_skill
from saga.skills.metadata_caption.skill import run_metadata_caption_skill
from saga.skills.model_to_pov_scene.skill import run_model_to_pov_scene_compiler_skill
from saga.skills.pseudocolor.skill import run_pseudocolor_skill
from saga.skills.raysar.skill import run_raysar_synthesis_skill
from saga.skills.raysar_sweep.skill import run_raysar_sweep_synthesis_skill
from saga.skills.sar_preprocess.skill import run_sar_preprocess_skill
from saga.skills.style_transfer.skill import run_style_transfer_skill
from saga.skills.target_background_composition.skill import run_target_background_composition_skill
from saga.skills.traditional_augmentation.skill import run_traditional_augmentation_skill


def main() -> None:
    parser = argparse.ArgumentParser(prog="saga", description="SAGA SAR data augmentation agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser("diagnose", help="Diagnose a user dataset")
    diagnose.add_argument("--dataset", required=True, help="Dataset YAML/JSON config")
    diagnose.add_argument("--output", required=True, help="Output directory")
    diagnose.add_argument("--skip-image-size", action="store_true", help="Do not open images to read dimensions")

    inspect = subparsers.add_parser("inspect-format", help="Sample paths/text so a dataset format can be configured")
    inspect.add_argument("--root", required=True, help="Dataset root")
    inspect.add_argument("--output", required=True, help="Output directory")
    inspect.add_argument("--sample-limit", type=int, default=120, help="Number of image paths to sample")

    profile = subparsers.add_parser(
        "profile-dataset",
        help="Create a progressive Dataset Profiler report from raw scan + optional user request",
    )
    profile.add_argument("--root", required=True, help="Dataset root to scan")
    profile.add_argument("--output", required=True, help="Output directory")
    profile.add_argument("--request", default="", help="Natural-language user request and optional format hint")
    profile.add_argument("--sample-limit", type=int, default=120, help="Number of image paths to sample")
    profile.add_argument("--image-probe-limit", type=int, default=200, help="Number of images to open for header/stat probes")
    profile.add_argument("--txt-preview-chars", type=int, default=300, help="Characters to preview from sampled txt files")

    infer = subparsers.add_parser("infer-format", help="Use an LLM to infer a DatasetFormatSpec from inspection output")
    infer.add_argument("--inspection", required=True, help="Path to format_inspection.json")
    infer.add_argument("--output", required=True, help="Output DatasetFormatSpec YAML")
    infer.add_argument("--llm-config", required=True, help="LLM config YAML/JSON")
    infer.add_argument("--description", default="", help="User-provided explanation of the dataset format")
    infer.add_argument("--prompt-sample-limit", type=int, default=40, help="Number of sampled paths to send to LLM")
    infer.add_argument("--txt-preview-limit", type=int, default=10, help="Number of txt previews to send to LLM")
    infer.add_argument("--no-response-format", action="store_true", help="Do not send response_format=json_object")

    understand = subparsers.add_parser(
        "understand-format",
        help="Use an LLM to summarize dataset layout as semantic path templates",
    )
    understand.add_argument("--inspection", required=True, help="Path to format_inspection.json")
    understand.add_argument("--output", required=True, help="Output format_understanding JSON")
    understand.add_argument("--llm-config", required=True, help="LLM config YAML/JSON")
    understand.add_argument("--description", default="", help="User-provided explanation of the dataset format")
    understand.add_argument("--prompt-group-limit", type=int, default=30, help="Number of path groups to send to LLM")
    understand.add_argument("--prompt-sample-limit", type=int, default=40, help="Number of sampled paths to send to LLM")
    understand.add_argument("--txt-preview-limit", type=int, default=8, help="Number of txt previews to send to LLM")
    understand.add_argument("--no-response-format", action="store_true", help="Do not send response_format=json_object")

    build_spec = subparsers.add_parser(
        "build-format-spec",
        help="Build a DatasetFormatSpec from format understanding and/or inspection heuristics",
    )
    build_spec.add_argument("--understanding", default=None, help="Path to format_understanding JSON/YAML")
    build_spec.add_argument("--inspection", default=None, help="Optional path to format_inspection.json")
    build_spec.add_argument("--output", required=True, help="Output DatasetFormatSpec YAML")
    build_spec.add_argument("--name", default=None, help="Override generated format spec name")
    build_spec.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Do not add programmatic SAR filename heuristics from inspection",
    )

    validate = subparsers.add_parser("validate-format", help="Validate a DatasetFormatSpec on sampled dataset items")
    validate.add_argument("--dataset", required=True, help="Dataset YAML/JSON config with format_spec")
    validate.add_argument("--output", required=True, help="Output directory")
    validate.add_argument("--sample-limit", type=int, default=200, help="Number of samples to validate")
    validate.add_argument(
        "--required-fields",
        default="class",
        help="Comma-separated required normalized fields, e.g. class,azimuth_deg",
    )

    bridge_format = subparsers.add_parser(
        "bridge-format",
        help="Compile DatasetFormatSpec from profile-dataset outputs and immediately validate it",
    )
    bridge_format.add_argument("--profile", required=True, help="Directory produced by profile-dataset")
    bridge_format.add_argument("--output", default=None, help="Output directory, defaults to --profile")
    bridge_format.add_argument("--sample-limit", type=int, default=500, help="Number of samples to validate")
    bridge_format.add_argument(
        "--required-fields",
        default=None,
        help="Comma-separated required fields. Defaults to class + fields from hints/filters.",
    )
    bridge_format.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Only compile rules from FormatHint; do not add high-precision SAR filename heuristics.",
    )

    agent_run = subparsers.add_parser(
        "agent-run",
        help="Run SAGA intent/profile/bridge/recipe pipeline from one natural-language request",
    )
    agent_run.add_argument("--request", required=True, help="Natural-language user request")
    agent_run.add_argument("--output", required=True, help="Output run directory")
    agent_run.add_argument("--dataset-root", default=None, help="Dataset root; defaults to parsed content source")
    agent_run.add_argument("--config", default=None, help="Optional style transfer skill config")
    agent_run.add_argument("--diffusion-lora-config", default=None, help="Optional diffusion LoRA skill config")
    agent_run.add_argument("--traditional-aug-config", default=None, help="Optional traditional augmentation skill config")
    agent_run.add_argument("--saga-config", default=None, help="Optional global SAGA policy/config YAML/JSON")
    agent_run.add_argument("--llm-config", default=None, help="Optional LLM config YAML/JSON for hybrid planner")
    agent_run.add_argument(
        "--use-llm-intent",
        action="store_true",
        help="Ask an LLM for an intent proposal, then guardrail it before dataset profiling",
    )
    agent_run.add_argument(
        "--use-llm-planner",
        action="store_true",
        help="Ask an LLM for a planner proposal, then guardrail it before recipe generation",
    )
    agent_run.add_argument("--sample-limit", type=int, default=120, help="Raw profile sample limit")
    agent_run.add_argument("--image-probe-limit", type=int, default=200, help="Raw profile image probe limit")
    agent_run.add_argument("--bridge-sample-limit", type=int, default=500, help="Format validation sample limit")
    agent_run.add_argument("--memory-dir", default="runs/memory", help="Policy memory directory for planner retrieval")
    agent_run.add_argument("--no-memory", action="store_true", help="Do not retrieve prior policy memory for planner context")
    agent_run.add_argument("--dry-run", action="store_true", help="Generate recipe and commands without running expensive skills")
    agent_run.add_argument("--run", action="store_true", help="Actually execute expensive recipe steps; agent-run defaults to dry-run")
    agent_run.add_argument("--no-execute", action="store_true", help="Stop after recipe generation")

    agent_continue = subparsers.add_parser(
        "agent-continue",
        help="Continue a previous agent-run with a user clarification answer",
    )
    agent_continue.add_argument("--run-dir", required=True, help="Previous agent-run directory")
    agent_continue.add_argument("--answer", required=True, help="User clarification answer")
    agent_continue.add_argument("--output", default=None, help="Output continuation directory")
    agent_continue.add_argument("--llm-config", default=None, help="Optional LLM config YAML/JSON")
    agent_continue.add_argument("--use-llm-intent", action="store_true", help="Use LLM intent recognizer for the continued request")
    agent_continue.add_argument("--use-llm-planner", action="store_true", help="Use LLM planner for the continued request")
    agent_continue.add_argument("--config", default=None, help="Optional style transfer skill config")
    agent_continue.add_argument("--diffusion-lora-config", default=None, help="Optional diffusion LoRA skill config")
    agent_continue.add_argument("--traditional-aug-config", default=None, help="Optional traditional augmentation skill config")
    agent_continue.add_argument("--saga-config", default=None, help="Optional global SAGA policy/config YAML/JSON")
    agent_continue.add_argument("--memory-dir", default="runs/memory", help="Policy memory directory for planner retrieval")
    agent_continue.add_argument("--no-memory", action="store_true", help="Do not retrieve prior policy memory")
    agent_continue.add_argument("--dry-run", action="store_true", help="Generate recipe and commands without running expensive skills")
    agent_continue.add_argument("--run", action="store_true", help="Actually execute expensive recipe steps; defaults to dry-run")
    agent_continue.add_argument("--no-execute", action="store_true", help="Stop after recipe generation")

    execute_recipe_cmd = subparsers.add_parser(
        "execute-recipe",
        help="Execute a SagaRecipe DAG, with dry-run/checkpoint-style step reports",
    )
    execute_recipe_cmd.add_argument("--recipe", required=True, help="Recipe YAML/JSON path")
    execute_recipe_cmd.add_argument("--output", required=True, help="Execution output directory")
    execute_recipe_cmd.add_argument("--dry-run", action="store_true", help="Do not run expensive skills")
    execute_recipe_cmd.add_argument("--continue-on-error", action="store_true", help="Continue after failed steps")

    smoke_matrix = subparsers.add_parser(
        "smoke-matrix",
        help="Run a configured natural-language agent-run smoke matrix",
    )
    smoke_matrix.add_argument("--config", default="configs/smoke/p0_p2_agent_skill_smoke.yaml", help="Smoke matrix YAML/JSON config")
    smoke_matrix.add_argument("--output", required=True, help="Output directory")
    smoke_matrix.add_argument("--case", action="append", default=None, help="Run only a named case; can be repeated")
    smoke_matrix.add_argument("--run-real", action="store_true", help="Allow cases marked allow_real=true to run without dry-run")
    smoke_matrix.add_argument("--memory-dir", default=None, help="Optional memory directory for smoke runs")

    evaluate_output = subparsers.add_parser(
        "evaluate-output",
        help="Evaluate an output image directory and write observer/repair reports",
    )
    evaluate_output.add_argument("--input", required=True, help="Output image directory to evaluate")
    evaluate_output.add_argument("--output", required=True, help="Directory for evaluation artifacts")
    evaluate_output.add_argument("--expected-count", type=int, default=None, help="Expected number of generated images")
    evaluate_output.add_argument("--sample-limit", type=int, default=100, help="Number of images to sample")
    evaluate_output.add_argument("--dry-run", action="store_true", help="Record what would be evaluated")
    evaluate_output.add_argument("--max-trials", type=int, default=3, help="Repair policy maximum trials")

    evaluate_distribution_cmd = subparsers.add_parser(
        "evaluate-distribution",
        help="Compare generated images against reference images with FID-style distribution metrics",
    )
    evaluate_distribution_cmd.add_argument("--reference", required=True, help="Reference image directory")
    evaluate_distribution_cmd.add_argument("--generated", required=True, help="Generated image directory")
    evaluate_distribution_cmd.add_argument("--output", required=True, help="Directory for distribution evaluation artifacts")
    evaluate_distribution_cmd.add_argument("--reference-sample-limit", type=int, default=200, help="Reference images to sample")
    evaluate_distribution_cmd.add_argument("--generated-sample-limit", type=int, default=50, help="Generated images to sample")
    evaluate_distribution_cmd.add_argument("--image-size", type=int, default=64, help="Size for lightweight SAR features")
    evaluate_distribution_cmd.add_argument("--no-standard-fid", action="store_true", help="Skip pytorch_fid standard backend")
    evaluate_distribution_cmd.add_argument("--dry-run", action="store_true", help="Record what would be evaluated")

    evaluate_sar_artifacts_cmd = subparsers.add_parser(
        "evaluate-sar-artifacts",
        help="Evaluate SAR-specific visual artifacts such as stripes, gradients, and target compactness",
    )
    evaluate_sar_artifacts_cmd.add_argument("--input", required=True, help="Output image directory to evaluate")
    evaluate_sar_artifacts_cmd.add_argument("--output", required=True, help="Directory for SAR artifact evaluation artifacts")
    evaluate_sar_artifacts_cmd.add_argument("--sample-limit", type=int, default=100, help="Number of images to sample")
    evaluate_sar_artifacts_cmd.add_argument("--image-size", type=int, default=128, help="Resize size for artifact metrics")
    evaluate_sar_artifacts_cmd.add_argument("--dry-run", action="store_true", help="Record what would be evaluated")

    repair_parameters_cmd = subparsers.add_parser(
        "repair-parameters",
        help="Create a bounded parameter-repair plan from a recipe and execution/evaluation evidence",
    )
    repair_parameters_cmd.add_argument("--recipe", required=True, help="Original SagaRecipe YAML/JSON path")
    repair_parameters_cmd.add_argument("--execution", default=None, help="Optional recipe_execution.json with step reports")
    repair_parameters_cmd.add_argument("--quality-report", default=None, help="Optional quality_evaluation.json")
    repair_parameters_cmd.add_argument("--distribution-report", default=None, help="Optional distribution_evaluation.json")
    repair_parameters_cmd.add_argument("--output", required=True, help="Output directory for repair plan artifacts")
    repair_parameters_cmd.add_argument("--max-trials", type=int, default=3, help="Maximum bounded repair trials")
    repair_parameters_cmd.add_argument("--trial-index", type=int, default=0, help="Current zero-based repair trial index")
    repair_parameters_cmd.add_argument("--dry-run", action="store_true", help="Write what would be repaired without marking it as executable")

    repair_run_cmd = subparsers.add_parser(
        "repair-run",
        help="Create a bounded repair/replan artifact from an agent-run and optionally execute the revised recipe",
    )
    repair_run_cmd.add_argument("--run-dir", required=True, help="Agent-run directory containing recipe.yaml and execution report")
    repair_run_cmd.add_argument("--output", default=None, help="Output directory, defaults to RUN_DIR/repair_replan")
    repair_run_cmd.add_argument("--max-trials", type=int, default=3, help="Maximum bounded repair trials")
    repair_run_cmd.add_argument("--trial-index", type=int, default=0, help="Current zero-based repair trial index")
    repair_run_cmd.add_argument("--dry-run", action="store_true", help="Write repair artifacts and dry-run revised execution")
    repair_run_cmd.add_argument("--execute-revised", action="store_true", help="Execute the revised recipe if one is generated")
    repair_run_cmd.add_argument("--continue-on-error", action="store_true", help="Continue revised execution after failed steps")

    export_dataset = subparsers.add_parser(
        "export-dataset",
        help="Package generated outputs as a traceable SAGA augmented dataset",
    )
    export_dataset.add_argument("--source", required=True, help="Directory containing generated images")
    export_dataset.add_argument("--output", required=True, help="Augmented dataset export directory")
    export_dataset.add_argument("--recipe-id", default="manual_export", help="Recipe id for provenance")
    export_dataset.add_argument("--task", default="augmentation", help="Task name for the export card")
    export_dataset.add_argument(
        "--mode",
        default="copy",
        choices=["copy", "symlink", "manifest_only"],
        help="How generated images should be materialized",
    )
    export_dataset.add_argument("--selection-manifest", default=None, help="Optional selection_manifest.json")
    export_dataset.add_argument("--quality-report", default=None, help="Optional quality_evaluation.json")
    export_dataset.add_argument("--repair-policy", default=None, help="Optional repair_policy.json")
    export_dataset.add_argument("--include-originals", action="store_true", help="Also link selected original inputs")
    export_dataset.add_argument("--require-quality-pass", action="store_true", help="Block export when quality triggers exist")
    export_dataset.add_argument(
        "--write-caption-sidecars",
        action="store_true",
        help="Write image-stem .txt captions next to exported generated images",
    )
    export_dataset.add_argument("--dry-run", action="store_true", help="Write an export plan without materializing images")

    memory_cmd = subparsers.add_parser(
        "memory",
        help="Record or list SAGA policy memory entries",
    )
    memory_subparsers = memory_cmd.add_subparsers(dest="memory_command", required=True)
    memory_record = memory_subparsers.add_parser("record", help="Record an existing agent-run directory")
    memory_record.add_argument("--run", required=True, help="Agent run directory")
    memory_record.add_argument("--memory-dir", default="runs/memory", help="Policy memory directory")
    memory_list = memory_subparsers.add_parser("list", help="List recent policy memory entries")
    memory_list.add_argument("--memory-dir", default="runs/memory", help="Policy memory directory")
    memory_list.add_argument("--limit", type=int, default=10, help="Number of entries to show")
    memory_list.add_argument("--task", default=None, help="Optional task filter")
    memory_list.add_argument("--dataset-root", default=None, help="Optional dataset root filter")

    explain_plan = subparsers.add_parser(
        "explain-plan",
        help="Explain why an agent-run selected its augmentation plan",
    )
    explain_plan.add_argument("--run", required=True, help="Agent-run directory")
    explain_plan.add_argument("--output", default=None, help="Output directory, defaults to --run")

    compare_runs = subparsers.add_parser(
        "compare-runs",
        help="Compare multiple agent-run directories by planner choice, observers, export, and downstream evaluator results",
    )
    compare_runs.add_argument("runs", nargs="+", help="Agent-run directories to compare")
    compare_runs.add_argument("--output", required=True, help="Output directory for comparison artifacts")

    benefit_evidence_cmd = subparsers.add_parser(
        "benefit-evidence",
        help="Build a unified benefit evidence report for an agent-run",
    )
    benefit_evidence_cmd.add_argument("--run", required=True, help="Agent-run directory")
    benefit_evidence_cmd.add_argument("--output", default=None, help="Output directory, defaults to --run")

    plan_critic_cmd = subparsers.add_parser(
        "plan-critic",
        help="Critique an agent-run plan for task/skill/evaluator/claim risks",
    )
    plan_critic_cmd.add_argument("--run", required=True, help="Agent-run directory")
    plan_critic_cmd.add_argument("--output", default=None, help="Output directory, defaults to --run")

    pilot_recipes_cmd = subparsers.add_parser(
        "pilot-recipes",
        help="Compile top-k candidate planner recipes and optionally execute dry-run pilots",
    )
    pilot_recipes_cmd.add_argument("--run", required=True, help="Agent-run directory containing augmentation_plan.json")
    pilot_recipes_cmd.add_argument("--output", default=None, help="Output directory, defaults to RUN/candidate_pilot")
    pilot_recipes_cmd.add_argument("--top-k", type=int, default=3, help="Number of executable candidates to compile")
    pilot_recipes_cmd.add_argument("--pilot-sample-count", type=int, default=8, help="Cap generated samples for pilot recipes")
    pilot_recipes_cmd.add_argument("--execute", action="store_true", help="Execute candidate recipes")
    pilot_recipes_cmd.add_argument("--dry-run", action="store_true", help="Execute pilot recipes in dry-run mode")
    pilot_recipes_cmd.add_argument("--real-run", action="store_true", help="Actually run candidate recipe skills when --execute is set")
    pilot_recipes_cmd.add_argument("--continue-on-error", action="store_true", help="Continue candidate pilot after failed steps")

    auto_repair_cmd = subparsers.add_parser(
        "auto-repair-run",
        help="Run bounded repair planning and optionally execute the revised recipe",
    )
    auto_repair_cmd.add_argument("--run-dir", required=True, help="Agent-run directory containing recipe.yaml and execution report")
    auto_repair_cmd.add_argument("--output", default=None, help="Output directory, defaults to RUN/auto_repair")
    auto_repair_cmd.add_argument("--max-trials", type=int, default=3, help="Maximum bounded repair trials")
    auto_repair_cmd.add_argument("--trial-index", type=int, default=0, help="Current zero-based repair trial index")
    auto_repair_cmd.add_argument("--execute-revised", action="store_true", help="Execute revised recipe if generated")
    auto_repair_cmd.add_argument("--dry-run", action="store_true", help="Execute revised recipe in dry-run mode")
    auto_repair_cmd.add_argument("--run", action="store_true", help="Actually run revised recipe skills; defaults to dry-run")
    auto_repair_cmd.add_argument("--continue-on-error", action="store_true", help="Continue revised execution after failed steps")

    plan = subparsers.add_parser("plan", help="Create an augmentation recipe from diagnosis")
    plan.add_argument("--diagnosis", required=True, help="Path to machine_report.json")
    plan.add_argument("--output", required=True, help="Output recipe path, .yaml or .json")
    plan.add_argument("--request", default=None, help="Optional request YAML/JSON")
    plan.add_argument("--llm-config", default=None, help="Optional LLM YAML/JSON config")
    plan.add_argument("--use-llm", action="store_true", help="Use LLM planner instead of rule-based planner")

    style_transfer = subparsers.add_parser(
        "style-transfer",
        help="Run the myproject/stytransfer wrapper as a SAGA skill",
    )
    style_transfer.add_argument("--content", required=True, help="Content image or directory")
    style_transfer.add_argument("--style", required=True, help="Style image or directory")
    style_transfer.add_argument("--output", required=True, help="Output directory")
    style_transfer.add_argument("--config", default=None, help="Optional style transfer skill YAML/JSON config")
    style_transfer.add_argument("--dry-run", action="store_true", help="Write command/report without running model")

    diffusion_lora = subparsers.add_parser(
        "diffusion-lora",
        help="Wrap qinglong_trainer text-caption LoRA training and inference as a SAGA skill",
    )
    diffusion_lora.add_argument("--dataset", required=True, help="Directory containing paired images and .txt captions")
    diffusion_lora.add_argument("--output", required=True, help="Output directory for SAGA skill artifacts")
    diffusion_lora.add_argument("--config", default=None, help="Optional diffusion LoRA skill YAML/JSON config")
    diffusion_lora.add_argument("--model-family", default=None, help="Model family: flux, sd3, or sdxl")
    diffusion_lora.add_argument("--target-count", type=int, default=None, help="Number of generated samples to plan/run")
    diffusion_lora.add_argument("--output-name", default=None, help="LoRA output base name")
    diffusion_lora.add_argument("--prompt", default=None, help="Optional prompt to use before dataset sampled captions")
    diffusion_lora.add_argument("--lora-weights", default=None, help="Existing LoRA weights for inference-only or override")
    diffusion_lora.add_argument("--no-train", action="store_true", help="Skip LoRA training command/run")
    diffusion_lora.add_argument("--no-infer", action="store_true", help="Skip inference command/run")
    diffusion_lora.add_argument("--dry-run", action="store_true", help="Write commands/reports without model execution")
    diffusion_lora.add_argument("--run", action="store_true", help="Actually execute train/inference; defaults to dry-run")

    geodiff_sar = subparsers.add_parser(
        "geodiff-sar",
        help="Run GeoDiff-SAR composite skill: real GEFM extraction, ControlNet training, 3D GEFM rendering, and inference",
    )
    geodiff_sar.add_argument("--dataset", default=None, help="Real SAR image/txt dataset used for ControlNet training")
    geodiff_sar.add_argument("--model", default=None, help="3D model used to render inference GEFM conditions")
    geodiff_sar.add_argument("--output", required=True, help="Output directory for SAGA skill artifacts")
    geodiff_sar.add_argument("--config", default="configs/skills/geodiff_sar.yaml", help="Optional GeoDiff-SAR skill YAML/JSON config")
    geodiff_sar.add_argument("--azimuth-values", default=None, help="Comma-separated target azimuth values, e.g. 0,30,60")
    geodiff_sar.add_argument("--azimuth-start", type=float, default=None, help="Sweep start azimuth in degrees")
    geodiff_sar.add_argument("--azimuth-stop", type=float, default=None, help="Sweep stop azimuth in degrees")
    geodiff_sar.add_argument("--azimuth-step", type=float, default=None, help="Sweep step in degrees")
    geodiff_sar.add_argument("--depressions", default=None, help="Comma-separated depression angles, e.g. 20,40,60")
    geodiff_sar.add_argument("--target-count", type=int, default=None, help="Number of generated samples to plan/run")
    geodiff_sar.add_argument("--training-epochs", type=int, default=None, help="ControlNet training epochs")
    geodiff_sar.add_argument("--controlnet-weights", default=None, help="Existing ControlNet weights for inference or override")
    geodiff_sar.add_argument("--controlnet-init-weights", default=None, help="Initial ControlNet weights for continued training")
    geodiff_sar.add_argument("--prompt", default=None, help="Optional prompt override for generated samples")
    geodiff_sar.add_argument("--model-name", default=None, help="Stable model name used for raytracing outputs")
    geodiff_sar.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value for train/inference")
    geodiff_sar.add_argument("--no-train", action="store_true", help="Skip real-image GEFM extraction and ControlNet training")
    geodiff_sar.add_argument("--no-infer", action="store_true", help="Skip 3D GEFM rendering and ControlNet inference")
    geodiff_sar.add_argument("--no-extract-real-gefm", action="store_true", help="Use dataset directory directly as ControlNet condition dir")
    geodiff_sar.add_argument("--no-render-gefm", action="store_true", help="Skip 3D-model GEFM rendering")
    geodiff_sar.add_argument("--dry-run", action="store_true", help="Write commands/reports without model execution")
    geodiff_sar.add_argument("--run", action="store_true", help="Actually execute the expensive stages; defaults to dry-run")

    traditional_aug = subparsers.add_parser(
        "traditional-augment",
        help="Run fast SAR-aware traditional augmentation as a SAGA skill",
    )
    traditional_aug.add_argument("--dataset", required=True, help="Image dataset root")
    traditional_aug.add_argument("--output", required=True, help="Output directory for SAGA skill artifacts")
    traditional_aug.add_argument("--config", default=None, help="Optional traditional augmentation YAML/JSON config")
    traditional_aug.add_argument("--dataset-config", default=None, help="Optional validated dataset config for metadata filters")
    traditional_aug.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata include filter as field=value. Can be repeated; requires --dataset-config for semantic metadata.",
    )
    traditional_aug.add_argument(
        "--exclude-filter",
        action="append",
        default=[],
        help="Metadata exclude filter as field=value or field=a,b. Can be repeated; requires --dataset-config for semantic metadata.",
    )
    traditional_aug.add_argument(
        "--exclude-polarization",
        action="append",
        default=[],
        help="Convenience metadata exclude filter for polarization values such as pauli.",
    )
    traditional_aug.add_argument("--target-count", type=int, default=None, help="Number of augmented samples to plan/run")
    traditional_aug.add_argument("--multiplier", type=int, default=None, help="Fallback output multiplier over selected inputs")
    traditional_aug.add_argument("--dry-run", action="store_true", help="Write plan/report without materializing images")
    traditional_aug.add_argument("--run", action="store_true", help="Actually materialize augmented images; defaults to dry-run")

    sar_preprocess = subparsers.add_parser(
        "sar-preprocess",
        help="Normalize SAR intensity/bit-depth/size before generation or evaluation",
    )
    sar_preprocess.add_argument("--input", required=True, help="Input image directory")
    sar_preprocess.add_argument("--output", required=True, help="Output directory for preprocessing artifacts")
    sar_preprocess.add_argument("--config", default=None, help="Optional SAR preprocess YAML/JSON config")
    sar_preprocess.add_argument("--mode", default=None, help="Preprocess mode: percentile, log_percentile, minmax, none")
    sar_preprocess.add_argument("--resize", type=int, default=None, help="Optional square resize")
    sar_preprocess.add_argument("--dry-run", action="store_true", help="Write plan/report without materializing images")
    sar_preprocess.add_argument("--run", action="store_true", help="Actually materialize preprocessed images; defaults to dry-run")

    metadata_caption = subparsers.add_parser(
        "metadata-caption",
        help="Stage image/txt caption pairs from validated metadata",
    )
    metadata_caption.add_argument("--dataset", required=True, help="Dataset root")
    metadata_caption.add_argument("--output", required=True, help="Output directory")
    metadata_caption.add_argument("--dataset-config", default=None, help="Optional validated SAGA dataset config")
    metadata_caption.add_argument("--template", default=None, help="Optional Python format template using metadata fields")
    metadata_caption.add_argument("--mode", choices=["copy", "symlink", "manifest_only"], default="copy", help="How images are staged")
    metadata_caption.add_argument("--overwrite", action="store_true", help="Overwrite existing txt captions in staged output")
    metadata_caption.add_argument("--dry-run", action="store_true", help="Write plan/report without staging files")
    metadata_caption.add_argument("--run", action="store_true", help="Actually stage images and captions; defaults to dry-run")

    dataset_balancing = subparsers.add_parser(
        "dataset-balance",
        help="Create class/metadata deficit plan for target-oriented augmentation",
    )
    dataset_balancing.add_argument("--dataset", required=True, help="Dataset root")
    dataset_balancing.add_argument("--output", required=True, help="Output directory")
    dataset_balancing.add_argument("--dataset-config", default=None, help="Optional validated SAGA dataset config")
    dataset_balancing.add_argument("--fields", default="class,polarization,azimuth_deg", help="Comma-separated fields to balance")
    dataset_balancing.add_argument("--target-per-bin", type=int, default=None, help="Optional target count per bin")
    dataset_balancing.add_argument("--max-multiplier", type=float, default=3.0, help="Do not recommend more than current_count * max_multiplier")
    dataset_balancing.add_argument("--dry-run", action="store_true", help="Record a balancing plan")

    gan_generation = subparsers.add_parser(
        "gan-generate",
        help="Wrap myproject/dcgan or myproject/dcgan-cpu as a SAGA GAN generation skill",
    )
    gan_generation.add_argument("--dataset", required=True, help="Training image directory")
    gan_generation.add_argument("--output", required=True, help="Output directory")
    gan_generation.add_argument("--project-dir", default=None, help="Override GAN project directory")
    gan_generation.add_argument("--variant", choices=["gpu", "cpu"], default="gpu", help="Use myproject/dcgan or myproject/dcgan-cpu")
    gan_generation.add_argument("--target-count", type=int, default=100, help="Number of generated images")
    gan_generation.add_argument("--epochs", type=int, default=None, help="Optional training epochs override")
    gan_generation.add_argument("--model-path", default=None, help="Existing generator model path for inference")
    gan_generation.add_argument("--no-train", action="store_true", help="Skip training command")
    gan_generation.add_argument("--no-infer", action="store_true", help="Skip inference command")
    gan_generation.add_argument("--dry-run", action="store_true", help="Write commands/report without running GAN")
    gan_generation.add_argument("--run", action="store_true", help="Actually run generated GAN commands; defaults to dry-run")

    background_generation = subparsers.add_parser(
        "background-generate",
        help="Generate SAR background scenes with packaged segmentation-control weights",
    )
    background_generation.add_argument("scene_prompt", nargs="*", help="Scene words, e.g. '建筑 水体 道路'")
    background_generation.add_argument("--output", required=True, help="Output directory")
    background_generation.add_argument("--config", default="configs/skills/background_generation.yaml", help="Optional background skill YAML/JSON config")
    background_generation.add_argument("--project-dir", default=None, help="Override scene_gen_segment_large project directory")
    background_generation.add_argument("--num-images", type=int, default=None, help="Number of background images/control maps")
    background_generation.add_argument("--split", choices=["all", "train", "val", "test"], default=None, help="Control-map split")
    background_generation.add_argument("--top-k", type=int, default=None, help="Random pool among best matching controls")
    background_generation.add_argument("--selection-seed", type=int, default=None, help="Control-map selection seed")
    background_generation.add_argument("--seed", type=int, default=None, help="Base generation seed")
    background_generation.add_argument("--cns", type=float, default=None, help="ControlNet strength")
    background_generation.add_argument("--steps", type=int, default=None, help="Sampling steps")
    background_generation.add_argument("--scale", type=float, default=None, help="Guidance scale")
    background_generation.add_argument("--width", type=int, default=None, help="Output width")
    background_generation.add_argument("--height", type=int, default=None, help="Output height")
    background_generation.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 2")
    background_generation.add_argument("--cpu-threads", type=int, default=None, help="CPU threads for accelerate launch")
    background_generation.add_argument("--extra-prompt", default=None, help="Extra English prompt suffix")
    background_generation.add_argument("--clear-cache-each-image", action="store_true", help="Pass through to inference script")
    background_generation.add_argument("--show-top", type=int, default=None, help="Show top matching controls in the wrapped script")
    background_generation.add_argument("--dry-run", action="store_true", help="Select controls and write prompt plan without model inference")
    background_generation.add_argument("--run", action="store_true", help="Actually run model inference; defaults to dry-run")

    compose_target_background = subparsers.add_parser(
        "compose-target-background",
        help="Fuse SAR target images into background scenes with deterministic image blending",
    )
    compose_target_background.add_argument("--targets", required=True, help="Target image directory")
    compose_target_background.add_argument("--backgrounds", required=True, help="Background image directory")
    compose_target_background.add_argument("--output", required=True, help="Output directory")
    compose_target_background.add_argument("--config", default="configs/skills/target_background_composition.yaml", help="Optional composition skill YAML/JSON config")
    compose_target_background.add_argument("--masks", default=None, help="Optional target mask directory")
    compose_target_background.add_argument("--target-count", type=int, default=None, help="Number of composed images")
    compose_target_background.add_argument(
        "--blend-mode",
        choices=["feather", "laplacian", "poisson", "hard"],
        default=None,
        help="Fusion mode",
    )
    compose_target_background.add_argument("--mask-mode", default=None, help="Mask mode: auto or provided")
    compose_target_background.add_argument("--placement-policy", choices=["random", "center"], default=None, help="Target placement policy")
    compose_target_background.add_argument("--dry-run", action="store_true", help="Write composition plan/report without materializing images")
    compose_target_background.add_argument("--run", action="store_true", help="Actually materialize composed images; defaults to dry-run")

    pseudocolor = subparsers.add_parser(
        "pseudocolor",
        help="Apply lightweight SAR pseudocolor transform",
    )
    pseudocolor.add_argument("--input", required=True, help="Input image directory")
    pseudocolor.add_argument("--output", required=True, help="Output directory")
    pseudocolor.add_argument("--colormap", default="viridis", help="Colormap: viridis, hot, ocean, sar")
    pseudocolor.add_argument("--flat", action="store_true", help="Do not preserve input directory tree")
    pseudocolor.add_argument("--dry-run", action="store_true", help="Write plan/report without materializing images")
    pseudocolor.add_argument("--run", action="store_true", help="Actually materialize pseudocolor images; defaults to dry-run")

    model_to_pov = subparsers.add_parser(
        "model-to-pov",
        help="Compile OBJ/STL/PLY/GLB/point-cloud assets into a RaySAR-compatible POV scene",
    )
    model_to_pov.add_argument("--model", required=True, help="3D model file: obj/stl/ply/glb/gltf/off/dae/3ds or xyz txt")
    model_to_pov.add_argument("--output", required=True, help="Output directory for generated_scene.pov and report")
    model_to_pov.add_argument("--config", default="configs/skills/model_to_pov_scene.yaml", help="Optional compiler YAML/JSON config")
    model_to_pov.add_argument("--incidence-angle", type=float, default=None, help="SAR incidence angle in degrees")
    model_to_pov.add_argument("--depression-angle", type=float, default=None, help="SAR depression angle in degrees; incidence becomes 90 - depression")
    model_to_pov.add_argument("--azimuth", type=float, default=None, help="Target yaw/aspect rotation around the POV height axis in degrees")
    model_to_pov.add_argument("--pitch", type=float, default=None, help="Pitch rotation in degrees")
    model_to_pov.add_argument("--roll", type=float, default=None, help="Roll rotation in degrees")
    model_to_pov.add_argument("--target-extent", type=float, default=None, help="Scale the longest model extent to this many meters")
    model_to_pov.add_argument("--scale", type=float, default=None, help="Additional model scale factor")
    model_to_pov.add_argument("--input-up-axis", choices=["x", "y", "z"], default=None, help="Up axis in the input model")
    model_to_pov.add_argument("--pov-height-axis", choices=["x", "y", "z"], default=None, help="Height axis in the generated POV scene")
    model_to_pov.add_argument("--range-distance", type=float, default=None, help="Horizontal sensor distance used by the RaySAR-style camera")
    model_to_pov.add_argument("--scene-center", default=None, help="Scene center as x,y,z")
    model_to_pov.add_argument("--no-ground", action="store_true", help="Do not instantiate the generated ground plane object")
    model_to_pov.add_argument("--dry-run", action="store_true", help="Compile/report without expensive RaySAR rendering")
    model_to_pov.add_argument("--run", action="store_true", help="Compile scene; defaults to dry-run but still writes generated_scene.pov")

    raysar = subparsers.add_parser(
        "raysar-synthesis",
        help="Run RaySAR/POV-Ray physical SAR simulation as a SAGA skill",
    )
    raysar.add_argument("--pov-scene", default=None, help="RaySAR-compatible .pov scene")
    raysar.add_argument("--contributions-txt", default=None, help="Existing Contributions.txt for postprocess-only mode")
    raysar.add_argument("--parameters-file", default=None, help="RaySAR parameters.txt")
    raysar.add_argument("--output", required=True, help="Output directory")
    raysar.add_argument("--config", default="configs/skills/raysar.yaml", help="Optional RaySAR skill YAML/JSON config")
    raysar.add_argument("--adapted-povray", default=None, help="Compiled RaySAR-adapted POV-Ray executable")
    raysar.add_argument("--width", type=int, default=None, help="Render width")
    raysar.add_argument("--height", type=int, default=None, help="Render height")
    raysar.add_argument("--postprocess-only", action="store_true", help="Skip rendering and post-process --contributions-txt")
    raysar.add_argument("--no-postprocess", action="store_true", help="Only render Contributions.txt and preview image")
    raysar.add_argument("--no-auto-fix-scene", action="store_true", help="Do not patch SAR_Output_Data/SAR_Intersection or orthographic angle")
    raysar.add_argument("--no-build", action="store_true", help="Do not compile adapted POV-Ray if no executable is found")
    raysar.add_argument("--dry-run", action="store_true", help="Write plan/report without running RaySAR")
    raysar.add_argument("--run", action="store_true", help="Actually render/postprocess; defaults to dry-run")

    raysar_sweep = subparsers.add_parser(
        "raysar-sweep",
        help="Run multi-view RaySAR simulation from a 3D model by sweeping azimuth/aspect angles",
    )
    raysar_sweep.add_argument("--model", required=True, help="3D model file: obj/stl/ply/glb/gltf/off/dae/3ds or xyz txt")
    raysar_sweep.add_argument("--output", required=True, help="Output directory")
    raysar_sweep.add_argument("--config", default="configs/skills/raysar_sweep.yaml", help="Optional RaySAR sweep skill YAML/JSON config")
    raysar_sweep.add_argument("--azimuth-values", default=None, help="Comma-separated azimuth values, e.g. 0,10,20")
    raysar_sweep.add_argument("--azimuth-start", type=float, default=None, help="Sweep start azimuth in degrees")
    raysar_sweep.add_argument("--azimuth-stop", type=float, default=None, help="Sweep stop azimuth in degrees")
    raysar_sweep.add_argument("--azimuth-step", type=float, default=None, help="Sweep step in degrees")
    raysar_sweep.add_argument("--incidence-angle", type=float, default=None, help="SAR incidence angle in degrees")
    raysar_sweep.add_argument("--depression-angle", type=float, default=None, help="SAR depression angle in degrees")
    raysar_sweep.add_argument("--target-extent", type=float, default=None, help="Scale longest model extent to this many meters")
    raysar_sweep.add_argument("--scale", type=float, default=None, help="Additional model scale factor")
    raysar_sweep.add_argument("--input-up-axis", choices=["x", "y", "z"], default=None, help="Up axis in the input model")
    raysar_sweep.add_argument("--pov-height-axis", choices=["x", "y", "z"], default=None, help="Height axis in generated POV scenes")
    raysar_sweep.add_argument("--range-distance", type=float, default=None, help="Horizontal sensor distance used by the camera")
    raysar_sweep.add_argument("--sensor-plane", type=float, default=None, help="Orthographic sensor plane width/height in meters")
    raysar_sweep.add_argument("--parameters-file", default=None, help="Optional RaySAR parameters.txt")
    raysar_sweep.add_argument("--adapted-povray", default=None, help="Compiled RaySAR-adapted POV-Ray executable")
    raysar_sweep.add_argument("--width", type=int, default=None, help="Render width")
    raysar_sweep.add_argument("--height", type=int, default=None, help="Render height")
    raysar_sweep.add_argument("--no-postprocess", action="store_true", help="Only render Contributions.txt and preview images")
    raysar_sweep.add_argument("--dry-run", action="store_true", help="Write plan/report without running RaySAR")
    raysar_sweep.add_argument("--run", action="store_true", help="Actually render/postprocess; defaults to dry-run")

    metadata_slice_eval = subparsers.add_parser(
        "metadata-slice-eval",
        help="Summarize or evaluate results by SAR metadata slices",
    )
    metadata_slice_eval.add_argument("--dataset-config", required=True, help="Validated SAGA dataset config")
    metadata_slice_eval.add_argument("--output", required=True, help="Output directory")
    metadata_slice_eval.add_argument("--fields", default="class,polarization,azimuth_deg,band,resolution_m", help="Comma-separated slice fields")
    metadata_slice_eval.add_argument("--predictions-csv", default=None, help="Optional per-sample predictions CSV")
    metadata_slice_eval.add_argument("--dry-run", action="store_true", help="Write slice report")

    leakage_check = subparsers.add_parser(
        "leakage-check",
        help="Check baseline/augmented/validation datasets for leakage",
    )
    leakage_check.add_argument("--baseline-dataset", required=True, help="Baseline/original dataset")
    leakage_check.add_argument("--augmented-dataset", default=None, help="Optional augmented dataset")
    leakage_check.add_argument("--val-dataset", default=None, help="Optional validation/test dataset")
    leakage_check.add_argument("--output", required=True, help="Output directory")
    leakage_check.add_argument("--sample-limit", type=int, default=20000, help="Maximum images per dataset")
    leakage_check.add_argument("--dry-run", action="store_true", help="Write leakage report")

    duplicate_check = subparsers.add_parser(
        "duplicate-check",
        help="Detect duplicate and near-duplicate images",
    )
    duplicate_check.add_argument("--input", required=True, help="Input image directory")
    duplicate_check.add_argument("--output", required=True, help="Output directory")
    duplicate_check.add_argument("--hash-size", type=int, default=16, help="Average hash size")
    duplicate_check.add_argument("--hamming-threshold", type=int, default=6, help="Near-duplicate Hamming threshold")
    duplicate_check.add_argument("--sample-limit", type=int, default=5000, help="Maximum images to compare")
    duplicate_check.add_argument("--dry-run", action="store_true", help="Write duplicate report")

    classification_eval = subparsers.add_parser(
        "classification-eval",
        help="Create or run downstream classification baseline-vs-augmented evaluation commands",
    )
    classification_eval.add_argument("--baseline-dataset", required=True, help="Baseline/original classification dataset root")
    classification_eval.add_argument("--augmented-dataset", default=None, help="Optional augmented classification dataset root")
    classification_eval.add_argument("--val-dataset", default=None, help="Optional held-out validation dataset root")
    classification_eval.add_argument("--baseline-dataset-config", default=None, help="Optional SAGA dataset config for baseline labels/metadata")
    classification_eval.add_argument("--augmented-dataset-config", default=None, help="Optional SAGA dataset config for augmented labels/metadata")
    classification_eval.add_argument("--val-dataset-config", default=None, help="Optional SAGA dataset config for held-out validation labels/metadata")
    classification_eval.add_argument("--output", required=True, help="Output directory for evaluation artifacts")
    classification_eval.add_argument("--config", default=None, help="Optional classification evaluation YAML/JSON config")
    classification_eval.add_argument("--dry-run", action="store_true", help="Write commands/reports without training")
    classification_eval.add_argument("--run", action="store_true", help="Actually run training/validation commands; defaults to dry-run")

    agent_command = subparsers.add_parser(
        "agent-command",
        help="Run a natural-language SAGA command through the skill router",
    )
    agent_command.add_argument("--text", required=True, help="Natural-language user command")
    agent_command.add_argument("--output", required=True, help="Output run directory")
    agent_command.add_argument("--config", default=None, help="Optional skill YAML/JSON config")
    agent_command.add_argument("--dry-run", action="store_true", help="Route and report without running model")

    args = parser.parse_args()
    if args.command == "diagnose":
        config = load_dataset_config(args.dataset)
        report = run_diagnosis(
            config=config,
            output_dir=args.output,
            read_image_size=not args.skip_image_size,
        )
        print(f"Wrote diagnosis to {Path(args.output).resolve()}")
        print(f"Total images: {report['summary']['total_images']}")
        print(f"Parser schemas: {report['summary']['parser_schemas']}")
        print(f"Issues: {len(report['issues'])}")
    elif args.command == "inspect-format":
        report = inspect_dataset_format(
            root=args.root,
            output_dir=args.output,
            sample_limit=args.sample_limit,
        )
        print(f"Wrote format inspection to {Path(args.output).resolve()}")
        print(f"Total images: {report['total_images']}")
        print(f"Suffix counts: {report['suffix_counts']}")
    elif args.command == "profile-dataset":
        request_result = recognize_request(
            text=args.request,
            output_dir=args.output,
            dataset_root=args.root,
        )
        report = profile_dataset(
            root=args.root,
            output_dir=args.output,
            request=args.request,
            intent_spec=request_result["intent_spec"],
            format_hints=request_result["format_hints"],
            sample_limit=args.sample_limit,
            image_probe_limit=args.image_probe_limit,
            txt_preview_chars=args.txt_preview_chars,
        )
        dataset_profile = report["dataset_profile"]
        print(f"Wrote dataset profile to {Path(args.output).resolve()}")
        print(f"Profile level: {dataset_profile['profile_level']} ({dataset_profile['status']})")
        print(f"Task candidates: {dataset_profile['task_candidates']}")
        print(f"Clarification questions: {len(report['clarification_questions'])}")
    elif args.command == "infer-format":
        llm_config = LLMConfig.from_mapping(load_llm_config(args.llm_config))
        infer_format_spec_with_llm(
            inspection_path=args.inspection,
            output_path=args.output,
            llm_config=llm_config,
            user_description=args.description,
            prompt_sample_limit=args.prompt_sample_limit,
            txt_preview_limit=args.txt_preview_limit,
            use_response_format=not args.no_response_format,
        )
        print(f"Wrote inferred format spec to {Path(args.output).resolve()}")
    elif args.command == "understand-format":
        llm_config = LLMConfig.from_mapping(load_llm_config(args.llm_config))
        understanding = understand_format_with_llm(
            inspection_path=args.inspection,
            output_path=args.output,
            llm_config=llm_config,
            user_description=args.description,
            prompt_group_limit=args.prompt_group_limit,
            prompt_sample_limit=args.prompt_sample_limit,
            txt_preview_limit=args.txt_preview_limit,
            use_response_format=not args.no_response_format,
        )
        rules = understanding.get("format_understanding", {}).get("rules", [])
        unresolved = understanding.get("format_understanding", {}).get("unresolved", [])
        print(f"Wrote format understanding to {Path(args.output).resolve()}")
        print(f"Template rules: {len(rules)}")
        print(f"Unresolved items: {len(unresolved)}")
    elif args.command == "build-format-spec":
        if not args.understanding and not args.inspection:
            raise SystemExit("build-format-spec requires --understanding and/or --inspection")
        spec = build_format_spec(
            understanding_path=args.understanding,
            inspection_path=args.inspection,
            output_path=args.output,
            name=args.name,
            include_heuristics=not args.no_heuristics,
        )
        rules = spec.get("format_spec", {}).get("rules", [])
        print(f"Wrote format spec to {Path(args.output).resolve()}")
        print(f"Rules: {len(rules)}")
    elif args.command == "validate-format":
        config = load_dataset_config(args.dataset)
        required_fields = [field.strip() for field in args.required_fields.split(",") if field.strip()]
        report = validate_dataset_format(
            config=config,
            output_dir=args.output,
            sample_limit=args.sample_limit,
            required_fields=required_fields,
        )
        print(f"Wrote format validation to {Path(args.output).resolve()}")
        print(f"Validated samples: {report['validated_samples']}")
        print(f"Valid: {report['valid']}")
        print(f"Field coverage: {report['field_coverage']}")
    elif args.command == "bridge-format":
        required_fields = None
        if args.required_fields:
            required_fields = [field.strip() for field in args.required_fields.split(",") if field.strip()]
        report = compile_and_validate_format_from_profile_dir(
            profile_dir=args.profile,
            output_dir=args.output,
            sample_limit=args.sample_limit,
            required_fields=required_fields,
            include_heuristics=not args.no_heuristics,
        )
        print(f"Wrote format bridge report to {Path(args.output or args.profile).resolve()}")
        print(f"Valid: {report['valid']}")
        print(f"Profile level after validation: {report['profile_level_after_validation']}")
        print(f"Required fields: {report['required_fields']}")
        print(f"Field coverage: {report['validation']['field_coverage']}")
    elif args.command == "agent-run":
        state = run_agent(
            request=args.request,
            output_dir=args.output,
            dataset_root=args.dataset_root,
            style_transfer_config=args.config,
            diffusion_lora_config=args.diffusion_lora_config,
            traditional_aug_config=args.traditional_aug_config,
            saga_config_path=args.saga_config,
            llm_config=LLMConfig.from_mapping(load_llm_config(args.llm_config))
            if args.use_llm_planner or args.use_llm_intent
            else None,
            use_llm_intent=args.use_llm_intent,
            use_llm_planner=args.use_llm_planner,
            memory_dir=args.memory_dir,
            use_memory=not args.no_memory,
            dry_run=(not args.run) or args.dry_run,
            sample_limit=args.sample_limit,
            image_probe_limit=args.image_probe_limit,
            bridge_sample_limit=args.bridge_sample_limit,
            execute=not args.no_execute,
        )
        print(f"Wrote agent run to {Path(args.output).resolve()}")
        print(f"Recipe: {state.get('recipe_path')}")
        print(f"Bridge valid: {state.get('bridge_valid')}")
        print(f"Execution status: {state.get('execution_status')}")
    elif args.command == "agent-continue":
        state = continue_agent_run(
            run_dir=args.run_dir,
            answer=args.answer,
            output_dir=args.output,
            llm_config=LLMConfig.from_mapping(load_llm_config(args.llm_config))
            if args.use_llm_planner or args.use_llm_intent
            else None,
            use_llm_intent=args.use_llm_intent,
            use_llm_planner=args.use_llm_planner,
            dry_run=(not args.run) or args.dry_run,
            execute=not args.no_execute,
            memory_dir=args.memory_dir,
            use_memory=not args.no_memory,
            style_transfer_config=args.config,
            diffusion_lora_config=args.diffusion_lora_config,
            traditional_aug_config=args.traditional_aug_config,
            saga_config_path=args.saga_config,
        )
        print(f"Wrote continued agent run to {Path(state.get('output_dir')).resolve()}")
        print(f"Recipe: {state.get('recipe_path')}")
        print(f"Bridge valid: {state.get('bridge_valid')}")
        print(f"Execution status: {state.get('execution_status')}")
    elif args.command == "execute-recipe":
        report = execute_recipe(
            recipe_path=args.recipe,
            output_dir=args.output,
            dry_run=args.dry_run,
            stop_on_error=not args.continue_on_error,
        )
        print(f"Wrote recipe execution to {Path(args.output).resolve()}")
        print(f"Status: {report['status']}")
        print(f"Steps: {len(report['steps'])}")
    elif args.command == "smoke-matrix":
        report = run_smoke_matrix(
            config_path=args.config,
            output_dir=args.output,
            case_ids=args.case,
            run_real=args.run_real,
            memory_dir=args.memory_dir,
        )
        print(f"Wrote smoke matrix report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Passed: {report.get('passed_count')}/{report.get('case_count')}")
    elif args.command == "evaluate-output":
        evaluation = evaluate_output_directory(
            input_dir=args.input,
            output_dir=args.output,
            expected_count=args.expected_count,
            dry_run=args.dry_run,
            sample_limit=args.sample_limit,
        )
        repair = build_repair_policy_report(
            evaluation=evaluation,
            output_dir=Path(args.output) / "repair_policy",
            max_trials=args.max_trials,
            auto_rerun=False,
            dry_run=args.dry_run,
        )
        print(f"Wrote output evaluation to {Path(args.output).resolve()}")
        print(f"Status: {evaluation.get('status')}")
        print(f"Images: {evaluation.get('image_count')}")
        print(f"Repair policy: {repair.get('status')}")
    elif args.command == "evaluate-distribution":
        report = evaluate_distribution(
            reference_dir=args.reference,
            generated_dir=args.generated,
            output_dir=args.output,
            dry_run=args.dry_run,
            reference_sample_limit=args.reference_sample_limit,
            generated_sample_limit=args.generated_sample_limit,
            image_size=args.image_size,
            standard_fid=not args.no_standard_fid,
        )
        print(f"Wrote distribution evaluation to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Metrics: {report.get('metrics')}")
    elif args.command == "evaluate-sar-artifacts":
        report = evaluate_sar_artifacts(
            input_dir=args.input,
            output_dir=args.output,
            dry_run=args.dry_run,
            sample_limit=args.sample_limit,
            image_size=args.image_size,
        )
        print(f"Wrote SAR artifact evaluation to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Triggers: {len(report.get('triggers') or [])}")
    elif args.command == "repair-parameters":
        recipe = SagaRecipe.from_path(args.recipe)
        execution_report = load_optional_cli_mapping(args.execution) or {}
        evaluation = merge_cli_evaluations(
            quality=load_optional_cli_mapping(args.quality_report),
            distribution=load_optional_cli_mapping(args.distribution_report),
        )
        report = build_parameter_repair_plan(
            recipe=recipe,
            evaluation=evaluation,
            output_dir=args.output,
            max_trials=args.max_trials,
            trial_index=args.trial_index,
            dry_run=args.dry_run,
            execution_report=execution_report,
        )
        artifacts = report.get("artifacts") or {}
        print(f"Wrote parameter repair plan to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Patches: {report.get('patch_count')}")
        if artifacts.get("revised_recipe_yaml"):
            print(f"Revised recipe: {artifacts.get('revised_recipe_yaml')}")
    elif args.command == "repair-run":
        run_dir = Path(args.run_dir).expanduser().resolve()
        output = Path(args.output).expanduser().resolve() if args.output else run_dir / "repair_replan"
        recipe_path = run_dir / "recipe.yaml"
        execution_path = run_dir / "execution" / "recipe_execution.json"
        if not recipe_path.exists():
            raise SystemExit(f"Missing recipe: {recipe_path}")
        recipe = SagaRecipe.from_path(recipe_path)
        execution_report = load_optional_cli_mapping(execution_path.as_posix()) or {}
        report = build_parameter_repair_plan(
            recipe=recipe,
            evaluation=None,
            output_dir=output,
            max_trials=args.max_trials,
            trial_index=args.trial_index,
            dry_run=args.dry_run,
            execution_report=execution_report,
        )
        print(f"Wrote run repair plan to {output.resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Patches: {report.get('patch_count')}")
        revised = (report.get("artifacts") or {}).get("revised_recipe_yaml")
        if revised:
            print(f"Revised recipe: {revised}")
        if args.execute_revised:
            if not revised:
                print("No revised recipe was generated; skipping revised execution.")
            else:
                exec_report = execute_recipe(
                    recipe_path=revised,
                    output_dir=output / "revised_execution",
                    dry_run=args.dry_run,
                    stop_on_error=not args.continue_on_error,
                )
                print(f"Revised execution: {exec_report.get('status')}")
    elif args.command == "export-dataset":
        report = export_augmented_dataset(
            source_dir=args.source,
            output_dir=args.output,
            recipe_id=args.recipe_id,
            task=args.task,
            dry_run=args.dry_run,
            mode=args.mode,
            selection_manifest=args.selection_manifest,
            quality_report=load_optional_cli_mapping(args.quality_report),
            repair_policy=load_optional_cli_mapping(args.repair_policy),
            include_originals=args.include_originals,
            require_quality_pass=args.require_quality_pass,
            write_caption_sidecars=args.write_caption_sidecars,
        )
        print(f"Wrote dataset export to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Generated images: {report.get('generated_image_count')}")
        print(f"Manifest rows: {report.get('manifest_count')}")
        print(f"Caption sidecars: {report.get('caption_sidecar_count')}")
    elif args.command == "memory":
        if args.memory_command == "record":
            report = record_agent_run_memory(run_dir=args.run, memory_dir=args.memory_dir)
            print(f"Wrote policy memory to {Path(report['memory_dir']).resolve()}")
            print(f"Entries: {report.get('entry_count')}")
            print(f"Recipe: {report.get('entry', {}).get('recipe_id')}")
        elif args.memory_command == "list":
            entries = list_memory_entries(
                memory_dir=args.memory_dir,
                limit=args.limit,
                task=args.task,
                dataset_root=args.dataset_root,
            )
            print(f"Entries: {len(entries)}")
            for entry in entries:
                print(
                    f"- {entry.get('recipe_id')} | task={entry.get('task')} | "
                    f"exec={entry.get('execution', {}).get('status')} | "
                    f"quality={entry.get('quality', {}).get('status')} | "
                    f"run={entry.get('run_dir')}"
                )
    elif args.command == "explain-plan":
        report = explain_agent_run(run_dir=args.run, output_dir=args.output)
        selected = report.get("selected") or {}
        print(f"Wrote plan explanation to {Path(args.output or args.run).resolve()}")
        print(f"Selected skill: {selected.get('skill')}")
        print(f"Selected recipe task: {selected.get('recipe_task')}")
    elif args.command == "compare-runs":
        report = compare_agent_runs(run_dirs=args.runs, output_dir=args.output)
        print(f"Wrote run comparison to {Path(args.output).resolve()}")
        for item in report.get("ranking") or []:
            print(
                f"- {item.get('rank')}. {item.get('selected_skill')} | "
                f"score={item.get('score')} | run={item.get('run_dir')}"
            )
    elif args.command == "benefit-evidence":
        report = build_benefit_evidence_report(run_dir=args.run, output_dir=args.output)
        print(f"Wrote benefit evidence to {Path(args.output or args.run).resolve()}")
        print(f"Evidence level: {report.get('evidence_level')} ({report.get('evidence_label')})")
        print(f"Downstream claim allowed: {report.get('downstream_claim_allowed')}")
    elif args.command == "plan-critic":
        report = critique_agent_plan(run_dir=args.run, output_dir=args.output)
        print(f"Wrote plan critic to {Path(args.output or args.run).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Findings: {report.get('finding_count')}")
    elif args.command == "pilot-recipes":
        report = build_candidate_recipe_pilot(
            run_dir=args.run,
            output_dir=args.output,
            top_k=args.top_k,
            pilot_sample_count=args.pilot_sample_count,
            execute=args.execute,
            dry_run=(not args.real_run) or args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        print(f"Wrote candidate pilot to {Path(args.output or Path(args.run) / 'candidate_pilot').resolve()}")
        print(f"Candidates: {report.get('candidate_count')}")
        for item in report.get("ranking") or []:
            print(f"- {item.get('rank')}. {item.get('skill')} score={item.get('score')} status={item.get('status')}")
    elif args.command == "auto-repair-run":
        report = run_bounded_auto_repair(
            run_dir=args.run_dir,
            output_dir=args.output,
            max_trials=args.max_trials,
            trial_index=args.trial_index,
            execute_revised=args.execute_revised,
            dry_run=(not args.run) or args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        print(f"Wrote bounded auto repair to {Path(args.output or Path(args.run_dir) / 'auto_repair').resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Patch count: {(report.get('repair_plan') or {}).get('patch_count')}")
        print(f"Accepted: {report.get('accepted')}")
    elif args.command == "plan":
        recipe = run_plan(
            diagnosis_path=args.diagnosis,
            output_path=args.output,
            request_path=args.request,
            llm_config_path=args.llm_config,
            use_llm=args.use_llm,
        )
        print(f"Wrote recipe to {Path(args.output).resolve()}")
        print(f"Planner: {recipe.get('planner')}")
        print(f"Selected skills: {', '.join(recipe.get('selected_skills', []))}")
    elif args.command == "style-transfer":
        report = run_style_transfer_skill(
            content=args.content,
            style=args.style,
            output_dir=args.output,
            config_path=args.config,
            dry_run=args.dry_run,
        )
        print(f"Wrote style transfer report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
    elif args.command == "diffusion-lora":
        report = run_diffusion_lora_generation_skill(
            dataset_root=args.dataset,
            output_dir=args.output,
            config_path=args.config,
            model_family=args.model_family,
            target_count=args.target_count,
            output_name=args.output_name,
            prompt=args.prompt,
            lora_weights=args.lora_weights,
            train=not args.no_train,
            infer=not args.no_infer,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote diffusion LoRA report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
        print(f"Caption paired samples: {report.get('caption_profile', {}).get('paired_count')}")
        print(f"Train command: {report.get('artifacts', {}).get('train_command')}")
        print(f"Inference commands: {report.get('artifacts', {}).get('inference_commands')}")
    elif args.command == "geodiff-sar":
        azimuth_values = parse_cli_float_list(args.azimuth_values)
        azimuth_sweep = None
        sweep_parts = [args.azimuth_start, args.azimuth_stop, args.azimuth_step]
        if any(value is not None for value in sweep_parts):
            if not all(value is not None for value in sweep_parts):
                raise SystemExit("--azimuth-start, --azimuth-stop, and --azimuth-step must be provided together")
            azimuth_sweep = {
                "start": args.azimuth_start,
                "stop": args.azimuth_stop,
                "step": args.azimuth_step,
            }
        report = run_geodiff_sar_skill(
            dataset_root=args.dataset,
            model_file=args.model,
            output_dir=args.output,
            config_path=args.config,
            target_azimuths=azimuth_values,
            azimuth_sweep=azimuth_sweep,
            depressions=parse_cli_float_list(args.depressions),
            target_count=args.target_count,
            training_epochs=args.training_epochs,
            controlnet_weights=args.controlnet_weights,
            controlnet_init_weights=args.controlnet_init_weights,
            prompt=args.prompt,
            model_name=args.model_name,
            train=not args.no_train,
            infer=not args.no_infer,
            extract_real_gefm=not args.no_extract_real_gefm,
            render_gefm=not args.no_render_gefm,
            cuda=args.cuda,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote GeoDiff-SAR report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
        print(f"Generated output: {report.get('generated_output_dir')}")
        print(f"Expected ControlNet: {report.get('expected_controlnet_weights')}")
    elif args.command == "traditional-augment":
        report = run_traditional_augmentation_skill(
            dataset_root=args.dataset,
            output_dir=args.output,
            config_path=args.config,
            dataset_config=args.dataset_config,
            filters=parse_cli_filters(args.filter),
            exclude_filters=merge_filter_dicts(
                parse_cli_filters(args.exclude_filter, list_values=True),
                {"polarization": args.exclude_polarization} if args.exclude_polarization else {},
            ),
            target_count=args.target_count,
            multiplier=args.multiplier,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote traditional augmentation report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
        print(f"Input images: {report.get('input_count')}")
        print(f"Planned outputs: {report.get('planned_count')}")
        print(f"Generated outputs: {report.get('generated_count')}")
    elif args.command == "sar-preprocess":
        report = run_sar_preprocess_skill(
            input_dir=args.input,
            output_dir=args.output,
            config_path=args.config,
            mode=args.mode,
            resize=args.resize,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote SAR preprocess report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Planned: {report.get('planned_count')}")
        print(f"Processed: {report.get('processed_count')}")
    elif args.command == "metadata-caption":
        report = run_metadata_caption_skill(
            dataset_root=args.dataset,
            output_dir=args.output,
            dataset_config=args.dataset_config,
            template=args.template,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote metadata caption report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Planned captions: {report.get('planned_count')}")
        print(f"Captioned dataset: {report.get('captioned_dataset_dir')}")
    elif args.command == "dataset-balance":
        report = run_dataset_balancing_skill(
            dataset_root=args.dataset,
            output_dir=args.output,
            dataset_config=args.dataset_config,
            fields=[field.strip() for field in args.fields.split(",") if field.strip()],
            target_per_bin=args.target_per_bin,
            max_multiplier=args.max_multiplier,
            dry_run=True,
        )
        print(f"Wrote dataset balancing report to {Path(args.output).resolve()}")
        print(f"Samples: {report.get('sample_count')}")
        print(f"Recommended additions: {report.get('total_recommended_additions')}")
    elif args.command == "gan-generate":
        report = run_gan_generation_skill(
            dataset_root=args.dataset,
            output_dir=args.output,
            project_dir=args.project_dir,
            variant=args.variant,
            target_count=args.target_count,
            epochs=args.epochs,
            model_path=args.model_path,
            train=not args.no_train,
            infer=not args.no_infer,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote GAN generation report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Commands: {report.get('commands')}")
    elif args.command == "background-generate":
        report = run_background_generation_skill(
            scene_prompt=" ".join(args.scene_prompt).strip(),
            output_dir=args.output,
            config_path=args.config,
            project_dir=args.project_dir,
            num_images=args.num_images,
            split=args.split,
            top_k=args.top_k,
            selection_seed=args.selection_seed,
            seed=args.seed,
            cns=args.cns,
            steps=args.steps,
            scale=args.scale,
            width=args.width,
            height=args.height,
            cuda=args.cuda,
            cpu_threads=args.cpu_threads,
            extra_prompt=args.extra_prompt,
            clear_cache_each_image=args.clear_cache_each_image,
            show_top=args.show_top,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote background generation report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Generated dir: {report.get('generated_output_dir')}")
        print(f"Commands: {report.get('commands')}")
    elif args.command == "compose-target-background":
        report = run_target_background_composition_skill(
            target_dir=args.targets,
            background_dir=args.backgrounds,
            output_dir=args.output,
            config_path=args.config,
            mask_dir=args.masks,
            target_count=args.target_count,
            blend_mode=args.blend_mode,
            mask_mode=args.mask_mode,
            placement_policy=args.placement_policy,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote target/background composition report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
        print(f"Generated: {report.get('generated_count')}/{report.get('target_count')}")
        print(f"Generated dir: {report.get('generated_output_dir')}")
    elif args.command == "pseudocolor":
        report = run_pseudocolor_skill(
            input_dir=args.input,
            output_dir=args.output,
            colormap=args.colormap,
            preserve_tree=not args.flat,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote pseudocolor report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Planned: {report.get('planned_count')}")
        print(f"Materialized: {report.get('materialized_count')}")
    elif args.command == "model-to-pov":
        report = run_model_to_pov_scene_compiler_skill(
            model_file=args.model,
            output_dir=args.output,
            config_path=args.config,
            incidence_angle_deg=args.incidence_angle,
            depression_angle_deg=args.depression_angle,
            azimuth_deg=args.azimuth,
            pitch_deg=args.pitch,
            roll_deg=args.roll,
            target_extent_m=args.target_extent,
            scale_factor=args.scale,
            input_up_axis=args.input_up_axis,
            pov_height_axis=args.pov_height_axis,
            range_distance_m=args.range_distance,
            sensor_plane_m=args.sensor_plane,
            scene_center=parse_cli_vec3(args.scene_center) if args.scene_center else None,
            add_ground=not args.no_ground,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote model-to-POV report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Compiled scene: {report.get('compiled_scene')}")
        print(f"Faces: {report.get('mesh_stats', {}).get('face_count')}")
    elif args.command == "raysar-synthesis":
        if not args.pov_scene and not args.contributions_txt:
            raise SystemExit("raysar-synthesis requires --pov-scene or --contributions-txt")
        report = run_raysar_synthesis_skill(
            pov_scene=args.pov_scene,
            contributions_txt=args.contributions_txt,
            parameters_file=args.parameters_file,
            output_dir=args.output,
            config_path=args.config,
            adapted_povray=args.adapted_povray,
            width=args.width,
            height=args.height,
            render=not args.postprocess_only,
            postprocess=not args.no_postprocess,
            auto_fix_scene=not args.no_auto_fix_scene,
            build_if_missing=not args.no_build,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote RaySAR synthesis report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Contributions: {report.get('contributions_txt')}")
        print(f"Generated maps: {report.get('generated_output_dir')}")
        print(f"Map count: {report.get('map_count')}")
    elif args.command == "raysar-sweep":
        azimuth_values = parse_cli_float_list(args.azimuth_values)
        azimuth_sweep = None
        sweep_parts = [args.azimuth_start, args.azimuth_stop, args.azimuth_step]
        if any(value is not None for value in sweep_parts):
            if not all(value is not None for value in sweep_parts):
                raise SystemExit("--azimuth-start, --azimuth-stop, and --azimuth-step must be provided together")
            azimuth_sweep = {
                "start": args.azimuth_start,
                "stop": args.azimuth_stop,
                "step": args.azimuth_step,
            }
        if not azimuth_values and not azimuth_sweep:
            raise SystemExit("raysar-sweep requires --azimuth-values or --azimuth-start/--azimuth-stop/--azimuth-step")
        report = run_raysar_sweep_synthesis_skill(
            model_file=args.model,
            output_dir=args.output,
            config_path=args.config,
            azimuth_values=azimuth_values,
            azimuth_sweep=azimuth_sweep,
            incidence_angle_deg=args.incidence_angle,
            depression_angle_deg=args.depression_angle,
            target_extent_m=args.target_extent,
            scale_factor=args.scale,
            input_up_axis=args.input_up_axis,
            pov_height_axis=args.pov_height_axis,
            range_distance_m=args.range_distance,
            sensor_plane_m=args.sensor_plane,
            parameters_file=args.parameters_file,
            adapted_povray=args.adapted_povray,
            width=args.width,
            height=args.height,
            postprocess=not args.no_postprocess,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote RaySAR sweep report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Views: {report.get('view_count')}")
        print(f"Generated maps: {report.get('generated_output_dir')}")
        print(f"Map count: {report.get('map_count')} / {report.get('expected_count')}")
    elif args.command == "metadata-slice-eval":
        report = run_metadata_slice_evaluation_skill(
            dataset_config=args.dataset_config,
            output_dir=args.output,
            fields=[field.strip() for field in args.fields.split(",") if field.strip()],
            predictions_csv=args.predictions_csv,
            dry_run=args.dry_run,
        )
        print(f"Wrote metadata slice evaluation to {Path(args.output).resolve()}")
        print(f"Slices: {report.get('slice_count')}")
        print(f"Predictions available: {report.get('predictions_available')}")
    elif args.command == "leakage-check":
        report = run_leakage_check_skill(
            baseline_dataset=args.baseline_dataset,
            augmented_dataset=args.augmented_dataset,
            val_dataset=args.val_dataset,
            output_dir=args.output,
            sample_limit=args.sample_limit,
            dry_run=args.dry_run,
        )
        print(f"Wrote leakage check to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Triggers: {report.get('trigger_count')}")
    elif args.command == "duplicate-check":
        report = run_duplicate_near_duplicate_skill(
            input_dir=args.input,
            output_dir=args.output,
            hash_size=args.hash_size,
            hamming_threshold=args.hamming_threshold,
            sample_limit=args.sample_limit,
            dry_run=args.dry_run,
        )
        print(f"Wrote duplicate check to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Near-duplicate pairs: {report.get('pair_count')}")
    elif args.command == "classification-eval":
        report = run_classification_evaluation_skill(
            baseline_dataset=args.baseline_dataset,
            augmented_dataset=args.augmented_dataset,
            val_dataset=args.val_dataset,
            baseline_dataset_config=args.baseline_dataset_config,
            augmented_dataset_config=args.augmented_dataset_config,
            val_dataset_config=args.val_dataset_config,
            output_dir=args.output,
            config_path=args.config,
            dry_run=(not args.run) or args.dry_run,
        )
        print(f"Wrote classification evaluation report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Dry run: {report.get('dry_run')}")
        print(f"Commands: {len(report.get('commands') or {})}")
    elif args.command == "agent-command":
        report = run_agent_command(
            text=args.text,
            output_dir=args.output,
            config_path=args.config,
            dry_run=args.dry_run,
        )
        print(f"Wrote agent command report to {Path(args.output).resolve()}")
        print(f"Status: {report.get('status')}")
        print(f"Skill: {report.get('interpretation', {}).get('skill')}")
        print(f"Selected content images: {report.get('selection', {}).get('selected_count')}")


def load_optional_cli_mapping(path: str | None) -> dict | None:
    if not path:
        return None
    return load_llm_config(path)


def parse_cli_filters(values: list[str] | None, list_values: bool = False) -> dict:
    filters: dict[str, object] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid filter {raw!r}; expected field=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid filter {raw!r}; empty field")
        parsed: object
        if list_values:
            parsed = [coerce_cli_scalar(item.strip()) for item in value.split(",") if item.strip()]
        else:
            parsed = coerce_cli_scalar(value.strip())
        filters[key] = parsed
    return filters


def merge_filter_dicts(*items: dict) -> dict:
    merged: dict = {}
    for item in items:
        for key, value in (item or {}).items():
            if key in merged and isinstance(merged[key], list):
                existing = list(merged[key])
                if isinstance(value, list):
                    existing.extend(value)
                else:
                    existing.append(value)
                merged[key] = existing
            else:
                merged[key] = value
    return merged


def coerce_cli_scalar(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_cli_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit(f"Invalid vector {value!r}; expected x,y,z")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise SystemExit(f"Invalid vector {value!r}; expected numeric x,y,z") from exc


def parse_cli_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.replace("，", ",").replace("、", ",").split(",") if part.strip()]
    if not parts:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"Invalid float list {value!r}; expected values like 0,10,20") from exc


def merge_cli_evaluations(
    quality: dict | None,
    distribution: dict | None,
) -> dict:
    if quality and distribution:
        merged = dict(quality)
        merged["triggers"] = list(quality.get("triggers") or []) + list(distribution.get("triggers") or [])
        merged["secondary_evaluation"] = distribution
        return merged
    return quality or distribution or {}

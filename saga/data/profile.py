from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.core.profile import (
    DATASET_PROFILE_VERSION,
    LLM_CONTEXT_VERSION,
    RAW_PROFILE_VERSION,
    ClarificationQuestion,
)
from saga.data.discovery import IMAGE_EXTS, is_image, natural_key
from saga.data.path_patterns import build_path_groups, stratified_path_sample


ANNOTATION_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml"}
SIDECAR_EXTS = {".txt", ".json", ".xml", ".csv"}
MASK_DIR_NAMES = {"mask", "masks", "seg", "segmentation", "label_mask", "label_masks"}
LABEL_DIR_NAMES = {"label", "labels", "annotation", "annotations", "ann", "anns"}
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs", "jpegimages"}
NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")
TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?|[A-Za-z]+|[\u4e00-\u9fff]+")


def profile_dataset(
    root: str | Path,
    output_dir: str | Path,
    request: str = "",
    intent_spec: dict[str, Any] | None = None,
    format_hints: list[dict[str, Any]] | None = None,
    sample_limit: int = 120,
    image_probe_limit: int = 200,
    txt_preview_chars: int = 300,
) -> dict[str, Any]:
    """Create the Dataset Profiler MVP artifacts.

    The profiler only observes objective filesystem/image facts. Semantic
    interpretation is passed in through intent_spec / format_hints and remains
    explicitly unvalidated until a DatasetFormatSpec validator confirms it.
    """

    output_path = Path(output_dir).expanduser().resolve()
    raw_profile = build_raw_dataset_profile(
        root=root,
        sample_limit=sample_limit,
        image_probe_limit=image_probe_limit,
        txt_preview_chars=txt_preview_chars,
    )
    llm_context = build_llm_context_profile(raw_profile, request=request, format_hints=format_hints or [])
    dataset_profile = build_progressive_dataset_profile(
        raw_profile=raw_profile,
        intent_spec=intent_spec or {},
        format_hints=format_hints or [],
    )
    questions = build_clarification_questions(
        raw_profile=raw_profile,
        intent_spec=intent_spec or {},
        format_hints=format_hints or [],
    )
    artifact = {
        "schema_version": DATASET_PROFILE_VERSION,
        "root": raw_profile["root"],
        "request": request,
        "raw_profile_path": (output_path / "raw_profile.json").as_posix(),
        "llm_context_path": (output_path / "llm_context.json").as_posix(),
        "dataset_profile": dataset_profile,
        "clarification_questions": [question.to_dict() for question in questions],
        "principles": [
            "Observe before interpret.",
            "External user formats stay open; internal contracts must become fixed.",
            "LLMs propose semantic mappings; deterministic programs validate facts.",
            "Profiling is progressive: ask when uncertain instead of hallucinating.",
        ],
    }

    save_json(output_path / "raw_profile.json", raw_profile)
    save_json(output_path / "llm_context.json", llm_context)
    save_json(output_path / "dataset_profile.json", artifact)
    save_json(output_path / "clarification_questions.json", artifact["clarification_questions"])
    save_text(output_path / "profile_report.md", render_profile_markdown(artifact, raw_profile, llm_context))
    return artifact


def build_raw_dataset_profile(
    root: str | Path,
    sample_limit: int = 120,
    image_probe_limit: int = 200,
    txt_preview_chars: int = 300,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset root must be a directory: {root_path}")

    all_files = sorted((path for path in root_path.rglob("*") if path.is_file()), key=lambda p: natural_key(p.as_posix()))
    all_dirs = sorted((path for path in root_path.rglob("*") if path.is_dir()), key=lambda p: natural_key(p.as_posix()))
    image_paths = [path for path in all_files if is_image(path)]
    sampled_images = stratified_path_sample(image_paths, root=root_path, limit=sample_limit)
    probed_images = stratified_path_sample(image_paths, root=root_path, limit=image_probe_limit)

    suffix_counts = Counter(path.suffix.lower() or "<none>" for path in all_files)
    image_suffix_counts = Counter(path.suffix.lower() for path in image_paths)
    depth_counts = Counter(len(path.relative_to(root_path).parts) for path in image_paths)
    parent_counts = Counter(path.parent.relative_to(root_path).as_posix() for path in image_paths)
    path_groups = build_path_groups(image_paths, root=root_path)
    image_probes = [probe_image(path, root_path) for path in probed_images]
    image_probe_summary = summarize_image_probes(image_probes)
    sidecars = summarize_sidecars(root_path, all_files, image_paths, txt_preview_chars=txt_preview_chars)
    directory_signals = summarize_directory_signals(root_path, all_dirs, image_paths)
    filename_tokens = summarize_filename_tokens(root_path, sampled_images)
    standard_candidates = detect_standard_format_candidates(
        root=root_path,
        all_files=all_files,
        image_paths=image_paths,
        sidecars=sidecars,
        directory_signals=directory_signals,
    )

    samples = []
    for path in sampled_images:
        relpath = path.relative_to(root_path).as_posix()
        samples.append(
            {
                "relative_path": relpath,
                "filename": path.name,
                "stem": path.stem,
                "suffix": path.suffix.lower(),
                "parent": path.parent.relative_to(root_path).as_posix(),
                "depth": len(path.relative_to(root_path).parts),
                "numeric_tokens": NUMBER_RE.findall(path.name),
                "has_txt_sidecar": path.with_suffix(".txt").exists(),
            }
        )

    return {
        "schema_version": RAW_PROFILE_VERSION,
        "root": root_path.as_posix(),
        "filesystem": {
            "total_files": len(all_files),
            "total_dirs": len(all_dirs),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "annotation_suffix_counts": {
                ext: suffix_counts.get(ext, 0) for ext in sorted(ANNOTATION_EXTS) if suffix_counts.get(ext, 0)
            },
        },
        "images": {
            "total_images": len(image_paths),
            "suffix_counts": dict(sorted(image_suffix_counts.items())),
            "depth_counts": dict(sorted(depth_counts.items())),
            "top_parent_dirs": dict(parent_counts.most_common(50)),
            "probe_limit": image_probe_limit,
            "probe_summary": image_probe_summary,
            "probe_samples": image_probes[: min(40, len(image_probes))],
        },
        "sidecars": sidecars,
        "directory_signals": directory_signals,
        "path_groups": path_groups,
        "filename_tokens": filename_tokens,
        "standard_format_candidates": standard_candidates,
        "samples": samples,
        "notes": [
            "RawDatasetProfile contains objective scan results only.",
            "It does not claim semantic meanings for filename tokens unless another validated spec does so.",
            "Use llm_context.json plus user text for schema induction, then validate the resulting DatasetFormatSpec.",
        ],
    }


def build_llm_context_profile(
    raw_profile: dict[str, Any],
    request: str = "",
    format_hints: list[dict[str, Any]] | None = None,
    group_limit: int = 30,
    sample_limit: int = 40,
) -> dict[str, Any]:
    images = raw_profile.get("images", {})
    sidecars = raw_profile.get("sidecars", {})
    return {
        "schema_version": LLM_CONTEXT_VERSION,
        "root": raw_profile.get("root"),
        "user_request": request,
        "format_hints_from_request": format_hints or [],
        "objective_summary": {
            "total_files": raw_profile.get("filesystem", {}).get("total_files"),
            "total_images": images.get("total_images"),
            "image_suffix_counts": images.get("suffix_counts"),
            "image_mode_counts": images.get("probe_summary", {}).get("mode_counts"),
            "image_size_counts": images.get("probe_summary", {}).get("size_counts"),
            "same_stem_sidecar_counts": sidecars.get("same_stem_sidecar_counts"),
            "standard_format_candidates": raw_profile.get("standard_format_candidates", [])[:8],
        },
        "directory_signals": raw_profile.get("directory_signals", {}),
        "path_groups": compact_path_groups(raw_profile.get("path_groups", []), group_limit),
        "filename_token_observations": raw_profile.get("filename_tokens", {}),
        "sample_relative_paths": [
            sample.get("relative_path") for sample in raw_profile.get("samples", [])[:sample_limit]
        ],
        "rules_for_llm": [
            "Do not count files; counts are supplied by the profiler.",
            "Do not invent semantic field meanings that are not visible or stated by the user.",
            "Return candidate DatasetFormatSpec or FormatHint only; deterministic validation decides whether it is trusted.",
            "When the mapping is ambiguous, emit unresolved items or clarification questions.",
        ],
    }


def build_progressive_dataset_profile(
    raw_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    format_hints: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = raw_profile.get("standard_format_candidates", [])
    validated_format = None
    status = "raw_profile_only"
    level = 0
    if candidates:
        status = "format_candidates_available"
        level = 1
    if format_hints:
        status = "semantic_hints_unvalidated"
        level = 1
    task_candidates = rank_task_candidates(raw_profile, intent_spec)
    return {
        "status": status,
        "profile_level": level,
        "level_definition": {
            "0": "Raw filesystem and image facts only.",
            "1": "Format candidates or user semantic hints exist, but no validated DatasetFormatSpec yet.",
            "2": "A DatasetFormatSpec has been validated and semantic fields are trusted.",
            "3": "Task-ready profile with validated fields and downstream diagnosis.",
        },
        "task_candidates": task_candidates,
        "validated_format_spec": validated_format,
        "known": {
            "num_images": raw_profile.get("images", {}).get("total_images", 0),
            "image_suffix_counts": raw_profile.get("images", {}).get("suffix_counts", {}),
            "image_modes": raw_profile.get("images", {}).get("probe_summary", {}).get("mode_counts", {}),
            "image_sizes": raw_profile.get("images", {}).get("probe_summary", {}).get("size_counts", {}),
            "standard_format_candidates": candidates,
            "format_hints": format_hints,
        },
        "unknown": infer_unknowns(raw_profile, intent_spec, format_hints),
        "next_step": choose_next_step(raw_profile, intent_spec, format_hints),
    }


def build_clarification_questions(
    raw_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    format_hints: list[dict[str, Any]],
) -> list[ClarificationQuestion]:
    questions: list[ClarificationQuestion] = []
    total_images = raw_profile.get("images", {}).get("total_images", 0)
    if total_images == 0:
        questions.append(
            ClarificationQuestion(
                id="no_images_found",
                severity="critical",
                question="这个路径下没有发现支持的图像文件。用户数据是否放在子目录之外，或是否需要加入新的图像后缀？",
                reason="Dataset Profiler cannot proceed without image files.",
            )
        )
        return questions

    intent = intent_spec.get("intent", {})
    if intent.get("task") == "style_transfer":
        if not intent.get("content_source"):
            questions.append(
                ClarificationQuestion(
                    id="missing_content_source",
                    severity="critical",
                    question="这条风格迁移命令里，哪些图像应作为内容图被迁移？",
                    reason="Style transfer requires a content source.",
                )
            )
        if not intent.get("style_source"):
            questions.append(
                ClarificationQuestion(
                    id="missing_style_source",
                    severity="critical",
                    question="这条风格迁移命令里，哪些图像应作为风格图或指导图？",
                    reason="Style transfer requires a style source.",
                )
            )

    filter_fields = sorted((intent.get("filters") or {}).keys())
    hinted_fields = collect_hint_fields(format_hints)
    for field in filter_fields:
        if field not in hinted_fields and not field_appears_in_format_candidates(raw_profile, field):
            questions.append(
                ClarificationQuestion(
                    id=f"filter_field_mapping_{field}",
                    severity="warning",
                    question=f"用户要求按 `{field}` 筛选，但当前还没有经过验证的字段解析规则。这个字段在文件名、目录名或标签文件中的位置是什么？",
                    reason="A filter can only be executed reproducibly after its field mapping is known and validated.",
                    scope={"field": field},
                    examples=sample_paths(raw_profile, 5),
                )
            )

    numeric_summary = raw_profile.get("filename_tokens", {}).get("numeric_token_sequences", {})
    has_numeric_tokens = bool(numeric_summary.get("examples"))
    if has_numeric_tokens and not hinted_fields and not has_sar_named_metadata_candidate(raw_profile):
        questions.append(
            ClarificationQuestion(
                id="numeric_token_semantics_unknown",
                severity="info",
                question="文件名中存在规律性的数字 token。它们是否表示方位角、下视角/入射角、分辨率、波段、序号或其他 SAR 参数？",
                reason="The profiler can observe numeric tokens but cannot safely assign semantic meanings without hints or validation.",
                examples=[item.get("relative_path", "") for item in numeric_summary.get("examples", [])[:5]],
            )
        )

    candidates = raw_profile.get("standard_format_candidates", [])
    if not candidates:
        questions.append(
            ClarificationQuestion(
                id="label_source_unknown",
                severity="info",
                question="这些图像的类别或标签主要来自目录名、文件名、txt/json/xml/csv 标注，还是用户后续会提供单独说明？",
                reason="No standard annotation format candidate was detected with high confidence.",
                examples=sample_paths(raw_profile, 5),
            )
        )
    return questions


def has_sar_named_metadata_candidate(raw_profile: dict[str, Any]) -> bool:
    return any(
        candidate.get("format") == "sar_named_metadata_filename"
        for candidate in raw_profile.get("standard_format_candidates", [])
    )


def probe_image(path: Path, root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "suffix": path.suffix.lower(),
    }
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            item["mode"] = image.mode
            item["width"], item["height"] = image.size
            item["bands"] = list(image.getbands())
            item["bit_depth"] = infer_bit_depth(image.mode)
            item["extrema"] = normalize_extrema(image.getextrema())
            item["mean"] = [round(float(value), 4) for value in ImageStat.Stat(image).mean]
    except Exception as exc:  # pragma: no cover - defensive for uncommon image codecs
        item["probe_error"] = f"{type(exc).__name__}: {exc}"
    return item


def summarize_image_probes(probes: list[dict[str, Any]]) -> dict[str, Any]:
    mode_counts = Counter(str(item.get("mode", "unreadable")) for item in probes)
    size_counts = Counter(
        f"{item.get('width')}x{item.get('height')}"
        for item in probes
        if item.get("width") is not None and item.get("height") is not None
    )
    bit_depth_counts = Counter(str(item.get("bit_depth", "unknown")) for item in probes)
    errors = [item for item in probes if item.get("probe_error")]
    return {
        "probed_images": len(probes),
        "readable_images": len(probes) - len(errors),
        "mode_counts": dict(mode_counts.most_common()),
        "size_counts": dict(size_counts.most_common(20)),
        "bit_depth_counts": dict(bit_depth_counts.most_common()),
        "probe_errors": errors[:20],
    }


def summarize_sidecars(
    root: Path,
    all_files: list[Path],
    image_paths: list[Path],
    txt_preview_chars: int = 300,
) -> dict[str, Any]:
    sidecar_counts = Counter(path.suffix.lower() for path in all_files if path.suffix.lower() in SIDECAR_EXTS)
    same_stem_sidecar_counts: dict[str, int] = {}
    for ext in sorted(SIDECAR_EXTS):
        same_stem_sidecar_counts[ext] = sum(1 for path in image_paths if path.with_suffix(ext).exists())

    txt_previews = []
    for path in all_files:
        if path.suffix.lower() != ".txt":
            continue
        try:
            preview = path.read_text(encoding="utf-8", errors="replace")[:txt_preview_chars]
        except OSError:
            preview = ""
        txt_previews.append({"relative_path": path.relative_to(root).as_posix(), "preview": preview})
        if len(txt_previews) >= 20:
            break

    return {
        "suffix_counts": dict(sorted(sidecar_counts.items())),
        "same_stem_sidecar_counts": same_stem_sidecar_counts,
        "txt_previews": txt_previews,
    }


def summarize_directory_signals(root: Path, all_dirs: list[Path], image_paths: list[Path]) -> dict[str, Any]:
    keyword_dirs: dict[str, list[str]] = defaultdict(list)
    for path in all_dirs:
        name = path.name.lower()
        rel = path.relative_to(root).as_posix()
        if name in MASK_DIR_NAMES:
            keyword_dirs["mask_dirs"].append(rel)
        if name in LABEL_DIR_NAMES:
            keyword_dirs["label_dirs"].append(rel)
        if name in IMAGE_DIR_NAMES:
            keyword_dirs["image_dirs"].append(rel)

    leaf_image_counts = Counter(path.parent.relative_to(root).as_posix() for path in image_paths)
    first_level_counts = Counter(
        path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "."
        for path in image_paths
    )
    return {
        "keyword_dirs": {key: values[:50] for key, values in keyword_dirs.items()},
        "leaf_image_dir_count": len(leaf_image_counts),
        "top_leaf_image_dirs": dict(leaf_image_counts.most_common(50)),
        "top_first_level_dirs": dict(first_level_counts.most_common(50)),
    }


def summarize_filename_tokens(root: Path, sampled_images: list[Path]) -> dict[str, Any]:
    token_length_counts: Counter[str] = Counter()
    numeric_length_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for path in sampled_images:
        relpath = path.relative_to(root).as_posix()
        tokens = TOKEN_RE.findall(path.stem)
        numeric_tokens = NUMBER_RE.findall(path.stem)
        token_length_counts[str(len(tokens))] += 1
        numeric_length_counts[str(len(numeric_tokens))] += 1
        if numeric_tokens and len(examples) < 40:
            examples.append(
                {
                    "relative_path": relpath,
                    "stem": path.stem,
                    "tokens": tokens[:30],
                    "numeric_tokens": numeric_tokens,
                }
            )
    return {
        "token_length_counts": dict(token_length_counts.most_common()),
        "numeric_token_sequences": {
            "numeric_token_count_distribution": dict(numeric_length_counts.most_common()),
            "examples": examples,
        },
    }


def detect_standard_format_candidates(
    root: Path,
    all_files: list[Path],
    image_paths: list[Path],
    sidecars: dict[str, Any],
    directory_signals: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    total_images = max(1, len(image_paths))
    txt_sidecars = sidecars.get("same_stem_sidecar_counts", {}).get(".txt", 0)
    json_sidecars = sidecars.get("same_stem_sidecar_counts", {}).get(".json", 0)

    numeric_txt_pairs = sum(
        1 for path in image_paths if path.stem.isdigit() and path.with_suffix(".txt").exists()
    )
    if numeric_txt_pairs:
        candidates.append(
            {
                "format": "saga_pair_or_numeric_txt_sidecar",
                "task_candidates": ["classification", "generation_dataset"],
                "confidence": min(0.95, numeric_txt_pairs / total_images),
                "evidence": {
                    "numeric_image_txt_pairs": numeric_txt_pairs,
                    "total_images": len(image_paths),
                },
            }
        )

    if txt_sidecars:
        yolo_like = count_yolo_like_txt_files(all_files)
        candidates.append(
            {
                "format": "txt_sidecar",
                "task_candidates": ["classification", "object_detection", "generation_dataset"],
                "confidence": min(0.9, txt_sidecars / total_images),
                "evidence": {
                    "same_stem_txt_sidecars": txt_sidecars,
                    "yolo_like_txt_files_sampled": yolo_like,
                },
            }
        )
        if yolo_like:
            candidates.append(
                {
                    "format": "yolo_detection",
                    "task_candidates": ["object_detection"],
                    "confidence": min(0.85, 0.35 + yolo_like / max(1, min(50, txt_sidecars))),
                    "evidence": {"sampled_yolo_like_txt_files": yolo_like},
                }
            )

    if json_sidecars:
        candidates.append(
            {
                "format": "json_sidecar",
                "task_candidates": ["classification", "object_detection", "segmentation"],
                "confidence": min(0.8, json_sidecars / total_images),
                "evidence": {"same_stem_json_sidecars": json_sidecars},
            }
        )

    coco = probe_coco_json(all_files)
    if coco:
        candidates.append(coco)

    voc_xml = count_voc_like_xml_files(all_files)
    if voc_xml:
        candidates.append(
            {
                "format": "voc_xml_detection",
                "task_candidates": ["object_detection"],
                "confidence": min(0.85, 0.35 + voc_xml / 50),
                "evidence": {"sampled_voc_like_xml_files": voc_xml},
            }
        )

    keyword_dirs = directory_signals.get("keyword_dirs", {})
    if keyword_dirs.get("mask_dirs"):
        candidates.append(
            {
                "format": "mask_folder_segmentation",
                "task_candidates": ["semantic_segmentation", "instance_segmentation"],
                "confidence": 0.65,
                "evidence": {"mask_dirs": keyword_dirs.get("mask_dirs", [])[:10]},
            }
        )

    named_metadata = count_named_metadata_paths(root, image_paths)
    if named_metadata["support"]:
        candidates.append(
            {
                "format": "sar_named_metadata_filename",
                "task_candidates": ["classification", "ATR"],
                "confidence": min(0.9, 0.45 + named_metadata["support"] / max(1, len(image_paths))),
                "evidence": named_metadata,
            }
        )

    first_level = directory_signals.get("top_first_level_dirs", {})
    non_root_first_level = {key: count for key, count in first_level.items() if key != "."}
    if len(non_root_first_level) >= 2 and sum(non_root_first_level.values()) / total_images > 0.8:
        candidates.append(
            {
                "format": "image_folder_classification",
                "task_candidates": ["classification", "ATR"],
                "confidence": min(0.9, sum(non_root_first_level.values()) / total_images),
                "evidence": {
                    "candidate_class_dirs": dict(list(non_root_first_level.items())[:20]),
                },
            }
        )

    return sorted(candidates, key=lambda item: (-float(item.get("confidence", 0)), item.get("format", "")))


def count_named_metadata_paths(root: Path, image_paths: list[Path]) -> dict[str, Any]:
    patterns = {
        "azimuth_deg": re.compile(r"(?i)(?:azim|azimuth|az)[-_]?-?\d+(?:\.\d+)?"),
        "incidence_angle_deg": re.compile(r"(?i)(?:inci|incidence|inc)[-_]?-?\d+(?:\.\d+)?"),
        "elevation_angle_deg": re.compile(r"(?i)(?:elev|elevation|el)[-_]?-?\d+(?:\.\d+)?"),
        "resolution_m": re.compile(r"(?i)(?:res|resolution)[-_]?-?\d+(?:\.\d+)?"),
    }
    field_counts: Counter[str] = Counter()
    examples: list[str] = []
    for path in image_paths:
        relpath = path.relative_to(root).as_posix()
        matched = False
        for field, pattern in patterns.items():
            if pattern.search(relpath):
                field_counts[field] += 1
                matched = True
        if matched and len(examples) < 10:
            examples.append(relpath)
    support = max(field_counts.values()) if field_counts else 0
    return {
        "support": support,
        "field_counts": dict(field_counts.most_common()),
        "examples": examples,
    }


def count_yolo_like_txt_files(all_files: list[Path], limit: int = 50) -> int:
    count = 0
    checked = 0
    for path in all_files:
        if path.suffix.lower() != ".txt":
            continue
        checked += 1
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError:
            lines = []
        if lines and all(re.match(r"^\d+(?:\s+-?\d+(?:\.\d+)?){4,}", line) for line in lines[:10]):
            count += 1
        if checked >= limit:
            break
    return count


def probe_coco_json(all_files: list[Path], limit: int = 20) -> dict[str, Any] | None:
    checked = 0
    for path in all_files:
        if path.suffix.lower() != ".json":
            continue
        checked += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            data = None
        if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
            return {
                "format": "coco_detection_or_segmentation",
                "task_candidates": ["object_detection", "instance_segmentation"],
                "confidence": 0.9,
                "evidence": {"json_file": path.as_posix()},
            }
        if checked >= limit:
            break
    return None


def count_voc_like_xml_files(all_files: list[Path], limit: int = 50) -> int:
    count = 0
    checked = 0
    for path in all_files:
        if path.suffix.lower() != ".xml":
            continue
        checked += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:4000].lower()
        except OSError:
            text = ""
        if "<annotation" in text and "<object" in text and "<bndbox" in text:
            count += 1
        if checked >= limit:
            break
    return count


def infer_unknowns(
    raw_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    format_hints: list[dict[str, Any]],
) -> list[str]:
    unknowns: list[str] = []
    if (
        not format_hints
        and raw_profile.get("filename_tokens", {}).get("numeric_token_sequences", {}).get("examples")
        and not has_sar_named_metadata_candidate(raw_profile)
    ):
        unknowns.append("semantic_meaning_of_filename_numeric_tokens")
    if not raw_profile.get("standard_format_candidates"):
        unknowns.append("label_source")
    intent = intent_spec.get("intent", {})
    for field in sorted((intent.get("filters") or {}).keys()):
        if field not in collect_hint_fields(format_hints):
            unknowns.append(f"validated_mapping_for_filter_{field}")
    return unknowns


def choose_next_step(
    raw_profile: dict[str, Any],
    intent_spec: dict[str, Any],
    format_hints: list[dict[str, Any]],
) -> str:
    if raw_profile.get("images", {}).get("total_images", 0) == 0:
        return "Ask user to provide a directory containing supported image files or extend IMAGE_EXTS."
    if format_hints:
        return "Compile candidate DatasetFormatSpec from hints and raw profile, then run deterministic validation."
    if raw_profile.get("standard_format_candidates"):
        return "Validate the highest-confidence standard format candidate or ask the user to confirm the label source."
    if raw_profile.get("filename_tokens", {}).get("numeric_token_sequences", {}).get("examples"):
        return "Ask for the semantic meaning of filename tokens, then create a DatasetFormatSpec."
    return "Ask the user for label source and task intent."


def rank_task_candidates(raw_profile: dict[str, Any], intent_spec: dict[str, Any]) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    intent_task = intent_spec.get("intent", {}).get("task")
    if intent_task and intent_task != "unknown":
        scores[intent_task] = 1.0
    for candidate in raw_profile.get("standard_format_candidates", []):
        confidence = float(candidate.get("confidence", 0.0))
        for task in candidate.get("task_candidates", []):
            scores[task] = max(scores.get(task, 0.0), confidence)
    if not scores:
        scores["unknown"] = 0.1
    return [
        {"task": task, "confidence": round(score, 3)}
        for task, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def collect_hint_fields(format_hints: list[dict[str, Any]]) -> set[str]:
    fields: set[str] = set()
    for hint in format_hints:
        rule = hint.get("rule") or {}
        for item in rule.get("fields_after_anchor", []):
            if isinstance(item, dict) and item.get("normalized"):
                fields.add(item["normalized"])
            elif isinstance(item, str):
                fields.add(item)
    return fields


def field_appears_in_format_candidates(raw_profile: dict[str, Any], field: str) -> bool:
    field_lower = field.lower()
    for candidate in raw_profile.get("standard_format_candidates", []):
        text = json.dumps(candidate, ensure_ascii=False).lower()
        if field_lower in text:
            return True
    return False


def sample_paths(raw_profile: dict[str, Any], limit: int) -> list[str]:
    return [sample.get("relative_path", "") for sample in raw_profile.get("samples", [])[:limit]]


def compact_path_groups(groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    compacted = []
    for group in groups[:limit]:
        compacted.append(
            {
                "signature": group.get("signature"),
                "count": group.get("count"),
                "examples": group.get("examples", [])[:4],
                "parent_examples": group.get("parent_examples", [])[:4],
                "suffix_counts": group.get("suffix_counts"),
                "depth_counts": group.get("depth_counts"),
            }
        )
    return compacted


def infer_bit_depth(mode: str) -> int | str:
    if mode == "1":
        return 1
    if mode in {"L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"}:
        return 8
    if mode.startswith("I;16"):
        return 16
    if mode == "I":
        return 32
    if mode == "F":
        return 32
    return "unknown"


def normalize_extrema(value: Any) -> Any:
    if isinstance(value, tuple):
        return [normalize_extrema(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    return value


def render_profile_markdown(
    artifact: dict[str, Any],
    raw_profile: dict[str, Any],
    llm_context: dict[str, Any],
) -> str:
    dataset_profile = artifact["dataset_profile"]
    lines = [
        "# SAGA Dataset Profile",
        "",
        "## Status",
        "",
        f"- Root: `{artifact['root']}`",
        f"- Profile level: {dataset_profile['profile_level']} ({dataset_profile['status']})",
        f"- Next step: {dataset_profile['next_step']}",
        "",
        "## Objective Scan",
        "",
        f"- Total files: {raw_profile.get('filesystem', {}).get('total_files')}",
        f"- Total images: {raw_profile.get('images', {}).get('total_images')}",
        f"- Image suffixes: `{raw_profile.get('images', {}).get('suffix_counts')}`",
        f"- Image modes: `{raw_profile.get('images', {}).get('probe_summary', {}).get('mode_counts')}`",
        f"- Image sizes: `{raw_profile.get('images', {}).get('probe_summary', {}).get('size_counts')}`",
        "",
        "## Standard Format Candidates",
        "",
    ]
    candidates = raw_profile.get("standard_format_candidates", [])
    if candidates:
        for candidate in candidates[:10]:
            lines.append(
                f"- `{candidate.get('format')}` confidence={candidate.get('confidence')} tasks={candidate.get('task_candidates')}"
            )
    else:
        lines.append("- No high-confidence standard format candidate was detected.")

    lines.extend(["", "## Path Pattern Groups", ""])
    for group in raw_profile.get("path_groups", [])[:20]:
        examples = ", ".join(f"`{item}`" for item in group.get("examples", [])[:3])
        lines.append(f"- {group.get('count')} x `{group.get('signature')}`: {examples}")

    lines.extend(["", "## Format Hints From Request", ""])
    hints = dataset_profile.get("known", {}).get("format_hints", [])
    if hints:
        for hint in hints:
            lines.append(f"- `{hint.get('type')}`: {hint.get('rule')}")
    else:
        lines.append("- No semantic format hint was extracted from the request.")

    lines.extend(["", "## Clarification Questions", ""])
    questions = artifact.get("clarification_questions", [])
    if questions:
        for question in questions:
            lines.append(f"- [{question.get('severity')}] {question.get('question')}")
    else:
        lines.append("- No immediate clarification is required for the current profiling level.")

    lines.extend(
        [
            "",
            "## LLM Boundary",
            "",
            "- LLM may read the user request and `llm_context.json`.",
            "- LLM should not read the whole dataset or decide validation by itself.",
            "- Any semantic DatasetFormatSpec proposed by an LLM must pass deterministic validation.",
            "",
            "## LLM Context Preview",
            "",
            f"- Sample paths sent to LLM: {len(llm_context.get('sample_relative_paths', []))}",
            f"- Path groups sent to LLM: {len(llm_context.get('path_groups', []))}",
            "",
        ]
    )
    return "\n".join(lines)

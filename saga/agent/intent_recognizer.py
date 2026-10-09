from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from saga.core.config import save_json, save_text
from saga.core.profile import FORMAT_HINT_VERSION, INTENT_SPEC_VERSION, normalize_field_name


FIELD_SPLIT_RE = re.compile(r"\s*(?:、|，|,|/|和|以及|;|；)\s*")
RAYSAR_MODEL_EXTS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".off", ".dae", ".3ds", ".xyz"}


def recognize_request(
    text: str,
    output_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    intent_spec = build_intent_spec(text=text, dataset_root=dataset_root)
    format_hints = extract_format_hints(text)
    result = {
        "intent_spec": intent_spec,
        "format_hints": format_hints,
        "recognizer": {
            "mode": "rule_based_mvp",
            "llm_ready": True,
            "note": (
                "This module parses obvious command structure and format hints. "
                "A later LLM recognizer can replace or augment it, but execution "
                "must still wait for deterministic validation."
            ),
        },
    }
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        save_json(output_path / "intent_spec.json", intent_spec)
        save_json(output_path / "format_hints.json", format_hints)
        save_text(output_path / "intent_report.md", render_intent_markdown(result))
    return result


def build_intent_spec(text: str, dataset_root: str | Path | None = None) -> dict[str, Any]:
    task = infer_task(text)
    path_candidates = parse_path_candidates(text)
    filters = extract_filters(text)
    exclude_filters = extract_exclude_filters(text)
    style_source = parse_style_source(text)
    content_source = parse_content_source(text)
    baseline_dataset, augmented_dataset = parse_classification_evaluation_sources(
        task=task,
        path_candidates=path_candidates,
    )
    dataset_source = (
        baseline_dataset
        or parse_dataset_source(text)
        or content_source
        or (Path(dataset_root).as_posix() if dataset_root else None)
    )
    model_family = parse_model_family(text)
    target_count = parse_target_count(text)
    training_epochs = parse_training_epochs(text)
    pov_scene = parse_raysar_scene_source(text, path_candidates)
    model_file = parse_raysar_model_file(text, path_candidates)
    if model_file and dataset_source == model_file:
        dataset_source = first_non_model_path(path_candidates)
    parameters_file = parse_raysar_parameters_file(text, path_candidates)
    contributions_txt = parse_raysar_contributions_file(text, path_candidates)
    width, height = parse_render_size(text)
    raysar_geometry = parse_raysar_geometry(text, filters)
    scene_prompt = parse_scene_prompt(text, task)
    target_source, background_source, mask_source = parse_composition_sources(text, path_candidates, dataset_source)
    blend_mode = parse_blend_mode(text)
    goals = infer_goals(text, task)
    unresolved = []
    if task == "style_transfer" and not style_source:
        unresolved.append("style_source")
    if task == "style_transfer" and not content_source:
        unresolved.append("content_source")
    if task == "diffusion_lora_generation" and not dataset_source:
        unresolved.append("dataset_root")
    if task == "geodiff_sar_generation" and not dataset_source:
        unresolved.append("dataset_root")
    if task == "geodiff_sar_generation" and not model_file:
        unresolved.append("model_file")
    if task == "gan_generation" and not dataset_source:
        unresolved.append("dataset_root")
    if task == "traditional_augmentation" and not dataset_source:
        unresolved.append("dataset_root")
    if task == "pseudocolor_transform" and not dataset_source:
        unresolved.append("dataset_root")
    if task == "target_background_composition" and not target_source:
        unresolved.append("target_source")
    if task == "target_background_composition" and not background_source:
        unresolved.append("background_source")
    if task == "classification_evaluation" and not dataset_source:
        unresolved.append("baseline_dataset")
    if task == "raysar_synthesis" and not (pov_scene or model_file or contributions_txt):
        unresolved.append("pov_scene_or_model_file_or_contributions_txt")
    return {
        "schema_version": INTENT_SPEC_VERSION,
        "source": "rule_based_mvp",
        "raw_text": text,
        "dataset_root": Path(dataset_root).as_posix() if dataset_root else None,
        "intent": {
            "task": task,
            "goals": goals,
            "style_source": style_source,
            "content_source": content_source,
            "dataset_source": dataset_source,
            "baseline_dataset": baseline_dataset or dataset_source,
            "augmented_dataset": augmented_dataset,
            "path_candidates": path_candidates,
            "model_family": model_family,
            "target_count": target_count,
            "training_epochs": training_epochs,
            "pov_scene": pov_scene,
            "model_file": model_file,
            "parameters_file": parameters_file,
            "contributions_txt": contributions_txt,
            "width": width,
            "height": height,
            "scene_prompt": scene_prompt,
            "target_source": target_source,
            "background_source": background_source,
            "mask_source": mask_source,
            "blend_mode": blend_mode,
            "raysar_geometry": raysar_geometry,
            "filters": filters,
            "exclude_filters": exclude_filters,
        },
        "confidence": estimate_intent_confidence(task, style_source, content_source, filters, dataset_source),
        "unresolved": unresolved,
        "notes": [
            "Natural-language intent is allowed to be flexible.",
            "This IntentSpec is not an execution plan; planner/rules still need to verify paths, fields, and skill constraints.",
        ],
    }


def infer_task(text: str) -> str:
    lowered = text.lower()
    eval_keywords = ["评估", "验证", "比较", "对比", "下游", "收益", "准确率", "效果", "benchmark", "evaluator", "accuracy"]
    strong_eval_keywords = ["验证", "比较", "对比", "收益", "准确率", "benchmark", "accuracy"]
    class_keywords = ["分类", "识别", "atr", "ATR", "classification"]
    augmentation_keywords = ["增广", "增强", "扩增", "生成", "增广计划", "augmentation", "augment"]
    has_eval = any(keyword in text or keyword in lowered for keyword in eval_keywords)
    has_strong_eval = any(keyword in text or keyword in lowered for keyword in strong_eval_keywords)
    has_class = any(keyword in text or keyword in lowered for keyword in class_keywords)
    has_augmentation = any(keyword in text or keyword in lowered for keyword in augmentation_keywords)
    explicit_raysar_requested = any(keyword in lowered for keyword in ["raysar", "pov-ray", "povray"]) or any(
        keyword in text for keyword in ["物理仿真", "射线追踪", "散射贡献", "Contributions.txt", "三维模型仿真", "3D模型仿真", "3D模型成像", "模型成像"]
    )
    ray_or_physics_requested = explicit_raysar_requested or any(keyword in lowered for keyword in [".pov", ".obj", ".stl", ".ply", ".glb"]) or any(
        keyword in text for keyword in ["物理仿真", "射线追踪", "散射贡献", "Contributions.txt", "三维模型仿真", "3D模型仿真", "3D模型成像", "模型成像"]
    )
    explicit_geodiff_requested = any(keyword in lowered for keyword in ["geo-diff", "geodiff-sar", "controlnet"]) or any(
        keyword in text
        for keyword in [
            "几何扩散",
            "物理先验扩散",
            "扩散物理先验",
            "条件图",
            "GEFM",
            "gefm",
            "稀疏方位角补全",
            "目标稀疏方位角补全",
        ]
    )
    geodiff_requested = explicit_geodiff_requested or ("geodiff" in lowered and "geodiff_components" not in lowered)
    gaussian_splatting_requested = any(keyword in lowered for keyword in ["gaussian splatting", "splatting", "sar gs"]) or any(
        keyword in text for keyword in ["高斯泼溅", "高斯溅射", "3DGS", "SAR GS", "低显存补全", "轻量补全"]
    )
    if explicit_raysar_requested and not explicit_geodiff_requested and not excludes_physics_simulation(text):
        return "raysar_synthesis"
    if geodiff_requested and not excludes_physics_simulation(text):
        return "geodiff_sar_generation"
    if gaussian_splatting_requested:
        return "gaussian_splatting_completion"
    if ray_or_physics_requested and not excludes_physics_simulation(text):
        return "raysar_synthesis"
    if has_eval and has_class and (has_strong_eval or not has_augmentation) and not has_augmentation:
        return "classification_evaluation"
    if any(keyword in lowered for keyword in ["lora", "lo-ra"]) or any(
        keyword in text for keyword in ["训练LoRA", "训练lora", "lora训练", "LoRA训练", "扩散模型增广"]
    ):
        return "diffusion_lora_generation"
    if any(keyword in text for keyword in ["训练一个Flux", "训练一个SD3", "训练一个SDXL"]):
        return "diffusion_lora_generation"
    if any(keyword in lowered for keyword in ["gan", "dcgan"]) or any(keyword in text for keyword in ["图生图", "简单目标生成"]):
        return "gan_generation"
    if any(keyword in text for keyword in ["风格迁移", "特征迁移", "作为风格图", "作为指导图"]) or "style transfer" in lowered:
        return "style_transfer"
    if any(keyword in text for keyword in ["伪彩", "伪彩色", "可视化", "人工检查", "weicaise"]) or "pseudocolor" in lowered:
        return "pseudocolor_transform"
    if any(keyword in text for keyword in ["图像融合", "融合", "合成到", "目标背景合成", "目标和背景合成", "目标-背景合成", "目标和场景合成", "目标-场景合成", "场景合成"]) or any(
        keyword in lowered for keyword in ["image fusion", "image blending", "composition", "composite target", "target background"]
    ):
        return "target_background_composition"
    if any(keyword in text for keyword in ["背景生成", "生成背景", "背景图生成", "生成场景背景", "SAR背景", "sar背景"]) or any(
        keyword in lowered for keyword in ["background generation", "generate background", "scene background"]
    ):
        return "background_generation"
    if (
        any(keyword in text for keyword in ["传统增广", "传统增强", "基础增广", "基础增强", "快速增广", "快速增强"])
        or ("传统" in text and has_augmentation)
        or ("基础" in text and has_augmentation)
    ) or any(keyword in lowered for keyword in ["traditional augmentation", "basic augmentation", "fast augmentation"]) or re.search(
        r"\b(?:traditional|basic|fast)\s+(?:sar\s+(?:data\s+)?)?augmentation\b",
        lowered,
    ):
        return "traditional_augmentation"
    if any(keyword in text for keyword in ["检测", "目标检测", "框"]) or "detection" in lowered:
        return "object_detection"
    if any(keyword in text for keyword in ["分割", "语义分割", "实例分割"]) or "segmentation" in lowered:
        return "segmentation"
    if any(keyword in text for keyword in ["分类", "识别", "atr", "ATR"]):
        return "classification"
    if any(keyword in text for keyword in ["生成", "补全", "增广", "增强"]):
        return "augmentation_or_generation"
    return "unknown"


def excludes_physics_simulation(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in text
        for keyword in [
            "不需要物理仿真",
            "不要物理仿真",
            "不做物理仿真",
            "不用物理仿真",
            "不需要RaySAR",
            "不要RaySAR",
            "不做RaySAR",
            "不用RaySAR",
        ]
    ) or any(keyword in lowered for keyword in ["no raysar", "without raysar", "no physics simulation", "without physics simulation"])


def infer_goals(text: str, task: str) -> list[str]:
    goals = []
    def add_goal(goal: str) -> None:
        if goal not in goals:
            goals.append(goal)

    if task == "classification_evaluation":
        add_goal("downstream_benefit_validation")
    if requests_downstream_classification_evaluation(text):
        add_goal("downstream_benefit_validation")
    if requests_lightweight_probe(text) or "downstream_benefit_validation" not in goals:
        add_goal("lightweight_benefit_probe")
    if task == "traditional_augmentation":
        add_goal("fast_conservative_augmentation")
    if task == "style_transfer":
        add_goal("cross_domain_or_feature_transfer")
    if task == "diffusion_lora_generation":
        add_goal("train_text_caption_lora")
        add_goal("generate_augmented_samples")
    if task == "geodiff_sar_generation":
        add_goal("complete_sparse_azimuth")
        add_goal("physical_prior_generation")
        add_goal("generate_augmented_samples")
    if task == "gaussian_splatting_completion":
        add_goal("complete_sparse_azimuth")
        add_goal("lightweight_sparse_view_completion")
        add_goal("generate_augmented_samples")
    if task == "gan_generation":
        add_goal("fast_gan_image_to_image_generation")
        add_goal("generate_augmented_samples")
    if task == "raysar_synthesis":
        add_goal("physics_interpretable_simulation")
        add_goal("generate_augmented_samples")
    if task == "pseudocolor_transform":
        add_goal("pseudocolor_visualization")
    if task == "background_generation":
        add_goal("generate_backgrounds")
    if task == "target_background_composition":
        add_goal("target_background_composition")
        add_goal("image_fusion")
    if any(keyword in text for keyword in ["补类别", "类别不平衡", "少数类"]):
        add_goal("rebalance_classes")
    if any(keyword in text for keyword in ["补姿态", "方位角补全", "稀疏方位", "补方位"]):
        add_goal("complete_sparse_azimuth")
    if any(keyword in text for keyword in ["小目标"]):
        add_goal("improve_small_object_coverage")
    if any(keyword in text for keyword in ["背景", "场景合成", "目标和场景", "目标和背景", "目标-背景", "图像融合"]):
        add_goal("target_background_composition")
    if any(keyword in text for keyword in ["跨载荷", "跨域", "特征迁移", "风格迁移"]):
        add_goal("domain_adaptation")
    if any(keyword in text for keyword in ["极化方式", "极化生成", "全极化", "polarization"]):
        add_goal("polarization_conditioned_generation")
    return goals or ["unspecified"]


def requests_downstream_classification_evaluation(text: str) -> bool:
    lowered = text.lower()
    if excludes_downstream_classification_evaluation(text):
        return False
    if any(
        keyword in text
        for keyword in [
            "下游",
            "准确率",
            "识别率",
            "分类评估",
            "分类收益",
            "分类效果",
            "训练分类",
            "分类模型",
            "下游分类",
        ]
    ):
        return True
    if any(
        keyword in lowered
        for keyword in [
            "accuracy",
            "classification evaluation",
            "classification benchmark",
            "downstream",
            "benchmark",
            "train classifier",
            "classifier accuracy",
        ]
    ):
        return True
    return "收益" in text and any(keyword in text for keyword in ["分类", "识别", "下游", "准确率", "模型"])


def excludes_downstream_classification_evaluation(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in text
        for keyword in [
            "不训练分类",
            "不训练分类器",
            "不训练分类模型",
            "不需要训练分类",
            "不需要训练分类器",
            "不需要训练分类模型",
            "不要训练分类",
            "不要训练分类器",
            "不要训练分类模型",
            "不做分类评估",
            "不需要分类评估",
            "不要分类评估",
            "不做下游",
            "不需要下游",
            "不要下游",
            "不评估下游",
            "不看准确率",
        ]
    ) or any(
        keyword in lowered
        for keyword in [
            "do not train classifier",
            "don't train classifier",
            "no classifier training",
            "no classification evaluation",
            "without downstream",
            "no downstream",
        ]
    )


def requests_lightweight_probe(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in text for keyword in ["评估", "验证", "质量", "分布", "指标", "效果", "伪影", "重复"]) or any(
        keyword in lowered for keyword in ["fid", "mmd", "quality", "distribution", "artifact", "duplicate", "metric"]
    )


def parse_style_source(text: str) -> str | None:
    patterns = [
        r"[，,]\s*(?P<path>.+?)作为风格图",
        r"把(?P<path>.+?)作为风格图",
        r"把(?P<path>.+?)作为指导图",
        r"(?P<path>\S+)作为风格图",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_path_text(match.group("path"))
    return None


def parse_content_source(text: str) -> str | None:
    patterns = [
        r"把(?P<path>.+?)的数据作为内容图",
        r"把(?P<path>.+?)作为内容图",
        r"迁移到(?P<path>.+?)(?:的数据上|数据上|的数据|上)",
        r"迁到(?P<path>.+?)(?:的数据上|数据上|的数据|上)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return clean_path_text(match.group("path"))
    return None


def parse_dataset_source(text: str) -> str | None:
    path_candidate = first_existing_path_like(text)
    if path_candidate:
        return path_candidate
    patterns = [
        r"用(?P<path>.+?)(?:训练|来训练|作为训练数据|作为数据集)",
        r"用(?P<path>.+?)(?:做|进行)?(?:传统增广|传统增强|基础增广|基础增强|快速增广|快速增强)",
        r"使用(?P<path>.+?)(?:训练|来训练|作为训练数据|作为数据集)",
        r"使用(?P<path>.+?)(?:做|进行)?(?:传统增广|传统增强|基础增广|基础增强|快速增广|快速增强)",
        r"数据集(?:就)?(?:用|使用)\s*(?P<path>\S+)",
        r"数据集(?:为|是|:|：)?\s*(?P<path>\S+)",
        r"输入数据集(?:为|是|:|：)?\s*(?P<path>\S+)",
        r"(?P<path>\S+)(?:训练一个|训练|来训练)(?:Flux|SD3|SDXL|LoRA|lora)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = clean_path_text(match.group("path"))
            if is_plausible_dataset_path(candidate):
                return candidate
    return None


def parse_classification_evaluation_sources(task: str, path_candidates: list[str]) -> tuple[str | None, str | None]:
    if task != "classification_evaluation":
        return None, None
    baseline = path_candidates[0] if path_candidates else None
    augmented = path_candidates[1] if len(path_candidates) > 1 else None
    return baseline, augmented


def parse_raysar_scene_source(text: str, path_candidates: list[str]) -> str | None:
    for candidate in path_candidates:
        if Path(candidate).suffix.lower() == ".pov":
            return candidate
    patterns = [
        r"(?:pov_scene|POV场景|场景文件|\.pov文件)(?:为|是|:|：)?\s*(?P<path>\S+\.pov)",
        r"(?:用|使用|输入)\s*(?P<path>\S+\.pov)(?:做|进行|作为)?(?:RaySAR|raysar|物理仿真|射线追踪)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_path_text(match.group("path"))
    return None


def parse_raysar_model_file(text: str, path_candidates: list[str]) -> str | None:
    for candidate in path_candidates:
        if Path(candidate).suffix.lower() in RAYSAR_MODEL_EXTS:
            return candidate
    patterns = [
        r"(?:3D模型|三维模型|模型文件|model_file|model)(?:为|是|:|：)?\s*(?P<path>\S+\.(?:obj|stl|ply|glb|gltf|off|dae|3ds|xyz))",
        r"(?:用|使用|输入)\s*(?P<path>\S+\.(?:obj|stl|ply|glb|gltf|off|dae|3ds|xyz))(?:做|进行|作为)?(?:RaySAR|raysar|物理仿真|射线追踪|成像)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_path_text(match.group("path"))
    return None


def parse_raysar_parameters_file(text: str, path_candidates: list[str]) -> str | None:
    for candidate in path_candidates:
        suffix = Path(candidate).suffix.lower()
        name = Path(candidate).name.lower()
        if suffix in {".txt", ".yaml", ".yml", ".json"} and ("parameter" in name or "param" in name or "参数" in name):
            return candidate
    patterns = [
        r"(?:parameters_file|参数文件|仿真参数)(?:为|是|:|：)?\s*(?P<path>\S+\.(?:txt|yaml|yml|json))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_path_text(match.group("path"))
    return None


def parse_raysar_contributions_file(text: str, path_candidates: list[str]) -> str | None:
    for candidate in path_candidates:
        if Path(candidate).name.lower() in {"contributions.txt", "contribution.txt"}:
            return candidate
    match = re.search(r"(?:contributions_txt|Contributions\.txt|贡献文件)(?:为|是|:|：)?\s*(?P<path>\S+\.txt)", text, flags=re.IGNORECASE)
    if match:
        return clean_path_text(match.group("path"))
    return None


def parse_raysar_geometry(text: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    filters = filters or {}
    azimuth_sweep = parse_azimuth_sweep(text)
    azimuth_values = parse_azimuth_values(text)
    if azimuth_sweep:
        geometry["azimuth_sweep"] = azimuth_sweep
    if azimuth_values:
        geometry["azimuth_values"] = azimuth_values
    if filters.get("incidence_angle_deg") is not None:
        geometry["incidence_angle_deg"] = filters["incidence_angle_deg"]
    if filters.get("depression_angle_deg") is not None:
        geometry["depression_angle_deg"] = filters["depression_angle_deg"]
    if filters.get("azimuth_deg") is not None:
        geometry["azimuth_deg"] = filters["azimuth_deg"]
    patterns = [
        ("target_extent_m", r"(?:目标尺寸|目标尺度|最长边|模型尺寸)(?:为|=|是)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?:m|米)?"),
        ("sensor_plane_m", r"(?:场景尺寸|成像场景|传感器平面|sensor_plane)(?:为|=|是)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?:m|米)?"),
        ("range_distance_m", r"(?:距离|斜距|水平距离|range_distance)(?:为|=|是)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?:m|米)?"),
    ]
    for field, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            geometry[field] = parse_number(match.group("value"))
    up_match = re.search(r"(?:up[_ -]?axis|高度轴|上方向)(?:为|=|是)?\s*(?P<value>[xyzXYZ])", text)
    if up_match:
        geometry["input_up_axis"] = up_match.group("value").lower()
    return geometry


def parse_azimuth_sweep(text: str) -> dict[str, Any] | None:
    patterns = [
        r"方位角[^。；;\n]{0,12}?(?:从|由)?\s*(?P<start>-?\d+(?:\.\d+)?)\s*(?:到|至|-|~|—|–)\s*(?P<stop>-?\d+(?:\.\d+)?)[^。；;\n]{0,20}?(?:每隔|间隔|步长|步进|每|step)\s*(?P<step>\d+(?:\.\d+)?)\s*(?:度|deg|°)?",
        r"(?:从|由)\s*(?P<start>-?\d+(?:\.\d+)?)\s*(?:到|至|-|~|—|–)\s*(?P<stop>-?\d+(?:\.\d+)?)[^。；;\n]{0,12}?方位角[^。；;\n]{0,20}?(?:每隔|间隔|步长|步进|每|step)\s*(?P<step>\d+(?:\.\d+)?)\s*(?:度|deg|°)?",
        r"(?:绕|围绕)[^。；;\n]{0,12}?(?:轴|方向)[^。；;\n]{0,12}?(?:从|由)?\s*(?P<start>-?\d+(?:\.\d+)?)\s*(?:度|deg|°)?\s*(?:转到|转至|到|至|-|~|—|–)\s*(?P<stop>-?\d+(?:\.\d+)?)\s*(?:度|deg|°)?[^。；;\n]{0,20}?(?:每隔|间隔|步长|步进|每|step)\s*(?P<step>\d+(?:\.\d+)?)\s*(?:度|deg|°)?",
        r"azimuth[^。；;\n]{0,12}?(?P<start>-?\d+(?:\.\d+)?)\s*(?:to|-|~)\s*(?P<stop>-?\d+(?:\.\d+)?)[^。；;\n]{0,20}?(?:step|interval|every)\s*(?P<step>\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return {
                "start": parse_number(match.group("start")),
                "stop": parse_number(match.group("stop")),
                "step": parse_number(match.group("step")),
            }
    return None


def parse_azimuth_values(text: str) -> list[Any]:
    patterns = [
        r"方位角(?:为|=|是|:|：)?\s*(?P<values>-?\d+(?:\.\d+)?(?:\s*(?:、|,|，|/|和|以及|及)\s*-?\d+(?:\.\d+)?){1,})\s*(?:度|deg|°)?",
        r"azimuth(?:s)?(?:\s*=|\s*:)?\s*(?P<values>-?\d+(?:\.\d+)?(?:\s*(?:,|/|and)\s*-?\d+(?:\.\d+)?){1,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        values = []
        for item in re.split(r"\s*(?:、|,|，|/|和|以及|及|and)\s*", match.group("values")):
            if item.strip():
                values.append(parse_number(item.strip()))
        return values
    return []


def parse_render_size(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"(?P<w>\d{2,5})\s*(?:x|×|乘|by)\s*(?P<h>\d{2,5})", text, flags=re.IGNORECASE)
    if match:
        return int(match.group("w")), int(match.group("h"))
    match = re.search(r"(?:分辨率|渲染尺寸|图像尺寸)(?:为|是|=|:|：)?\s*(?P<size>\d{2,5})", text)
    if match:
        size = int(match.group("size"))
        return size, size
    return None, None


def parse_scene_prompt(text: str, task: str) -> str | None:
    if task != "background_generation":
        return None
    cleaned = text.strip()
    cleaned = re.sub(
        r"(?:，|,|。|；|;)?\s*(?:不需要|不用|无需|不要|不必)(?:重新)?训练(?:背景)?模型.*$",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"(?:，|,|。|；|;)?\s*(?:不训练|不用训练|无需训练|权重已训练好|权重训练好了|使用已有权重).*$",
        " ",
        cleaned,
    )
    include_match = re.search(
        r"(?:场景|背景|内容|画面)?\s*(?:包含|包括|含有|带有|具有|有|为|是|:|：)\s*(?P<prompt>[^。；;\n]+)",
        cleaned,
    )
    if include_match:
        cleaned = include_match.group("prompt")
    cleaned = re.sub(r"(?:请|帮我|给我)?(?:用|使用)\S+\s*", "", cleaned)
    cleaned = re.sub(r"(?:生成|出|做|制作|创建)?\s*\d+\s*(?:张|个|幅)?", "", cleaned)
    cleaned = re.sub(r"(?:SAR|sar)?(?:背景图生成|背景生成|生成背景图|生成背景|背景图|背景|场景)", " ", cleaned)
    cleaned = re.sub(r"(?:包含|包括|含有|带有|具有)", " ", cleaned)
    cleaned = re.sub(r"(?:和|及|与|、|/|，|,)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。；;")
    return cleaned or text.strip()


def parse_composition_sources(
    text: str,
    path_candidates: list[str],
    dataset_source: str | None,
) -> tuple[str | None, str | None, str | None]:
    target = parse_path_near_role(
        text,
        role_patterns=[
            r"(?P<path>[^\s，,。；;]+?)(?:作为|当作|用作)?(?:目标图|目标|前景)",
            r"(?P<path>[^\s，,。；;]+)(?:\s+as\s+|\s+)(?:target|foreground)\b",
            r"(?:目标图|目标|前景|target)(?:为|是|:|：)?\s*(?P<path>[^\s，,。；;]+)",
        ],
    )
    background = parse_path_near_role(
        text,
        role_patterns=[
            r"(?P<path>[^\s，,。；;]+?)(?:作为|当作|用作)?(?:背景图|背景|场景)",
            r"(?P<path>[^\s，,。；;]+)(?:\s+as\s+|\s+)(?:background|scene)\b",
            r"(?:背景图|背景|场景|background)(?:为|是|:|：)?\s*(?P<path>[^\s，,。；;]+)",
        ],
    )
    mask = parse_path_near_role(
        text,
        role_patterns=[
            r"(?P<path>[^\s，,。；;]+?)(?:作为|当作|用作)?(?:mask|掩码|目标mask|目标掩码)",
            r"(?:mask|掩码|目标mask|目标掩码)(?:为|是|:|：)?\s*(?P<path>[^\s，,。；;]+)",
        ],
    )
    if not target and path_candidates:
        target = dataset_source or path_candidates[0]
    if not background:
        for candidate in path_candidates:
            if candidate != target and candidate != mask:
                background = candidate
                break
    return target, background, mask


def parse_path_near_role(text: str, role_patterns: list[str]) -> str | None:
    for pattern in role_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group("path")
            for part in reversed([item for item in re.split(r"[\s，,。；;]+", raw) if item]):
                candidate = clean_path_text(part)
                if is_plausible_dataset_path(candidate):
                    return candidate
    return None


def parse_blend_mode(text: str) -> str | None:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["poisson", "seamlessclone", "seamless clone"]) or any(keyword in text for keyword in ["泊松", "无缝融合"]):
        return "poisson"
    if any(keyword in lowered for keyword in ["laplacian", "pyramid"]) or any(keyword in text for keyword in ["拉普拉斯", "金字塔"]):
        return "laplacian"
    if any(keyword in lowered for keyword in ["feather", "alpha"]) or any(keyword in text for keyword in ["羽化", "渐变", "透明度"]):
        return "feather"
    return None


def parse_path_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    prefixed_starts = [
        match.start()
        for match in re.finditer(r"(?:\.\.?/|exampledataset/|myproject/|runs/|configs/)", text)
    ]
    for idx, start in enumerate(prefixed_starts):
        next_start = prefixed_starts[idx + 1] if idx + 1 < len(prefixed_starts) else len(text)
        end = min(next_start, first_path_terminator(text, start))
        raw = re.sub(r"(?:和|与|以及|及)$", "", text[start:end].strip())
        add_path_candidate(candidates, seen, raw)

    for raw in re.findall(r"(?<![\w.-])/[^\s，,。；;\n]+", text):
        add_path_candidate(candidates, seen, raw)
    return candidates


def first_path_terminator(text: str, start: int) -> int:
    terminators = [" ", "\t", "\n", "，", ",", "。", "；", ";"]
    positions = [text.find(item, start) for item in terminators]
    soft_terminators = [
        "作为",
        "当作",
        "用作",
        "做",
        "进行",
        "生成",
        "增广",
        "增强",
        "迁移",
        "作为内容图",
        "作为指导图",
        "的数据",
        "下的数据",
    ]
    positions.extend(text.find(item, start) for item in soft_terminators)
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else len(text)


def add_path_candidate(candidates: list[str], seen: set[str], raw: str) -> None:
    cleaned = clean_path_text(raw)
    if not is_plausible_dataset_path(cleaned):
        return
    if cleaned not in seen:
        candidates.append(cleaned)
        seen.add(cleaned)


def first_existing_path_like(text: str) -> str | None:
    for candidate in parse_path_candidates(text):
        if is_plausible_dataset_path(candidate):
            return candidate
    return None


def first_non_model_path(path_candidates: list[str]) -> str | None:
    for candidate in path_candidates:
        if Path(candidate).suffix.lower() not in RAYSAR_MODEL_EXTS:
            return candidate
    return None


def is_plausible_dataset_path(value: str | None) -> bool:
    if not value:
        return False
    if "/" not in value and not str(value).startswith("."):
        return False
    path = Path(value).expanduser()
    if path.exists():
        return True
    return any(str(value).startswith(prefix) for prefix in ("exampledataset/", "myproject/", "runs/", "configs/", "./", "../", "/"))


def parse_model_family(text: str) -> str | None:
    lowered = text.lower()
    if "flux" in lowered:
        return "flux"
    if "sd3.5" in lowered or "sd35" in lowered or "sd3" in lowered:
        return "sd3"
    if "sdxl" in lowered:
        return "sdxl"
    return None


def parse_target_count(text: str) -> int | None:
    patterns = [
        r"(?:生成|出|增广|扩增)\s*(?P<count>\d+)\s*(?:张|个|幅)",
        r"(?P<count>\d+)\s*(?:张|个|幅)(?:的)?(?:增广|增强|生成|扩增)?(?:数据|图像|图片|样本)",
        r"(?:target_count|目标数量)\s*(?:=|为|是|:|：)?\s*(?P<count>\d+)",
        r"(?:需求|需要|要)[^。；;\n]{0,12}?(?P<count>\d+)\s*(?:张|个|幅)",
        r"扩增到\s*(?P<count>\d+)\s*(?:张|个|幅)?",
        r"\b(?:generate|create|augment(?:\s+to)?)\s+(?P<count>\d+)\s+(?:samples?|images?|chips?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group("count"))
    return None


def parse_training_epochs(text: str) -> int | None:
    patterns = [
        r"(?:训练|训|train)[^。；;\n]{0,12}?(?P<count>\d+)\s*(?:轮|个epoch|epochs?|epoch)",
        r"(?:设置|设为|跑|迭代)[^。；;\n]{0,12}?(?P<count>\d+)\s*(?:轮|个epoch|epochs?|epoch)",
        r"(?:max_train_epochs|训练轮数|epoch数)\s*(?:=|为|是|:|：)?\s*(?P<count>\d+)",
        r"(?P<count>\d+)\s*(?:轮|个epoch|epochs?|epoch)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group("count"))
    return None


def extract_filters(text: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    patterns = [
        ("depression_angle_deg", r"下视角(?:为|=|是)?\s*(?P<value>-?\d+(?:\.\d+)?)"),
        ("incidence_angle_deg", r"入射角(?:为|=|是)?\s*(?P<value>-?\d+(?:\.\d+)?)"),
        ("azimuth_deg", r"方位角(?:为|=|是)?\s*(?P<value>-?\d+(?:\.\d+)?)"),
        ("elevation_angle_deg", r"仰角(?:为|=|是)?\s*(?P<value>-?\d+(?:\.\d+)?)"),
        ("resolution_m", r"分辨率(?:为|=|是)?\s*(?P<value>-?\d+(?:\.\d+)?)"),
    ]
    for field, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            filters[field] = parse_number(match.group("value"))
    band_match = re.search(r"波段(?:为|=|是)?\s*(?P<value>[A-Za-z][A-Za-z0-9+-]*)", text)
    if band_match:
        filters["band"] = band_match.group("value")
    pol_match = re.search(r"极化(?:为|=|是)?\s*(?P<value>A?(?:HH|HV|VH|VV)|Pauli|pauli)", text)
    if pol_match:
        filters["polarization"] = pol_match.group("value")
    return filters


def extract_exclude_filters(text: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    lowered = text.lower()
    if re.search(r"(pauli|Pauli)[^。；;\n]{0,16}(?:不作为|不算|不是|排除|剔除|不要|不使用)", text) or re.search(
        r"(?:排除|剔除|不要|不使用|不包含|不包括)[^。；;\n]{0,16}(pauli|Pauli)",
        text,
    ):
        filters.setdefault("polarization", []).append("pauli")
    if "非pauli" in lowered or "not pauli" in lowered:
        filters.setdefault("polarization", []).append("pauli")
    return filters


def extract_format_hints(text: str) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for match in re.finditer(
        r"(?:命名|文件名|名称|filename)[^。；;\n]*?"
        r"(?P<anchor>-?\d+(?:\.\d+)?)[^。；;\n]{0,24}?"
        r"(?:数字|字段|token|参数)[^。；;\n]{0,12}?"
        r"(?:依次)?(?:为|是|代表|表示)(?P<fields>[^。；;\n]+)",
        text,
        flags=re.IGNORECASE,
    ):
        hint = build_anchor_hint(match.group("anchor"), match.group("fields"), match.group(0))
        key = hint_key(hint)
        if key not in seen:
            hints.append(hint)
            seen.add(key)

    for match in re.finditer(
        r"(?P<anchor>-?\d+(?:\.\d+)?)\s*(?:后面|后面的|之后|后的|后)[^。；;\n]{0,16}?"
        r"(?:数字|字段|token|参数)?[^。；;\n]{0,8}?"
        r"(?:依次)?(?:为|是|代表|表示)(?P<fields>[^。；;\n]+)",
        text,
        flags=re.IGNORECASE,
    ):
        hint = build_anchor_hint(match.group("anchor"), match.group("fields"), match.group(0))
        key = hint_key(hint)
        if key not in seen:
            hints.append(hint)
            seen.add(key)

    for match in re.finditer(
        r"(?:后面|后面的|最后|末尾|结尾|后缀)[^。；;\n]{0,16}?"
        r"(?:字母|字段|字符|token|参数)[^。；;\n]{0,12}?"
        r"(?:代表|表示|为|是)(?P<field>极化方式|极化|polarization|pol)",
        text,
        flags=re.IGNORECASE,
    ):
        hint = build_suffix_field_hint(match.group("field"), match.group(0))
        key = hint_key(hint)
        if key not in seen:
            hints.append(hint)
            seen.add(key)
    return hints


def build_anchor_hint(anchor: str, fields_text: str, source_text: str) -> dict[str, Any]:
    raw_fields = [field for field in FIELD_SPLIT_RE.split(fields_text.strip()) if field.strip()]
    fields = []
    for raw in raw_fields:
        cleaned = raw.strip().strip("。；;，, ")
        normalized = normalize_field_name(cleaned)
        fields.append({"raw": cleaned, "normalized": normalized})
    return {
        "schema_version": FORMAT_HINT_VERSION,
        "type": "filename_anchor_fields",
        "source": "user_request",
        "source_text": source_text.strip(),
        "rule": {
            "anchor": parse_number(anchor),
            "fields_after_anchor": fields,
        },
        "validation_required": True,
        "notes": [
            "This is a semantic hint, not a trusted parser.",
            "SAGA must compile and validate a DatasetFormatSpec before using it for filtering or diagnosis.",
        ],
    }


def build_suffix_field_hint(field_text: str, source_text: str) -> dict[str, Any]:
    cleaned = field_text.strip().strip("。；;，, ")
    normalized = normalize_field_name(cleaned)
    return {
        "schema_version": FORMAT_HINT_VERSION,
        "type": "filename_suffix_field",
        "source": "user_request",
        "source_text": source_text.strip(),
        "rule": {
            "field": {"raw": cleaned, "normalized": normalized},
            "position": "last_underscore_token_before_extension",
        },
        "validation_required": True,
        "notes": [
            "This is a semantic hint, not a trusted parser.",
            "SAGA must compile and validate a DatasetFormatSpec before using it for filtering or diagnosis.",
        ],
    }


def hint_key(hint: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    rule = hint.get("rule") or {}
    if hint.get("type") == "filename_suffix_field":
        field = rule.get("field") or {}
        return ("suffix", (str(field.get("normalized") or field.get("raw") or ""),))
    fields = []
    for field in rule.get("fields_after_anchor", []):
        if isinstance(field, dict):
            fields.append(field.get("normalized") or field.get("raw") or "")
        else:
            fields.append(str(field))
    return (str(rule.get("anchor")), tuple(fields))


def clean_path_text(value: str) -> str:
    cleaned = value.strip().strip("，,。；; ")
    cleaned = re.sub(r"^(将|把)\s*", "", cleaned)
    cleaned = re.sub(
        r"(?:作为|当作|用作)?(?:目标图|目标|前景|背景图|背景|场景|风格图|指导图|内容图|mask|掩码|目标mask|目标掩码)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip("，,。；; ")
    cleaned = re.sub(r"\s*(的数据|数据)$", "", cleaned)
    path_like = re.findall(r"[\w\u4e00-\u9fff./@:+()（）\[\]\-]+/[\w\u4e00-\u9fff./@:+()（）\[\]\-]+", cleaned)
    if path_like:
        cleaned = path_like[0].strip("，,。；; ")
        cleaned = re.split(
            r"(?:做|进行|作为|用于|拿来|来)?(?:SAR)?(?:分类|检测|分割)?(?:数据)?"
            r"(?:增广|增强|扩增|生成|训练|评估|验证|可视化|伪彩色|伪彩)",
            cleaned,
            maxsplit=1,
        )[0].strip("，,。；; ")
        cleaned = re.split(
            r"(?:这个|该)?(?:SAR)?(?:分类|检测|分割)?数据集|"
            r"(?:制定|做|进行)?(?:(?:扩散模型)?(?:LoRA|lora)|扩散模型|GAN|gan|DCGAN|dcgan|图生图)?(?:增广|增强|生成)|"
            r"(?:制定|做|进行)?(?:LoRA|lora|扩散模型|GAN|gan|DCGAN|dcgan|图生图|传统增广|传统增强|基础增广|基础增强|快速增广|快速增强|风格迁移|特征迁移|伪彩色|伪彩|可视化|训练|生成|增广|增强|下游|分类收益|准确率|比较|对比|评估|验证|计划)",
            cleaned,
            maxsplit=1,
        )[0].strip("，,。；; ")
    marker = re.search(r"(?:\.\.?/|/|exampledataset/|myproject/|runs/|configs/)", cleaned)
    if marker and marker.start() > 0:
        cleaned = cleaned[marker.start() :]
    extension_match = re.search(
        r"(?P<path>.+?\.(?:pov|obj|stl|ply|glb|gltf|off|dae|3ds|xyz|txt|json|yaml|yml|xml|csv|png|jpg|jpeg|bmp|tif|tiff|webp))",
        cleaned,
        flags=re.IGNORECASE,
    )
    if extension_match:
        cleaned = extension_match.group("path")
    cleaned = re.sub(
        r"(?:作为|当作|用作)?(?:目标图|目标|前景|背景图|背景|场景|风格图|指导图|内容图|mask|掩码|目标mask|目标掩码)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip("，,。；; ")
    cleaned = re.sub(r"的$", "", cleaned)
    return cleaned


def parse_number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def estimate_intent_confidence(
    task: str,
    style_source: str | None,
    content_source: str | None,
    filters: dict[str, Any],
    dataset_source: str | None = None,
) -> float:
    if task == "unknown":
        return 0.1
    score = 0.45
    if task == "style_transfer":
        if style_source:
            score += 0.2
        if content_source:
            score += 0.2
    if task in {"diffusion_lora_generation", "gan_generation", "geodiff_sar_generation"}:
        if dataset_source:
            score += 0.3
    if task == "raysar_synthesis":
        if dataset_source:
            score += 0.3
    if task == "background_generation":
        score += 0.25
    if task == "classification_evaluation":
        if dataset_source:
            score += 0.3
    if filters:
        score += 0.1
    return round(min(score, 0.95), 3)


def render_intent_markdown(result: dict[str, Any]) -> str:
    intent_spec = result["intent_spec"]
    intent = intent_spec["intent"]
    lines = [
        "# SAGA Intent Recognition",
        "",
        f"- Mode: {result['recognizer']['mode']}",
        f"- Task: `{intent.get('task')}`",
        f"- Confidence: {intent_spec.get('confidence')}",
        f"- Style source: `{intent.get('style_source')}`",
        f"- Content source: `{intent.get('content_source')}`",
        f"- Dataset source: `{intent.get('dataset_source')}`",
        f"- Model family: `{intent.get('model_family')}`",
        f"- Target count: `{intent.get('target_count')}`",
        f"- Training epochs: `{intent.get('training_epochs')}`",
        f"- POV scene: `{intent.get('pov_scene')}`",
        f"- 3D model: `{intent.get('model_file')}`",
        f"- RaySAR parameters: `{intent.get('parameters_file')}`",
        f"- Contributions: `{intent.get('contributions_txt')}`",
        f"- Scene prompt: `{intent.get('scene_prompt')}`",
        f"- RaySAR geometry: `{intent.get('raysar_geometry')}`",
        f"- Filters: `{intent.get('filters')}`",
        f"- Exclude filters: `{intent.get('exclude_filters')}`",
        f"- Goals: `{intent.get('goals')}`",
        "",
        "## Format Hints",
        "",
    ]
    hints = result.get("format_hints", [])
    if hints:
        for hint in hints:
            lines.append(f"- `{hint.get('type')}`: {hint.get('rule')}")
    else:
        lines.append("- No explicit format hint was extracted.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This module interprets user language only.",
            "- It does not scan files, validate labels, execute skills, or decide dataset truth.",
            "",
        ]
    )
    return "\n".join(lines)

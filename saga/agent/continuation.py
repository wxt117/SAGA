from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from saga.agent.llm_client import LLMConfig
from saga.agent.runtime import run_agent
from saga.core.config import load_mapping, save_json, save_text


AGENT_CONTINUATION_VERSION = "saga_agent_continuation_v1"


def continue_agent_run(
    run_dir: str | Path,
    answer: str,
    output_dir: str | Path | None = None,
    llm_config: LLMConfig | None = None,
    use_llm_intent: bool | None = None,
    use_llm_planner: bool | None = None,
    dry_run: bool = True,
    execute: bool = True,
    memory_dir: str | Path = "runs/memory",
    use_memory: bool = True,
    style_transfer_config: str | Path | None = None,
    diffusion_lora_config: str | Path | None = None,
    traditional_aug_config: str | Path | None = None,
    saga_config_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    state = load_mapping(run_path / "agent_state.json")
    questions = load_questions(run_path)
    original_request = str(state.get("request") or "")
    continued_request = build_continued_request(
        original_request=original_request,
        questions=questions,
        answer=answer,
    )
    out_path = Path(output_dir).expanduser().resolve() if output_dir else default_continuation_dir(run_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if use_llm_intent is None:
        use_llm_intent = bool(str(state.get("intent_mode") or "").startswith("llm"))
    if use_llm_planner is None:
        use_llm_planner = bool(str(state.get("planner_mode") or "").startswith("llm"))
    if (use_llm_intent or use_llm_planner) and llm_config is None:
        use_llm_intent = False
        use_llm_planner = False

    save_json(
        out_path / "continuation_input.json",
        {
            "schema_version": AGENT_CONTINUATION_VERSION,
            "source_run_dir": run_path.as_posix(),
            "original_request": original_request,
            "questions": questions,
            "answer": answer,
            "continued_request": continued_request,
            "use_llm_intent": use_llm_intent,
            "use_llm_planner": use_llm_planner,
        },
    )
    save_text(out_path / "continuation_request.txt", continued_request + "\n")
    result = run_agent(
        request=continued_request,
        output_dir=out_path,
        dataset_root=state.get("dataset_root"),
        style_transfer_config=style_transfer_config,
        diffusion_lora_config=diffusion_lora_config,
        traditional_aug_config=traditional_aug_config,
        saga_config_path=saga_config_path,
        llm_config=llm_config,
        use_llm_intent=bool(use_llm_intent),
        use_llm_planner=bool(use_llm_planner),
        memory_dir=memory_dir,
        use_memory=use_memory,
        dry_run=dry_run,
        execute=execute,
    )
    result["continuation_source_run_dir"] = run_path.as_posix()
    save_json(out_path / "agent_state.json", result)
    return result


def load_questions(run_path: Path) -> list[str]:
    for name in ("planner_clarification.json", "intent_guardrail.json"):
        path = run_path / name
        if not path.exists():
            continue
        data = load_mapping(path)
        if isinstance(data.get("questions"), list):
            return [str(item) for item in data["questions"]]
        if isinstance(data.get("clarification_questions"), list):
            return [str(item) for item in data["clarification_questions"]]
    return []


def build_continued_request(original_request: str, questions: list[str], answer: str) -> str:
    parts = [original_request.strip()]
    if questions:
        parts.append("SAGA 此前需要澄清的问题：")
        parts.extend(f"- {question}" for question in questions)
    parts.append("用户补充说明：")
    parts.append(answer.strip())
    return "\n".join(part for part in parts if part)


def default_continuation_dir(run_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return run_path / "continuations" / stamp

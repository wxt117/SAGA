from __future__ import annotations

from pathlib import Path

import yaml

from saga.agent.json_utils import loads_json_object
from saga.agent.llm_client import LLMConfig, OpenAICompatibleClient
from saga.core.config import load_mapping, save_text
from saga.data.inspect import build_format_spec_prompt


def infer_format_spec_with_llm(
    inspection_path: str | Path,
    output_path: str | Path,
    llm_config: LLMConfig,
    user_description: str = "",
    prompt_sample_limit: int = 40,
    txt_preview_limit: int = 10,
    use_response_format: bool = True,
) -> str:
    inspection = load_mapping(inspection_path)
    prompt = build_format_spec_prompt(
        inspection=inspection,
        user_description=user_description,
        sample_limit=prompt_sample_limit,
        txt_preview_limit=txt_preview_limit,
    )
    client = OpenAICompatibleClient(llm_config)
    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You infer dataset parsing specs for SAGA. "
                    "Return only JSON. Regex patterns must use Python named groups. "
                    "Keep the answer concise."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"} if use_response_format else None,
    )
    if not content.strip():
        raise RuntimeError("LLM returned an empty DatasetFormatSpec response.")
    data = loads_json_object(content)
    rendered = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    save_text(output_path, rendered)
    return rendered

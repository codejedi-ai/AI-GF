"""DeepSeek LLM provider (OpenAI-compatible API via livekit-plugins-openai)."""
from __future__ import annotations

import os

from livekit.plugins import openai

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def create(cfg: dict):
    """Factory for the provider registry: build a DeepSeek LLM from a config `llm` section."""
    model = cfg.get("model") or "deepseek-chat"
    base_url = cfg.get("url") or DEEPSEEK_BASE_URL
    return openai.LLM(
        model=model,
        base_url=base_url,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

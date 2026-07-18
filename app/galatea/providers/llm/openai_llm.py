"""
OpenAI (and OpenAI-compatible) LLM provider.

When the config's `url` is set, any OpenAI-compatible server works
(LM Studio, Ollama, vLLM, ...) — the model name is passed through as-is.
"""
from __future__ import annotations

from livekit.plugins import openai


def create(cfg: dict):
    """Factory for the provider registry: build an LLM from a config `llm` section."""
    model = cfg.get("model") or "gpt-4o-mini"
    base_url = cfg.get("url") or cfg.get("base_url")
    if base_url:
        return openai.LLM(model=model, base_url=base_url)
    return openai.LLM(model=model)

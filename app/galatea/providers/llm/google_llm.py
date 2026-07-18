"""Google Gemini LLM provider (wraps livekit-plugins-google; install it to use)."""
from __future__ import annotations

from livekit.plugins import google


def create(cfg: dict):
    """Factory for the provider registry: build a Gemini LLM from a config `llm` section."""
    kwargs = {}
    model = cfg.get("model")
    if model:
        kwargs["model"] = model
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    return google.LLM(**kwargs)

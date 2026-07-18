"""Anthropic Claude LLM provider (wraps livekit-plugins-anthropic; install it to use)."""
from __future__ import annotations

from livekit.plugins import anthropic


def create(cfg: dict):
    """Factory for the provider registry: build a Claude LLM from a config `llm` section."""
    kwargs = {}
    model = cfg.get("model")
    if model:
        kwargs["model"] = model
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    return anthropic.LLM(**kwargs)

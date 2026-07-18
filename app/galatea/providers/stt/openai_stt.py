"""OpenAI STT provider (wraps livekit-plugins-openai)."""
from __future__ import annotations

from livekit.plugins import openai


def create(cfg: dict):
    """Factory for the provider registry: build OpenAI STT from a config `stt` section."""
    kwargs = {}
    model = cfg.get("model")
    if model:
        kwargs["model"] = model
    language = cfg.get("language")
    if language:
        kwargs["language"] = language
    return openai.STT(**kwargs)

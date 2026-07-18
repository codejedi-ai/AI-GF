"""OpenAI TTS provider (wraps livekit-plugins-openai)."""
from __future__ import annotations

from livekit.plugins import openai


def create(cfg: dict):
    """Factory for the provider registry: build OpenAI TTS from a config `tts` section."""
    from galatea.config import voice_options

    vo = voice_options(cfg)
    kwargs = {}
    model = cfg.get("model") or vo.get("model")
    if model:
        kwargs["model"] = model
    voice = vo.get("voice") or vo.get("voice_id")
    if voice:
        kwargs["voice"] = voice
    if vo.get("speed") is not None:
        kwargs["speed"] = vo["speed"]
    return openai.TTS(**kwargs)

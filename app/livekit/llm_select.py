"""
Build an LLM instance from the same data/galatea/config.json used by the
voice entrypoint, so text channels (Telegram, CLI) answer with the same
model as the voice agent instead of a hardcoded choice.
"""
from __future__ import annotations

from livekit.agents import inference
from livekit.plugins import anthropic


def build_llm(cfg: dict):
    llm_model = cfg.get("llm_model", "claude-haiku-4-5")
    llm_provider = cfg.get("provider", "anthropic").lower()

    if llm_provider == "anthropic":
        return anthropic.LLM(model=llm_model)
    elif llm_provider == "openai":
        from livekit.plugins import openai

        return openai.LLM(model=llm_model)
    elif llm_provider == "google":
        from livekit.plugins import google

        return google.LLM(model=llm_model)
    return inference.LLM(llm_model)

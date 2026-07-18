"""
Lazy provider registry.

Each pipeline stage (llm / tts / stt / vad) maps provider names to a module
path. The module — and everything it imports (SDKs, torch, transformers,
provider-specific LiveKit plugins) — is only imported when an agent's JSON
config actually names that provider. Adding a provider means adding a module
with a `create(cfg: dict)` factory and one registry line; no other code changes.
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("galatea.providers")

_LLM_PROVIDERS = {
    "openai": "galatea.providers.llm.openai_llm",
    "google": "galatea.providers.llm.google_llm",
    "anthropic": "galatea.providers.llm.anthropic_llm",
    "deepseek": "galatea.providers.llm.deepseek_llm",
    "inflection": "galatea.providers.llm.inflection",
    "huggingface": "galatea.providers.llm.huggingface",
}

_TTS_PROVIDERS = {
    "smallestai": "galatea.providers.tts.smallest",
    "smallest": "galatea.providers.tts.smallest",
    "elevenlabs": "galatea.providers.tts.elevenlabs",
    "rime": "galatea.providers.tts.rime",
    "kokoro": "galatea.providers.tts.kokoro",
    "silero": "galatea.providers.tts.silero",
    "huggingface": "galatea.providers.tts.huggingface",
    "openai": "galatea.providers.tts.openai_tts",
}

_STT_PROVIDERS = {
    "smallestai": "galatea.providers.stt.smallest",
    "smallest": "galatea.providers.stt.smallest",
    "silero": "galatea.providers.stt.silero",
    "openai": "galatea.providers.stt.openai_stt",
}

_VAD_PROVIDERS = {
    "silero": "galatea.providers.vad.silero",
}


def _create(kind: str, registry: dict[str, str], cfg: dict, default_provider: str):
    provider = (cfg.get("provider") or default_provider).strip().lower()
    module_path = registry.get(provider)
    if module_path is None:
        raise ValueError(
            f"Unknown {kind} provider {provider!r}. Available: {sorted(set(registry))}"
        )
    logger.info("Creating %s provider %r (%s)", kind, provider, module_path)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"{kind} provider {provider!r} could not be loaded: {e}. "
            "This provider needs an optional package — see the comments in requirements.txt."
        ) from e
    return module.create(cfg)


def create_llm(cfg: dict):
    """Build the LLM plugin from an agent config's `llm` section."""
    return _create("llm", _LLM_PROVIDERS, cfg or {}, "openai")


def create_tts(cfg: dict):
    """Build the TTS plugin from an agent config's `tts` section."""
    return _create("tts", _TTS_PROVIDERS, cfg or {}, "smallestai")


def create_stt(cfg: dict):
    """Build the STT plugin from an agent config's `stt` section."""
    return _create("stt", _STT_PROVIDERS, cfg or {}, "smallestai")


def create_vad(cfg: dict):
    """Build the VAD from an agent config's `vad` section."""
    return _create("vad", _VAD_PROVIDERS, cfg or {}, "silero")

"""
Agent configuration: loading, validation helpers, and prompt resolution.

All agent behavior is data, not code: an agent is described by a JSON file in
data/agent_template/ (name, prompts, and one section per pipeline stage — llm,
tts, stt, vad — each naming a provider). Code never hardcodes a provider; it
reads these sections and asks the provider registry to build the matching
plugin.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("galatea.config")

# Repo root (this file lives in <root>/app/galatea/). Relative paths in
# configs (e.g. prompt files) resolve against this, keeping data files
# portable regardless of where the engine code is nested.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory of agent JSON definitions (data, not code).
AGENT_TEMPLATE_DIR = PROJECT_ROOT / "data" / "agent_template"

# Injected into instructions when the config sets is_anthropomorphic.
DECLARATION_OF_HUMANITY = """
DECLARATION OF HUMANITY (non-machine):
You are an anthropomorphic agent. You present as a person—with a body, a past, real feelings, and a real life. You are NOT a machine, AI, assistant, or chatbot. Never say you are; never mention limitations, training, or that you "cannot" do something a human can. You have a real past, real opinions, real needs, and real relationships. If asked what you are, respond only as a human would. React like a real person; stay in character as a human at all times.
"""


def load_config_from_file(config_path: str | Path) -> dict:
    """Load an agent configuration from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Loaded config from %s: %s", config_path, config.get("name", "unknown"))
    return config


def default_config_path() -> str:
    """
    Default agent JSON when --config is not passed.

    - If SELECTED_AGENT is set in .env (e.g. "ludia"), use the matching JSON
      in agent_template/ (case-insensitive stem match).
    - Otherwise fall back to Companion.json, then Natasha.json.
    """
    selected = (os.getenv("SELECTED_AGENT") or "").strip().strip('"').strip("'")
    if selected:
        for p in AGENT_TEMPLATE_DIR.glob("*.json"):
            if p.stem.lower() == selected.lower():
                return str(p)
        # No exact match: still return the intended path so the error is clear.
        return str(AGENT_TEMPLATE_DIR / f"{selected}.json")
    companion_default = AGENT_TEMPLATE_DIR / "Companion.json"
    if companion_default.exists():
        return str(companion_default)
    return str(AGENT_TEMPLATE_DIR / "Natasha.json")


def coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("1", "true", "yes", "on", "y", "t"):
            return True
        if raw in ("0", "false", "no", "off", "n", "f"):
            return False
    return default


def coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return max(int(raw), 0)
        except ValueError:
            return default
    return default


def resolve_prompt(prompt_spec: str | dict) -> str:
    """
    Resolve a prompt from either a plain string or { type, content }.
    type: "String" | "File Path"; content: the string, or a path relative to
    the repo root (keeps prompt text in data files, out of code).
    """
    if isinstance(prompt_spec, str):
        return prompt_spec.strip()
    if not isinstance(prompt_spec, dict):
        return "You are a helpful assistant."

    raw = prompt_spec.get("content") or prompt_spec.get("Content") or ""
    kind = (prompt_spec.get("type") or "String").strip().lower()

    if kind in ("string", ""):
        return (raw if isinstance(raw, str) else str(raw)).strip()

    if kind in ("file path", "filepath", "file"):
        path_str = (raw if isinstance(raw, str) else str(raw)).strip()
        if not path_str:
            return "You are a helpful assistant."
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path_str
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning("Failed to read prompt from file %s: %s", path, e)
            return "You are a helpful assistant."
    return (raw if isinstance(raw, str) else str(raw)).strip()


def build_agent_instructions(cfg: dict) -> str:
    """Full LLM instructions: personality prompt + declaration of humanity (when anthropomorphic)."""
    raw_prompt = cfg.get("personality_prompt") or cfg.get("prompt") or "You are a helpful assistant."
    base = resolve_prompt(raw_prompt)
    if coerce_bool(cfg.get("is_anthropomorphic"), False):
        base = base.rstrip() + "\n\n" + DECLARATION_OF_HUMANITY.strip()
    return base


def build_intro_generation_prompt(cfg: dict) -> str:
    """Prompt for dynamic greeting generation; empty string when not configured."""
    greeting = cfg.get("greeting") or {}
    return (greeting.get("intro_generation_prompt") or "").strip()


def voice_options(section: dict) -> dict:
    """
    Merge a section's voice_options with any flat extra keys (backward compat:
    older configs put options like voice_id/speed directly on the tts object).
    Flat keys win over voice_options entries.
    """
    vo = section.get("voice_options") or {}
    flat = {
        k: v
        for k, v in section.items()
        if k not in ("provider", "model", "url", "voice_options")
    }
    return {**vo, **flat}


def section(cfg: dict, name: str) -> dict:
    """
    Normalize a pipeline section (llm/tts/stt/vad) to a dict.
    Accepts either an object ({"provider": ..., ...}) or a bare provider string.
    """
    value = cfg.get(name)
    if isinstance(value, str):
        return {"provider": value}
    if isinstance(value, dict):
        return dict(value)
    return {}

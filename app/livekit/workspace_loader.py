"""Shared config.json/SOUL.md/SKILLS.md loader, used by the voice entrypoint
and the text-channel brain so both read the same persona data."""
from __future__ import annotations

import json
import logging

from app.livekit.utils.paths import PathManager

logger = logging.getLogger("livekit.workspace_loader")


def load_workspace() -> dict:
    """Load config.json, SOUL.md, and SKILLS.md from the data/galatea directory."""
    config_path = PathManager.get_config_path()
    soul_path = PathManager.get_soul_path()
    skills_path = PathManager.get_skills_path()

    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load config.json: {e}")

    soul = soul_path.read_text(encoding="utf-8") if soul_path.exists() else "You are a helpful assistant."
    skills = skills_path.read_text(encoding="utf-8") if skills_path.exists() else ""

    return {
        "config": config,
        "soul": soul,
        "skills": skills,
    }

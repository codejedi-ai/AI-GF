import json
import os
from pathlib import Path

def load_voice_configs():
    config_path = Path.home() / ".galatea" / "agentconfig.json"
    
    if not config_path.exists():
        # Fallback to check if agent_configs.py is in the same .galatea folder
        # or return empty if nothing found.
        # But we also have agent_configs.py in ~/.galatea now.
        try:
            import sys
            galatea_path = str(Path.home() / ".galatea")
            if galatea_path not in sys.path:
                sys.path.append(galatea_path)
            from agent_configs import VOICE_CONFIGS
            return VOICE_CONFIGS
        except ImportError:
            return {}

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    return configs

VOICE_CONFIGS = load_voice_configs()

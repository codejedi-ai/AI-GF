"""
Configuration settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Vision Model Configuration
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
STRATEGY_MODEL = "claude-3-5-haiku-20241022"
TAGGING_MODEL = "gpt-4o"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Navigation Settings
MAX_NAVIGATION_STEPS = 15
WAIT_TIMEOUT = 10000  # ms
STEP_DELAY = 5000  # ms
# File Paths
SCREENSHOT_DIR = "screenshots"
DATA_DIR = "data"
CONFIG_DIR = "config"

# GitHub URLs
GITHUB_HOMEPAGE = "https://github.com"
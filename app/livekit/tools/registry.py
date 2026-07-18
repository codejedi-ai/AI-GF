"""
Modular tool loader.

A tool is a pair of files: `<name>.json` (the declaration — name,
description, and a JSON-Schema `parameters`, in OpenAI function-calling
shape) and `<name>.py` (the definition — a `run(raw_arguments: dict)`
function). Adding, editing, or removing a tool is just adding, editing, or
removing that file pair in one of the search directories below; nothing
else needs to change. That makes the toolset itself data-plus-code the same
way agent_template/ configs are data — including editable by the agent
itself, if it's given the ability to write files here.

discover_tools() re-reads the directories on every call rather than caching,
so a tool added or edited between calls takes effect on the next call
without restarting the process.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

from livekit.agents import function_tool

logger = logging.getLogger("livekit.tools.registry")

TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_SEARCH_DIRS = [TOOLS_DIR / "builtin"]

_REQUIRED_DECLARATION_KEYS = ("name", "parameters")


def _load_declaration(json_path: Path) -> dict:
    declaration = json.loads(json_path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED_DECLARATION_KEYS if k not in declaration]
    if missing:
        raise ValueError(f"{json_path}: declaration missing required key(s) {missing}")
    return declaration


def _load_run_function(py_path: Path, tool_name: str):
    spec = importlib.util.spec_from_file_location(f"galatea_tool_{tool_name}", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if run is None:
        raise ValueError(f"{py_path}: must define a run(raw_arguments: dict) function")
    return run


def discover_tools(search_dirs: list[Path] | None = None) -> list:
    """Scan directories for <name>.json + <name>.py pairs and build LiveKit tools.

    A pair that fails to load (bad JSON, no matching .py, no run()) is
    skipped with a logged warning rather than failing the whole agent.
    """
    tools = []
    for directory in search_dirs or DEFAULT_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for json_path in sorted(directory.glob("*.json")):
            tool_name = json_path.stem
            py_path = directory / f"{tool_name}.py"
            if not py_path.exists():
                logger.warning("Skipping tool %r: no matching %s", tool_name, py_path.name)
                continue
            try:
                declaration = _load_declaration(json_path)
                run = _load_run_function(py_path, tool_name)
                tools.append(function_tool(run, raw_schema=declaration))
            except Exception:
                logger.exception("Failed to load tool %r from %s", tool_name, directory)
    return tools

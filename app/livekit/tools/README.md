# Modular tools

A tool is a pair of files in this directory (or a subfolder registered in
`registry.py`'s `DEFAULT_SEARCH_DIRS`):

- `<name>.json` — the **declaration**: `name`, `description`, and a
  JSON-Schema `parameters` object, in OpenAI function-calling shape
  (`{"name": ..., "description": ..., "parameters": {...}}`).
- `<name>.py` — the **definition**: a single function

  ```python
  async def run(raw_arguments: dict) -> str:
      ...
  ```

  `raw_arguments` is the parsed JSON matching the declaration's
  `parameters` schema. `run` can be sync or async and should return a
  short string the LLM can read back.

`registry.discover_tools()` scans a directory for matching `<name>.json` +
`<name>.py` pairs and turns each into a LiveKit `function_tool` (via
`raw_schema`, so the JSON *is* the schema — no Python type hints or
docstring parsing involved). It re-scans on every call rather than caching,
so adding, editing, or deleting a pair takes effect on the next agent
session without restarting the process.

This is deliberately the same shape as `data/agent_template/*.json` for
agent configs: the declaration is data, kept separate from the code that
implements it. It also means a tool can be added or modified by writing
these two files — including by the agent itself, if it's later given
filesystem access to this directory.

## Example

See `builtin/send_message.json` + `builtin/send_message.py`.

## Adding a tool

1. Write `<name>.json` with `name`, `description`, `parameters`.
2. Write `<name>.py` with a `run(raw_arguments: dict)` function.
3. Drop both into `builtin/` (or another directory passed to
   `discover_tools(search_dirs=[...])`). Nothing else needs to change —
   `app/livekit/__main__.py` already wires `discover_tools()`'s output into
   the agent's tool list.

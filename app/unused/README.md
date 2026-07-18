# Unused code

Archived material that predates the current `app/galatea/` engine. Nothing in
here is imported by the running agent — it is kept for reference only. This
folder holds code that isn't currently used, so it lives under `app/` (code)
rather than at the repo root.

- `agentics/` — the previous voice-agent workspace. Its Smallest TTS/STT,
  ElevenLabs/Rime/Kokoro/Silero/HuggingFace providers, config handling, and
  intro generation were all ported into `app/galatea/`. What remains here is
  the old entrypoints (`agent.py`, `main.py`, `rime_agent.py`), experiment
  scripts, background-task prototypes, the Snowflake RAG tool, and misc notes.
- `misc/` — stray files found around the repo (e.g. a frontend Supabase
  middleware snippet that was sitting in `data/`).

The older text/Telegram system lives in `app/livekit` (formerly
`galatea_livekit`) and in git history.

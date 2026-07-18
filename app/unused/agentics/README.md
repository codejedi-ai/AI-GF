# Galatea-Atom

This repo evolves in **two ways**:

| Track | Use case | Stack |
|-------|----------|--------|
| **Original / Atom** | Outbound phone calls, Atoms API | Smallest AI **Atoms** (phone), optional Waves/Pulse |
| **LiveKit** | Real-time voice in a room | **Smallest agent**: **Waves** (TTS) + **Pulse** (STT) only |

The **LiveKit** track is the main "spinal cord" here: a LiveKit-based voice agent using **Smallest AI** — **Waves** for TTS and **Pulse** for STT — in the same pattern as Galatea-LiveKit. Single entrypoint: **`main.py`**.

## Setup

### 1. Environment
Create a `.env` in the project root:

```env
LIVEKIT_URL=wss://<your-livekit-server>.livekit.cloud
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>
OPENAI_API_KEY=<your-openai-api-key>
SMALLEST_API_KEY=<your-smallest-ai-api-key>
```

### 2. Install
```bash
pip install -r requirements.txt
# or: uv sync
```

## Running the agent

**Entrypoint:** `python main.py` (or `python agent.py` for the worker only).

### Voice agent (LiveKit room) — JSON config
Uses an agent template JSON (same shape as Galatea-LiveKit `agent_template/*.json`):

```bash
# Default config (Companion.json or SELECTED_AGENT from .env)
python main.py dev

# Or with a specific JSON config
python main.py --config agent_template/Companion.json dev
python main.py --config agent_template/Natasha.json console
```

### Voice agent — built-in voice key (no JSON file)
Uses `agent_configs.VOICE_CONFIGS` (e.g. `ludia`, `antigravity`, `dong_sook`):

```bash
python main.py --voice ludia dev
python main.py --voice antigravity console
```

### Outbound phone call (Smallest Atoms)
Place a call via Smallest AI Atoms (requires `smallestai` and `SMALLEST_API_KEY`):

```bash
python main.py call --config agent_template/Companion.json --phone +1234567890
python main.py call --voice ludia --phone +1234567890
```

## Repo evolution (Atom vs LiveKit / Smallest)

- **Original repo direction**: uses **Atom** for outbound phone (`main.py call`, `atoms_phone.py`).
- **LiveKit direction**: uses the **Smallest agent** only — **Waves** (TTS) + **Pulse** (STT) — in `agent.py` for room-based voice. No Atom dependency for the LiveKit worker.

## Structure (after dissolving "Galatea-Atoms - Uses Atom")

- **`main.py`** — Single entrypoint: run LiveKit agent or `call` (Atoms phone).
- **`agent.py`** — LiveKit agent worker: **Smallest** only — **Waves** (TTS) + **Pulse** (STT); `--config` JSON or `--voice` from `agent_configs`.
- **`atoms_phone.py`** — Smallest Atoms outbound call: load config from JSON or `--voice`.
- **`agent_configs.py`** — Optional built-in voice configs (`ludia`, `dong_sook`, `antigravity`).
- **`agent_template/*.json`** — Agent definitions (personality, TTS/STT/LLM, greeting).
- **`plugins/`** — Smallest **Waves** (TTS) and **Pulse** (STT) plugins.

## Technical details

- **TTS**: **Smallest AI Waves** (models: `lightning`, `lightning-large` / `waves_lightning_large`)
- **STT**: **Smallest AI Pulse** (32+ languages; config: `stt.mode: "pulse"`; optional `lightning` mode)
- **LLM**: OpenAI (or OpenAI-compatible `llm.url` in JSON)
- **Pattern**: Same as Galatea-LiveKit (`entrypoint`, `prewarm`, `AgentSession`, `--config` JSON).

### Waves and Pulse in config

- In `agent_template/*.json` or `--voice` configs, TTS is **Waves** (no extra field; `tts.provider: "smallestai"` uses Waves).
- STT is **Pulse** by default; set `stt.mode: "pulse"` explicitly or leave unset. Use `stt.mode: "lightning"` for Lightning STT.

## Tips

- Default agent: set `SELECTED_AGENT="name"` in `.env` to pick a default `agent_template/<name>.json`.
- Run worker only: `python agent.py dev` or `python agent.py --config agent_template/Companion.json console`.

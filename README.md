# Galatea AI — Voice Agent

A LiveKit-based voice agent where **agents are data, not code**. Each agent is a JSON file in `data/agent_template/` that defines its personality, greeting, and — per pipeline stage — which provider to use. Provider code is loaded lazily: a provider's module (and its SDK/deps) is only imported when an agent's JSON actually names it.

## Repository layout

The repo separates code and data at the top level: **code lives under `app/`,
data lives under `data/`**, and everything else that's neither (docs, papers,
the engineering notebook) is its own top-level folder.

```
app/                      CODE — everything that is imported or executed
  main.py                 Entrypoint — runs the LiveKit worker
  galatea/                The agent engine
    agent.py              LiveKit worker (config-driven, provider-agnostic)
    config.py             JSON config loading + prompt resolution
    intro.py              Optional dynamic greeting generation
    providers/            Lazy provider registry
      llm/                openai (+ OpenAI-compatible urls), google, anthropic, deepseek, inflection, huggingface
      tts/                smallestai, elevenlabs, rime, kokoro, silero, huggingface, openai
      stt/                smallestai, silero, openai
      vad/                silero
  frontend/               Web frontend (Vite/React) — self-contained, own package.json
  browser-control/        Browser-automation MCP server + agent — self-contained,
                          own pyproject.toml/venv/Dockerfile; its data/ subfolder
                          stays nested here (see note below)
  livekit/                Older LiveKit text/Telegram system
  unused/                 Code not currently wired into anything, kept for
                          reference only (formerly top-level legacy/)
    agentics/
    misc/

data/                     DATA — nothing executable lives here
  agent_template/         Agent definitions (one JSON per agent)
  prompts/                Prompt text files referenced by configs ("File Path" prompts)
  galatea/                Runtime config + character data for app/livekit
                          (formerly the repo-root .galatea/ folder)
    config.json
    SOUL.md
    SKILLS.md
    data/galatea_bus.db

docs/                     Character sheets and what it takes to run this
engineering_notebook/     Hackathon engineering notebook
papers/                   Research papers

LICENSE, README.md, requirements.txt, .env(.example)
                          Project-level meta files, stay at the repo root
```

Rules of thumb this follows:

- **`app/` is exclusively code.** Anything imported, executed, or run as a
  worker lives under `app/`. Code that isn't currently wired into anything
  (the old pre-refactor agent) lives in `app/unused/` instead of the repo
  root — it's still code, just dormant.
- **`data/` is exclusively data.** Agent JSONs, prompt text, and the
  config/character files the older `app/livekit` system reads at runtime
  (`config.json`, `SOUL.md`, `SKILLS.md`, its sqlite bus db) all live here,
  organized by subfolder per consumer (`agent_template/`, `prompts/`,
  `galatea/`) so ownership stays clear even though it's one shared tree.
- **Everything else is a peer of `app/` and `data/`** — `docs/`,
  `engineering_notebook/`, `papers/` aren't code or data in the runtime
  sense, so they keep their own top-level folders.
- **Exception:** `app/browser-control/` keeps its own `data/` subfolder
  in place rather than splitting it out to the top-level `data/`. It's a
  vendored, self-contained sibling tool with its own `pyproject.toml`,
  virtualenv, and Dockerfile that all assume a `data/` folder relative to
  its own directory — pulling that apart would break its build/run scripts
  for no real benefit. Same reasoning applies to `app/frontend/`, which
  keeps its own `node_modules`/build config untouched.

## Setup

```bash
pip install -r requirements.txt   # core deps
cp .env.example .env              # then fill in your keys
```

Only install optional provider packages for providers your agents actually use — see the comments in `requirements.txt`.

## Running

Run from the repo root, so relative `--config` paths resolve:

```bash
python app/main.py dev --config data/agent_template/Ludia.json
python app/main.py console --config data/agent_template/Natasha.json   # local console mode
python app/main.py dev                                                # SELECTED_AGENT from .env, or default
```

## Agent JSON format

```jsonc
{
  "name": "ludia",
  "is_anthropomorphic": true,              // appends the Declaration of Humanity
  "personality_prompt": "…",               // or { "type": "File Path", "content": "data/prompts/ludia.md" }
  "greeting": {
    "intro_phrase": "hey cutie...",        // static greeting (fallback)
    "intro_generation_prompt": "…",        // optional: generate the greeting each call
    "intro_generation_model": "Pi-3.1",
    "gen_temperature": 0.9
  },
  "llm": { "provider": "openai", "model": "gpt-4o-mini", "url": null },
  "tts": {
    "provider": "rime",                    // smallestai | elevenlabs | rime | kokoro | silero | huggingface | openai
    "model": "arcana",
    "voice_options": { "speaker": "celeste", "speed_alpha": 1.5 }
  },
  "stt": { "provider": "openai", "model": "gpt-4o-mini-transcribe" },  // smallestai | silero | openai
  "vad": { "provider": "silero" },
  "requirements": ["livekit-plugins-rime @ git+https://github.com/rimelabs/livekit-agents.git@bddcc0d265176bb6b3a6b32d6773e7980e08790c#subdirectory=livekit-plugins/livekit-plugins-rime"]
}
```

Notes:

- `llm.url` makes any OpenAI-compatible server work (Ollama, LM Studio, vLLM…).
- `personality_prompt` accepts a plain string or `{ "type": "File Path", "content": "data/prompts/<file>" }` — paths resolve from the repo root, keeping long prompts out of the JSON.
- Unknown providers fail fast with the list of available ones.
- `requirements` (optional array of pip specs) declares the extra packages *this agent's* providers need, beyond the always-installed core in `requirements-core.txt`. It's how a config that names, say, `"llm": "google"` says so itself instead of that being hardcoded anywhere in code.

## Adding a provider

1. Add a module under `app/galatea/providers/<stage>/` exposing `create(cfg: dict)`.
2. Register its name in the matching table in `app/galatea/providers/__init__.py`.
3. If it needs an extra pip package, note it in the `requirements` array of any agent JSON that uses it (see above).

Nothing else changes — configs can name the new provider immediately, and its imports stay lazy.

## Deploying to LiveKit Cloud

Each Cloud Agent deployment runs one image built from one `requirements.txt`,
dedicated to whichever agent config it defaults to. Before deploying (or
switching which agent a deployment runs), regenerate `requirements.txt` from
that agent's own declared `requirements`:

```bash
python app/galatea/build_requirements.py --config data/agent_template/Natasha.json
lk agent deploy .   # or `lk agent create .` for a first deploy
```

`requirements.txt` is generated — edit `requirements-core.txt` (always
installed) or the target agent's JSON `requirements` array instead of editing
`requirements.txt` by hand.

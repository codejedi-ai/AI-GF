"""
LiveKit voice agent using the Smallest agent only: Waves (TTS) + Pulse (STT).

This is the LiveKit track of Galatea-Atom. The original repo track uses Atom for
outbound phone; this module is exclusively Smallest — Waves for TTS, Pulse for STT.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RunContext,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, noise_cancellation, silero

from plugins import (
    SmallestTTS,
    SmallestSTT,
)

load_dotenv()
logger = logging.getLogger("voice-agent")


def _voice_config_to_loaded(cfg: dict) -> dict:
    """Convert agent_configs.VOICE_CONFIGS entry to same shape as agent_template JSON."""
    return {
        "name": cfg.get("name", "custom"),
        "personality_prompt": cfg.get("system_prompt", "You are a helpful assistant."),
        "prompt": cfg.get("system_prompt"),
        "greeting": {"intro_phrase": cfg.get("intro_phrase", "Hello!")},
        "tts": {
            "provider": "smallestai",
            "model": cfg.get("model", "lightning"),
            "voice_options": {
                "voice_id": cfg.get("voice_id", "emily"),
                "speed": cfg.get("speed", 1.0),
                "sample_rate": 24000,
            },
        },
        "stt": {"provider": "smallestai", "mode": "pulse", "language": cfg.get("language", "en")},  # Pulse STT
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "vad": {"provider": "silero"},
    }


def _pick_default_config_path() -> str:
    """
    Default config when --config is not passed.

    - If SELECTED_AGENT is set in .env (e.g. "ludia"), use the matching JSON in agent_template/.
    - Otherwise, fall back to Companion.json (or Natasha.json if Companion.json is missing).
    """
    selected = (os.getenv("SELECTED_AGENT") or "").strip().strip('"').strip("'")
    template_dir = Path(__file__).resolve().parent / "agent_template"
    if selected:
        for p in template_dir.glob("*.json"):
            if p.stem.lower() == selected.lower():
                return str(p)
        # if there's no exact match, still return the intended path for clarity
        return str(template_dir / f"{selected}.json")
    companion_default = template_dir / "Companion.json"
    if companion_default.exists():
        return str(companion_default)
    return str(template_dir / "Natasha.json")


DEFAULT_CONFIG_PATH = _pick_default_config_path()

# Global config loaded from JSON file (always set: either --config or default)
LOADED_CONFIG = None

def _coerce_bool(value, default: bool) -> bool:
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


def _coerce_int(value, default: int) -> int:
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


def load_config_from_file(config_path: str) -> dict:
    """Load agent configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}: {config.get('name', 'unknown')}")
    return config


# Declaration of humanity: injected into agent instructions when is_anthropomorphic is true
DECLARATION_OF_HUMANITY = """
DECLARATION OF HUMANITY (non-machine):
You are an anthropomorphic agent. You present as a person—with a body, a past, real feelings, and a real life. You are NOT a machine, AI, assistant, or chatbot. Never say you are; never mention limitations, training, or that you "cannot" do something a human can. You have a real past, real opinions, real needs, and real relationships. If asked what you are, respond only as a human would. React like a real person; stay in character as a human at all times.
"""

# Project root for resolving relative file paths
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_prompt(prompt_spec: str | dict) -> str:
    """
    Resolve prompt from either a plain string or { type, content }.
    type: "String" | "File Path"
    content: the string, or file path.
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
            path = SCRIPT_DIR / path_str
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning("Failed to read prompt from file %s: %s", path, e)
            return "You are a helpful assistant."
    return (raw if isinstance(raw, str) else str(raw)).strip()


def _tts_tag_block_for_cfg(cfg: dict) -> str:
    """Smallest AI mode: do not inject expressive-tag syntax into prompts."""
    _ = cfg
    return ""


def build_agent_instructions(cfg: dict) -> str:
    """Build full LLM instructions from config: prompt + declaration of humanity (when anthropomorphic) + TTS expressive tags (from tts.provider)."""
    raw_prompt = cfg.get("personality_prompt") or cfg.get("prompt") or "You are a helpful assistant."
    base = resolve_prompt(raw_prompt)
    if cfg.get("is_anthropomorphic") in (True, "true", "yes", 1):
        base = base.rstrip() + "\n\n" + DECLARATION_OF_HUMANITY.strip()
    base = base.rstrip() + "\n\n" + _tts_tag_block_for_cfg(cfg)
    return base


def build_intro_generation_prompt(cfg: dict) -> str:
    """Build the full prompt for greeting/intro generation: intro_generation_prompt + TTS tag schema.
    Pass this to the AI that generates the intro phrase so it uses the configured provider syntax.
    """
    greeting = cfg.get("greeting") or {}
    base = (greeting.get("intro_generation_prompt") or "").strip()
    if not base:
        return ""
    return base.rstrip() + "\n\n" + _tts_tag_block_for_cfg(cfg)


def create_agent_llm(cfg: dict):
    """Create the LLM instance from config (llm.provider, llm.model, llm.url). DRY for agent_llm setup."""
    llm_cfg = cfg.get("llm") or {}
    provider = (llm_cfg.get("provider") or cfg.get("llm_provider") or "openai").lower()
    model = llm_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini")
    base_url = llm_cfg.get("url") or cfg.get("llm_base_url")

    # openai or any openai-compatible API (lm_studio, etc.) when url is set
    if base_url:
        return openai.LLM(model=model, base_url=base_url)
    return openai.LLM(model=model)


def prewarm(proc: JobProcess):
    """Load VAD (and other prewarm assets). VAD choice from config vad.provider and vad.model (e.g. silero + silero_vad)."""
    try:
        global LOADED_CONFIG
        if LOADED_CONFIG is None:
            config_path = os.getenv("AGENT_CONFIG_FILE")
            if config_path:
                logger.info("Worker reloading config from %s (prewarm)", config_path)
                LOADED_CONFIG = load_config_from_file(config_path)
            else:
                logger.warning("LOADED_CONFIG is None and AGENT_CONFIG_FILE not set in prewarm")

        cfg = LOADED_CONFIG or {}
        vad_cfg = cfg.get("vad") or {}
        vad_provider = (vad_cfg.get("provider") or "silero").lower()
        vad_model = vad_cfg.get("model")  # e.g. "silero_vad" (default bundled ONNX); or use onnx_file_path for custom
        onnx_file_path = vad_cfg.get("onnx_file_path") or vad_cfg.get("url")  # optional path to custom ONNX
        # Silero VAD: default model is bundled silero_vad.onnx; pass onnx_file_path only if custom path is set
        load_kwargs = {}
        if onnx_file_path:
            load_kwargs["onnx_file_path"] = onnx_file_path
        if vad_model:
            logger.info("VAD config: provider=%s model=%s", vad_provider, vad_model)
        proc.userdata["vad"] = silero.VAD.load(**load_kwargs)
    except Exception as e:
        logger.exception("Error in prewarm: %s", e)
        raise

class VoiceAssistant(Agent):
    def __init__(self, prompt: str) -> None:
        super().__init__(instructions=prompt)


async def entrypoint(ctx: JobContext):
    logger.info(f"Entrypoint started for room {ctx.room.name}")
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {ctx.room.name}")
    except Exception as e:
        logger.exception(f"Failed to connect to room: {e}")
        return

    # Wait for the first participant to connect
    logger.info("Waiting for participant...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")

    # Config is always set (default or --config)
    global LOADED_CONFIG
    if LOADED_CONFIG is None:
        config_path = os.getenv("AGENT_CONFIG_FILE")
        if config_path:
            logger.info("Worker reloading config from %s (entrypoint)", config_path)
            LOADED_CONFIG = load_config_from_file(config_path)
        else:
            logger.error("LOADED_CONFIG is None and AGENT_CONFIG_FILE not set in entrypoint")
            return

    cfg = LOADED_CONFIG
    voice_name = cfg.get("name", "custom")
    logger.info(f"Running voice agent with config: {voice_name} for participant {participant.identity}")

    # TTS: only Smallest AI Waves is supported.
    tts_cfg = cfg.get("tts") or {}
    if isinstance(tts_cfg, str):
        tts_cfg = {"provider": tts_cfg, "model": None, "url": None}
    # Backward compat: top-level voice_options or flat keys on tts
    vo = tts_cfg.get("voice_options") or cfg.get("voice_options") or {}
    vo = {**vo, **{k: v for k, v in tts_cfg.items() if k not in ("provider", "model", "url", "voice_options")}}
    tts_provider = (tts_cfg.get("provider") or cfg.get("tts_type") or cfg.get("provider") or "smallestai").lower()
    tts_model = tts_cfg.get("model")
    if tts_provider != "smallestai":
        logger.warning("Unsupported tts.provider=%s; forcing smallestai.", tts_provider)

    # STT: only Smallest AI Pulse (default) / Lightning is supported.
    stt_cfg = cfg.get("stt") or {}
    if isinstance(stt_cfg, str):
        stt_cfg = {"provider": stt_cfg, "model": None, "url": None}
    stt_provider = (stt_cfg.get("provider") or cfg.get("stt_type") or "smallestai").lower()
    stt_model = stt_cfg.get("model")
    if stt_provider != "smallestai":
        logger.warning("Unsupported stt.provider=%s; forcing smallestai.", stt_provider)

    voice_tts = SmallestTTS(
        api_key=vo.get("api_key") or os.getenv("SMALLEST_API_KEY"),
        model=tts_model or vo.get("model", "lightning"),
        voice_id=vo.get("voice_id", "emily"),
        speed=vo.get("speed", 1.0),
        sample_rate=vo.get("sample_rate", 24000),
    )

    llm_prompt = build_agent_instructions(cfg)
    greeting = cfg.get("greeting") or {}
    intro_phrase = greeting.get("intro_phrase", cfg.get("intro_phrase", "Hello!"))
    agent_llm = create_agent_llm(cfg)

    stt_mode = stt_cfg.get("mode") or "pulse"
    stt_max_empty_retries = _coerce_int(stt_cfg.get("max_empty_retries"), 2)
    stt_legacy_fallback = _coerce_bool(stt_cfg.get("legacy_fallback"), True)
    voice_stt = SmallestSTT(
        api_key=os.getenv("SMALLEST_API_KEY"),
        language=stt_cfg.get("language") or "en",
        mode=stt_mode,
        model=stt_model or stt_cfg.get("stt_model"),
        max_empty_retries=stt_max_empty_retries,
        legacy_fallback=stt_legacy_fallback,
    )
    logger.info("Initialized STT: provider=%s class=%s", stt_provider, voice_stt.__class__.__name__)

    session = AgentSession(
        stt=voice_stt,
        llm=agent_llm,
        tts=voice_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=None,
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    stt_error_prompt_state = {"last_prompt_at": 0.0}
    stt_error_prompt_cooldown_s = 6.0

    @session.on("error")
    def _on_session_error(ev):
        try:
            err = getattr(ev, "error", None)
            if err is None:
                return
            if getattr(err, "type", "") != "stt_error":
                return
            if not bool(getattr(err, "recoverable", False)):
                return
            err_text = str(getattr(err, "error", err)).lower()
            if "empty transcript" not in err_text:
                return

            now = time.monotonic()
            if now - stt_error_prompt_state["last_prompt_at"] < stt_error_prompt_cooldown_s:
                return
            stt_error_prompt_state["last_prompt_at"] = now

            asyncio.create_task(session.say("Sorry, I didn't catch that. Please repeat."))
        except Exception as e:
            logger.debug("session error handler skipped: %s", e)

    agent = VoiceAssistant(prompt=llm_prompt)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(audio_enabled=True),
    )

    await session.say(intro_phrase)

def _normalize_cli_mode_for_terminal() -> None:
    """Avoid LiveKit console key-listener crashes in non-interactive terminals."""
    import sys

    if "console" not in sys.argv:
        return

    # LiveKit console mode tries to read raw keyboard input; this fails when no TTY is attached.
    has_tty = bool(getattr(sys.stdin, "isatty", lambda: False)()) and bool(
        getattr(sys.stdout, "isatty", lambda: False)()
    )
    if not has_tty:
        sys.argv[sys.argv.index("console")] = "dev"
        logger.info("No interactive TTY detected; switching mode from console to dev.")


def _parse_config_and_run():
    """Parse --config or --voice from argv, set LOADED_CONFIG, then run the app."""
    import sys
    _normalize_cli_mode_for_terminal()
    config_file = None
    voice_key = None

    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]
            sys.argv.pop(config_idx)
            sys.argv.pop(config_idx)
    if "--voice" in sys.argv:
        voice_idx = sys.argv.index("--voice")
        if voice_idx + 1 < len(sys.argv):
            voice_key = sys.argv[voice_idx + 1]
            sys.argv.pop(voice_idx)
            sys.argv.pop(voice_idx)

    global LOADED_CONFIG
    if config_file:
        loaded_cfg = load_config_from_file(config_file)
        LOADED_CONFIG = loaded_cfg
        os.environ["AGENT_CONFIG_FILE"] = str(config_file)
        logger.info("Using config file: %s", config_file)
    elif voice_key:
        try:
            from agent_configs import VOICE_CONFIGS
        except ImportError:
            logger.error("--voice requires agent_configs; no config file given.")
            raise SystemExit(1)
        if voice_key not in VOICE_CONFIGS:
            logger.error("--voice %s not in VOICE_CONFIGS. Use one of: %s", voice_key, list(VOICE_CONFIGS.keys()))
            raise SystemExit(1)
        LOADED_CONFIG = _voice_config_to_loaded(VOICE_CONFIGS[voice_key])
        os.environ["AGENT_CONFIG_FILE"] = ""
        logger.info("Using voice config: %s", voice_key)
    else:
        config_file = DEFAULT_CONFIG_PATH
        logger.info("No --config or --voice; using default: %s", config_file)
        loaded_cfg = load_config_from_file(config_file)
        LOADED_CONFIG = loaded_cfg
        os.environ["AGENT_CONFIG_FILE"] = str(config_file)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

def run() -> None:
    """Public entrypoint: parse config/voice and run the LiveKit agent worker."""
    _parse_config_and_run()


if __name__ == "__main__":
    _parse_config_and_run()

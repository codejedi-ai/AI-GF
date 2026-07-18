"""
LiveKit voice agent worker.

Fully config-driven: the agent's personality, greeting, and every pipeline
stage (LLM, TTS, STT, VAD) come from a JSON file in data/agent_template/.
Provider code is imported lazily by the provider registry, so only the
providers named in the config are ever loaded.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import noise_cancellation

from galatea import config as cfg_mod
from galatea import providers

load_dotenv()
logger = logging.getLogger("galatea.agent")

# Config loaded from JSON (always set before the worker starts: --config or default).
LOADED_CONFIG: dict | None = None


def _ensure_config_loaded(where: str) -> dict | None:
    """Worker subprocesses reload the config from AGENT_CONFIG_FILE."""
    global LOADED_CONFIG
    if LOADED_CONFIG is None:
        config_path = os.getenv("AGENT_CONFIG_FILE")
        if config_path:
            logger.info("Worker reloading config from %s (%s)", config_path, where)
            LOADED_CONFIG = cfg_mod.load_config_from_file(config_path)
        else:
            logger.error("LOADED_CONFIG is None and AGENT_CONFIG_FILE not set in %s", where)
    return LOADED_CONFIG


def prewarm(proc: JobProcess):
    """Load VAD (and other prewarm assets) per the config's vad section."""
    try:
        cfg = _ensure_config_loaded("prewarm") or {}
        proc.userdata["vad"] = providers.create_vad(cfg_mod.section(cfg, "vad"))
    except Exception as e:
        logger.exception("Error in prewarm: %s", e)
        raise


class VoiceAssistant(Agent):
    def __init__(self, prompt: str) -> None:
        super().__init__(instructions=prompt)


async def _resolve_intro(cfg: dict) -> str:
    """Greeting: generate dynamically when configured, else the static intro_phrase."""
    greeting = cfg.get("greeting") or {}
    fallback = greeting.get("intro_phrase", cfg.get("intro_phrase", "Hello!"))
    gen_prompt = cfg_mod.build_intro_generation_prompt(cfg)
    if not gen_prompt:
        return fallback
    try:
        from galatea.intro import generate_intro

        generated = await generate_intro(
            gen_prompt,
            model=greeting.get("intro_generation_model"),
            temperature=greeting.get("gen_temperature", 0.9),
        )
        if generated:
            return generated
    except Exception as e:
        logger.warning("Intro generation failed, using intro_phrase: %s", e)
    return fallback


async def entrypoint(ctx: JobContext):
    logger.info("Entrypoint started for room %s", ctx.room.name)
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("Connected to room: %s", ctx.room.name)
    except Exception as e:
        logger.exception("Failed to connect to room: %s", e)
        return

    logger.info("Waiting for participant...")
    participant = await ctx.wait_for_participant()
    logger.info("Participant connected: %s", participant.identity)

    cfg = _ensure_config_loaded("entrypoint")
    if cfg is None:
        return

    logger.info(
        "Running voice agent %r for participant %s", cfg.get("name", "custom"), participant.identity
    )

    # Every pipeline stage is built from its config section; only the named
    # providers get imported.
    voice_tts = providers.create_tts(cfg_mod.section(cfg, "tts"))
    voice_stt = providers.create_stt(cfg_mod.section(cfg, "stt"))
    agent_llm = providers.create_llm(cfg_mod.section(cfg, "llm"))
    logger.info(
        "Pipeline: llm=%s tts=%s stt=%s",
        agent_llm.__class__.__name__,
        voice_tts.__class__.__name__,
        voice_stt.__class__.__name__,
    )

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

    agent = VoiceAssistant(prompt=cfg_mod.build_agent_instructions(cfg))

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        room_output_options=RoomOutputOptions(audio_enabled=True),
    )

    await session.say(await _resolve_intro(cfg))


def _normalize_cli_mode_for_terminal() -> None:
    """Avoid LiveKit console key-listener crashes in non-interactive terminals."""
    if "console" not in sys.argv:
        return

    # LiveKit console mode reads raw keyboard input; that fails without a TTY.
    has_tty = bool(getattr(sys.stdin, "isatty", lambda: False)()) and bool(
        getattr(sys.stdout, "isatty", lambda: False)()
    )
    if not has_tty:
        sys.argv[sys.argv.index("console")] = "dev"
        logger.info("No interactive TTY detected; switching mode from console to dev.")


def run() -> None:
    """Parse --config from argv, load the agent JSON, and run the LiveKit worker."""
    _normalize_cli_mode_for_terminal()
    config_file = None

    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]
            sys.argv.pop(config_idx)
            sys.argv.pop(config_idx)

    global LOADED_CONFIG
    if not config_file:
        config_file = cfg_mod.default_config_path()
        logger.info("No --config given; using default: %s", config_file)
    LOADED_CONFIG = cfg_mod.load_config_from_file(config_file)
    # Worker subprocesses don't inherit module globals; pass the path via env.
    os.environ["AGENT_CONFIG_FILE"] = str(config_file)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )


if __name__ == "__main__":
    run()

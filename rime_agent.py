import logging
import random
import os
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    tts,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import (
    openai,
    noise_cancellation,
    rime,
    silero,
    tavus,
)
from livekit.agents.tokenize import tokenizer

from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent_configs import VOICE_CONFIGS

load_dotenv()
logger = logging.getLogger("voice-agent")

VOICE_NAMES = ["celeste"]
# randomly select a voice from the list
VOICE = random.choice(VOICE_NAMES)

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class RimeAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=VOICE_CONFIGS[VOICE]["llm_prompt"])


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()

    logger.info(f"Running Rime voice agent for voice config {VOICE} and participant {participant.identity}")

    rime_tts = rime.TTS(
        **VOICE_CONFIGS[VOICE]["tts_options"]
    )
    if VOICE_CONFIGS[VOICE].get("sentence_tokenizer"):
        sentence_tokenizer = VOICE_CONFIGS[VOICE].get("sentence_tokenizer")
        if not isinstance(sentence_tokenizer, tokenizer.SentenceTokenizer):
            raise TypeError(
                f"Expected sentence_tokenizer to be an instance of tokenizer.SentenceTokenizer, got {type(sentence_tokenizer)}"
            )
        rime_tts = tts.StreamAdapter(tts=rime_tts, sentence_tokenizer=sentence_tokenizer)

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=rime_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel()
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    persona_id = os.getenv("TAVUS_PERSONA_ID")
    replica_id = os.getenv("TAVUS_REPLICA_ID")

    # --- Tavus integration ---
    avatar = tavus.AvatarSession(
        replica_id=replica_id,      # Replace with your actual replica ID
        persona_id=persona_id,      # Replace with your actual persona ID
        # Optional: avatar_participant_name="Tavus-avatar-agent"
    )
    await avatar.start(session, room=ctx.room)
    # -------------------------

    await session.start(
        room=ctx.room,
        agent=RimeAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=False  # Tavus handles audio separately
        )
    )

    await session.say(VOICE_CONFIGS[VOICE]["intro_phrase"])

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )

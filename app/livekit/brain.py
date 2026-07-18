"""
Text-channel brain: turns InboundMessage from non-voice channels (telegram,
cli) into an LLM reply and publishes it back as OutboundMessage.

The voice channel already gets its replies from its own live AgentSession
turn-taking (see __main__.py's entrypoint) and is excluded here — otherwise
a voice turn would get answered twice.
"""
from __future__ import annotations

import logging
import os

from livekit.agents.llm import ChatContext

from app.livekit.bus.events import InboundMessage, OutboundMessage
from app.livekit.bus.queue import MessageBus
from app.livekit.llm_select import build_llm
from app.livekit.media import synthesize_voice_reply, transcribe_voice_note
from app.livekit.workspace_loader import load_workspace

logger = logging.getLogger("livekit.brain")

VOICE_SOURCE = "voice"


async def _generate_reply(cfg: dict, system_prompt: str, user_text: str) -> str:
    llm = build_llm(cfg)
    ctx = ChatContext.empty()
    ctx.add_message(role="system", content=system_prompt)
    ctx.add_message(role="user", content=user_text)
    response = await llm.chat(chat_ctx=ctx).collect()
    return response.text.strip()


async def _handle_message(bus: MessageBus, msg: InboundMessage) -> None:
    workspace = load_workspace()
    cfg = workspace["config"]
    system_prompt = f"{workspace['soul']}\n\nCORE SKILLS AND CAPABILITIES:\n{workspace['skills']}"

    user_text = msg.text
    if msg.media_type == "voice" and msg.media_url:
        user_text = await transcribe_voice_note(msg.media_url)

    if not user_text:
        return

    reply_text = await _generate_reply(cfg, system_prompt, user_text)
    outbound = OutboundMessage(
        target=msg.source,
        user_id=msg.user_id,
        chat_id=msg.chat_id,
        text=reply_text,
    )

    if msg.media_type == "voice":
        voice_id = cfg.get("voice_id") or os.getenv("ELEVEN_VOICE_ID", "95XPUDALaQL1LY3I023E")
        try:
            outbound.payload["audio_bytes"] = await synthesize_voice_reply(reply_text, voice_id)
            outbound.media_type = "voice"
        except Exception:
            logger.exception("Voice reply synthesis failed for chat_id=%s; sending text only", msg.chat_id)

    await bus.publish_outbound(outbound)


async def run_brain(bus: MessageBus) -> None:
    """Consume inbound messages from text channels and publish LLM replies."""
    async for msg in bus.subscribe_inbound():
        if msg.source == VOICE_SOURCE:
            continue
        try:
            await _handle_message(bus, msg)
        except Exception:
            logger.exception("Failed to handle inbound message from %s", msg.source)

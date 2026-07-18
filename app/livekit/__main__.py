# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "livekit-agents",
#     "livekit-plugins-elevenlabs",
#     "livekit-plugins-openai",
#     "livekit-plugins-anthropic",
#     "livekit-plugins-google",
#     "livekit-plugins-silero",
#     "livekit-plugins-noise-cancellation",
#     "python-dotenv",
#     "requests",
#     "aiohttp",
#     "aiogram",
# ]
# ///

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    FunctionTool,
    JobContext,
    ModelSettings,
    cli,
    function_tool,
)
from livekit.plugins import silero

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.livekit.providers import ElevenLabsTTS, ElevenLabsSTT
from app.livekit.bus.events import InboundMessage, OutboundMessage
from app.livekit.bus.queue import MessageBus
from app.livekit.llm_select import build_llm
from app.livekit.tools.registry import discover_tools
from app.livekit.workspace_loader import load_workspace

# Configure logging
logger = logging.getLogger("galatea-voice-agent")
logger.setLevel(logging.INFO)

load_dotenv()

class GalateaVoiceAgent(Agent):
    def __init__(self, instructions: str, tools: list[FunctionTool]) -> None:
        super().__init__(instructions=instructions, tools=tools)

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"Starting Galatea Voice Session: {ctx.room.name}")
    
    # 1. Load Workspace
    workspace = load_workspace()
    cfg = workspace["config"]
    
    system_prompt = f"{workspace['soul']}\n\nCORE SKILLS AND CAPABILITIES:\n{workspace['skills']}"
    
    agent_name = cfg.get("name", "Natasha")

    # 2. Data Structures: Queues for Bus Integration
    bus = MessageBus()
    chat_id = f"voice_{ctx.room.name}"

    # 3. Tools for Agent to interact with the Galatea Ecosystem
    @function_tool(description=f"Send a physical or system command to {agent_name}'s processing loop.")
    async def command_body(
        text: Annotated[str, "The command or intent to send to the internal system"]
    ) -> str:
        logger.info(f"Tool Call: command_body text='{text}'")
        msg = InboundMessage(
            source="voice",
            user_id="user",
            chat_id=chat_id,
            text=text
        )
        await bus.publish_inbound(msg)
        return "Command sent to system bus."

    # Modular tools: each is a <name>.json declaration + <name>.py definition
    # under app/livekit/tools/ (see registry.py). Re-scanned every session so
    # a tool added or edited on disk is picked up without a restart.
    modular_tools = discover_tools()
    agent_tools = [command_body, *modular_tools]

    # 4. Initialize Agent & Session
    agent = GalateaVoiceAgent(
        instructions=system_prompt,
        tools=agent_tools,
    )

    # Use ElevenLabs for BOTH TTS and STT
    voice_id = cfg.get("voice_id") or os.getenv("ELEVEN_VOICE_ID", "95XPUDALaQL1LY3I023E")
    llm = build_llm(cfg)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=ElevenLabsSTT(), 
        llm=llm,
        tts=ElevenLabsTTS(voice_id=voice_id),
        tools=agent_tools
    )

    # 5. Listen for Outbound Queue (Responses from System)
    async def _listen_outbound():
        async for msg in bus.subscribe_outbound("voice"):
            if msg.chat_id == chat_id:
                # 1. Strip bracketed/narrated text: *smiles*, (laughs), etc.
                text = re.sub(r'[*({].*?[*)}]', '', msg.text)
                
                # 2. Strip narration patterns only (word + punctuation at start of text)
                # This leaves "That makes me smile" but removes "Smiles. That makes me..."
                narration_pattern = r'^(Chuckles|Sighs|Smiles|Nods|Laughs|Exhales)[.!?,]\s*'
                text = re.sub(narration_pattern, '', text, flags=re.IGNORECASE)
                
                filtered_text = text.strip()
                if filtered_text:
                    logger.info(f"Speaking outbound message (filtered): {filtered_text}")
                    await session.say(filtered_text)

    asyncio.create_task(_listen_outbound())

    # 6. Bridge Speech Commitment directly to Inbound Queue
    @session.on("user_speech_committed")
    def _on_speech(msg):
        text = msg.text.strip()
        if text:
            asyncio.create_task(bus.publish_inbound(InboundMessage(
                source="voice",
                user_id="user",
                chat_id=chat_id,
                text=text
            )))

    await session.start(agent, room=ctx.room)
    
    # Ready and listening without robotic greeting
    logger.info(f"{agent_name} is now connected and listening.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("console")
    cli.run_app(server)

"""
One-shot speech helpers for text channels (Telegram voice notes), as
opposed to the realtime streaming STT/TTS used by the live voice pipeline
in __main__.py's AgentSession. A voice note is a single finished audio
file, so a plain request/response call is simpler and more reliable here
than adapting the streaming plugin classes.
"""
from __future__ import annotations

import os

import httpx
from openai import AsyncOpenAI

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


async def transcribe_voice_note(media_url: str) -> str:
    """Download a Telegram voice-note URL and transcribe it via OpenAI."""
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(media_url)
        resp.raise_for_status()
        audio_bytes = resp.content

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    transcription = await client.audio.transcriptions.create(
        file=("voice.ogg", audio_bytes),
        model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
    )
    return transcription.text.strip()


async def synthesize_voice_reply(text: str, voice_id: str) -> bytes:
    """Synthesize `text` to MP3 bytes via the ElevenLabs REST API."""
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
            },
        )
        resp.raise_for_status()
        return resp.content

"""
Run the Telegram channel together with the text-channel brain.

    python app/livekit/run_telegram.py

Requires TELEGRAM_BOT_TOKEN in .env (get one from @BotFather on Telegram).
Uses the same data/galatea/config.json persona and LLM provider/model as
the voice agent.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit.run_telegram")


async def main() -> None:
    from app.livekit.bus.queue import MessageBus
    from app.livekit.channels.telegram import TelegramChannel
    from app.livekit.brain import run_brain

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set — add it to .env (get one from @BotFather).")

    bus = MessageBus()
    channel = TelegramChannel(token=token, bus=bus)

    logger.info("Starting Telegram channel + brain...")
    await asyncio.gather(
        channel.start(),
        run_brain(bus),
    )


if __name__ == "__main__":
    asyncio.run(main())

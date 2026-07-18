"""Definition for the send_message tool (declared in send_message.json)."""
from app.livekit.bus.events import OutboundMessage
from app.livekit.bus.queue import MessageBus

_bus: MessageBus | None = None


def _get_bus() -> MessageBus:
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus


async def run(raw_arguments: dict) -> str:
    target = raw_arguments["target"]
    user_id = raw_arguments["user_id"]
    chat_id = raw_arguments["chat_id"]
    text = raw_arguments["text"]

    await _get_bus().publish_outbound(
        OutboundMessage(target=target, user_id=user_id, chat_id=chat_id, text=text)
    )
    return f"Message queued for {target}:{chat_id}"

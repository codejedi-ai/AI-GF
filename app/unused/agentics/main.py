"""
Galatea-Atom: single entrypoint.

LiveKit track (Smallest agent: Waves TTS + Pulse STT):
  python main.py [dev|console] --config agent_template/Companion.json
  python main.py --config agent_template/Companion.json dev

Atom track (outbound phone via Smallest Atoms):
  python main.py call --config agent_template/Companion.json --phone +1234567890
  python main.py call --voice ludia --phone +1234567890
"""
import asyncio
import logging
import sys
from pathlib import Path

# Run from project root so imports resolve (agent, plugins, agent_configs, atoms_phone)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("galatea-atom")


def _run_call(argv: list[str]) -> None:
    """Parse call --config/--voice --phone and run atoms_phone."""
    import argparse
    from atoms_phone import ensure_agent_and_call

    parser = argparse.ArgumentParser(prog="main.py call", description="Smallest Atoms outbound call")
    parser.add_argument("--config", help="Path to agent JSON config")
    parser.add_argument("--voice", help="Voice key from agent_configs (e.g. ludia, antigravity)")
    parser.add_argument("--phone", required=True, help="Target phone number")
    args, _ = parser.parse_known_args(argv)

    asyncio.run(
        ensure_agent_and_call(
            config_path=args.config or None,
            voice_key=args.voice or None,
            phone_number=args.phone,
        )
    )


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0].lower() == "call":
        _run_call(argv[1:])
        return

    # Run LiveKit agent (pattern of rime_agent): --config optional, then dev/console/etc.
    import agent as agent_module
    agent_module.run()


if __name__ == "__main__":
    main()

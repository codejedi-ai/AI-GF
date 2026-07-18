"""
Galatea voice agent — single entrypoint.

Run a LiveKit voice agent defined by a JSON config in data/agent_template/
(run from the repo root, so relative --config paths resolve):

  python app/main.py dev --config data/agent_template/Ludia.json
  python app/main.py console --config data/agent_template/Natasha.json
  python app/main.py dev                 # uses SELECTED_AGENT from .env, or the default template

The JSON config chooses every provider (llm/tts/stt/vad); only the providers
named in the config are imported.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    from galatea.agent import run

    run()


if __name__ == "__main__":
    main()

# Galatea LiveKit Agent

## Project Summary

This project is a Python-based, real-time conversational AI agent system built on top of [LiveKit](https://livekit.io/). It enables hyper-realistic, character-driven voice agents that can join LiveKit audio rooms, respond to users in natural language, and speak with expressive, customizable voices. The system leverages **ElevenLabs** for TTS, **Silero** for VAD, and OpenAI LLMs (large language models).

---

## Folder Structure

```
Galatea-LiveKit/
│
├── .env                  # Environment variables (API keys, URLs)
├── galatea_agent.py      # Main agent logic and entrypoint (Galatea Assistant)
├── requirements.txt      # Python dependencies
├── README.md             # Basic project info
├── tools/                # Agent tools (File management, Snowflake RAG)
└── plugins/              # Specialized plugins (ElevenLabs TTS)
```

**Configurations and Memories:**
All agent configurations (templates) and personality definitions are stored in the `~/.galatea/` directory on your machine.

---

## Key Components

### 1. `galatea_agent.py`

- Main entry point for the agent.
- Handles LiveKit room connection, session management, plugin integration, and event loop.
- Integrates TTS (ElevenLabs), LLM (OpenAI/Gemini/etc), STT (OpenAI/Whisper), and VAD (Silero).

### 2. File Management Tools

- The agent can list, read, and edit files in the workspace via OS-agnostic tools in `tools/file_tools.py`.

---

## Setup & Installation

### 1. Create and Activate a Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables & API Keys

Create a `.env` file in the project root with the following keys:

```env
LIVEKIT_URL=wss://<your-livekit-server>.livekit.cloud
LIVEKIT_API_KEY=<your-livekit-api-key>
LIVEKIT_API_SECRET=<your-livekit-api-secret>

OPENAI_API_KEY=<your-openai-api-key>
ELEVEN_API_KEY=<your-elevenlabs-api-key>
```

---

## Running the Agent

### Start the Natasha Agent (Windows)

```powershell
./run_natasha.ps1
```

---

## Technical Notes

- **TTS**: Exclusively uses **ElevenLabs**.
- **VAD**: Exclusively uses **Silero VAD**.
- **Tools**: Includes OS-agnostic file management (list, read, edit).
- **Configurations**: Located in `~/.galatea/agent_template/`.

---

## Troubleshooting

- **Python not found**: Ensure you are using the virtual environment's Python. The `run_natasha.ps1` script is pre-configured to use `.venv\Scripts\python.exe`.
- **API Quota**: Check your ElevenLabs and OpenAI billing/usage.

---

_This project has been specialized for the Galatea-LiveKit ecosystem._

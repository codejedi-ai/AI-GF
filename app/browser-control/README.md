# GitHub Release Navigator (Service 2)

A robust, vision-based autonomous agent designed to navigate GitHub and extraction semantic release information without relying on fragile CSS selectors.

## 🚀 Overview

The **GitHub Release Navigator** solves the "brittle scraper" problem. Instead of looking for specific classes or IDs (which often change), this agent uses **Computer Vision (GPT-4o)** to "see" the page and **Reinforcement Learning (RL) heuristics** to "decide" where to click. It runs on a local **Model Context Protocol (MCP)** server that controls a Playwright browser instance.

### Key Capabilities
- **Resilient Navigation**: Finds "Releases" buttons even if the DOM layout changes, by recognizing visual cues (buttons, links, layout).
- **Intelligent Planning**: Uses an A* search algorithm (Lesser/Greater A*) to score potential actions based on a weighted prediction model specific to the navigation goal.
- **Vision-Augmented Extraction**: Extracts specific data (Version, Commit Hash, Author) by visually analyzing the release card and correlating it with the HTML structure.
- **Natural Language Control**: Can act on prompts like "Find the latest release for pytorch/pytorch".

---

## 🏗️ Architectural Approach: The "Split-Brain" Model

Rather than using a single monolithic LLM for all tasks, we designed a **biologically-inspired architecture** that separates perception from reasoning:

| Component | Model | Role |
|-----------|-------|------|
| **Eyes** | **GPT-4o (Vision)** | Visual perception and page description. |
| **Brain** | **Claude 3.5 Sonnet** | Strategic reasoning, goal-checking, heuristic generation. |
| **Spinal Cord** | **MCP Server + Playwright** | Low-level browser actions, DOM queries, semantic element filtering. |

### Why This Design?
1.  **GPT-4o excels at visual understanding**: It accurately identifies UI elements, badges ("Latest"), and page structure from screenshots.
2.  **Claude excels at reasoning under constraints**: It generates effective heuristics and makes strategic navigation decisions.
3.  **MCP (Model Context Protocol)**: Provides fast, reliable browser control, handling the "muscle memory" of clicking, typing, and querying the DOM without expensive LLM calls.

---

## 🔑 Key Innovations

### 1. Prediction Model (Upfront Heuristic Generation)
**Problem**: Traditional agents call the LLM for every link evaluation, leading to O(n) API costs per page.
**Solution**: At the start of navigation, the Brain generates a **Weighted Bag-of-Words** model. This allows the agent to score all links **locally** using fast string matching, drastically reducing API costs.

### 2. Dual A* Search Strategy
*   **Lesser A* (DOM Searcher)**: Fast candidate retrieval using CSS selectors.
*   **Greater A* (Page Searcher)**: Semantic scoring using `Score = WordValue × Count × FontCoefficient`. The **Font Coefficient** assigns higher value to prominent elements (e.g., `<h1>` = 3.0), mimicking how humans prioritize visually prominent elements.

### 3. Self-Supervised Q-Learning
The agent maintains a **Knowledge Base** (Q-Table) that persists across runs. On subsequent runs for the same repository, the agent immediately prioritizes known-successful paths, bypassing exploration entirely.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- **OpenAI API Key (Required for Vision)**
- **Anthropic API Key (Required for Strategic Reasoning/Brain)**

### Steps

1.  **Clone the repository** (if not already done).

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright Browsers**:
    ```bash
    playwright install chromium
    ```

4.  **Environment Setup**:
    Create a `.env` file in the `service-2` directory. **Both keys are mandatory**:
    ```ini
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-... 
    VISION_MODEL=gpt-4o
    ```

## run Usage

### Basic CLI
Run the navigator for a specific repository:

```bash
python navigate.py --repo owner/repo
```

**Example:**
```bash
python navigate.py --repo openclaw/openclaw
```

### Advanced Usage

**Using a Custom Start URL & Prompt:**
```bash
python navigate.py --url "https://github.com/explore" --prompt "Find the trending python repo and get its latest release"
```

**Visual vs Headless:**
Currently defaults to **Headed** mode (you will see the browser open) for debugging purposes. To change this, modify `mcp_server.py` line 135: `headless=True`.

## 📂 Project Structure

| File | Description |
|------|-------------|
| `navigate.py` | **Main Entry Point**. Runs the autonomous agent loop. |
| `mcp_server.py` | **Tool Layer**. The MCP server interacting with Playwright. |
| `vision_helper.py`| **AI Layer**. Handlers for GPT-4o/Claude vision & logic. |
| `knowledge.py` | **Memory**. Q-learning table implementation. |
| `page_tracker.py` | Utils for tracking page state and history. |
| `config.py` | Configuration constants and environment loading. |
| `debug_artifacts/`| Stores screenshots and DOM dumps during runs. |
| `data/` | Output directory for results (`output.json`) and cost logs. |

## 🧪 Testing

Run the test suite to verify components:

```bash
pytest tests/
```

## 🤝 Contribution

1.  Fork the Project
2.  Create your Feature Branch
3.  Commit your Changes
4.  Push to the Branch
5.  Open a Pull Request

---
*Built with ❤️ by the Advanced Agentic Coding Team*
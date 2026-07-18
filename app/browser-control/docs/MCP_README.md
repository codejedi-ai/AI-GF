# GitHub Navigator MCP Server

This MCP (Model Context Protocol) server exposes browser navigation "limbs" as tools that AI models can use.

## Architecture

The system is separated into three layers:

### 🧠 **Brain (Claude/GPT-4)**
- Makes strategic decisions
- Calculates heuristics for A* search
- Determines when goal is reached
- Lives in: `vision_helper.py`

### 👁️ **Eyes (GPT-4o Vision)**
- Analyzes screenshots
- Provides visual descriptions to Brain
- Lives in: `vision_helper.py`

### 🦾 **Limbs (Browser + A* - THIS MCP SERVER)**
- Controls browser via Playwright
- Provides primitives: navigate, get links, screenshot, etc.
- A* navigation logic
- Exposed via MCP tools

## MCP Tools Available

### `get_page_state`
Get everything about current page in one call:
- Current URL
- HTML content (truncated)
- All links with text
- Base64 screenshot

**Usage:**
```json
{
  "include_screenshot": true,
  "html_max_length": 20000
}
```

### `navigate_to_url`
Navigate to a URL
```json
{
  "url": "https://github.com/openclaw/openclaw"
}
```

### `get_current_url`
Get current page URL

### `get_page_html`
Get HTML of current page
```json
{
  "max_length": 50000
}
```

### `get_all_links`
Get all href links from page
```json
{
  "include_text": true
}
```

### `get_screenshot`
Capture screenshot
```json
{
  "full_page": false
}
```

### `click_element`
Click an element by CSS selector
```json
{
  "selector": "#releases-tab"
}
```

## Setup

### 1. Install MCP SDK
```bash
pip install mcp
```

### 2. Run the MCP Server Standalone
```bash
python mcp_server.py
```

### 3. Integrate with Claude Desktop

Copy the contents of `mcp_config.json` to your Claude Desktop config:
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`

Then restart Claude Desktop.

### 4. Use from Claude

Claude can now use these tools! Example prompt:

```
Use the github-navigator tools to:
1. Navigate to https://github.com/openclaw/openclaw
2. Get the page state
3. Find and navigate to the latest release
4. Extract the version, author, and release notes
```

## How It Works

```
┌─────────────────────────────────────┐
│  Claude Desktop / AI Model          │
│  (Brain - Strategic Decisions)      │
└──────────────┬──────────────────────┘
               │ MCP Protocol
               │ (JSON-RPC)
               ▼
┌─────────────────────────────────────┐
│  MCP Server (mcp_server.py)         │
│  🦾 Limbs - Browser Control         │
│  - navigate_to_url                  │
│  - get_page_state                   │
│  - get_all_links                    │
│  - click_element                    │
│  - get_screenshot                   │
└──────────────┬──────────────────────┘
               │ Playwright API
               ▼
┌─────────────────────────────────────┐
│  Chromium Browser                   │
│  (Actual Web Pages)                 │
└─────────────────────────────────────┘
```

## Example: AI-Driven Navigation

The Brain (Claude) can orchestrate navigation:

1. **Brain**: Call `get_page_state` to see current state
2. **Eyes** (in Brain): Analyze screenshot from `get_page_state`
3. **Brain**: Evaluate all links, score them
4. **Brain**: Call `navigate_to_url` with highest-scoring link
5. **Brain**: Call `get_page_state` again
6. **Brain**: Decide if goal reached
7. Repeat until goal found

## Advantages of MCP Architecture

- ✅ **Separation of Concerns**: Brain logic separate from browser control
- ✅ **Reusable**: Any AI model can use these tools
- ✅ **Language Agnostic**: Brain can be in Python, TypeScript, etc.
- ✅ **Testable**: Can test tools independently
- ✅ **Extensible**: Easy to add new tools

## Next Steps

You can now build a GitHub CLI scraper where:
- Claude is the Brain making decisions
- This MCP server provides the Limbs
- GPT-4o Vision provides the Eyes (can be integrated in Brain prompts)

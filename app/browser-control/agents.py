"""
Agents Module
Contains the specific Agent implementations (VisionAgent, StrategyAgent) and the Coordinator (VisionAssistant).
Powered by Hugging Face smolagents.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                        CENTRAL BRAIN (Claude)                       │
│   Strategic reasoning, goal verification, heuristic generation      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   EYES (GPT-4o) │   │  GREATER A*     │   │   LESSER A*     │
│   Visual Desc.  │   │  Page Searcher  │   │   DOM Searcher  │
│   Screenshot    │   │  URL Navigation │   │   Element Click │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MCP SERVER (Spinal Cord)                         │
│   Playwright Browser Control, DOM Queries, Screenshot Capture       │
│   Tools: navigate_to_url, click_element, get_dom_tree, etc.         │
└─────────────────────────────────────────────────────────────────────┘
"""

import json
import os
import base64
import io
import asyncio
from typing import Dict, Any, Optional
import config
from pathlib import Path
from PIL import Image
# Hybrid Imports
try:
    from smolagents import CodeAgent, LiteLLMModel, tool
    import smolagents
except ImportError:
    print("DEBUG: smolagents not installed.")
    CodeAgent = None
    LiteLLMModel = None
    tool = lambda x: x # Dummy decorator

import httpx
import json

# Define MCP Tool for Strategy Agent
@tool
def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    """
    Call a tool on the MCP server (browser).
    Args:
        tool_name: The name of the tool (e.g., 'navigate_to_url', 'click_element', 'get_page_state')
        arguments: Dictionary of arguments for the tool.
    Returns:
        The result of the tool call.
    """
    try:
        resp = httpx.post("http://localhost:8000/call_tool", json={"name": tool_name, "arguments": arguments}, timeout=30.0)
        if resp.status_code == 200:
            return str(resp.json())
        return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Connection Error: {e}"

try:
    from agno.agent import Agent as AgnoAgent
    from agno.models.openai import OpenAIChat
    from agno.media import Image as AgnoImage
except ImportError:
    print("DEBUG: Agno not installed.")
    AgnoAgent = None
    OpenAIChat = None
    AgnoImage = None

# --- Agent 1: The Eyes (Vision) ---
class VisionAgent:
    """
    The 'Eyes' of the operation.
    Responsibility: Visual Perception.
    Model: GPT-4o (Vision) via Agno.
    """

    # --- Vision Prompts ---
    EYES_ANALYSIS_PROMPT = """## VISUAL ANALYSIS TASK
You are the "Eyes" of an autonomous navigation agent. Your role is to describe
what you SEE on this page to help the "Brain" make strategic decisions.

### Focus On:
1. **Page Type**: What kind of page is this? (homepage, search results, product page, etc.)
2. **Visual Hierarchy**: What elements are most prominent?
3. **Interactive Elements**: Buttons, links, forms visible
4. **Status Indicators**: Badges, labels, alerts (e.g., "Latest", "New", "Sale")
5. **Content Structure**: Headers, sections, lists

### Output Format
Provide a concise but specific description. Example:
"This is a GitHub releases page. The most prominent element is a release card
showing version 'v2.0.0' with a green 'Latest' badge. Below are download links
for source code (zip, tar.gz). Navigation shows: Code, Issues, Pull requests, Releases."

Be objective. Describe what you see, not what you think should happen."""

    EXTRACTION_SYSTEM_PROMPT = """## DATA EXTRACTION ROLE

You extract structured data from web pages using both visual and DOM information.

### Data Sources Available
1. **Screenshot**: Visual representation of the page
2. **HTML Content**: Raw markup (may be truncated)
3. **DOM Tree**: Structured element hierarchy

### Extraction Principles
1. **Prefer DOM data**: More reliable than visual OCR
2. **Cross-reference**: Verify visual matches against DOM
3. **Handle missing data**: Return empty string, not guesses
4. **Exact values only**: No interpretation or summarization"""

    def __init__(self, model_name: str = None):
        self.model_id = model_name if model_name else config.TAGGING_MODEL
        
        # Use Agno for Vision
        if AgnoAgent and OpenAIChat:
            print(f"   [VisionAgent] Initializing Agno Agent with {self.model_id}...")
            self.agent = AgnoAgent(
                model=OpenAIChat(id=self.model_id, api_key=config.OPENAI_API_KEY),
                description="You are the Vision Agent. You see screenshots and describe them."
            )
        else:
            print("   [VisionAgent] Agno not installed or failing.")
            self.agent = None

    def _base64_to_data_uri(self, b64_str: str) -> str:
        """Convert base64 string to Data URI for Agno/OpenAI"""
        return f"data:image/png;base64,{b64_str}"

    async def analyze_page(self, screenshot_data: str, prompt_override: str = None) -> str:
        """
        Agent Action: See the page and describe it.
        Uses Agno Framework.
        """
        if not self.agent:
            raise ValueError("Vision Agent not initialized (Agno missing).")

        eyes_prompt = prompt_override if prompt_override else self.EYES_ANALYSIS_PROMPT
        full_prompt = f"You are the Eyes. {eyes_prompt}\n\nPlease describe what you see in the provided image."
        
        print(f"   [VisionAgent] Analyzing visual field with {self.model_id} (Agno)...")
        
        # Agno supports 'images' argument in run()
        # It accepts list of image URLs or base64 data URIs
        # Note: Agno runs synchronously by default? Usually. 
        # We wrap in to_thread to keep our async loop happy.
        
        # Agno run() returns a RunResponse object. 
        # response.content is the text.
        response = await asyncio.to_thread(
            self.agent.run,
            full_prompt,
            images=[AgnoImage(url=self._base64_to_data_uri(screenshot_data))]
        )
        
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

    async def extract_data(self, screenshot_data: str, html_content: str, prompt: str, system_prompt: str = None) -> str:
        """
        Agent Action: Extract specific data fields using Vision + Text.
        """
        if not self.agent: raise ValueError("Vision Agent not initialized.")

        sys_prompt = system_prompt if system_prompt else self.EXTRACTION_SYSTEM_PROMPT
        task = f"{sys_prompt}\n\n{prompt}\n\nHTML Content (snippet): {html_content[:1000]}..."
        
        response = await asyncio.to_thread(
            self.agent.run,
            task,
            images=[AgnoImage(url=self._base64_to_data_uri(screenshot_data))]
        )
        
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

# --- Agent 2: The Brain (Strategy) ---
class StrategyAgent:
    """
    The 'Brain' of the operation.
    Responsibility: Reasoning, Planning, and Decision Making.
    Model: Claude 3.5 Sonnet via smolagents (LiteLLM).
    
    NOTE: This agent is initialized with `tools=[]` (Code Act disabled). 
    It acts as a pure Reasoning Engine, outputting structured decisions (JSON) 
    which are executed by the `Navigator` (Captain) via the MCP Client.
    """

    # --- Strategy Prompts ---
    MACHINE_CONTEXT = """## MACHINE NAVIGATION CONTEXT

You are an autonomous navigation agent operating as a MACHINE, not a human.

### Your Capabilities (via MCP Server)

You have direct programmatic access to:

1. **URL & Page State**
   - Current URL (exact string)
   - Full HTML content (raw bytes)
   - Page load state and network status

2. **DOM Tree (Structured)**
   - Complete DOM hierarchy as JSON
   - All element attributes (id, class, href, aria-*, data-*)
   - Semantic element filtering (a, button, input, nav, main, etc.)
   - Text content of each node

3. **Outgoing Links (Graph)**
   - All `<a>` elements with href attributes
   - Link text and surrounding context
   - Relative vs absolute URL resolution

4. **Interactive Elements**
   - Buttons, inputs, textareas, selects
   - Click targets with CSS selectors
   - Form structure and submission endpoints

5. **Visual Snapshot (Screenshot)**
   - Base64-encoded PNG of viewport
   - Used for visual verification, not primary navigation

### Your Advantages Over Human Navigation

| Human                          | Machine (You)                     |
|--------------------------------|-----------------------------------|
| Scans page visually            | Parses full DOM instantly         |
| Clicks based on visual cues    | Navigates directly via URL/href   |
| Reads text sequentially        | Searches all text simultaneously  |
| Limited working memory         | Caches entire page state          |
| Guesses at link destinations   | Knows exact href before clicking  |

### Decision Strategy

1. **Prefer URL Navigation (Greater A*)**: If you know the target URL, navigate directly.
2. **Use DOM Interaction (Lesser A*)**: Only click elements when URL is unknown.
3. **Trust Structured Data**: href attributes are ground truth; visual text may be misleading.
4. **Minimize Actions**: Every action has cost. Prefer fewer, precise actions."""

    MCP_TOOLS_REF = """## MCP SERVER TOOLS

The following tools are available via the MCP (Model Context Protocol) server:

### Navigation Tools
- `navigate_to_url(url)` → Go directly to a URL
- `click_element(selector)` → Click element by CSS selector
- `type_input(selector, text)` → Type into input field
- `press_key(key)` → Press keyboard key (Enter, Escape, Tab)

### Observation Tools
- `get_page_state(include_screenshot)` → Get URL, HTML, links, and optional screenshot
- `get_dom_tree(include_attributes, max_depth)` → Get hierarchical DOM structure
- `get_clean_dom_tree(semantic_only)` → Get filtered DOM with semantic elements only
- `query_elements(selector)` → Query elements by CSS selector
- `find_elements_by_text(text)` → Find elements containing specific text
- `get_all_links(include_text)` → Get all href links on page
- `get_screenshot(full_page)` → Capture viewport or full page screenshot

### Search Tools
- `find_path_to_element(target, strategy)` → A* pathfinding to element
- `find_element_on_page(target, search_strategy)` → Locate element efficiently

### Tool Response Format
All tools return JSON with structured data. Example:
```json
{
    "status": "success",
    "url": "https://github.com/...",
    "dom_tree": {...},
    "links": [{"href": "...", "text": "..."}]
}
```"""

    def __init__(self, model_name: str = None):
        self.model_id = model_name if model_name else config.STRATEGY_MODEL
        
        # Ensure correct formatting for smolagents/litellm
        if "anthropic/" not in self.model_id and "claude" in self.model_id:
            self.model_id = f"anthropic/{self.model_id}"
            
        if CodeAgent and LiteLLMModel:
            print(f"   [StrategyAgent] Initializing smolagents with {self.model_id}...")
            self.model = LiteLLMModel(
                model_id=self.model_id,
                api_key=config.ANTHROPIC_API_KEY
            )
            self.agent = CodeAgent(tools=[call_mcp_tool], model=self.model, add_base_tools=False)
        else:
            print("   [StrategyAgent] smolagents not installed or failing.")
            self.agent = None

    @classmethod
    def get_navigation_system_prompt(cls) -> str:
        return f"""{cls.MACHINE_CONTEXT}\n\n{cls.MCP_TOOLS_REF}\n\n## YOUR TASK

Analyze the current page state and decide the next action to reach the goal.

### Available Actions
1. **navigate** - Go directly to a known URL (preferred when href is available)
2. **click** - Click an element (use when URL is unknown, element has no href)
3. **type** - Enter text into an input field
4. **press** - Press a keyboard key
5. **extract** - Extract structured data from current page
6. **done** - Goal has been reached

### Response Format (JSON only)
{{
    "action": "navigate|click|type|press|extract|done",
    "url": "https://... (for navigate)",
    "selector": "CSS selector (for click/type)",
    "text": "text to type (for type)",
    "key": "Enter|Tab|Escape (for press)",
    "reasoning": "brief explanation of why this action"
}}"""

    @classmethod
    def get_brain_heuristic_system_prompt(cls, goal_description: str) -> str:
        return f"""{cls.MACHINE_CONTEXT}

## HEURISTIC SCORING ROLE (Greater A*)

You are scoring candidate links to find the optimal path to the current goal:
'{goal_description}'

### Scoring Guidelines (0-1000)

| Score Range | Link Type                                    |
|-------------|----------------------------------------------|
| 900-1000    | Direct link to goal target                   |
| 700-899     | Links containing text strongly related to goal|
| 500-699     | Links to relevant categories/sections        |
| 300-499     | General navigation, weak relevance           |
| 100-299     | Unlikely to be relevant                      |
| 0-99        | Irrelevant (external links, unrelated)       |

### Key Signals (from structured data)
- **href attribute**: Check for URL patterns matching the goal
- **Link text**: Check for keywords matching the goal
- **Context**: Parent element, surrounding text

### Response Format (JSON array)
[
    {{"index": 0, "heuristic_value": 850, "reasoning": "Direct match to goal"}},
    {{"index": 1, "heuristic_value": 200, "reasoning": "Unrelated navigation"}}
]"""

    @classmethod
    def get_brain_goal_check_system(cls) -> str:
         return f"""{cls.MACHINE_CONTEXT}

## GOAL VERIFICATION ROLE

You are the "Brain" of the navigation agent. Your task is to decide if the
current page state satisfies the navigation goal.

You receive:
1. **URL** - The current page URL (machine-readable)
2. **DOM Summary** - Structured representation of page content
3. **Visual Report** - Description from the Eyes (human-readable interpretation)

### Decision Criteria
- URL patterns are strong signals
- DOM content is ground truth for text matching
- Visual report provides context that may not be in DOM

### Response Format (JSON)
{{
    "goal_reached": true/false,
    "confidence": 0-100,
    "reasoning": "specific evidence from URL, DOM, and visual report"
}}"""

    @classmethod
    def get_prediction_model_system_prompt(cls) -> str:
        return """## PREDICTION MODEL GENERATOR

You generate "Weighted Bag of Words" models for navigation heuristics.

### Purpose
These models allow the agent to score links LOCALLY (without calling you again)
by matching keywords in link text/href against weighted terms.

### Model Structure
```json
{
    "weights": {
        "exact_match_term": 1.0,
        "strong_signal": 0.8,
        "moderate_signal": 0.5,
        "weak_signal": 0.3
    },
    "decay_rate": 0.6
}
```

### Guidelines
1. **weights**: Keywords most associated with the goal (case-insensitive matching)
2. **decay_rate**: How quickly relevance decreases with distance from match (0.0-1.0)"""

    async def reason(self, user_prompt: str, system_prompt: str = None, max_tokens: int = 1000) -> str:
        """
        Agent Action: Think and respond.
        Uses smolagents (Claude).
        """
        if not self.agent: raise ValueError("Strategy Agent not initialized.")

        # Default to the main navigation system prompt if not specified
        if system_prompt is None:
             system_prompt = self.get_navigation_system_prompt()

        task = f"System Instruction: {system_prompt}\n\nUser Task: {user_prompt}"
        
        # No try/except: Let it crash if model fails
        print(f"   [StrategyAgent] Reasoning with {self.model_id} (smolagents)...")
        result = await asyncio.to_thread(
            self.agent.run,
            task
        )
        return str(result)

# --- Coordinator: The Bridge ---
class VisionAssistant:
    """
    Coordinator class that bridges Agent A (Vision) and Agent B (Strategy).
    Implements A-to-A communication.
    """
    def __init__(self, vision_model: str = None):
        if vision_model:
            config.VISION_MODEL = vision_model
            
        self.vision_agent = VisionAgent(model_name=config.TAGGING_MODEL)
        self.strategy_agent = StrategyAgent(model_name=config.STRATEGY_MODEL)
        
    async def calculate_link_heuristics(self, candidates: list, goal_description: str, screenshot_data: str = None, dom_tree: str = None) -> list:
        """
        Coordination Flow:
        1. VisionAgent observes the scene.
        2. VisionAgent passes observation -> StrategyAgent.
        3. StrategyAgent evaluates candidates based on observation.
        """
        if not candidates:
            return []
        
        # 1. Communication: Vision Agent -> Text Description
        visual_description = ""
        if screenshot_data:
            visual_description = await self.vision_agent.analyze_page(screenshot_data)
        
        # 2. Data Prep
        choice_text = "\n".join([f"{i}. Text: '{c['text'][:50]}' URL: {c['url']}" for i, c in enumerate(candidates)])
        candidate_summary = []
        for i, cand in enumerate(candidates):
             candidate_summary.append({
                "index": i,
                "href": cand.get("url", ""),
                "text": cand.get("text", "")[:100],
                "tag": cand.get("tag", "a")
            })
        
        # 3. Communication: Text Description + Data -> Strategy Agent
        brain_system_prompt = StrategyAgent.get_brain_heuristic_system_prompt(goal_description)
        brain_user_prompt = f"""## HEURISTIC SCORING TASK

**Target**: {goal_description}

### Available Data

**Visual Context (from Eyes)**:
{visual_description}

**DOM Excerpt**:
{dom_tree[:2000] if dom_tree else "Not provided"}

**Candidate Links** ({len(candidates)} total):
```json
{json.dumps(candidate_summary, indent=2)}
```

### Instructions
Score each candidate based on relevance to the target goal.
1. href pattern (matches expected patterns?)
2. Link text (keywords match?)
3. Visual context (prominence?)

Return JSON array with index, heuristic_value (0-1000), and reasoning."""
        
        response_text = await self.strategy_agent.reason(user_prompt=brain_user_prompt, system_prompt=brain_system_prompt, max_tokens=3000)
        
        try:
            # Cleanup JSON
            content = response_text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            scores = json.loads(content)
            
            score_map = {s["index"]: s["heuristic_value"] for s in scores}
            for i, cand in enumerate(candidates):
                cand["heuristic_value"] = score_map.get(i, 0)
                
            return candidates
        except Exception as e:
            print(f"Error parsing strategy response: {e}")
            for cand in candidates:
                cand["heuristic_value"] = 50
            return candidates

    async def is_goal_reached(self, screenshot_data: str, url: str, dom_tree: str, verification_prompt: str) -> dict:
        """
        Coordination Flow: Vision Agent (Verify) -> Strategy Agent (Decide)
        """
        # 1. Vision Agent: "What do I see?"
        eyes_prompt = "Describe this page in detail. Focus on the main content, headers, and any status indicators."
        visual_report = await self.vision_agent.analyze_page(screenshot_data, prompt_override=eyes_prompt)
        
        # 2. Strategy Agent: "Given what eyes see, are we there?"
        brain_prompt = f"""## GOAL VERIFICATION

### Current State
**URL**: {url}

**Visual Report (from Eyes)**:
{visual_report}

**DOM Content (excerpt)**:
{dom_tree[:2000]}

### Verification Task
{verification_prompt}

### Decision
Return JSON with goal_reached, confidence, and reasoning."""

        system_msg = StrategyAgent.get_brain_goal_check_system()
        
        response_text = await self.strategy_agent.reason(user_prompt=brain_prompt, system_prompt=system_msg, max_tokens=500)
        
        try:
            content = response_text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {"goal_reached": False, "confidence": 0, "reasoning": "Error"}

    async def extract_with_vision_and_html(self, screenshot_data: str, html_content: str, extract_fields: list = None, instruction_hint: str = "") -> Dict[str, Any]:
        """
        Extraction prioritizes Vision Agent (GPT-4o) as it is multimodal expert.
        """
        if extract_fields is None:
            extract_fields = ["version", "commit", "author"]
            
        field_descriptions = {
            "version": "Version/release name (e.g., v1.0.0, 2026.2.1)",
            "commit": "Git commit hash, short form (e.g., abc1234)",
            "author": "Username of release author/uploader",
            "release_notes": "Summary of release notes (first 200 chars)",
            "published_at": "Publication date (ISO format preferred)",
            "downloads": "List of downloadable assets with name and URL"
        }
        
        fields_list = "\n".join([
            f"{i+1}. **{field}**: {field_descriptions.get(field, field)}" 
            for i, field in enumerate(extract_fields)
        ])
        
        example_json = {field: f"<{field}>" for field in extract_fields}

        prompt = f"""## EXTRACTION TASK

Extract the following fields from this page:

{fields_list}

### Response Format (JSON only)
```json
{json.dumps(example_json, indent=2)}
```

### Rules
- Extract exact values as they appear
- Use empty string "" for missing fields
- Do not guess or infer values
- For lists (downloads), include all items found"""
        
        # Allow caller to provide specific patterns via instruction_hint
        extraction_system_prompt = VisionAgent.EXTRACTION_SYSTEM_PROMPT
        if instruction_hint:
             extraction_system_prompt += f"\n\n{instruction_hint}"
        
        try:
            # Direct task delegation to Vision Agent
            print(f"   [Coordinator] Delegating extraction to VisionAgent...")
            content = await self.vision_agent.extract_data(screenshot_data, html_content, prompt, system_prompt=extraction_system_prompt)
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
        except Exception as e:
            print(f"Extraction failed: {e}")
            return {}

    async def generate_prediction_model(self, goal: str, context: str) -> Dict[str, Any]:
        """
        Pure Strategy Task -> Delegated to Strategy Agent
        """
        cache_path = Path("data/prediction_cache.json")
        
        # Strategy Agent Job
        system_prompt = StrategyAgent.get_prediction_model_system_prompt()
        user_prompt = f"""## GENERATE PREDICTION MODEL

**Goal**: {goal}
**Context**: '{context}'

Generate a weighted keyword model that will help identify relevant links.

### Requirements
1. Include 5-10 weighted keywords
2. Higher weights (0.8-1.0) for exact goal matches
3. Lower weights (0.3-0.5) for related but indirect terms
4. Consider URL patterns
5. Set appropriate decay_rate (0.5-0.7 typical)

### Response Format (JSON only)
{{
    "weights": {{
        "keyword1": 1.0,
        "keyword2": 0.8,
        ...
    }},
    "decay_rate": 0.6,
    "explanation": "Brief reasoning for keyword selection"
}}"""
        
        response_text = await self.strategy_agent.reason(user_prompt=user_prompt, system_prompt=system_prompt)
        
        try:
            content = response_text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            result = json.loads(content)
            
            return result
        except:
            return {"weights": {"Releases": 1.0, "Latest": 0.9}, "decay_rate": 0.5}

    # Backward compatibility wrapper
    async def get_navigation_instruction(self, *args, **kwargs):
        return None

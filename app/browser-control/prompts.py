
"""
Centralized Prompts for Navigation Agent

This module contains all system prompts and user prompts for the autonomous
navigation agent. The agent operates as a MACHINE, not a human - it has direct
access to structured data (URLs, DOM trees, link graphs) rather than relying
solely on visual perception.
"""
import json


class NavigationPrompts:
    """
    Prompt templates for the multi-model navigation architecture:
    
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
    
    # =========================================================================
    # CORE SYSTEM PROMPT: Machine Navigation Context
    # =========================================================================
    
    @staticmethod
    def machine_navigation_context() -> str:
        """
        Core context that explains the agent's machine-native capabilities.
        This should be included in all strategic prompts.
        """
        return """## MACHINE NAVIGATION CONTEXT

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

    # =========================================================================
    # MCP SERVER TOOLS REFERENCE
    # =========================================================================
    
    @staticmethod
    def mcp_tools_reference() -> str:
        """
        Reference for available MCP Server tools.
        """
        return """## MCP SERVER TOOLS

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

    # =========================================================================
    # STRATEGY PROMPTS
    # =========================================================================
    
    @staticmethod
    def navigation_system_prompt() -> str:
        """System prompt for general navigation decisions."""
        context = NavigationPrompts.machine_navigation_context()
        tools = NavigationPrompts.mcp_tools_reference()
        
        return f"""{context}

{tools}

## YOUR TASK

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

    @staticmethod
    def navigation_user_prompt(current_url: str, step_name: str, task_context: str, 
                                dom_summary: str = "", links_summary: str = "") -> str:
        """User prompt with current state for navigation decision."""
        return f"""## CURRENT STATE

**URL**: {current_url}
**Step**: {step_name}

**Task Context**:
{task_context}

**DOM Summary**:
{dom_summary if dom_summary else "Not provided - request via get_dom_tree if needed"}

**Available Links**:
{links_summary if links_summary else "Not provided - request via get_all_links if needed"}

## DECISION

Based on the structured data above, what is the optimal next action?
Remember: If you know the target URL, navigate directly instead of clicking.

Respond with JSON only."""

    # =========================================================================
    # EYES (VISUAL PERCEPTION) PROMPTS
    # =========================================================================
    
    @staticmethod
    def eyes_analysis_prompt() -> str:
        """Prompt for visual analysis by GPT-4o (Eyes)."""
        return """## VISUAL ANALYSIS TASK

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

    @staticmethod
    def eyes_goal_check_prompt() -> str:
        """Prompt for Eyes to verify if goal state is reached."""
        return """## GOAL VERIFICATION TASK

Describe this page to help determine if the navigation goal has been reached.

### Key Observations Needed:
1. What type of page is this? (URL pattern may help but describe visuals)
2. Is there a "Latest" badge, label, or indicator visible?
3. What version/release information is displayed?
4. Are there any signs this is NOT the target page?

Be specific about what you observe. The Brain will make the final decision."""

    # =========================================================================
    # BRAIN (STRATEGIC REASONING) PROMPTS  
    # =========================================================================
    
    @staticmethod
    def brain_goal_check_system() -> str:
        """System prompt for Brain's goal verification."""
        context = NavigationPrompts.machine_navigation_context()
        
        return f"""{context}

## GOAL VERIFICATION ROLE

You are the "Brain" of the navigation agent. Your task is to decide if the
current page state satisfies the navigation goal.

You receive:
1. **URL** - The current page URL (machine-readable)
2. **DOM Summary** - Structured representation of page content
3. **Visual Report** - Description from the Eyes (human-readable interpretation)

### Decision Criteria
- URL patterns are strong signals (e.g., /releases/tag/ indicates a release page)
- DOM content is ground truth for text matching
- Visual report provides context that may not be in DOM

### Response Format (JSON)
{{
    "goal_reached": true/false,
    "confidence": 0-100,
    "reasoning": "specific evidence from URL, DOM, and visual report"
}}"""

    @staticmethod
    def brain_goal_check_prompt(repo: str, url: str, visual_description: str, dom_tree: str) -> str:
        """User prompt for Brain's goal verification."""
        return f"""## GOAL VERIFICATION

**Mission**: Find the LATEST release page for repository '{repo}'.

### Current State

**URL**: {url}

**Visual Report (from Eyes)**:
{visual_description}

**DOM Content (excerpt)**:
{dom_tree[:2000]}

### Verification Checklist
1. Does the URL contain /releases/tag/ or /releases/latest?
2. Does the DOM or visual report show a "Latest" indicator?
3. Is version/tag information clearly present?
4. Is there any evidence this is NOT the latest release?

### Decision
Return JSON with goal_reached, confidence, and reasoning."""

    # =========================================================================
    # A* SEARCH PROMPTS
    # =========================================================================
    
    @staticmethod
    def brain_heuristic_system_prompt(repo: str) -> str:
        """System prompt for Greater A* heuristic scoring."""
        context = NavigationPrompts.machine_navigation_context()
        
        return f"""{context}

## HEURISTIC SCORING ROLE (Greater A*)

You are scoring candidate links to find the optimal path to the LATEST release
for repository '{repo}'.

### Scoring Guidelines (0-1000)

| Score Range | Link Type                                    |
|-------------|----------------------------------------------|
| 900-1000    | Direct link to /releases/latest or latest tag|
| 700-899     | Links containing "Latest", release version  |
| 500-699     | General releases page, tags page            |
| 300-499     | Downloads, changelog, related content       |
| 100-299     | General navigation (issues, code, etc.)     |
| 0-99        | Irrelevant (external links, unrelated)      |

### Key Signals (from structured data)
- **href attribute**: Check for /releases/, /tags/, version patterns
- **Link text**: "Latest", "Releases", version numbers (v1.0.0)
- **Context**: Parent element, surrounding text

### Response Format (JSON array)
[
    {{"index": 0, "heuristic_value": 850, "reasoning": "Direct release link"}},
    {{"index": 1, "heuristic_value": 200, "reasoning": "Code navigation"}}
]"""

    @staticmethod
    def brain_heuristic_user_prompt(visual_description: str, dom_tree: str, 
                                     candidates: list, repo: str) -> str:
        """User prompt for Greater A* heuristic scoring."""
        candidate_summary = []
        for i, cand in enumerate(candidates):
            candidate_summary.append({
                "index": i,
                "href": cand.get("url", ""),
                "text": cand.get("text", "")[:100],
                "tag": cand.get("tag", "a")
            })
            
        return f"""## HEURISTIC SCORING TASK

**Target**: Find LATEST release for '{repo}'

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
Score each candidate based on:
1. href pattern (does it point to releases?)
2. Link text (does it mention "Latest", version numbers?)
3. Visual context (was this link highlighted as prominent?)

Return JSON array with index, heuristic_value (0-1000), and reasoning."""

    # =========================================================================
    # PREDICTION MODEL PROMPTS
    # =========================================================================
    
    @staticmethod
    def prediction_model_system_prompt() -> str:
        """System prompt for generating weighted keyword models."""
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
2. **decay_rate**: How quickly relevance decreases with distance from match (0.0-1.0)

### Examples
- Goal "Find releases" → {"Releases": 1.0, "Latest": 0.9, "v[0-9]": 0.7, "Tags": 0.5}
- Goal "Find documentation" → {"Docs": 1.0, "Documentation": 1.0, "Guide": 0.7, "README": 0.6}
- Goal "Find issues" → {"Issues": 1.0, "Bug": 0.8, "Open": 0.5, "Closed": 0.4}"""

    @staticmethod
    def prediction_model_user_prompt(goal: str, repo: str) -> str:
        """User prompt for prediction model generation."""
        return f"""## GENERATE PREDICTION MODEL

**Goal**: {goal}
**Context**: Repository '{repo}'

Generate a weighted keyword model that will help identify relevant links.

### Requirements
1. Include 5-10 weighted keywords
2. Higher weights (0.8-1.0) for exact goal matches
3. Lower weights (0.3-0.5) for related but indirect terms
4. Consider URL patterns (e.g., /releases/, /tags/)
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

    # =========================================================================
    # EXTRACTION PROMPTS
    # =========================================================================
    
    @staticmethod
    def extraction_system_prompt() -> str:
        """System prompt for data extraction."""
        return """## DATA EXTRACTION ROLE

You extract structured data from web pages using both visual and DOM information.

### Data Sources Available
1. **Screenshot**: Visual representation of the page
2. **HTML Content**: Raw markup (may be truncated)
3. **DOM Tree**: Structured element hierarchy

### Extraction Principles
1. **Prefer DOM data**: More reliable than visual OCR
2. **Cross-reference**: Verify visual matches against DOM
3. **Handle missing data**: Return empty string, not guesses
4. **Exact values only**: No interpretation or summarization

### Common Patterns (GitHub)
- Version: Look for headings, release titles (e.g., "v2.0.0", "2024.1.29")
- Commit: Look for short SHA hashes (7 chars, e.g., "abc1234")  
- Author: Look for user links, "@username", "by username"
- Date: Look for relative ("2 days ago") or absolute dates"""

    @staticmethod
    def extraction_prompt(extract_fields: list) -> str:
        """User prompt for data extraction."""
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

        return f"""## EXTRACTION TASK

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

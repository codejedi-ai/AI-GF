#!/usr/bin/env python3
"""
MCP Server for Browser Navigation Limbs
Exposes browser control and A* navigation as tools for AI models
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# Import BeautifulSoup for cleaner DOM parsing
from bs4 import BeautifulSoup
import argparse
import config

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    # Dummy classes to prevent NameError in type hints if evaluated at runtime
    FastAPI = object
    BaseModel = object
    HTTPException = Exception



class TabManager:
    """
    Manages browser tabs to:
    - Track all open tabs by URL and ID
    - Reuse existing tabs instead of opening new ones
    - Switch between tabs
    """
    
    def __init__(self):
        self.tabs: Dict[str, Page] = {}  # tab_id -> Page
        self.url_to_tab: Dict[str, str] = {}  # normalized_url -> tab_id
        self.active_tab_id: Optional[str] = None
        self._counter = 0
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison (remove fragments, trailing slashes)."""
        parsed = urlparse(url)
        # Keep scheme, netloc, path (normalized)
        path = parsed.path.rstrip('/')
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    
    def _generate_tab_id(self) -> str:
        """Generate a unique tab ID."""
        self._counter += 1
        return f"tab_{self._counter}"
    
    def register_tab(self, page: Page, url: str = None) -> str:
        """Register a new tab and return its ID."""
        tab_id = self._generate_tab_id()
        self.tabs[tab_id] = page
        
        if url:
            normalized = self._normalize_url(url)
            self.url_to_tab[normalized] = tab_id
        
        self.active_tab_id = tab_id
        return tab_id
    
    def update_tab_url(self, tab_id: str, new_url: str):
        """Update the URL mapping for a tab."""
        # Remove old URL mapping
        old_urls = [url for url, tid in self.url_to_tab.items() if tid == tab_id]
        for old_url in old_urls:
            del self.url_to_tab[old_url]
        
        # Add new URL mapping
        normalized = self._normalize_url(new_url)
        self.url_to_tab[normalized] = tab_id
    
    def find_tab_by_url(self, url: str) -> Optional[str]:
        """Find existing tab with this URL."""
        normalized = self._normalize_url(url)
        return self.url_to_tab.get(normalized)
    
    def get_tab(self, tab_id: str) -> Optional[Page]:
        """Get a tab by ID."""
        return self.tabs.get(tab_id)
    
    def get_active_tab(self) -> Optional[Page]:
        """Get the currently active tab."""
        if self.active_tab_id:
            return self.tabs.get(self.active_tab_id)
        return None
    
    def set_active_tab(self, tab_id: str) -> bool:
        """Set the active tab."""
        if tab_id in self.tabs:
            self.active_tab_id = tab_id
            return True
        return False
    
    def list_tabs(self) -> Dict[str, str]:
        """List all open tabs with their URLs."""
        result = {}
        for tab_id, page in self.tabs.items():
            try:
                result[tab_id] = page.url
            except:
                result[tab_id] = "unknown"
        return result
    
    def close_tab(self, tab_id: str):
        """Close a tab and remove from tracking."""
        if tab_id in self.tabs:
            # Remove URL mappings
            urls_to_remove = [url for url, tid in self.url_to_tab.items() if tid == tab_id]
            for url in urls_to_remove:
                del self.url_to_tab[url]
            
            del self.tabs[tab_id]
            
            if self.active_tab_id == tab_id:
                self.active_tab_id = next(iter(self.tabs.keys()), None)


# Global browser state
browser: Browser = None
context: BrowserContext = None
page: Page = None
playwright_instance = None
tab_manager = TabManager()


async def initialize_browser():
    """Initialize the browser if not already running"""
    global browser, context, page, playwright_instance, tab_manager

    if page is not None:
        return

    playwright_instance = await async_playwright().start()
    context = await playwright_instance.chromium.launch_persistent_context(
        user_data_dir="browser_data",
        headless=False,
        viewport={'width': 1920, 'height': 1080},
        args=['--no-sandbox', '--disable-dev-shm-usage']
    )

    if len(context.pages) > 0:
        page = context.pages[0]
    else:
        page = await context.new_page()
    
    # Register the initial page with tab manager
    tab_manager.register_tab(page, page.url)


async def cleanup_browser():
    """Cleanup browser resources"""
    global page, context, playwright_instance

    if page:
        await page.close()
    if context:
        await context.close()
    if playwright_instance:
        await playwright_instance.stop()


# Create MCP server
server = Server("github-navigator-limbs")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available browser control tools"""
    return [
        Tool(
            name="navigate_to_url",
            description="Navigate browser to a specific URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="get_current_url",
            description="Get the current page URL",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_page_html",
            description="Get the HTML content of the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of HTML to return (default: 50000)",
                        "default": 50000
                    }
                }
            }
        ),
        Tool(
            name="get_dom_tree",
            description="Get a simplified DOM tree representation of the current page with element attributes",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Include element attributes (default: true)",
                        "default": True
                    },
                    "include_text": {
                        "type": "boolean",
                        "description": "Include element text content (default: true)",
                        "default": True
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse in the DOM (default: 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="get_clean_dom_tree",
            description="Get a cleaned DOM tree using BeautifulSoup with simplified structure and semantic elements",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_attributes": {
                        "type": "boolean",
                        "description": "Include element attributes (default: true)",
                        "default": True
                    },
                    "include_text": {
                        "type": "boolean",
                        "description": "Include element text content (default: true)",
                        "default": True
                    },
                    "semantic_only": {
                        "type": "boolean",
                        "description": "Only include semantic elements (default: false)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="find_path_to_element",
            description="Use A* algorithm to find the optimal path to a target element from the current page state",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_element": {
                        "type": "string",
                        "description": "Description or text of the target element to find"
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Strategy for pathfinding: 'shortest', 'least_clicks', 'semantic' (default: 'shortest')",
                        "default": "shortest",
                        "enum": ["shortest", "least_clicks", "semantic"]
                    }
                },
                "required": ["target_element"]
            }
        ),
        Tool(
            name="find_valuable_pages",
            description="Use A* algorithm concepts to find the top 3 most valuable pages related to a topic, starting from the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or subject to find valuable pages for"
                    },
                    "depth_limit": {
                        "type": "integer",
                        "description": "Maximum depth to explore from current page (default: 2)",
                        "default": 2
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum number of pages to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="find_element_on_page",
            description="Use A* algorithm to find a specific element on the current page efficiently",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_element": {
                        "type": "string",
                        "description": "Description or text of the target element to find on the current page"
                    },
                    "search_strategy": {
                        "type": "string",
                        "description": "Strategy for searching: 'breadth_first', 'depth_first', 'by_importance' (default: 'by_importance')",
                        "default": "by_importance",
                        "enum": ["breadth_first", "depth_first", "by_importance"]
                    }
                },
                "required": ["target_element"]
            }
        ),
        Tool(
            name="query_elements",
            description="Query elements on the page using CSS selectors and return their properties",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to query elements"
                    },
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific attributes to retrieve (default: all common attributes)"
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="find_elements_by_text",
            description="Find elements containing specific text content",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to search for in elements"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case sensitive (default: false)",
                        "default": False
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="get_all_links",
            description="Get all href links from the current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_text": {
                        "type": "boolean",
                        "description": "Include link text (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="get_screenshot",
            description="Capture screenshot of current page",
            inputSchema={
                "type": "object",
                "properties": {
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture full scrollable page (default: false - viewport only)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="click_element",
            description="Click an element on the page using CSS selector",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to click"
                    }
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="get_page_state",
            description="Get complete page state: URL, HTML, links, and screenshot in one call",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_screenshot": {
                        "type": "boolean",
                        "description": "Include base64 screenshot (default: true)",
                        "default": True
                    },
                    "html_max_length": {
                        "type": "integer",
                        "description": "Max HTML length (default: 20000)",
                        "default": 20000
                    }
                }
            }
        ),
        Tool(
            name="type_input",
            description="Type text into an element",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector"},
                    "text": {"type": "string", "description": "Text to type"}
                },
                "required": ["selector", "text"]
            }
        ),
        Tool(
            name="press_key",
            description="Press a keyboard key",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key name (e.g. Enter, Escape)"}
                },
                "required": ["key"]
            }
        ),
        Tool(
            name="is_page_loaded",
            description="Check if a specific tab has finished loading (no LLM call, fast check)",
            inputSchema={
                "type": "object",
                "properties": {
                    "tab_id": {
                        "type": "string",
                        "description": "Tab ID to check (optional, defaults to active tab)"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Max time to wait for load (default: 5000ms)",
                        "default": 5000
                    }
                }
            }
        ),
        Tool(
            name="click_button",
            description="Click a button element ONLY. Asserts that the element is a button/input/summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector of button to click"}
                },
                "required": ["selector"]
            }
        ),
        Tool(
            name="list_tabs",
            description="List all open tabs with their IDs and URLs",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="switch_tab",
            description="Switch to a specific tab by ID or switch to an existing tab if URL is already open",
            inputSchema={
                "type": "object",
                "properties": {
                    "tab_id": {"type": "string", "description": "Tab ID to switch to"},
                    "url": {"type": "string", "description": "Alternatively, URL to find and switch to"}
                }
            }
        ),
        Tool(
            name="close_tab",
            description="Close a tab by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "tab_id": {"type": "string", "description": "Tab ID to close"}
                },
                "required": ["tab_id"]
            }
        )
    ]


async def handle_tool_call(name: str, arguments: Any) -> list[TextContent | ImageContent]:
    """Core browser automation logic - Shared by StdIO and HTTP wrappers"""
    global page
    await initialize_browser()

    try:
        if name == "navigate_to_url":
            url = arguments["url"]
            
            # Check if URL is already open in an existing tab
            existing_tab_id = tab_manager.find_tab_by_url(url)
            
            if existing_tab_id:
                # Switch to existing tab instead of navigating
                existing_page = tab_manager.get_tab(existing_tab_id)
                if existing_page:
                    tab_manager.set_active_tab(existing_tab_id)
                    page = existing_page
                    await page.bring_to_front()
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "status": "success",
                            "action": "switched_to_existing_tab",
                            "tab_id": existing_tab_id,
                            "current_url": page.url
                        })
                    )]
            
            # Navigate in current tab
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            
            # Update tab manager with new URL
            if tab_manager.active_tab_id:
                tab_manager.update_tab_url(tab_manager.active_tab_id, page.url)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "action": "navigated",
                    "tab_id": tab_manager.active_tab_id,
                    "current_url": page.url
                })
            )]

        elif name == "get_current_url":
            return [TextContent(
                type="text",
                text=json.dumps({
                    "url": page.url
                })
            )]

        elif name == "get_page_html":
            max_length = arguments.get("max_length", 50000)
            html = await page.content()
            if len(html) > max_length:
                html = html[:max_length]
            return [TextContent(
                type="text",
                text=json.dumps({
                    "html": html,
                    "truncated": len(await page.content()) > max_length
                })
            )]

        elif name == "get_all_links":
            include_text = arguments.get("include_text", True)

            if include_text:
                links = await page.evaluate("""() => {
                    const results = [];
                    const anchors = Array.from(document.querySelectorAll('a[href]'));
                    anchors.forEach(a => {
                        results.push({
                            href: a.href.split('#')[0],
                            text: a.innerText.trim()
                        });
                    });
                    return results;
                }""")
            else:
                links = await page.evaluate("""() => {
                    const anchors = Array.from(document.querySelectorAll('a[href]'));
                    return anchors.map(a => a.href.split('#')[0]);
                }""")

            return [TextContent(
                type="text",
                text=json.dumps({
                    "links": links,
                    "count": len(links)
                })
            )]

        elif name == "get_screenshot":
            full_page = arguments.get("full_page", False)
            screenshot_path = Path("screenshots") / "mcp_screenshot.png"
            screenshot_path.parent.mkdir(exist_ok=True)

            await page.screenshot(path=screenshot_path, full_page=full_page)

            with open(screenshot_path, "rb") as f:
                screenshot_data = base64.b64encode(f.read()).decode('utf-8')

            return [TextContent(
                type="text",
                text=json.dumps({
                    "screenshot_base64": screenshot_data,
                    "width": page.viewport_size["width"],
                    "height": page.viewport_size["height"]
                })
            )]

        elif name == "click_element":
            selector = arguments["selector"]
            element = page.locator(selector).first
            await element.click()
            await page.wait_for_load_state("networkidle")

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "current_url": page.url
                })
            )]

        elif name == "get_page_state":
            include_screenshot = arguments.get("include_screenshot", True)
            html_max_length = arguments.get("html_max_length", 20000)

            # Get all state at once
            current_url = page.url
            html = await page.content()
            if len(html) > html_max_length:
                html = html[:html_max_length]

            links = await page.evaluate("""() => {
                const results = [];
                const anchors = Array.from(document.querySelectorAll('a[href]'));
                anchors.forEach(a => {
                    results.push({
                        href: a.href.split('#')[0],
                        text: a.innerText.trim()
                    });
                });
                return results;
            }""")

            result = {
                "url": current_url,
                "html": html,
                "html_truncated": len(await page.content()) > html_max_length,
                "links": links,
                "link_count": len(links)
            }

            if include_screenshot:
                screenshot_path = Path("screenshots") / "mcp_state.png"
                screenshot_path.parent.mkdir(exist_ok=True)
                await page.screenshot(path=screenshot_path, full_page=False)

                with open(screenshot_path, "rb") as f:
                    result["screenshot_base64"] = base64.b64encode(f.read()).decode('utf-8')

            return [TextContent(
                type="text",
                text=json.dumps(result)
            )]

        elif name == "type_input":
            selector = arguments["selector"]
            text = arguments["text"]
            element = page.locator(selector).first
            await element.fill(text)
            return [TextContent(
                type="text",
                text=json.dumps({"status": "success"})
            )]

        elif name == "is_page_loaded":
            # Fast, no-LLM check for page load state
            tab_id = arguments.get("tab_id", tab_manager.active_tab_id)
            timeout_ms = arguments.get("timeout_ms", 5000)
            
            # Get the specific tab to check
            target_page = tab_manager.get_tab(tab_id) if tab_id else page
            if not target_page:
                target_page = page
            
            try:
                # Wait for network idle (no pending requests)
                await target_page.wait_for_load_state("networkidle", timeout=timeout_ms)
                
                # Additional check: document ready state
                ready_state = await target_page.evaluate("document.readyState")
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "loaded": True,
                        "tab_id": tab_id,
                        "ready_state": ready_state,
                        "url": target_page.url
                    })
                )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "loaded": False,
                        "tab_id": tab_id,
                        "error": str(e),
                        "url": target_page.url if target_page else "unknown"
                    })
                )]

        elif name == "click_button":
            # Click with assertion that element is a button-like element
            selector = arguments["selector"]
            
            # Get element tag name
            element = page.locator(selector).first
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            
            # Assert: Only allow button-like elements
            allowed_tags = ["button", "input", "summary"]
            if tag_name not in allowed_tags:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": f"AssertionError: click_button requires element to be {allowed_tags}. Got: {tag_name}",
                        "element_tag": tag_name
                    })
                )]
            
            # Click the button
            await element.click()
            await page.wait_for_load_state("networkidle")
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "element_tag": tag_name,
                    "current_url": page.url
                })
            )]

        elif name == "list_tabs":
            # List all open tabs with their IDs and URLs
            tabs_info = tab_manager.list_tabs()
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "active_tab": tab_manager.active_tab_id,
                    "tabs": tabs_info,
                    "tab_count": len(tabs_info)
                })
            )]

        elif name == "switch_tab":
            # Switch to a specific tab by ID or URL
            tab_id = arguments.get("tab_id")
            url = arguments.get("url")
            
            target_tab_id = None
            
            if tab_id:
                target_tab_id = tab_id
            elif url:
                target_tab_id = tab_manager.find_tab_by_url(url)
            
            if not target_tab_id:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": f"Tab not found. Provide tab_id or URL of an open tab."
                    })
                )]
            
            target_page = tab_manager.get_tab(target_tab_id)
            if target_page:
                tab_manager.set_active_tab(target_tab_id)
                page = target_page
                await page.bring_to_front()
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "tab_id": target_tab_id,
                        "current_url": page.url
                    })
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": f"Tab {target_tab_id} exists but page is not accessible"
                    })
                )]

        elif name == "close_tab":
            # Close a tab by ID
            tab_id = arguments["tab_id"]
            
            target_page = tab_manager.get_tab(tab_id)
            if target_page:
                await target_page.close()
                tab_manager.close_tab(tab_id)
                
                # Update global page reference if needed
                if tab_manager.active_tab_id:
                    page = tab_manager.get_tab(tab_manager.active_tab_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "closed_tab": tab_id,
                        "active_tab": tab_manager.active_tab_id
                    })
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "error": f"Tab {tab_id} not found"
                    })
                )]

        elif name == "get_dom_tree":
            include_attributes = arguments.get("include_attributes", True)
            include_text = arguments.get("include_text", True)
            max_depth = arguments.get("max_depth", 10)

            # Convert Python booleans to JavaScript booleans
            js_include_attributes = "true" if include_attributes else "false"
            js_include_text = "true" if include_text else "false"

            # JavaScript to build a simplified DOM tree
            js_code = f"""
            (function() {{
                const maxDepth = {max_depth};

                function buildTree(element, currentDepth = 0) {{
                    if (!element || element.nodeType !== Node.ELEMENT_NODE || currentDepth > maxDepth) {{
                        return null;
                    }}

                    const node = {{
                        tagName: element.tagName ? element.tagName.toLowerCase() : 'unknown',
                        children: []
                    }};

                    // Add attributes if requested
                    if ({js_include_attributes}) {{
                        const attrs = {{}};
                        if (element.attributes) {{
                            for (let i = 0; i < element.attributes.length; i++) {{
                                const attr = element.attributes[i];
                                attrs[attr.name] = attr.value;
                            }}
                        }}
                        node.attributes = attrs;
                    }}

                    // Add text content if requested and element has no meaningful children
                    if ({js_include_text}) {{
                        const textContent = element.textContent ? element.textContent.trim() : '';
                        // Only add text content if it's substantial and element has no element children
                        const hasElementChildren = element.children && element.children.length > 0;
                        if (textContent && textContent.length > 0 && !hasElementChildren && textContent.length < 100) {{
                            node.text = textContent;
                        }}
                    }}

                    // Process child nodes (only element nodes, not text nodes)
                    if (element.children) {{
                        for (let i = 0; i < element.children.length; i++) {{
                            const child = element.children[i];
                            const childNode = buildTree(child, currentDepth + 1);
                            if (childNode) {{
                                node.children.push(childNode);
                            }}
                        }}
                    }}

                    return node;
                }}

                const rootElement = document.documentElement || document.body;
                return buildTree(rootElement, 0);
            }})();
            """

            dom_tree = await page.evaluate(js_code)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "dom_tree": dom_tree
                })
            )]

        elif name == "get_clean_dom_tree":
            include_attributes = arguments.get("include_attributes", True)
            include_text = arguments.get("include_text", True)
            semantic_only = arguments.get("semantic_only", False)

            # Get the raw HTML content
            html_content = await page.content()

            # Parse with BeautifulSoup for cleaner DOM
            soup = BeautifulSoup(html_content, 'html.parser')

            # Define semantic elements if filtering
            semantic_elements = {
                'header', 'nav', 'main', 'section', 'article', 'aside',
                'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'p', 'div', 'span', 'a', 'button', 'form', 'input',
                'label', 'select', 'textarea', 'table', 'thead', 'tbody',
                'tr', 'td', 'th', 'ul', 'ol', 'li', 'figure', 'figcaption'
            } if semantic_only else None

            def clean_element_to_dict(tag, current_depth=0, max_depth=10):
                """Convert BeautifulSoup tag to dictionary representation"""
                if current_depth > max_depth or not hasattr(tag, 'name'):
                    return None

                # Skip non-element tags (comments, etc.)
                if isinstance(tag, str):
                    # Handle text nodes
                    if include_text and tag.strip():
                        return {
                            "tagName": "#text",
                            "text": tag.strip()[:100],
                            "children": []
                        }
                    return None
                elif hasattr(tag, 'name') and tag.name and tag.name.startswith('[^]'):  # Skip comments
                    return None

                # If semantic_only is enabled, skip non-semantic elements
                if semantic_only and hasattr(tag, 'name') and tag.name not in semantic_elements:
                    # For non-semantic elements, return their processed children directly
                    result_children = []
                    for child in tag.children:
                        child_dict = clean_element_to_dict(child, current_depth, max_depth)
                        if child_dict:
                            if isinstance(child_dict, list):
                                result_children.extend(child_dict)
                            else:
                                result_children.append(child_dict)
                    return result_children  # Return children directly, not wrapped in an element

                # Create the element dictionary for semantic elements
                element_dict = {
                    "tagName": tag.name if hasattr(tag, 'name') and tag.name else "#text",
                    "children": []
                }

                # Add attributes if requested
                if include_attributes and hasattr(tag, 'attrs'):
                    element_dict["attributes"] = dict(tag.attrs) if hasattr(tag, 'name') and tag.name != '#text' else {}

                # Add text content if requested and element has no meaningful children
                if include_text and hasattr(tag, 'get_text'):
                    text_content = tag.get_text(strip=True)
                    # Only add text if the element has no element children
                    element_children = [child for child in tag.children if hasattr(child, 'name') and child.name]
                    if text_content and not element_children and len(text_content) < 100:
                        element_dict["text"] = text_content

                # Process children
                for child in tag.children:
                    child_dict = clean_element_to_dict(child, current_depth + 1, max_depth)
                    if child_dict:
                        if isinstance(child_dict, list):
                            element_dict["children"].extend(child_dict)
                        else:
                            element_dict["children"].append(child_dict)

                return element_dict

            # Start from the body or html element
            root_tag = soup.find('body') or soup.find('html') or soup
            clean_dom_tree = clean_element_to_dict(root_tag, max_depth=8)

            # If the result is a list (when semantic_only causes root element to be skipped),
            # wrap it in a generic container
            if isinstance(clean_dom_tree, list):
                clean_dom_tree = {
                    "tagName": "root",
                    "children": clean_dom_tree,
                    "attributes": {}
                }

            return [TextContent(
                type="text",
                text=json.dumps({
                    "clean_dom_tree": clean_dom_tree
                })
            )]

        elif name == "find_path_to_element":
            target_element = arguments["target_element"]
            strategy = arguments.get("strategy", "shortest")

            # Get the current page state to build the graph
            current_url = page.url
            html_content = await page.content()

            # Parse the DOM to create a graph representation
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all clickable elements (links, buttons, etc.)
            clickable_elements = soup.find_all(['a', 'button', 'input', 'area'])

            # Create a graph representation of the page
            graph = {}
            start_node = "start_node"  # Use a consistent identifier

            # Add the start node
            graph[start_node] = []

            # Add nodes for each clickable element
            for i, element in enumerate(clickable_elements):
                element_id = f"element_{i}"
                element_text = element.get_text(strip=True)[:50]  # First 50 chars of text
                element_href = element.get('href', '')
                element_type = element.name

                # Calculate cost based on strategy
                if strategy == "least_clicks":
                    cost = 1  # Uniform cost for least clicks
                elif strategy == "semantic":
                    # Lower cost for semantic elements that might lead to target
                    semantic_bonus = 0.5 if element_type in ['nav', 'main', 'section', 'article', 'aside', 'footer'] else 1.0
                    cost = semantic_bonus
                else:  # shortest
                    cost = 1  # Default uniform cost

                # Add edge from start to this element
                graph[start_node].append({
                    "to": element_id,
                    "cost": cost,
                    "element": {
                        "id": element_id,
                        "text": element_text,
                        "href": element_href,
                        "type": element_type,
                        "selector": _generate_selector(element)
                    }
                })

                # Add the element as a node
                graph[element_id] = []

            # Now implement A* algorithm to find path to target element
            def heuristic(node, target):
                # Simple heuristic: similarity of text to target
                if isinstance(node, dict) and 'element' in node:
                    element_text = node['element'].get('text', '').lower()
                    target_lower = target.lower()
                    if target_lower in element_text:
                        return 0  # Perfect match
                    else:
                        # Calculate similarity as inverse of edit distance (simplified)
                        return max(len(target), len(element_text))  # Rough estimate
                return len(target)  # Default heuristic

            def reconstruct_path(came_from, current):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # A* implementation
            open_set = [(0, start_node)]  # (f_score, node)
            came_from = {}
            g_score = {start_node: 0}
            f_score = {start_node: heuristic({"element": {"text": ""}}, target_element)}

            import heapq

            while open_set:
                current_f, current = heapq.heappop(open_set)

                # Check if current node contains the target element
                if current.startswith("element_"):
                    element_data = next((item["element"] for item in graph.get(start_node, []) if item["to"] == current), None)
                    if element_data:
                        element_text = element_data.get("text", "").lower()
                        target_lower = target_element.lower()
                        if target_lower in element_text or target_lower in element_data.get("href", "").lower():
                            # Found the target!
                            path = reconstruct_path(came_from, current)

                            # Convert path to actionable steps
                            steps = []
                            for p in path[1:]:  # Skip start node
                                if p.startswith("element_"):
                                    element_data = next((item["element"] for item in graph.get(start_node, []) if item["to"] == p), None)
                                    if element_data:
                                        steps.append({
                                            "action": "click",
                                            "selector": element_data["selector"],
                                            "element_text": element_data["text"],
                                            "element_type": element_data["type"]
                                        })

                            return [TextContent(
                                type="text",
                                text=json.dumps({
                                    "path_found": True,
                                    "steps": steps,
                                    "total_cost": g_score[current],
                                    "target_element": target_element
                                })
                            )]

                for neighbor_data in graph.get(current, []):
                    neighbor = neighbor_data["to"]
                    tentative_g_score = g_score[current] + neighbor_data["cost"]

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor_data, target_element)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # If we get here, no path was found
            return [TextContent(
                type="text",
                text=json.dumps({
                    "path_found": False,
                    "steps": [],
                    "total_cost": float('inf'),
                    "target_element": target_element,
                    "message": f"No path found to element containing '{target_element}'"
                })
            )]

        elif name == "find_valuable_pages":
            topic = arguments["topic"]
            depth_limit = arguments.get("depth_limit", 2)
            max_pages = arguments.get("max_pages", 3)

            # Define a bag of words with weights for the topic
            def calculate_topic_weights(topic_str):
                """Calculate weighted words for the topic"""
                # Split topic into words and assign weights
                words = topic_str.lower().split()
                weights = {}

                # Assign equal weight to each word in the topic initially
                for word in words:
                    # Remove punctuation
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word:
                        weights[clean_word] = 1.0  # Default weight

                return weights

            # Calculate weights for the topic
            topic_weights = calculate_topic_weights(topic)

            # Get current page content and links
            current_url = page.url
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract all links from the current page
            links = soup.find_all('a', href=True)
            page_links = []

            for link in links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    from urllib.parse import urljoin
                    absolute_url = urljoin(current_url, href)
                    link_text = link.get_text(strip=True)

                    # Calculate heuristic score using bag of words
                    link_words = link_text.lower().split()
                    total_weighted_score = 0
                    matching_words_count = 0

                    for word in link_words:
                        clean_word = ''.join(c for c in word if c.isalnum())
                        if clean_word in topic_weights:
                            total_weighted_score += topic_weights[clean_word]
                            matching_words_count += 1

                    # Calculate heuristic: sum of weighted words / total words in link text
                    heuristic_score = total_weighted_score / len(link_words) if link_words else 0

                    page_links.append({
                        "url": absolute_url,
                        "text": link_text,
                        "heuristic_score": heuristic_score,
                        "element": link
                    })

            # Sort by heuristic score and return top pages
            page_links.sort(key=lambda x: x["heuristic_score"], reverse=True)
            top_pages = []

            for link_info in page_links[:max_pages]:
                top_pages.append({
                    "url": link_info["url"],
                    "text": link_info["text"],
                    "heuristic_score": link_info["heuristic_score"],
                    "relevance": "high" if link_info["heuristic_score"] > 0.5 else "medium" if link_info["heuristic_score"] > 0.1 else "low"
                })

            return [TextContent(
                type="text",
                text=json.dumps({
                    "topic": topic,
                    "top_pages": top_pages,
                    "total_found": len(page_links),
                    "message": f"Found {len(top_pages)} valuable pages related to '{topic}' using A* heuristic scoring"
                })
            )]

        elif name == "find_element_on_page":
            target_element = arguments["target_element"]
            search_strategy = arguments.get("search_strategy", "by_importance")

            # Define a bag of words with weights for the target element
            def calculate_element_weights(target_str):
                """Calculate weighted words for the target element"""
                # Split target into words and assign weights
                words = target_str.lower().split()
                weights = {}

                # Assign equal weight to each word in the target initially
                for word in words:
                    # Remove punctuation
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word:
                        weights[clean_word] = 1.0  # Default weight

                return weights

            # Calculate weights for the target element
            element_weights = calculate_element_weights(target_element)

            # Get the current page DOM
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find elements matching the target description
            found_elements = []

            # Search by text content first
            all_text_elements = soup.find_all(string=True)

            for text_element in all_text_elements:
                if text_element.strip():  # Only consider non-empty text
                    text_content = text_element.strip().lower()
                    if any(word in text_content for word in element_weights.keys()):
                        parent = text_element.parent
                        while parent and parent.name != '[document]':
                            # Calculate heuristic score for this element
                            element_text = text_element.strip().lower()
                            element_words = element_text.split()

                            total_weighted_score = 0
                            matching_words_count = 0

                            for word in element_words:
                                clean_word = ''.join(c for c in word if c.isalnum())
                                if clean_word in element_weights:
                                    total_weighted_score += element_weights[clean_word]
                                    matching_words_count += 1

                            # Calculate heuristic: sum of weighted words / total words in text
                            heuristic_score = total_weighted_score / len(element_words) if element_words else 0

                            element_info = {
                                "element": parent,
                                "text": text_element.strip(),
                                "tag": parent.name,
                                "selector": _generate_selector(parent),
                                "attributes": dict(parent.attrs) if parent.attrs else {},
                                "heuristic_score": heuristic_score
                            }
                            if element_info not in found_elements:
                                found_elements.append(element_info)
                            parent = parent.parent

            # Also search by attributes that might contain the target
            all_elements = soup.find_all(lambda tag: tag.name != '[document]')

            for elem in all_elements:
                # Check various attributes for matches
                attr_values = []
                for attr_name, attr_value in elem.attrs.items():
                    if isinstance(attr_value, str):
                        attr_values.append(attr_value.lower())
                    elif isinstance(attr_value, list):
                        attr_values.extend([str(val).lower() for val in attr_value])

                combined_attrs = ' '.join(attr_values).lower()

                # Calculate heuristic score based on attribute matches
                attr_words = combined_attrs.split()
                total_attr_score = 0
                matching_attr_words = 0

                for word in attr_words:
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if clean_word in element_weights:
                        total_attr_score += element_weights[clean_word]
                        matching_attr_words += 1

                attr_heuristic_score = total_attr_score / len(attr_words) if attr_words else 0

                # If there's a match in attributes, add the element
                if attr_heuristic_score > 0:
                    element_info = {
                        "element": elem,
                        "text": elem.get_text(strip=True)[:100],
                        "tag": elem.name,
                        "selector": _generate_selector(elem),
                        "attributes": dict(elem.attrs) if elem.attrs else {},
                        "heuristic_score": attr_heuristic_score
                    }
                    if element_info not in found_elements:
                        found_elements.append(element_info)

            # Apply search strategy
            if search_strategy == "by_importance":
                # Sort by element importance and heuristic score (prioritize semantic elements with high scores)
                importance_order = {
                    'button': 10, 'a': 9, 'input': 8, 'h1': 7, 'h2': 6, 'h3': 5,
                    'h4': 4, 'h5': 3, 'h6': 2, 'nav': 1, 'main': 1, 'section': 1,
                    'article': 1, 'aside': 1, 'footer': 1, 'header': 1
                }
                # Sort by importance first, then by heuristic score
                found_elements.sort(key=lambda x: (importance_order.get(x['tag'], 0), x['heuristic_score']), reverse=True)
            elif search_strategy == "breadth_first":
                # Sort by appearance order (earlier in document first)
                found_elements.sort(key=lambda x: html_content.find(str(x['element'])))
            elif search_strategy == "depth_first":
                # Sort by nesting depth (deeper elements first)
                found_elements.sort(key=lambda x: len(x['selector']), reverse=True)

            # Format results
            formatted_results = []
            for elem_info in found_elements:
                formatted_results.append({
                    "selector": elem_info["selector"],
                    "tag": elem_info["tag"],
                    "text": elem_info["text"],
                    "attributes": elem_info["attributes"],
                    "heuristic_score": elem_info["heuristic_score"]
                })

            return [TextContent(
                type="text",
                text=json.dumps({
                    "target_element": target_element,
                    "found_elements": formatted_results,
                    "total_found": len(formatted_results),
                    "search_strategy": search_strategy,
                    "message": f"Found {len(formatted_results)} elements matching '{target_element}' using {search_strategy} strategy with A* heuristic scoring"
                })
            )]

        elif name == "query_elements":
            selector = arguments["selector"]
            attributes = arguments.get("attributes", [])

            # Query elements and return their properties
            js_code = f"""
            (function() {{
                const elements = Array.from(document.querySelectorAll('{selector}'));
                const requestedAttributes = {json.dumps(attributes)};

                return elements.map(el => {{
                    const result = {{
                        tagName: el.tagName ? el.tagName.toLowerCase() : '',
                        textContent: el.textContent ? el.textContent.trim() : '',
                        boundingBox: el.getBoundingClientRect ? {{
                            x: el.getBoundingClientRect().x,
                            y: el.getBoundingClientRect().y,
                            width: el.getBoundingClientRect().width,
                            height: el.getBoundingClientRect().height
                        }} : null
                    }};

                    // Add specific attributes if requested
                    if (requestedAttributes.length > 0) {{
                        result.attributes = {{}};
                        for (const attrName of requestedAttributes) {{
                            if (el.hasAttribute(attrName)) {{
                                result.attributes[attrName] = el.getAttribute(attrName);
                            }}
                        }}
                    }} else {{
                        // Add common attributes
                        const commonAttrs = ['id', 'class', 'href', 'src', 'alt', 'title', 'name', 'value', 'type', 'role', 'aria-label'];
                        result.attributes = {{}};
                        for (const attrName of commonAttrs) {{
                            if (el.hasAttribute(attrName)) {{
                                result.attributes[attrName] = el.getAttribute(attrName);
                            }}
                        }}
                    }}

                    return result;
                }});
            }})();
            """

            elements = await page.evaluate(js_code)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "elements": elements,
                    "count": len(elements)
                })
            )]

        elif name == "find_elements_by_text":
            text = arguments["text"]
            case_sensitive = arguments.get("case_sensitive", False)

            # Find elements containing specific text
            comparison = "textContent.includes(text)" if case_sensitive else "textContent.toLowerCase().includes(text.toLowerCase())"

            js_code = f"""
            (function() {{
                const text = '{text}';
                const caseSensitive = {str(case_sensitive).lower()};

                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );

                const elements = new Set();
                let node;

                while (node = walker.nextNode()) {{
                    const textContent = node.textContent || '';
                    const comparison = caseSensitive ?
                        textContent.includes(text) :
                        textContent.toLowerCase().includes(text.toLowerCase());

                    if (comparison) {{
                        let parent = node.parentElement;
                        while (parent && parent !== document.body) {{
                            elements.add(parent);
                            parent = parent.parentElement;
                        }}
                    }}
                }}

                // Convert to array and get properties
                return Array.from(elements).map(el => {{
                    return {{
                        tagName: el.tagName ? el.tagName.toLowerCase() : '',
                        textContent: el.textContent ? el.textContent.trim().substring(0, 100) : '',
                        outerHTML: el.outerHTML.substring(0, 200),
                        attributes: {{
                            id: el.id || '',
                            class: el.className || '',
                            'data-testid': el.dataset.testid || ''
                        }},
                        boundingBox: el.getBoundingClientRect ? {{
                            x: el.getBoundingClientRect().x,
                            y: el.getBoundingClientRect().y,
                            width: el.getBoundingClientRect().width,
                            height: el.getBoundingClientRect().height
                        }} : null
                    }};
                }});
            }})();
            """

            elements = await page.evaluate(js_code)
            return [TextContent(
                type="text",
                text=json.dumps({
                    "elements": elements,
                    "count": len(elements)
                })
            )]

        elif name == "press_key":
            key = arguments["key"]
            await page.keyboard.press(key)
            await page.wait_for_load_state("networkidle")
            return [TextContent(
                type="text",
                text=json.dumps({"status": "success"})
            )]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "error": f"Unknown tool: {name}"
                })
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "tool": name
            })
        )]


def _generate_selector(element):
    """Helper function to generate a CSS selector for an element"""
    if element.get('id'):
        return f"#{element.get('id')}"
    elif element.get('class'):
        classes = '.'.join(element.get('class', []))
        return f"{element.name}.{classes}"
    elif element.get('name'):
        return f"{element.name}[name='{element.get('name')}']"
    elif element.get('href'):
        return f"{element.name}[href='{element.get('href')}']"
    else:
        # Generate a more specific selector based on position
        parent = element.parent
        if parent:
            siblings = [s for s in parent.find_all(element.name, recursive=False)]
            index = siblings.index(element) + 1 if element in siblings else 1
            return f"{parent.name} > {element.name}:nth-of-type({index})"
        return element.name


# --- Wrapper 1: MCP StdIO Wrapper ---
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent]:
    """MCP Wrapper for browser automation"""
    return await handle_tool_call(name, arguments)


# --- HTTP Server Support ---
if HTTP_AVAILABLE:
    app = FastAPI(title="MCP Browser Server")

    class ToolCallRequest(BaseModel):
        name: str
        arguments: Dict[str, Any] = {}

    @app.post("/call_tool")
    async def http_call_tool(request: ToolCallRequest):
        """Execute a tool via HTTP"""
        try:
            result = await handle_tool_call(request.name, request.arguments)
            if result and hasattr(result[0], 'text'):
                try:
                    return json.loads(result[0].text)
                except json.JSONDecodeError:
                    return {"result": result[0].text}
            return {"result": str(result)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "active_tab": tab_manager.active_tab_id}
else:
    app = None

async def run_http_server(port: int = 8000):
    """Run the server in HTTP mode using Uvicorn"""
    if not HTTP_AVAILABLE:
        print("Error: HTTP dependencies (fastapi, uvicorn) not installed.")
        return

    print(f"Starting MCP HTTP Server on port {port}...")
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        await cleanup_browser()

async def main():
    """Run the MCP server"""
    parser = argparse.ArgumentParser(description="MCP Browser Server")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()

    if args.http:
        await run_http_server(args.port)
    else:
        # STDIO Mode (Default for legacy navigate.py)
        from mcp.server.stdio import stdio_server
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options()
                )
        finally:
            await cleanup_browser()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
GitHub Release Information Extractor using Vision Models
Uses an MCP Server for browser automation and vision models for navigation.
"""

import json
import argparse
import base64
import os
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import config
import config
from agents import VisionAssistant
from knowledge import KnowledgeBase

import httpx

# HTTP-based MCP Client
class MCPClient:
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process = None
        self.base_url = "http://localhost:8000"
        self.client = httpx.AsyncClient(timeout=30.0)

    async def start(self):
        """Start the MCP server subprocess in HTTP mode"""
        print("Starting MCP Server (HTTP Mode)...")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        cmd = [sys.executable, self.server_script, "--http", "--port", "8000"]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=sys.stdout, # Pipe stdout to console for debug
            stderr=sys.stderr,
            env=env
        )
        
        # Wait for server to be ready
        print("Waiting for MCP Server to come online...")
        retries = 20
        while retries > 0:
            try:
                resp = await self.client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    print("MCP Initialized (HTTP).")
                    return
            except Exception:
                await asyncio.sleep(0.5)
                retries -= 1
        
        raise RuntimeError("Failed to start MCP Server (HTTP)")

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server via HTTP"""
        if arguments is None:
            arguments = {}
            
        try:
            response = await self.client.post(
                f"{self.base_url}/call_tool",
                json={"name": name, "arguments": arguments}
            )
            response.raise_for_status()
            
            # The server returns {"result": ...} where result is the tool output (often JSON dict)
            data = response.json()
            return data.get("result", data)
            
        except Exception as e:
            print(f"MCP Call Failed: {e}")
            raise e

    async def stop(self):
        """Stop the MCP server"""
        await self.client.aclose()
        if self.process:
             self.process.terminate()
             self.process.wait()
             print("MCP Server stopped.")

import hashlib
import time

# --- RL & Cost Management ---

class CostManager:
    def __init__(self):
        self.total_cost = 0
        self.history = []

    def add_cost(self, category: str, description: str = ""):
        cost = 0
        if category == "vision_model":
            cost = 10  # Vision model calls are expensive
        elif category == "think":
            cost = 2 # Internal thought process
        elif category == "lesser_astar":
            cost = 1 # DOM query
        elif category == "greater_astar":
            cost = 0.1 # Local scoring
        elif category == "action":
            cost = 5 # Clicking/typing
        
        self.total_cost += cost
        self.history.append({"category": category, "description": description, "cost": cost, "timestamp": time.time()})

class Navigator:
    def __init__(self, vision_model: str):
        self.client = MCPClient(server_script="mcp_server.py")
        self.vision = VisionAssistant(vision_model=vision_model)
        
        # RL Components
        self.cost_manager = CostManager()
        self.brain = KnowledgeBase()
        self.dom_cache_dir = Path("debug_artifacts/dom_cache")
        self.dom_cache_dir.mkdir(exist_ok=True, parents=True)

    async def setup(self):
        await self.client.start()

    async def get_cached_dom(self, url: str) -> Optional[str]:
        """Get cached DOM tree if available (Simulating Memory)"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.dom_cache_dir / f"{url_hash}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    async def cache_dom(self, url: str, content: str):
        """Cache DOM tree"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = self.dom_cache_dir / f"{url_hash}.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)

    async def run_heuristic_search(self, start_url: str, repo_name: str, final_goal: str) -> Dict[str, Any]:
        """
        Main RL-driven heuristic search loop.
        """
        # Initialize
        await self.client.call_tool("navigate_to_url", {"url": start_url})
        current_url = start_url
        
        # RL Initialization
        self.trajectory = []
        self.last_state_key = None
        
        # 0. Generate Prediction Model (Weighted Bag of Words)
        self.cost_manager.add_cost("think", "Generating Prediction Model")
        self.prediction_model = await self.vision.generate_prediction_model(final_goal, context=f"Repository '{repo_name}'")
        print(f"   [Prediction Model] Weights: {self.prediction_model.get('weights')}")
        
        # GitHub Specific Instructions (The "Captain's Orders")
        verification_prompt = f"""**Mission**: Find the LATEST release page for repository '{repo_name}'.

### Verification Checklist
1. Does the URL contain /releases/tag/ or /releases/latest?
2. Does the DOM or visual report show a "Latest" indicator?
3. Is version/tag information clearly present?
4. Is there any evidence this is NOT the latest release?"""

        extraction_hint = """### Common Patterns (GitHub)
- Version: Look for headings, release titles (e.g., "v2.0.0", "2024.1.29")
- Commit: Look for short SHA hashes (7 chars, e.g., "abc1234")  
- Author: Look for user links, "@username", "by username"
- Date: Look for relative ("2 days ago") or absolute dates"""
        
        for step in range(1, 15):
            print(f"\n--- Step {step} ---")
            
            # 1. Observe Current State (Cost: See)
            self.cost_manager.add_cost("see", "Observing current state")
            state = await self.client.call_tool("get_page_state", {"include_screenshot": True})
            current_url = state.get("url", "")
            screenshot_data = state.get("screenshot_base64", "")
            
            # Get Clean DOM (Cost: A* Greater implied)
            dom_res = await self.client.call_tool("get_clean_dom_tree", {"include_attributes": True, "semantic_only": True})
            dom_tree = dom_res.get("dom_tree", "")
            await self.cache_dom(current_url, dom_tree)

            # 2. Check Goal State (Brain Judgment)
            # Cost: We use Brain to verify if we are done.
            self.cost_manager.add_cost("think", "Checking if goal reached")
            goal_check = await self.vision.is_goal_reached(screenshot_data, current_url, dom_tree, verification_prompt=verification_prompt)
            
            if goal_check.get("goal_reached"):
                print(f"Goal Reached! Reasoning: {goal_check.get('reasoning')}")
                
                # REWARD SIGNAL: Success!
                print("   [Learning] Goal Reached. Backpropagating Reward +1.0")
                for experience in self.trajectory:
                    self.brain.update(experience["state"], experience["action"], 1.0)
                
                # FINAL EXTRACTION (High Cost but necessary)
                self.cost_manager.add_cost("see", "Final Extraction")
                html_content = state.get("html", "")
                data = await self.vision.extract_with_vision_and_html(
                    screenshot_data,
                    html_content[:50000],
                    instruction_hint=extraction_hint
                )
                return data

            # 3. Explicit A* Search Steps
            
            # 3. Explicit A* Search Steps
            
            # --- LESSER A* SEARCH (The DOM Searcher) ---
            # Role: Scans the raw DOM to find valid interactive elements.
            # "The DOM Searcher" - Finds candidates based on structural properties (tags, attributes).
            self.cost_manager.add_cost("lesser_astar", "DOM Searcher: finding candidates")
            
            # Pass prediction model weights for preliminary scoring
            candidates = await self.lesser_astar_search(
                query_selector="a, button, summary",
                weights=self.prediction_model.get("weights", {})
            )
            
            # --- GREATER A* SEARCH (The Page Searcher) ---
            # Role: Evaluates WHERE to go next (URL/page-level navigation).
            # "The Page Searcher" - Uses Heuristics (Weighted Words, Q-Values) to guide the agent.
            
            self.cost_manager.add_cost("greater_astar", f"Page Searcher: scoring {len(candidates)} candidates")
            
            scored_candidates = await self.greater_astar_search(
                candidates=candidates,
                current_url=current_url,
                prediction_model=self.prediction_model
            )
            
            # Select Best Node (Page-level decision)
            if not scored_candidates:
                print("No scored candidates.")
                break
                
            best_node = max(scored_candidates, key=lambda x: x.get("final_score", 0))
            score = best_node.get("final_score", 0)
            target_url = best_node.get("url", "")
            
            # Greater A* Output: Page-level destination (URL)
            print(f"   [Greater A*] Target Page: '{target_url}' (Score: {score:.2f})")
            
            if score > 0.5: # Threshold
                target_href = best_node.get('url', '')
                element_tag = best_node.get('tag', '').lower()
                
                # Execute Action based on element type
                try:
                    if target_href and target_href.startswith(('http', '/')):
                        # --- NAVIGATE ACTION (for elements with href) ---
                        # Links (a tags) should be navigated to directly, not clicked
                        full_url = target_href if target_href.startswith('http') else f"https://github.com{target_href}"
                        
                        action_key = f"navigate:{best_node['text'][:50]}"
                        self.trajectory.append({"state": self.last_state_key, "action": action_key})
                        
                        print(f"   [Greater A*] Navigate to URL: '{full_url}'")
                        self.cost_manager.add_cost("navigate", f"Navigating to {full_url}")
                        await self.client.call_tool("navigate_to_url", {"url": full_url})
                        
                    elif element_tag in ('button', 'input', 'summary'):
                        # --- CLICK ACTION (ONLY for buttons/inputs) ---
                        # Only click on interactive elements that don't have href
                        action_key = f"click:{best_node['text'][:50]}"
                        self.trajectory.append({"state": self.last_state_key, "action": action_key})
                        
                        selector_display = best_node.get('selector') or 'text-based'
                        print(f"   [Lesser A*] Click Button: '{best_node['text'][:50]}...' (Tag: {element_tag})")
                        self.cost_manager.add_cost("click", f"Clicking button {best_node['text']}")
                        
                        # Use click_button tool which asserts element is a button
                        if best_node.get("selector") and len(best_node["selector"]) > 5:
                            result = await self.client.call_tool("click_button", {"selector": best_node["selector"]})
                        else:
                            result = await self.client.call_tool("click_button", {"selector": f"text={best_node['text']}"})
                        
                        # Check assertion result
                        if result.get("status") == "error":
                            print(f"   [Assert Failed] {result.get('error')}")
                            continue
                    else:
                        # --- INVALID: Element has no href and is not a button ---
                        print(f"   [Warning] Skipping non-clickable element: '{best_node['text'][:30]}...' (Tag: {element_tag})")
                        print(f"   [Assert] Click action requires tag in: button, input, summary. Got: {element_tag}")
                        continue
                        
                except Exception as e:
                     print(f"Action failed: {e}")
                
                # Wait for page to load (fast check)
                await self.client.call_tool("is_page_loaded", {"timeout_ms": 3000})
                
                # Additional delay for visual mode (non-headless) so user can see
                # TODO: Skip this in headless mode
                await asyncio.sleep(2)
            else:
                print("   [Greater A*] No nodes satisfy threshold (>0.5). Search halted.")
                break
                
        return {}

    async def lesser_astar_search(self, query_selector: str, weights: dict = None) -> list:
        """
        Lesser A* Search (The DOM Searcher).
        Objective: Efficiently retrieve candidate elements from the DOM and applying structural heuristics.
        Heuristic: Score = Word_Value * Occurrence_Count * Font_Coefficient
        """
        res = await self.client.call_tool("query_elements", {"selector": query_selector})
        elements = res.get("elements", [])
        
        # Load Element Vocabulary (Font Coefficients)
        try:
            with open(os.path.join("data", "element_vocab.json"), "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                element_map = vocab_data.get("elements", {})
                default_coef = vocab_data.get("default_coefficient", 1.0)
                
            # Convert to simple key-value map for speed
            font_coefficients = {k: v["font_coefficient"] for k, v in element_map.items()}
        except Exception as e:
            print(f"Warning: Could not load element_vocab.json: {e}")
            font_coefficients = {"h1": 3.0, "a": 1.2, "button": 1.2} # Minimal Fallback
            default_coef = 1.0
        
        candidates = []
        for el in elements:
            text = el.get("textContent", "").strip()
            href = el.get("attributes", {}).get("href", "")
            selector = el.get("cssSelector")
            tag = el.get("tagName", "").lower()
            
            # Basic validation
            if len(text) > 1 or href:
                # Calculate Lesser Heuristic Score
                heuristic_score = 0
                if weights:
                    combined_text = (text + " " + href).lower()
                    font_coef = font_coefficients.get(tag, default_coef)
                    
                    for word, val in weights.items():
                        word_lower = word.lower()
                        count = combined_text.count(word_lower)
                        if count > 0:
                            # Formula: Word_Value * Occurrence * Font_Coefficient
                            term_score = val * count * font_coef
                            heuristic_score += term_score

                candidates.append({
                    "text": text[:100],
                    "url": href,
                    "selector": selector,
                    "tag": tag,
                    "lesser_score": heuristic_score
                })
        
        # Sort by Lesser Score (descending) to prioritize likely candidates
        candidates.sort(key=lambda x: x["lesser_score"], reverse=True)
        return candidates

    async def greater_astar_search(self, candidates: list, current_url: str, prediction_model: dict) -> list:
        """
        Greater A* Search (The Page Searcher).
        Objective: Determine the best path by scoring candidates against the Page Goal.
        Heuristic: H(n) = ActionValue + LocalScore + Q-Value
        """
        weights = prediction_model.get("weights", {})
        decay = prediction_model.get("decay_rate", 0.5)
        
        state_key = self.brain.get_state_key(current_url)
        self.last_state_key = state_key # Remember for update
        
        for cand in candidates:
            # 1. Base Action Value
            # "words give action meaning... text input button they are action"
            action_score = 0
            tag = cand.get("tag", "")
            if tag in ["button", "input", "a"]:
                action_score = 0.2  # Base value for being an interactive element
            
            # 2. Local Heuristic Score (Weighted Bag of Words)
            # "determined by the words of it's children like buttons href etc"
            local_score = 0
            
            # Combine all text sources: visible text, href attribute, title, etc.
            # identifying words in children effectively comes from aggregating textContent
            text_sources = [
                cand["text"].lower(),
                cand.get("url", "").lower() # href is a strong signal
            ]
            
            combined_text = " ".join(text_sources)
            
            for keyword, weight in weights.items():
                if keyword.lower() in combined_text:
                    local_score += weight
            
            # Boost if it looks like a version number (heuristic logic)
            import re
            if re.search(r"v\d+\.\d+", combined_text):
                 local_score += 0.5
            
            # 3. Q-Value (Self-Supervised Memory)
            action_key = f"click:{cand['text']}"
            q_value = self.brain.get_q_value(state_key, action_key)
            
            # 4. Final Combined Score
            # specific actions have higher value determined by words
            final_score = action_score + local_score + (q_value * 2.0)
            
            cand["local_score"] = local_score + action_score
            cand["q_value"] = q_value
            cand["final_score"] = final_score
            
        return candidates


    async def cleanup(self):
        """Clean up MCP resources"""
        await self.client.stop()

async def main():
    parser = argparse.ArgumentParser(description="GitHub Release Navigator (RL Powered)")
    parser.add_argument("--repo", help="Repository in format 'owner/repo'")
    parser.add_argument("--url", default=None, help="Starting URL")
    parser.add_argument("--prompt", help="Natural language prompt")
    parser.add_argument("--output", default=os.path.join("data", "output.json"), help="Output JSON file path")
    parser.add_argument("--vision-model", default=config.VISION_MODEL, help="Vision model to use")
    
    args = parser.parse_args()
    
    # RL Logic: Always start at Repo Root to discover path
    start_url = "about:blank"
    if args.repo:
        start_url = f"https://github.com/{args.repo}"
        prompt = f"Find latest release for {args.repo}"
    elif args.url:
        start_url = args.url
        prompt = args.prompt or "Navigate"
        
    print(f"Goal: {prompt}")
    print(f"Starting at: {start_url}")
    
    navigator = Navigator(vision_model=args.vision_model)
    
    try:
        await navigator.setup()
        
        # Check if we have a repo to guide the RL
        repo = args.repo if args.repo else "unknown"
        
        result = await navigator.run_heuristic_search(start_url, repo, prompt)
        
        # User Output (Clean)
        user_output = {
            "repository": repo,
            "latest_release": result
        }
        
        # Cost Log (Separate)
        cost_output = {
            "timestamp": time.time(),
            "repository": repo,
            "total_cost": navigator.cost_manager.total_cost,
            "cost_history": navigator.cost_manager.history
        }
        
        # Save User Output
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(user_output, f, indent=2)
            
        # Save Cost Log
        cost_log_path = Path(args.output).parent / "cost_log.json"
        with open(cost_log_path, 'w', encoding='utf-8') as f:
            json.dump(cost_output, f, indent=2)
            
        print(f"Results saved to {args.output}")
        print(f"Cost log saved to {cost_log_path}")
        print(f"Total Cost: {navigator.cost_manager.total_cost}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await navigator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
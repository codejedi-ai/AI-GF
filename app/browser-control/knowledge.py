"""
Knowledge Base - Q-Learning storage with semantic state representation.

Uses readable, transferable state keys instead of hashes for:
- Human-readable Q-table inspection
- Knowledge transfer across repositories
- Easy querying of learned experience
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse


class KnowledgeBase:
    """
    Q-Table with semantic state representation.
    
    State Key Format: "domain:page_type"
    Examples:
    - "github:repo_home" (any repo homepage)
    - "github:releases" (any releases page)
    - "github:release_tag" (specific release tag page)
    - "github:search_results" (search results page)
    
    This allows knowledge to transfer across different repositories.
    """
    
    # Page type patterns for GitHub
    PAGE_PATTERNS = {
        "release_tag": ["/releases/tag/", "/tree/"],
        "releases": ["/releases"],
        "issues": ["/issues"],
        "pull_requests": ["/pulls", "/pull/"],
        "code": ["/blob/", "/tree/"],
        "commits": ["/commits", "/commit/"],
        "actions": ["/actions"],
        "wiki": ["/wiki"],
        "search_results": ["/search"],
        "repo_home": None  # Default for repo root
    }
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.q_table_path = self.data_dir / "knowledge_q_table.json"
        self.q_table = self._load_q_table()
        self.learning_rate = 0.5  # Alpha
        self.discount_factor = 0.9  # Gamma
    
    def _load_q_table(self) -> Dict[str, Any]:
        """Load Q-table from disk."""
        if self.q_table_path.exists():
            try:
                with open(self.q_table_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load Q-Table: {e}. Starting fresh.")
        return {}
    
    def _save_q_table(self):
        """Save Q-table to disk."""
        try:
            with open(self.q_table_path, "w", encoding="utf-8") as f:
                json.dump(self.q_table, f, indent=2)
        except Exception as e:
            print(f"Failed to save Q-Table: {e}")
    
    def get_page_type(self, url: str) -> str:
        """
        Extract semantic page type from URL.
        
        Examples:
        - https://github.com/openclaw/openclaw → "repo_home"
        - https://github.com/openclaw/openclaw/releases → "releases"
        - https://github.com/openclaw/openclaw/releases/tag/v1.0 → "release_tag"
        """
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check against known patterns
        for page_type, patterns in self.PAGE_PATTERNS.items():
            if patterns:
                for pattern in patterns:
                    if pattern in path:
                        return page_type
        
        # Default: count path segments to determine type
        segments = [s for s in path.split('/') if s]
        if len(segments) <= 2:
            return "repo_home"
        
        return "other"
    
    def get_state_key(self, url: str) -> str:
        """
        Generate a human-readable, transferable state key.
        
        Format: "domain:page_type"
        
        Examples:
        - "github:repo_home"
        - "github:releases"
        - "github:release_tag"
        """
        parsed = urlparse(url)
        
        # Extract domain (simplified)
        domain = parsed.netloc.replace("www.", "").split(".")[0]
        
        # Get semantic page type
        page_type = self.get_page_type(url)
        
        return f"{domain}:{page_type}"
    
    def get_action_key(self, action_text: str, action_type: str = "click") -> str:
        """
        Generate a normalized action key.
        
        Cleans up action text to be more generalizable:
        - Removes timestamps, specific versions
        - Keeps semantic meaning
        """
        # Clean up common noise
        text = action_text.strip()
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        # Truncate very long text
        if len(text) > 50:
            text = text[:50] + "..."
        
        # Prefix with action type
        return f"{action_type}:{text}"
    
    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-Value for a state-action pair. Default is 0.0"""
        state_data = self.q_table.get(state_key, {})
        actions = state_data.get("actions", {})
        return actions.get(action_key, {}).get("q_value", 0.0)
    
    def get_all_q_values(self, state_key: str) -> Dict[str, float]:
        """Get all action Q-values for a state."""
        state_data = self.q_table.get(state_key, {})
        actions = state_data.get("actions", {})
        return {k: v.get("q_value", 0.0) for k, v in actions.items()}
    
    def update(self, state_key: str, action_key: str, reward: float):
        """
        Update Q-Value using Q-Learning formula.
        
        Q(s,a) = Q(s,a) + α * (reward - Q(s,a))
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {"actions": {}, "visit_count": 0}
        
        self.q_table[state_key]["visit_count"] = self.q_table[state_key].get("visit_count", 0) + 1
        
        actions = self.q_table[state_key]["actions"]
        if action_key not in actions:
            actions[action_key] = {"q_value": 0.0, "count": 0}
        
        current_q = actions[action_key]["q_value"]
        
        # Q-Learning Update
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        actions[action_key]["q_value"] = new_q
        actions[action_key]["count"] += 1
        
        self._save_q_table()
        print(f"   [Knowledge] Updated Q({state_key}, {action_key[:30]}...) = {new_q:.4f} (Reward: {reward})")
    
    def get_best_action(self, state_key: str) -> Optional[str]:
        """Get the action with highest Q-value for a state."""
        q_values = self.get_all_q_values(state_key)
        if not q_values:
            return None
        return max(q_values, key=q_values.get)
    
    def query_by_page_type(self, page_type: str) -> Dict[str, Any]:
        """
        Query all states matching a page type.
        
        Example: query_by_page_type("releases") returns all release page experiences.
        """
        results = {}
        for state_key, state_data in self.q_table.items():
            if f":{page_type}" in state_key:
                results[state_key] = state_data
        return results
    
    def query_by_domain(self, domain: str) -> Dict[str, Any]:
        """
        Query all states for a domain.
        
        Example: query_by_domain("github") returns all GitHub experiences.
        """
        results = {}
        for state_key, state_data in self.q_table.items():
            if state_key.startswith(f"{domain}:"):
                results[state_key] = state_data
        return results
    
    def get_experience_summary(self) -> Dict[str, Any]:
        """Get a summary of all learned experience."""
        summary = {
            "total_states": len(self.q_table),
            "states_by_domain": {},
            "states_by_page_type": {},
            "top_actions": []
        }
        
        for state_key, state_data in self.q_table.items():
            parts = state_key.split(":")
            if len(parts) == 2:
                domain, page_type = parts
                summary["states_by_domain"][domain] = summary["states_by_domain"].get(domain, 0) + 1
                summary["states_by_page_type"][page_type] = summary["states_by_page_type"].get(page_type, 0) + 1
            
            # Find top actions
            for action_key, action_data in state_data.get("actions", {}).items():
                if action_data.get("q_value", 0) > 0.5:
                    summary["top_actions"].append({
                        "state": state_key,
                        "action": action_key,
                        "q_value": action_data["q_value"],
                        "count": action_data["count"]
                    })
        
        # Sort top actions
        summary["top_actions"].sort(key=lambda x: x["q_value"], reverse=True)
        summary["top_actions"] = summary["top_actions"][:10]
        
        return summary
    
    def __repr__(self):
        return f"KnowledgeBase(states={len(self.q_table)}, path={self.q_table_path})"

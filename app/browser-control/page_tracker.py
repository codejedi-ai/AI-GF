"""
Page Tracker - Pydantic-based data structures for tracking navigation state.

Uses Pydantic v2 for:
- Type validation
- JSON serialization/deserialization
- Schema generation
- Immutable configurations
"""
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

from pydantic import BaseModel, Field, computed_field, model_validator


class PageState(BaseModel):
    """
    Represents the state of a single visited page.
    
    Pydantic model with automatic validation and serialization.
    """
    
    # Core identifiers
    url: str = Field(..., description="Full URL of the page")
    url_hash: str = Field(default="", description="MD5 hash of URL (8 chars)")
    title: str = Field(default="", description="Page title from <title> tag")
    timestamp: str = Field(default="", description="ISO timestamp of visit")
    
    # DOM Statistics
    dom_node_count: int = Field(default=0, ge=0, description="Total DOM nodes")
    link_count: int = Field(default=0, ge=0, description="Number of <a> elements")
    form_count: int = Field(default=0, ge=0, description="Number of <form> elements")
    interactive_count: int = Field(default=0, ge=0, description="Buttons, inputs, etc.")
    
    # Navigation metadata
    came_from_url: Optional[str] = Field(default=None, description="Previous page URL")
    came_from_action: Optional[str] = Field(default=None, description="Action taken: navigate, click, back")
    
    # Outgoing links (for graph building)
    outgoing_links: List[str] = Field(default_factory=list, description="All href links on page")
    
    # Extraction results
    extracted_data: Optional[Dict[str, Any]] = Field(default=None, description="Data extracted from page")
    
    # Flags
    is_goal_page: bool = Field(default=False, description="True if this page satisfies the goal")
    is_error_page: bool = Field(default=False, description="True if page had errors (404, etc)")
    visited_count: int = Field(default=1, ge=1, description="Number of times visited")
    
    # Scoring (from A* search)
    heuristic_score: float = Field(default=0.0, description="Score from Greater A*")
    q_value: float = Field(default=0.0, description="Q-value from knowledge base")
    
    model_config = {
        "extra": "allow",  # Allow additional fields
        "json_schema_extra": {
            "example": {
                "url": "https://github.com/openclaw/openclaw/releases",
                "title": "Releases · openclaw/openclaw",
                "link_count": 45,
                "is_goal_page": False
            }
        }
    }
    
    @model_validator(mode="after")
    def set_defaults(self) -> "PageState":
        """Set computed defaults after validation."""
        if not self.url_hash:
            self.url_hash = hashlib.md5(self.url.encode()).hexdigest()[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        return self
    
    @computed_field
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(self.url).netloc
    
    @computed_field
    @property
    def path(self) -> str:
        """Extract path from URL."""
        from urllib.parse import urlparse
        return urlparse(self.url).path


class NavigationEdge(BaseModel):
    """Represents a navigation action between two pages."""
    
    from_url_hash: str = Field(..., description="Source page hash")
    to_url_hash: str = Field(..., description="Destination page hash")
    action: str = Field(..., description="Action type: navigate, click, type")
    element_text: Optional[str] = Field(default=None, description="Text of clicked element")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    cost: float = Field(default=1.0, ge=0, description="Action cost for A*")


class SessionState(BaseModel):
    """
    Complete navigation session state.
    
    Tracks all pages, navigation graph, and session metadata.
    """
    
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    goal: str = Field(default="", description="Navigation goal description")
    start_url: str = Field(default="", description="Starting URL")
    
    # Page storage
    pages: Dict[str, PageState] = Field(default_factory=dict, description="url_hash -> PageState")
    
    # Navigation history
    history: List[str] = Field(default_factory=list, description="List of url_hashes in order")
    edges: List[NavigationEdge] = Field(default_factory=list, description="Navigation actions taken")
    
    # Current state
    current_url_hash: Optional[str] = Field(default=None)
    
    # Session metrics
    total_cost: float = Field(default=0.0, ge=0, description="Total navigation cost")
    goal_reached: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)
    
    model_config = {"extra": "allow"}
    
    @computed_field
    @property
    def page_count(self) -> int:
        """Number of unique pages visited."""
        return len(self.pages)
    
    @computed_field
    @property
    def action_count(self) -> int:
        """Number of navigation actions taken."""
        return len(self.history)


class PageTracker:
    """
    High-level API for tracking navigation state.
    
    Wraps SessionState with convenient methods for navigation flow.
    """
    
    def __init__(self, goal: str = "", start_url: str = "", session_id: str = None):
        self.session = SessionState(
            session_id=session_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            goal=goal,
            start_url=start_url
        )
    
    @property
    def current_page(self) -> Optional[PageState]:
        """Get current page state."""
        if self.session.current_url_hash:
            return self.session.pages.get(self.session.current_url_hash)
        return None
    
    def add_page(
        self,
        url: str,
        title: str = "",
        dom_node_count: int = 0,
        link_count: int = 0,
        form_count: int = 0,
        outgoing_links: List[str] = None,
        action: str = "navigate",
        element_text: str = None,
        cost: float = 1.0
    ) -> PageState:
        """Add or update a page in the tracker."""
        
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Check if already visited
        if url_hash in self.session.pages:
            page = self.session.pages[url_hash]
            # Update with new visit
            page.visited_count += 1
            page.timestamp = datetime.now().isoformat()
        else:
            # Create new page
            page = PageState(
                url=url,
                url_hash=url_hash,
                title=title,
                dom_node_count=dom_node_count,
                link_count=link_count,
                form_count=form_count,
                outgoing_links=outgoing_links or [],
                came_from_url=self.current_page.url if self.current_page else None,
                came_from_action=action
            )
            self.session.pages[url_hash] = page
        
        # Record navigation edge
        if self.session.current_url_hash:
            edge = NavigationEdge(
                from_url_hash=self.session.current_url_hash,
                to_url_hash=url_hash,
                action=action,
                element_text=element_text,
                cost=cost
            )
            self.session.edges.append(edge)
            self.session.total_cost += cost
        
        # Update current state
        self.session.current_url_hash = url_hash
        self.session.history.append(url_hash)
        
        return page
    
    def is_visited(self, url: str) -> bool:
        """Check if URL has been visited."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return url_hash in self.session.pages
    
    def get_visit_count(self, url: str) -> int:
        """Get visit count for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        page = self.session.pages.get(url_hash)
        return page.visited_count if page else 0
    
    def detect_cycle(self, url: str, lookback: int = 5) -> bool:
        """Check if navigating to URL would create a cycle."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        recent = self.session.history[-lookback:] if len(self.session.history) >= lookback else self.session.history
        return recent.count(url_hash) >= 2
    
    def get_unvisited_links(self, links: List[str]) -> List[str]:
        """Filter to only unvisited links."""
        return [link for link in links if not self.is_visited(link)]
    
    def mark_goal_reached(self, url: str = None):
        """Mark page as goal page."""
        if url:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            page = self.session.pages.get(url_hash)
        else:
            page = self.current_page
            
        if page:
            page.is_goal_page = True
            self.session.goal_reached = True
    
    def set_extracted_data(self, data: Dict[str, Any], url: str = None):
        """Store extraction results."""
        if url:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            page = self.session.pages.get(url_hash)
        else:
            page = self.current_page
            
        if page:
            page.extracted_data = data
    
    def get_navigation_path(self) -> List[Dict[str, Any]]:
        """Get ordered list of pages visited."""
        path = []
        for url_hash in self.session.history:
            page = self.session.pages.get(url_hash)
            if page:
                path.append({
                    "url": page.url,
                    "title": page.title,
                    "action": page.came_from_action or "start",
                    "timestamp": page.timestamp,
                    "is_goal": page.is_goal_page
                })
        return path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            "session_id": self.session.session_id,
            "goal": self.session.goal,
            "start_url": self.session.start_url,
            "pages_visited": self.session.page_count,
            "actions_taken": self.session.action_count,
            "total_cost": self.session.total_cost,
            "goal_reached": self.session.goal_reached,
            "current_url": self.current_page.url if self.current_page else None,
            "path": self.get_navigation_path()
        }
    
    def save(self, path: str = None) -> str:
        """Save session to JSON file."""
        if path is None:
            path = f"data/session_{self.session.session_id}.json"
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.session.model_dump_json(indent=2))
            
        return path
    
    @classmethod
    def load(cls, path: str) -> "PageTracker":
        """Load session from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
            
        session = SessionState.model_validate_json(data)
        tracker = cls()
        tracker.session = session
        return tracker
    
    def __repr__(self) -> str:
        return f"PageTracker(session={self.session.session_id}, pages={self.session.page_count}, goal_reached={self.session.goal_reached})"


# Factory function
def create_tracker(goal: str = "", start_url: str = "") -> PageTracker:
    """Create a new page tracker."""
    return PageTracker(goal=goal, start_url=start_url)

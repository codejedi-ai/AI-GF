#!/usr/bin/env python3
"""
Test script for the new A* pathfinding capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def test_astar_pathfinding():
    """Test the new A* pathfinding capabilities"""
    print("Testing A* Pathfinding Capabilities...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Test the A* pathfinding to find a specific element
        print("\n2. Testing A* pathfinding to find 'Issues'...")
        path_result = await client.call_tool("find_path_to_element", {
            "target_element": "Issues",
            "strategy": "shortest"
        })
        print(f"A* pathfinding result: {path_result}")
        
        # Test the A* pathfinding to find 'Pull requests'
        print("\n3. Testing A* pathfinding to find 'Pull requests'...")
        pr_path_result = await client.call_tool("find_path_to_element", {
            "target_element": "Pull requests",
            "strategy": "least_clicks"
        })
        print(f"Pull requests pathfinding result: {pr_path_result}")
        
        # Test the A* pathfinding with semantic strategy
        print("\n4. Testing A* pathfinding with semantic strategy...")
        semantic_path_result = await client.call_tool("find_path_to_element", {
            "target_element": "Sign in",
            "strategy": "semantic"
        })
        print(f"Semantic pathfinding result: {semantic_path_result}")
        
        print("\nAll A* pathfinding tests completed!")
        
    except Exception as e:
        print(f"Error during A* pathfinding testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_astar_pathfinding())
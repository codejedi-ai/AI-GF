#!/usr/bin/env python3
"""
Test script for the new A* Page Finder and A* DOM Finder capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def test_advanced_astar_features():
    """Test the new A* Page Finder and A* DOM Finder capabilities"""
    print("Testing Advanced A* Features: Page Finder & DOM Finder...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Test the A* Page Finder to find valuable pages about "documentation"
        print("\n2. Testing A* Page Finder for 'documentation'...")
        page_finder_result = await client.call_tool("find_valuable_pages", {
            "topic": "documentation",
            "max_pages": 3
        })
        print(f"Page Finder result: {page_finder_result}")
        
        # Test the A* Page Finder for "open source"
        print("\n3. Testing A* Page Finder for 'open source'...")
        oss_finder_result = await client.call_tool("find_valuable_pages", {
            "topic": "open source",
            "max_pages": 3
        })
        print(f"Open Source Page Finder result: {oss_finder_result}")
        
        # Test the A* DOM Finder to find "Sign in" element
        print("\n4. Testing A* DOM Finder for 'Sign in'...")
        dom_finder_result = await client.call_tool("find_element_on_page", {
            "target_element": "Sign in",
            "search_strategy": "by_importance"
        })
        print(f"DOM Finder result: Found {dom_finder_result.get('total_found', 0)} elements")
        if dom_finder_result.get('found_elements'):
            first_elem = dom_finder_result['found_elements'][0]
            print(f"  First element: {first_elem['tag']} with selector {first_elem['selector']}")
        
        # Test the A* DOM Finder with different strategy
        print("\n5. Testing A* DOM Finder with 'breadth_first' strategy...")
        bf_result = await client.call_tool("find_element_on_page", {
            "target_element": "Issues",
            "search_strategy": "breadth_first"
        })
        print(f"Breadth-first DOM Finder result: Found {bf_result.get('total_found', 0)} elements")
        
        # Test the A* DOM Finder for search input
        print("\n6. Testing A* DOM Finder for 'search'...")
        search_result = await client.call_tool("find_element_on_page", {
            "target_element": "search",
            "search_strategy": "by_importance"
        })
        print(f"Search DOM Finder result: Found {search_result.get('total_found', 0)} elements")
        
        print("\nAll Advanced A* features tests completed!")
        
    except Exception as e:
        print(f"Error during Advanced A* features testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_advanced_astar_features())
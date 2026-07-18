#!/usr/bin/env python3
"""
Test script for the new DOM querying capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from navigate import MCPClient


async def test_dom_querying():
    """Test the new DOM querying capabilities"""
    print("Testing MCP DOM Querying Capabilities...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Test get_dom_tree
        print("\n2. Testing get_dom_tree...")
        dom_result = await client.call_tool("get_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "max_depth": 5
        })
        dom_tree = dom_result.get("dom_tree", {})
        print(f"DOM tree retrieved. Root tag: {dom_tree.get('tagName', 'N/A')}")
        print(f"Number of top-level children: {len(dom_tree.get('children', []))}")
        
        # Test query_elements
        print("\n3. Testing query_elements...")
        query_result = await client.call_tool("query_elements", {
            "selector": "input[name='q']"  # GitHub search box
        })
        elements = query_result.get("elements", [])
        print(f"Found {len(elements)} search input elements")
        if elements:
            print(f"First element tag: {elements[0].get('tagName')}")
            print(f"Attributes: {elements[0].get('attributes', {})}")
        
        # Test find_elements_by_text
        print("\n4. Testing find_elements_by_text...")
        text_search_result = await client.call_tool("find_elements_by_text", {
            "text": "Sign in",
            "case_sensitive": False
        })
        text_elements = text_search_result.get("elements", [])
        print(f"Found {len(text_elements)} elements containing 'Sign in'")
        if text_elements:
            print(f"First element tag: {text_elements[0].get('tagName')}")
            print(f"Text content: {text_elements[0].get('textContent', '')[:50]}...")
        
        # Test more complex selector
        print("\n5. Testing complex selector query...")
        complex_query = await client.call_tool("query_elements", {
            "selector": "a[href*='/login']"
        })
        login_links = complex_query.get("elements", [])
        print(f"Found {len(login_links)} login-related links")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_dom_querying())
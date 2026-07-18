#!/usr/bin/env python3
"""
Test script for the new BeautifulSoup-enhanced DOM querying capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def test_beautifulsoup_dom():
    """Test the new BeautifulSoup-enhanced DOM querying capabilities"""
    print("Testing BeautifulSoup-Enhanced MCP DOM Capabilities...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Test get_clean_dom_tree
        print("\n2. Testing get_clean_dom_tree (BeautifulSoup-powered)...")
        clean_dom_result = await client.call_tool("get_clean_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "semantic_only": False
        })
        clean_dom_tree = clean_dom_result.get("clean_dom_tree", {})
        print(f"Clean DOM tree retrieved. Root tag: {clean_dom_tree.get('tagName', 'N/A')}")
        print(f"Number of top-level children: {len(clean_dom_tree.get('children', []))}")
        
        # Test get_clean_dom_tree with semantic filtering
        print("\n3. Testing get_clean_dom_tree with semantic filtering...")
        semantic_dom_result = await client.call_tool("get_clean_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "semantic_only": True
        })
        semantic_dom_tree = semantic_dom_result.get("clean_dom_tree", {})
        print(f"Semantic DOM tree retrieved. Root tag: {semantic_dom_tree.get('tagName', 'N/A')}")
        print(f"Number of top-level children: {len(semantic_dom_tree.get('children', []))}")
        
        # Compare with original DOM tree
        print("\n4. Comparing with original DOM tree...")
        original_dom_result = await client.call_tool("get_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "max_depth": 3
        })
        original_dom_tree = original_dom_result.get("dom_tree", {})
        print(f"Original DOM tree root tag: {original_dom_tree.get('tagName', 'N/A')}")
        print(f"Original DOM tree top-level children: {len(original_dom_tree.get('children', []))}")
        
        # Test query_elements
        print("\n5. Testing query_elements...")
        query_result = await client.call_tool("query_elements", {
            "selector": "input[name='q']"  # GitHub search box
        })
        elements = query_result.get("elements", [])
        print(f"Found {len(elements)} search input elements")
        if elements:
            print(f"First element tag: {elements[0].get('tagName')}")
            print(f"Attributes: {elements[0].get('attributes', {})}")
        
        # Test find_elements_by_text
        print("\n6. Testing find_elements_by_text...")
        text_search_result = await client.call_tool("find_elements_by_text", {
            "text": "Sign in",
            "case_sensitive": False
        })
        text_elements = text_search_result.get("elements", [])
        print(f"Found {len(text_elements)} elements containing 'Sign in'")
        if text_elements:
            print(f"First element tag: {text_elements[0].get('tagName')}")
            print(f"Text content: {text_elements[0].get('textContent', '')[:50]}...")
        
        print("\nAll BeautifulSoup-enhanced tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_beautifulsoup_dom())
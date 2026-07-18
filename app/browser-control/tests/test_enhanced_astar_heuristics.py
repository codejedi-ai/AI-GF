#!/usr/bin/env python3
"""
Test script for the enhanced A* algorithms with proper heuristic scoring
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def test_enhanced_astar_heuristics():
    """Test the enhanced A* algorithms with proper heuristic scoring"""
    print("Testing Enhanced A* Algorithms with Heuristic Scoring...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Test the enhanced A* Page Finder with heuristic scoring
        print("\n2. Testing Enhanced A* Page Finder for 'open source software'...")
        page_finder_result = await client.call_tool("find_valuable_pages", {
            "topic": "open source software",
            "max_pages": 3
        })
        print(f"Enhanced Page Finder result: {page_finder_result['message']}")
        top_pages = page_finder_result.get("top_pages", [])
        for i, page in enumerate(top_pages, 1):
            print(f"  {i}. '{page['text']}' -> {page['url']}")
            print(f"     Heuristic Score: {page['heuristic_score']:.4f}, Relevance: {page['relevance']}")
        
        # Test the enhanced A* Page Finder for "documentation"
        print("\n3. Testing Enhanced A* Page Finder for 'api documentation'...")
        api_docs_result = await client.call_tool("find_valuable_pages", {
            "topic": "api documentation",
            "max_pages": 3
        })
        print(f"API Docs Page Finder result: {api_docs_result['message']}")
        api_pages = api_docs_result.get("top_pages", [])
        for i, page in enumerate(api_pages, 1):
            print(f"  {i}. '{page['text']}' -> {page['url']}")
            print(f"     Heuristic Score: {page['heuristic_score']:.4f}, Relevance: {page['relevance']}")
        
        # Test the enhanced A* DOM Finder with heuristic scoring
        print("\n4. Testing Enhanced A* DOM Finder for 'sign in'...")
        dom_finder_result = await client.call_tool("find_element_on_page", {
            "target_element": "sign in",
            "search_strategy": "by_importance"
        })
        print(f"Enhanced DOM Finder result: {dom_finder_result['message']}")
        found_elements = dom_finder_result.get("found_elements", [])
        print(f"  Found {len(found_elements)} elements")
        if found_elements:
            top_element = found_elements[0]
            print(f"  Top element: {top_element['tag']} with selector '{top_element['selector']}'")
            print(f"  Heuristic Score: {top_element['heuristic_score']:.4f}")
            print(f"  Text: '{top_element['text']}'")
        
        # Test the enhanced A* DOM Finder for "pull requests"
        print("\n5. Testing Enhanced A* DOM Finder for 'pull requests'...")
        pr_finder_result = await client.call_tool("find_element_on_page", {
            "target_element": "pull requests",
            "search_strategy": "by_importance"
        })
        print(f"P/R Finder result: {pr_finder_result['message']}")
        pr_elements = pr_finder_result.get("found_elements", [])
        print(f"  Found {len(pr_elements)} elements")
        if pr_elements:
            top_pr_element = pr_elements[0]
            print(f"  Top element: {top_pr_element['tag']} with selector '{top_pr_element['selector']}'")
            print(f"  Heuristic Score: {top_pr_element['heuristic_score']:.4f}")
            print(f"  Text: '{top_pr_element['text']}'")
        
        print("\nAll Enhanced A* Heuristic Scoring tests completed!")
        
    except Exception as e:
        print(f"Error during Enhanced A* heuristic testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_enhanced_astar_heuristics())
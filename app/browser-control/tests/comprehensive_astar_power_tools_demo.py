#!/usr/bin/env python3
"""
Comprehensive demonstration of all A* capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def comprehensive_astar_power_tools_demo():
    """Comprehensive demonstration of all A* capabilities as power tools for Claude"""
    print("A* Power Tools for Claude - Comprehensive Demo")
    print("=" * 60)
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to GitHub homepage
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"   [OK] Navigation successful: {result['current_url']}")
        
        print("\n" + "="*60)
        print("PART 1: A* PAGE FINDER - Finding valuable pages")
        print("="*60)
        
        # Test A* Page Finder for documentation
        print("\n2. A* Page Finder: Finding documentation pages...")
        docs_result = await client.call_tool("find_valuable_pages", {
            "topic": "documentation",
            "max_pages": 3
        })
        print(f"   [OK] Found {len(docs_result['top_pages'])} valuable documentation pages:")
        for i, page in enumerate(docs_result['top_pages'], 1):
            print(f"     {i}. {page['text']} -> {page['url']} (Heuristic Score: {page['heuristic_score']:.4f})")
        
        # Test A* Page Finder for open source
        print("\n3. A* Page Finder: Finding open source related pages...")
        oss_result = await client.call_tool("find_valuable_pages", {
            "topic": "open source",
            "max_pages": 3
        })
        print(f"   [OK] Found {len(oss_result['top_pages'])} open source related pages:")
        for i, page in enumerate(oss_result['top_pages'], 1):
            print(f"     {i}. {page['text']} -> {page['url']}")
        
        print("\n" + "="*60)
        print("PART 2: A* DOM FINDER - Finding elements on current page")
        print("="*60)
        
        # Test A* DOM Finder for Sign in
        print("\n4. A* DOM Finder: Locating 'Sign in' elements...")
        signin_result = await client.call_tool("find_element_on_page", {
            "target_element": "Sign in",
            "search_strategy": "by_importance"
        })
        print(f"   [OK] Found {signin_result['total_found']} 'Sign in' elements")
        if signin_result['found_elements']:
            top_signin = signin_result['found_elements'][0]
            print(f"   [OK] Top result: {top_signin['tag']} element with selector '{top_signin['selector']}'")
        
        # Test A* DOM Finder for Issues
        print("\n5. A* DOM Finder: Locating 'Issues' elements...")
        issues_result = await client.call_tool("find_element_on_page", {
            "target_element": "Issues",
            "search_strategy": "by_importance"
        })
        print(f"   [OK] Found {issues_result['total_found']} 'Issues' elements")
        if issues_result['found_elements']:
            top_issues = issues_result['found_elements'][0]
            print(f"   [OK] Top result: {top_issues['tag']} element with selector '{top_issues['selector']}'")
        
        print("\n" + "="*60)
        print("PART 3: A* PATHFINDING - Finding optimal paths to elements")
        print("="*60)
        
        # Test A* Pathfinding
        print("\n6. A* Pathfinding: Finding path to 'Pull requests'...")
        pr_result = await client.call_tool("find_path_to_element", {
            "target_element": "Pull requests",
            "strategy": "shortest"
        })
        print(f"   [OK] Path found: {pr_result['path_found']}")
        if pr_result['path_found']:
            print(f"   [OK] Steps required: {len(pr_result['steps'])}")
            print(f"   [OK] Total cost: {pr_result['total_cost']}")
            for i, step in enumerate(pr_result['steps'], 1):
                print(f"     Step {i}: {step['action']} '{step['element_text']}' using selector '{step['selector']}'")
        
        print("\n" + "="*60)
        print("PART 4: COMBINING A* POWER TOOLS")
        print("="*60)
        
        # Demonstrate how Claude can use multiple A* tools together
        print("\n7. Combining A* tools: Find documentation pages, then elements on those pages...")
        print("   This shows how Claude can use A* Page Finder followed by A* DOM Finder")
        
        # Find a specific page using Page Finder
        specific_page_result = await client.call_tool("find_valuable_pages", {
            "topic": "api documentation",
            "max_pages": 1
        })
        
        if specific_page_result['top_pages']:
            api_doc_page = specific_page_result['top_pages'][0]
            print(f"   [OK] Found API documentation page: {api_doc_page['text']}")
            print(f"   [INFO] In a real scenario, Claude would navigate to: {api_doc_page['url']}")
            print("   [INFO] Then use A* DOM Finder to locate specific elements on that page")
        
        print("\n" + "="*60)
        print("ALL A* POWER TOOLS DEMONSTRATED SUCCESSFULLY!")
        print("="*60)
        
        print("\nA* Power Tools Summary for Claude:")
        print("[OK] A* Page Finder: Find top 3 valuable pages on any topic")
        print("[OK] A* DOM Finder: Locate specific elements on current page efficiently")
        print("[OK] A* Pathfinding: Find optimal paths to target elements")
        print("[OK] Multiple search strategies for different needs")
        print("[OK] Reduced computational load on Claude model")
        print("[OK] Faster, more reliable navigation")
        print("[OK] Combined usage for complex navigation tasks")
        
    except Exception as e:
        print(f"Error during A* power tools demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(comprehensive_astar_power_tools_demo())
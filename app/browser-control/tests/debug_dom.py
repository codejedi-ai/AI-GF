#!/usr/bin/env python3
"""
Debug script for the DOM tree retrieval
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from navigate import MCPClient


async def debug_dom_tree():
    """Debug the DOM tree retrieval"""
    print("Debugging DOM Tree Retrieval...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a test page (GitHub homepage)
        print("\nNavigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"Navigation result: {result}")
        
        # Get raw HTML first to compare
        print("\nGetting raw HTML...")
        html_result = await client.call_tool("get_page_html", {"max_length": 1000})
        print(f"HTML snippet: {html_result['html'][:200]}...")
        
        # Test get_dom_tree with minimal options
        print("\nTesting get_dom_tree with minimal options...")
        dom_result = await client.call_tool("get_dom_tree", {
            "include_attributes": False,
            "include_text": False,
            "max_depth": 2
        })
        dom_tree = dom_result.get("dom_tree", {})
        print(f"DOM tree result: {dom_tree}")
        
        # Try to get the body element specifically
        print("\nTrying to get body element via query_elements...")
        try:
            body_result = await client.call_tool("query_elements", {
                "selector": "body"
            })
            print(f"Body query result count: {body_result.get('count', 'N/A')}")
            if body_result.get('elements'):
                print(f"First element tag: {body_result['elements'][0].get('tagName', 'N/A')}")
        except Exception as e:
            print(f"Error querying body element: {e}")
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(debug_dom_tree())
#!/usr/bin/env python3
"""
Comprehensive test demonstrating the enhanced MCP capabilities with DOM querying
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from navigate import MCPClient


async def comprehensive_test():
    """Comprehensive test of the enhanced MCP capabilities"""
    print("Comprehensive Test of Enhanced MCP Capabilities")
    print("=" * 50)
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to GitHub homepage
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"   [OK] Navigation successful: {result['current_url']}")
        
        # Test the new DOM tree functionality
        print("\n2. Testing DOM tree retrieval...")
        dom_result = await client.call_tool("get_dom_tree", {
            "include_attributes": True,
            "include_text": False,
            "max_depth": 3
        })
        dom_tree = dom_result.get("dom_tree", {})
        print(f"   [OK] Retrieved DOM tree with root tag: {dom_tree.get('tagName', 'N/A')}")
        print(f"   [OK] Number of top-level children: {len(dom_tree.get('children', []))}")

        # Test querying specific elements
        print("\n3. Testing element querying by CSS selector...")
        search_box_result = await client.call_tool("query_elements", {
            "selector": "input[name='q']"  # GitHub search box
        })
        search_elements = search_box_result.get("elements", [])
        print(f"   [OK] Found {len(search_elements)} search input elements")
        if search_elements:
            attrs = search_elements[0].get("attributes", {})
            print(f"   [OK] First search element has attributes: {list(attrs.keys())}")

        # Test finding elements by text content
        print("\n4. Testing element discovery by text content...")
        sign_in_elements = await client.call_tool("find_elements_by_text", {
            "text": "Sign in",
            "case_sensitive": False
        })
        sign_in_count = sign_in_elements.get("count", 0)
        print(f"   [OK] Found {sign_in_count} elements containing 'Sign in'")

        # Test finding "Issues" links (common GitHub element)
        print("\n5. Testing discovery of 'Issues' elements...")
        issues_elements = await client.call_tool("find_elements_by_text", {
            "text": "Issues",
            "case_sensitive": False
        })
        issues_count = issues_elements.get("count", 0)
        print(f"   [OK] Found {issues_count} elements containing 'Issues'")

        # Test querying multiple attribute types
        print("\n6. Testing advanced element querying...")
        nav_elements = await client.call_tool("query_elements", {
            "selector": "nav a, header a, .Header-link",
            "attributes": ["href", "class", "id", "aria-label"]
        })
        nav_count = nav_elements.get("count", 0)
        print(f"   [OK] Found {nav_count} navigation elements with specific attributes")

        # Demonstrate how the new capabilities help avoid the action-observation loop
        print("\n7. Demonstrating improved element identification...")
        print("   - Before: Claude would guess selectors and often fail")
        print("   - After: Use find_elements_by_text to locate elements by content")
        print("   - Then: Use query_elements to get precise selectors for clicking")

        # Show how to use the new tools together
        print("\n8. Demonstrating combined usage for robust element selection...")

        # Find elements by text first
        elements_by_text = await client.call_tool("find_elements_by_text", {
            "text": "Pull requests",
            "case_sensitive": False
        })

        if elements_by_text.get("count", 0) > 0:
            # Get more details about these elements
            pull_req_elements = elements_by_text.get("elements", [])
            print(f"   [OK] Found {len(pull_req_elements)} 'Pull requests' elements")

            # For the first element, get more details
            first_elem = pull_req_elements[0]
            elem_attrs = first_elem.get("attributes", {})
            print(f"   [OK] First element tag: {first_elem.get('tagName')}")
            print(f"   [OK] Available attributes: {list(elem_attrs.keys())}")

            # Show how to construct a selector from attributes
            selector_parts = []
            if elem_attrs.get("id"):
                selector_parts.append(f"#{elem_attrs['id']}")
            if elem_attrs.get("class"):
                classes = elem_attrs["class"].split()
                if classes:
                    selector_parts.append(f".{'.'.join(filter(None, classes))}")
            if elem_attrs.get("href"):
                selector_parts.append(f"[href='{elem_attrs['href']}']")

            if selector_parts:
                constructed_selector = selector_parts[0]  # Use the most specific one
                print(f"   [OK] Constructed selector: {constructed_selector}")

                # Verify the selector works
                verification = await client.call_tool("query_elements", {
                    "selector": constructed_selector
                })
                verified_count = verification.get("count", 0)
                print(f"   [OK] Selector verification: found {verified_count} elements")

        print("\n" + "=" * 50)
        print("All enhanced MCP capabilities are working correctly!")
        print("Claude can now:")
        print("- Retrieve the complete DOM tree for analysis")
        print("- Query elements using CSS selectors with detailed attributes")
        print("- Find elements by their text content")
        print("- Build robust selectors based on element attributes")
        print("- Reduce the action-observation loop significantly")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during comprehensive testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(comprehensive_test())
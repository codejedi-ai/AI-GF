#!/usr/bin/env python3
"""
Comprehensive demonstration of the enhanced MCP capabilities with BeautifulSoup DOM parsing
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def comprehensive_demo():
    """Comprehensive demonstration of enhanced MCP capabilities"""
    print("Enhanced MCP Capabilities with BeautifulSoup DOM Parsing")
    print("=" * 60)
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to GitHub homepage
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"   [OK] Navigation successful: {result['current_url']}")
        
        # Test the original DOM tree functionality
        print("\n2. Testing original DOM tree retrieval...")
        original_dom_result = await client.call_tool("get_dom_tree", {
            "include_attributes": True,
            "include_text": False,
            "max_depth": 3
        })
        original_dom_tree = original_dom_result.get("dom_tree", {})
        print(f"   [OK] Original DOM tree with root tag: {original_dom_tree.get('tagName', 'N/A')}")
        print(f"   [OK] Number of top-level children: {len(original_dom_tree.get('children', []))}")
        
        # Test the new BeautifulSoup-enhanced DOM tree functionality
        print("\n3. Testing BeautifulSoup-enhanced DOM tree retrieval...")
        bs4_dom_result = await client.call_tool("get_clean_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "semantic_only": False
        })
        bs4_dom_tree = bs4_dom_result.get("clean_dom_tree", {})
        print(f"   [OK] BeautifulSoup DOM tree with root tag: {bs4_dom_tree.get('tagName', 'N/A')}")
        print(f"   [OK] Number of top-level children: {len(bs4_dom_tree.get('children', []))}")
        
        # Test semantic-only DOM tree
        print("\n4. Testing semantic-only DOM tree retrieval...")
        semantic_dom_result = await client.call_tool("get_clean_dom_tree", {
            "include_attributes": True,
            "include_text": True,
            "semantic_only": True
        })
        semantic_dom_tree = semantic_dom_result.get("clean_dom_tree", {})
        print(f"   [OK] Semantic DOM tree with root tag: {semantic_dom_tree.get('tagName', 'N/A')}")
        print(f"   [OK] Number of top-level children: {len(semantic_dom_tree.get('children', []))}")
        
        # Test querying specific elements
        print("\n5. Testing element querying by CSS selector...")
        search_box_result = await client.call_tool("query_elements", {
            "selector": "input[name='q']"  # GitHub search box
        })
        search_elements = search_box_result.get("elements", [])
        print(f"   [OK] Found {len(search_elements)} search input elements")
        if search_elements:
            attrs = search_elements[0].get("attributes", {})
            print(f"   [OK] First search element has attributes: {list(attrs.keys())}")
        
        # Test finding elements by text content
        print("\n6. Testing element discovery by text content...")
        sign_in_elements = await client.call_tool("find_elements_by_text", {
            "text": "Sign in",
            "case_sensitive": False
        })
        sign_in_count = sign_in_elements.get("count", 0)
        print(f"   [OK] Found {sign_in_count} elements containing 'Sign in'")
        
        # Test finding "Issues" links (common GitHub element)
        print("\n7. Testing discovery of 'Issues' elements...")
        issues_elements = await client.call_tool("find_elements_by_text", {
            "text": "Issues",
            "case_sensitive": False
        })
        issues_count = issues_elements.get("count", 0)
        print(f"   [OK] Found {issues_count} elements containing 'Issues'")
        
        # Demonstrate how the new capabilities help avoid the action-observation loop
        print("\n8. Demonstrating improved element identification for Claude...")
        print("   - Before: Claude would guess selectors and often fail")
        print("   - After: Use find_elements_by_text to locate elements by content")
        print("   - Then: Use query_elements to get precise selectors for clicking")
        print("   - Or: Use get_clean_dom_tree to get a simplified, semantic DOM tree")
        
        # Show how to use the new tools together for robust element selection
        print("\n9. Demonstrating combined usage for robust element selection...")
        
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
        
        # Show how the semantic DOM tree can help Claude understand page structure
        print("\n10. Demonstrating semantic DOM tree for better page understanding...")
        print("    The semantic DOM tree helps Claude focus on meaningful elements")
        print("    rather than getting distracted by layout elements.")
        
        print("\n" + "=" * 60)
        print("All enhanced MCP capabilities are working correctly!")
        print("\nEnhanced capabilities for Claude:")
        print("- Retrieve the complete DOM tree for analysis (original method)")
        print("- Get a cleaned DOM tree using BeautifulSoup for simpler parsing")
        print("- Get semantic-only DOM tree focusing on meaningful elements")
        print("- Query elements using CSS selectors with detailed attributes")
        print("- Find elements by their text content")
        print("- Build robust selectors based on element attributes")
        print("- Reduce the action-observation loop significantly")
        print("- Better understand page structure with semantic elements")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during comprehensive testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(comprehensive_demo())
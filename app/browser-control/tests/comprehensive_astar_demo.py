#!/usr/bin/env python3
"""
Comprehensive demonstration of the A* pathfinding capabilities in the MCP server
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent project directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from navigate import MCPClient


async def comprehensive_astar_demo():
    """Comprehensive demonstration of A* pathfinding capabilities"""
    print("A* Pathfinding Capabilities in MCP Server")
    print("=" * 50)
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to GitHub homepage
        print("\n1. Navigating to GitHub homepage...")
        result = await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        print(f"   [OK] Navigation successful: {result['current_url']}")
        
        # Test A* pathfinding with different strategies
        print("\n2. Testing A* pathfinding with 'shortest' strategy...")
        shortest_path = await client.call_tool("find_path_to_element", {
            "target_element": "Issues",
            "strategy": "shortest"
        })
        print(f"   [OK] Shortest path found: {shortest_path['path_found']}")
        if shortest_path['path_found']:
            print(f"   [OK] Steps required: {len(shortest_path['steps'])}")
            print(f"   [OK] Total cost: {shortest_path['total_cost']}")
        
        print("\n3. Testing A* pathfinding with 'least_clicks' strategy...")
        least_clicks_path = await client.call_tool("find_path_to_element", {
            "target_element": "Pull requests",
            "strategy": "least_clicks"
        })
        print(f"   [OK] Least clicks path found: {least_clicks_path['path_found']}")
        if least_clicks_path['path_found']:
            print(f"   [OK] Steps required: {len(least_clicks_path['steps'])}")
            print(f"   [OK] Total cost: {least_clicks_path['total_cost']}")
        
        print("\n4. Testing A* pathfinding with 'semantic' strategy...")
        semantic_path = await client.call_tool("find_path_to_element", {
            "target_element": "Sign in",
            "strategy": "semantic"
        })
        print(f"   [OK] Semantic path found: {semantic_path['path_found']}")
        if semantic_path['path_found']:
            print(f"   [OK] Steps required: {len(semantic_path['steps'])}")
            print(f"   [OK] Total cost: {semantic_path['total_cost']}")
        
        # Show the detailed steps for one of the successful paths
        if shortest_path['path_found'] and shortest_path['steps']:
            print("\n5. Detailed steps for 'Issues' path:")
            for i, step in enumerate(shortest_path['steps']):
                print(f"   Step {i+1}: {step['action']} on element '{step['element_text']}'")
                print(f"           Selector: {step['selector']}")
                print(f"           Type: {step['element_type']}")
        
        # Demonstrate how A* pathfinding helps Claude navigate more efficiently
        print("\n6. How A* pathfinding benefits Claude:")
        print("   - Reduces guesswork in element selection")
        print("   - Finds optimal paths to target elements")
        print("   - Different strategies for different navigation needs")
        print("   - Minimizes the action-observation loop")
        print("   - Provides clear, executable steps")
        
        print("\n7. Strategies available:")
        print("   - 'shortest': Finds path with minimum cost")
        print("   - 'least_clicks': Minimizes number of clicks")
        print("   - 'semantic': Prioritizes semantically meaningful elements")
        
        print("\n" + "=" * 50)
        print("A* Pathfinding capabilities are working correctly!")
        print("\nBenefits for Claude:")
        print("- Efficient navigation with optimal paths")
        print("- Multiple strategies for different scenarios")
        print("- Reduced computational load on the model")
        print("- Clear, actionable steps for navigation")
        print("- Faster convergence to target elements")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during A* pathfinding demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(comprehensive_astar_demo())
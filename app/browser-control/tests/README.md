# MCP Server Tests

This directory contains test files for the enhanced MCP server capabilities with DOM querying, BeautifulSoup parsing, and A* pathfinding.

## Test Files

- `test_bs4_dom.py` - Tests the BeautifulSoup-enhanced DOM querying capabilities
- `comprehensive_demo.py` - Comprehensive demonstration of DOM features
- `test_astar_pathfinding.py` - Tests the A* pathfinding functionality
- `comprehensive_astar_demo.py` - Comprehensive demonstration of A* features
- `test_advanced_astar.py` - Tests advanced A* features (Page Finder & DOM Finder)
- `comprehensive_astar_power_tools_demo.py` - Comprehensive demonstration of A* power tools
- `test_enhanced_astar_heuristics.py` - Tests enhanced A* algorithms with proper heuristic scoring
- `test_mcp_dom.py` - Basic test of DOM querying capabilities
- `debug_dom.py` - Debug script for DOM tree retrieval
- `simple_dom_test.py` - Simple test for DOM tree functionality

## Running Tests

To run the tests:

```bash
# Run a specific test
python tests/test_bs4_dom.py

# Run the comprehensive DOM demo
python tests/comprehensive_demo.py

# Run the A* pathfinding test
python tests/test_astar_pathfinding.py

# Run the comprehensive A* demo
python tests/comprehensive_astar_demo.py
```

## Enhanced Capabilities

The tests verify the following enhanced capabilities:

1. **Original DOM Tree Retrieval** - Traditional DOM tree with full structure
2. **BeautifulSoup-Enhanced DOM Tree** - Cleaned DOM tree using BeautifulSoup
3. **Semantic-Only DOM Tree** - DOM tree with only semantic elements
4. **A* Pathfinding** - Optimal pathfinding to target elements
5. **Element Querying** - Query elements using CSS selectors
6. **Text-Based Element Discovery** - Find elements by text content
7. **Robust Selector Construction** - Build selectors from element attributes
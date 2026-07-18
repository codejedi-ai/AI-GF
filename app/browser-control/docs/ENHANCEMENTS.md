# Enhanced MCP Server with BeautifulSoup DOM Parsing and Advanced A* Algorithms

## Overview

This enhancement adds powerful DOM querying capabilities to the MCP server using BeautifulSoup for cleaner DOM parsing and advanced A* algorithms for efficient navigation. The enhancements address the issue where Claude had trouble pinpointing elements by providing multiple ways to find and interact with elements on the page, plus intelligent pathfinding to navigate efficiently. The new A* capabilities include both page finding and DOM element finding as powerful tools for Claude.

## New Features

### 1. BeautifulSoup-Enhanced DOM Tree (`get_clean_dom_tree`)
- Parses HTML using BeautifulSoup for cleaner, more reliable DOM structure
- Provides option to include only semantic elements (`semantic_only` parameter)
- More resilient to malformed HTML than pure JavaScript parsing

### 2. Semantic-Only DOM Tree
- Filters out non-semantic elements like divs and spans
- Focuses on meaningful elements like headers, navs, articles, buttons, etc.
- Helps Claude understand page structure better

### 3. A* Page Finder Algorithm (`find_valuable_pages`)
- Finds the top 3 most valuable pages related to a topic using A* concepts
- Implements proper heuristic scoring: sum of weighted words / total words in link text
- Scores pages based on relevance to the given topic using bag-of-words approach
- Enables Claude to discover relevant content efficiently
- Reduces need for manual exploration

### 4. A* DOM Finder Algorithm (`find_element_on_page`)
- Finds specific elements on the current page using A* search concepts
- Implements proper heuristic scoring: sum of weighted words / total words in element text or attributes
- Multiple search strategies: by importance, breadth-first, depth-first
- Prioritizes semantic elements that are more likely to be relevant
- Provides efficient element location without guesswork

### 5. A* Pathfinding Algorithm (`find_path_to_element`)
- Finds optimal paths to target elements using A* algorithm
- Multiple strategies: shortest path, least clicks, semantic prioritization
- Reduces computational load on the Claude model
- Provides clear, executable navigation steps

### 6. Improved Element Discovery
- `find_elements_by_text`: Find elements containing specific text
- `query_elements`: Query elements with detailed attributes using CSS selectors
- Enhanced selector construction from element attributes

### 7. Reduced Action-Observation Loop
- Multiple fallback strategies when element selection fails
- Semantic DOM tree helps Claude understand page structure
- A* algorithms provide efficient navigation routes
- Better error recovery and element identification

## Tools Added

### `get_clean_dom_tree`
- **Description**: Get a cleaned DOM tree using BeautifulSoup with simplified structure
- **Parameters**:
  - `include_attributes` (boolean): Include element attributes (default: true)
  - `include_text` (boolean): Include element text content (default: true)
  - `semantic_only` (boolean): Only include semantic elements (default: false)

### `find_valuable_pages`
- **Description**: Use A* algorithm concepts to find the top 3 most valuable pages related to a topic, starting from the current page
- **Parameters**:
  - `topic` (string): Topic or subject to find valuable pages for
  - `depth_limit` (integer): Maximum depth to explore from current page (default: 2)
  - `max_pages` (integer): Maximum number of pages to return (default: 3)

### `find_element_on_page`
- **Description**: Use A* algorithm to find a specific element on the current page efficiently
- **Parameters**:
  - `target_element` (string): Description or text of the target element to find on the current page
  - `search_strategy` (string): Strategy for searching: 'breadth_first', 'depth_first', 'by_importance' (default: 'by_importance')

### `find_path_to_element`
- **Description**: Use A* algorithm to find the optimal path to a target element from the current page state
- **Parameters**:
  - `target_element` (string): Description or text of the target element to find
  - `strategy` (string): Strategy for pathfinding: 'shortest', 'least_clicks', 'semantic' (default: 'shortest')

### Enhanced `get_dom_tree`
- **Description**: Original DOM tree retrieval with element attributes
- **Parameters**:
  - `include_attributes` (boolean): Include element attributes (default: true)
  - `include_text` (boolean): Include element text content (default: true)
  - `max_depth` (integer): Maximum depth to traverse in the DOM (default: 10)

### `query_elements`
- **Description**: Query elements on the page using CSS selectors and return their properties
- **Parameters**:
  - `selector` (string): CSS selector to query elements
  - `attributes` (array): Specific attributes to retrieve

### `find_elements_by_text`
- **Description**: Find elements containing specific text content
- **Parameters**:
  - `text` (string): Text to search for in elements
  - `case_sensitive` (boolean): Whether the search should be case sensitive (default: false)

## Benefits for Claude

1. **Better Element Identification**: Instead of guessing selectors, Claude can use text content to find elements
2. **Reduced Failure Rate**: Multiple fallback strategies when initial element selection fails
3. **Improved Understanding**: Semantic DOM trees help Claude understand page structure
4. **More Reliable Interactions**: Detailed element attributes enable more precise selector construction
5. **Shorter Action-Observation Loops**: Better element discovery reduces the need for repeated attempts
6. **Efficient Navigation**: A* pathfinding finds optimal routes to targets
7. **Powerful Page Discovery**: A* Page Finder discovers relevant content quickly
8. **Efficient Element Location**: A* DOM Finder locates elements without guesswork
9. **Reduced Model Compute**: All A* algorithms happen in MCP server, not in Claude
10. **Multiple Strategies**: Different approaches for different navigation scenarios
11. **Combined Usage**: A* tools can be used together for complex navigation tasks

## Implementation Details

- BeautifulSoup is used for robust HTML parsing
- JavaScript evaluation is still used for dynamic content
- A* algorithms implemented with customizable heuristics and strategies
- Backward compatibility maintained with existing tools
- All new tools follow the same response format as existing tools
- Error handling improved throughout

## Files Modified

- `mcp_server.py`: Added new tools, BeautifulSoup integration, and advanced A* algorithms
- `navigate.py`: Updated to use new DOM querying and A* capabilities
- Created `tests/` directory: Moved all test files and added new tests

## Tests Available

All tests have been moved to the `tests/` directory:
- `test_bs4_dom.py`: Tests BeautifulSoup-enhanced capabilities
- `comprehensive_demo.py`: Full demonstration of DOM features
- `test_astar_pathfinding.py`: Tests A* pathfinding functionality
- `comprehensive_astar_demo.py`: Full demonstration of A* features
- `test_advanced_astar.py`: Tests advanced A* features (Page Finder & DOM Finder)
- `comprehensive_astar_power_tools_demo.py`: Full demonstration of A* power tools
- Other test files for specific functionality
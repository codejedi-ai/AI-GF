#!/usr/bin/env python3
"""
Simple test for DOM tree retrieval
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from navigate import MCPClient


async def simple_test():
    """Simple test for DOM tree retrieval"""
    print("Simple DOM Tree Test...")
    
    # Start the MCP server client
    client = MCPClient("mcp_server.py")
    
    try:
        await client.start()
        
        # Navigate to a simple test page
        print("\nNavigating to GitHub homepage...")
        await client.call_tool("navigate_to_url", {"url": "https://github.com"})
        
        # Test a simple JavaScript evaluation to see if there's an issue
        print("\nTesting simple JS evaluation...")
        simple_js_result = await client.call_tool("get_page_html", {"max_length": 100})
        print(f"Simple JS result: {simple_js_result['html'][:50]}...")
        
        # Try to get the DOM tree with a very simple approach
        print("\nTesting DOM tree with simple JS...")
        # Let's create a custom call to test the JS directly
        import json
        
        _request_id = 100  # Using a high ID to avoid conflicts
        request = {
            "jsonrpc": "2.0",
            "id": _request_id,
            "method": "tools/call",
            "params": {
                "name": "get_dom_tree",
                "arguments": {"max_depth": 1}
            }
        }

        # Send Request
        request_json = json.dumps(request)
        client.process.stdin.write(request_json.encode('utf-8') + b"\n")
        await client.process.stdin.drain()

        # Read Response
        while True:
            line = await client.process.stdout.readline()
            if not line:
                raise Exception("MCP Server closed connection unexpectedly")

            try:
                message = json.loads(line.decode('utf-8'))
                if message.get("id") == _request_id:
                    if "error" in message:
                        print(f"MCP Error: {message['error']}")
                        break

                    # MCP returns a list of content items (TextContent/ImageContent)
                    # We usually want the text from the first item which contains our JSON result
                    result = message.get("result", {})
                    content = result.get("content", [])
                    if content and content[0]["type"] == "text":
                        # The server returns JSON string inside the text field
                        try:
                            dom_result = json.loads(content[0]["text"])
                            print(f"DOM tree result: {dom_result}")
                            break
                        except:
                            print(f"DOM tree result (raw): {content[0]['text']}")
                            break
                    print(f"DOM tree result (content): {content}")
                    break
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(simple_test())
#!/usr/bin/env python3
"""
Development runner for Service 2 - Runs the WebSocket server locally
"""
import os
import sys

# Set up the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set default environment variables if not present
if not os.getenv("VISION_MODEL"):
    os.environ["VISION_MODEL"] = "gpt-4o"

# Run the app wrapper
if __name__ == "__main__":
    from app_wrapper import app
    import uvicorn
    
    print("=" * 60)
    print("Darci Navigation Service - Development Mode")
    print("=" * 60)
    print("\nEndpoints:")
    print("  - HTTP API:    http://localhost:8000/")
    print("  - WebSocket:   ws://localhost:8000/ws")
    print("  - API Docs:    http://localhost:8000/docs")
    print("  - Health:      http://localhost:8000/health")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

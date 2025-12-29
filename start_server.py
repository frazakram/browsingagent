#!/usr/bin/env python3
"""
Start the web server for the browsing agent.
"""
import asyncio
import sys
import socket

# Windows requires ProactorEventLoop for subprocess support (needed by Playwright)
# This MUST be set before importing uvicorn or any other async library
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port: int = 8000) -> int:
    """Find a free port starting from start_port."""
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

if __name__ == "__main__":
    port = 8000
    
    # Check if port is in use
    if is_port_in_use(port):
        print(f"⚠️  Port {port} is already in use.")
        if sys.platform == "win32":
            print(f"💡 Tip: Kill existing processes with: netstat -ano | findstr :{port}")
        else:
            print(f"💡 Tip: Kill existing processes with: lsof -ti:{port} | xargs kill -9")
        print(f"🔄 Trying to find a free port...")
        port = find_free_port(port)
        print(f"✅ Using port {port} instead")
    
    print(f"🚀 Starting server on http://localhost:{port}")
    print(f"📝 Press Ctrl+C to stop the server\n")
    
    # On Windows, disable reload because the subprocess doesn't inherit the event loop policy
    # The StatReload mechanism spawns a new Python process with default SelectorEventLoop
    use_reload = sys.platform != "win32"
    
    if sys.platform == "win32":
        print("ℹ️  Auto-reload disabled on Windows (restart server manually for code changes)\n")
    
    uvicorn.run(
        "src.web_server:app",
        host="0.0.0.0",
        port=port,
        reload=use_reload,
        log_level="info",
    )


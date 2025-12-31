"""
Test script to verify Chrome CDP connection.
Run this AFTER starting Chrome with remote debugging.

Steps:
1. Close ALL Chrome windows
2. Open Command Prompt and run: start chrome --remote-debugging-port=9222
3. Run this script: python test_chrome_connection.py
"""

import asyncio
import sys
import subprocess
import urllib.request
import json

# Windows requires ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def check_chrome_debug_port():
    """Check if Chrome is running with debug port open."""
    print("=" * 60)
    print("CHROME DEBUG PORT TEST")
    print("=" * 60)
    
    try:
        # Try to connect to Chrome's debug endpoint
        url = "http://localhost:9222/json/version"
        print(f"\n🔍 Checking {url}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5)
        data = json.loads(response.read().decode())
        
        print("\n✅ SUCCESS! Chrome is running with remote debugging enabled!")
        print(f"\n📋 Chrome Info:")
        print(f"   Browser: {data.get('Browser', 'Unknown')}")
        print(f"   Protocol: {data.get('Protocol-Version', 'Unknown')}")
        print(f"   User Agent: {data.get('User-Agent', 'Unknown')[:50]}...")
        print(f"   WebSocket URL: {data.get('webSocketDebuggerUrl', 'Unknown')}")
        return True
        
    except urllib.error.URLError as e:
        print(f"\n❌ FAILED to connect to Chrome debug port!")
        print(f"   Error: {e.reason}")
        return False
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


async def test_playwright_connection():
    """Test Playwright connection to Chrome."""
    print("\n" + "=" * 60)
    print("PLAYWRIGHT CONNECTION TEST")
    print("=" * 60)
    
    try:
        from playwright.async_api import async_playwright
        
        print("\n🔍 Starting Playwright...")
        async with async_playwright() as p:
            print("🔍 Attempting to connect via CDP...")
            browser = await p.chromium.connect_over_cdp("http://localhost:9222")
            print("✅ Playwright connected successfully!")
            
            # Get contexts
            contexts = browser.contexts
            print(f"📂 Found {len(contexts)} browser context(s)")
            
            # Open a new tab
            if contexts:
                context = contexts[0]
            else:
                context = await browser.new_context()
            
            page = await context.new_page()
            print("✅ New tab created successfully!")
            
            # Navigate to a test page
            print("🔍 Navigating to example.com...")
            await page.goto("https://example.com")
            title = await page.title()
            print(f"✅ Page loaded! Title: {title}")
            
            # Clean up - close only the tab we created
            await page.close()
            print("✅ Test tab closed (your other Chrome tabs are untouched)")
            
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED! Your browsing agent should work correctly.")
            print("=" * 60)
            return True
            
    except Exception as e:
        print(f"\n❌ Playwright connection failed: {e}")
        return False


def show_instructions():
    """Show instructions if tests fail."""
    print("\n" + "=" * 60)
    print("📋 HOW TO FIX")
    print("=" * 60)
    print("""
1. CLOSE ALL CHROME WINDOWS
   - Make sure Chrome is completely closed
   - Check Task Manager (Ctrl+Shift+Esc) and end any chrome.exe processes

2. START CHROME WITH REMOTE DEBUGGING
   Open Command Prompt (cmd) and run:
   
   start chrome --remote-debugging-port=9222

   Or run the start_chrome_debug.bat file in this folder.

3. VERIFY CHROME IS RUNNING
   - Chrome should open normally
   - You can use it like usual, browse websites, etc.

4. RUN THIS TEST AGAIN
   python test_chrome_connection.py

5. IF IT STILL DOESN'T WORK:
   - Try a different port: start chrome --remote-debugging-port=9223
   - Check your firewall settings
   - Make sure no other program is using port 9222
""")


async def main():
    print("\n🧪 Testing Chrome Remote Debugging Connection...\n")
    
    # Test 1: Check if port is open
    port_ok = check_chrome_debug_port()
    
    if not port_ok:
        show_instructions()
        return
    
    # Test 2: Test Playwright connection
    playwright_ok = await test_playwright_connection()
    
    if not playwright_ok:
        show_instructions()


if __name__ == "__main__":
    asyncio.run(main())


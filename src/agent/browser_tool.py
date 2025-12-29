from __future__ import annotations

import asyncio
import re
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, Page

from .config import settings


def clean_and_truncate_html(html: str, max_length: int = None) -> str:
    """
    Clean HTML by removing scripts, styles, and excessive whitespace,
    then truncate to stay within token limits.
    """
    if max_length is None:
        max_length = settings.max_content_length
    
    # Remove script tags and their content
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style tags and their content
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # Remove SVG elements (often very large)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '[SVG]', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove noscript tags
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove inline styles (data-* attributes and style attributes can be verbose)
    html = re.sub(r'\s+style="[^"]*"', '', html)
    html = re.sub(r"\s+style='[^']*'", '', html)
    
    # Remove data-* attributes (often used for tracking/analytics)
    html = re.sub(r'\s+data-[a-zA-Z0-9-]+="[^"]*"', '', html)
    
    # Remove class attributes that are very long (likely Tailwind or similar)
    def truncate_class(match):
        classes = match.group(1)
        if len(classes) > 100:
            return f' class="{classes[:100]}..."'
        return match.group(0)
    html = re.sub(r'\s+class="([^"]*)"', truncate_class, html)
    
    # Normalize whitespace
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)
    
    # Truncate if still too long
    if len(html) > max_length:
        truncated = html[:max_length]
        # Try to truncate at a tag boundary
        last_close = truncated.rfind('>')
        if last_close > max_length * 0.8:
            truncated = truncated[:last_close + 1]
        return truncated + "\n\n[... Content truncated to stay within token limits ...]"
    
    return html

# Windows requires ProactorEventLoop for subprocess support (needed by Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class BrowserController:
    """
    Thin async wrapper around Playwright providing high-level actions
    that the LLM agent can invoke. Maintains a single browser session.
    """

    CDP_URL = "http://localhost:9222"

    def __init__(self) -> None:
        self._playwright: Optional[Any] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._initialized: bool = False
        self._connected_to_existing: bool = False

    async def ensure_session(self) -> None:
        """
        Ensure browser session is initialized.
        First tries to connect to an existing Chrome browser via CDP (opens new tab).
        Falls back to launching a new browser if no existing browser is available.
        """
        if not self._initialized:
            self._playwright = await async_playwright().start()
            
            # Try to connect to existing Chrome browser via CDP first
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(self.CDP_URL)
                self._connected_to_existing = True
                # Get existing context or create new one, then open a new tab
                contexts = self._browser.contexts
                if contexts:
                    context = contexts[0]
                else:
                    context = await self._browser.new_context()
                self._page = await context.new_page()
                print("✅ Connected to existing Chrome browser - opened new tab")
            except Exception:
                # Fall back to launching a new browser
                print("ℹ️  No existing Chrome browser found, launching new browser...")
                print("💡 Tip: Start Chrome with remote debugging to use existing browser:")
                if sys.platform == "win32":
                    print('   chrome.exe --remote-debugging-port=9222')
                else:
                    print('   google-chrome --remote-debugging-port=9222')
                
                self._connected_to_existing = False
                launch_options = {
                    "headless": settings.headless,
                }
                # Try to use system Chrome if available, otherwise use Chromium
                try:
                    self._browser = await self._playwright.chromium.launch(
                        **launch_options,
                        channel="chrome"
                    )
                except Exception:
                    self._browser = await self._playwright.chromium.launch(**launch_options)
                
                self._page = await self._browser.new_page()
                # Set user agent to appear more like a real browser
                await self._page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })
            
            self._page.set_default_timeout(settings.navigation_timeout_ms)
            self._initialized = True

    async def close_session(self) -> None:
        """Close browser session. Only closes the tab if connected to existing browser."""
        if self._page:
            await self._page.close()  # Close the tab we created
            self._page = None
        if self._browser and not self._connected_to_existing:
            # Only close browser if we launched it ourselves
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._initialized = False
        self._connected_to_existing = False

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser page not initialized. Call ensure_session() first.")
        return self._page

    async def navigate(self, url: str) -> str:
        await self.page.goto(url)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def click(self, selector: str) -> str:
        """
        Click an element. Handles hidden elements by:
        1. First trying normal click
        2. If element not visible, try scrolling into view
        3. If still not visible, use JavaScript click (bypasses visibility)
        """
        try:
            # First, try normal click with reduced timeout
            await self.page.click(selector, timeout=5000)
        except Exception as e:
            error_msg = str(e).lower()
            if "not visible" in error_msg or "timeout" in error_msg:
                # Element exists but not visible - try alternative approaches
                try:
                    # Try scrolling element into view first
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)  # Wait for any animations
                        
                        # Try clicking again
                        try:
                            await self.page.click(selector, timeout=5000)
                        except Exception:
                            # Last resort: JavaScript click (bypasses visibility checks)
                            await self.page.evaluate(
                                """(selector) => {
                                    const el = document.querySelector(selector);
                                    if (el) el.click();
                                }""",
                                selector
                            )
                    else:
                        raise Exception(f"Element not found: {selector}")
                except Exception as inner_e:
                    raise Exception(f"Click failed after all attempts: {inner_e}")
            else:
                raise e
        
        # Wait for any navigation or dynamic content
        await asyncio.sleep(0.5)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def hover(self, selector: str) -> str:
        """
        Hover over an element. Useful for revealing dropdown menus.
        """
        await self.page.hover(selector)
        await asyncio.sleep(0.5)  # Wait for dropdown/menu to appear
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def fill(self, selector: str, text: str) -> str:
        await self.page.fill(selector, text)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def press(self, key: str) -> str:
        await self.page.keyboard.press(key)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def scroll(self, amount: int = 1000) -> str:
        await self.page.evaluate(
            "(amount) => { window.scrollBy(0, amount); }", amount
        )
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def extract_text(self, selector: str) -> str:
        elements = await self.page.query_selector_all(selector)
        texts: List[str] = []
        for el in elements:
            txt = await el.inner_text()
            texts.append(txt.strip())
        return "\n".join(texts)


async def demo():
    """
    Simple manual test to verify Playwright is wired correctly.
    Not used by the main agent, but handy during development.
    """

    controller = BrowserController()
    try:
        await controller.ensure_session()
        html = await controller.navigate("https://example.com")
        print("Loaded page length:", len(html))
    finally:
        await controller.close_session()


if __name__ == "__main__":
    asyncio.run(demo())



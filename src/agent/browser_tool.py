from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright, Browser, Page

from .config import settings

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
        return await self.page.content()

    async def click(self, selector: str) -> str:
        await self.page.click(selector)
        return await self.page.content()

    async def fill(self, selector: str, text: str) -> str:
        await self.page.fill(selector, text)
        return await self.page.content()

    async def press(self, key: str) -> str:
        await self.page.keyboard.press(key)
        return await self.page.content()

    async def scroll(self, amount: int = 1000) -> str:
        await self.page.evaluate(
            "(amount) => { window.scrollBy(0, amount); }", amount
        )
        return await self.page.content()

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



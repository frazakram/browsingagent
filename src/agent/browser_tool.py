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
    Aggressively clean HTML by removing ads, scripts, styles, and non-essential content.
    Focuses on keeping only actionable elements (links, buttons, forms, main content).
    """
    if max_length is None:
        max_length = settings.max_content_length
    
    # === REMOVE ENTIRE SECTIONS ===
    
    # Scripts and styles
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<link[^>]*>', '', html, flags=re.IGNORECASE)
    html = re.sub(r'<meta[^>]*>', '', html, flags=re.IGNORECASE)
    
    # Comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # SVG, canvas, video, audio, iframe (media bloat)
    html = re.sub(r'<svg[^>]*>.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<canvas[^>]*>.*?</canvas>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<video[^>]*>.*?</video>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<audio[^>]*>.*?</audio>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<iframe[^>]*>.*?</iframe>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<picture[^>]*>.*?</picture>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # === REMOVE ADS AND PROMOTIONAL CONTENT ===
    
    # Common ad containers (by class/id patterns)
    ad_patterns = [
        r'<[^>]*(class|id)="[^"]*\b(ad|ads|advert|advertisement|banner|promo|promotion|sponsor|sponsored|popup|modal|overlay|cookie|consent|newsletter|subscribe|social|share|widget|sidebar|recommended|trending|related|also-like)[^"]*"[^>]*>.*?</[^>]+>',
    ]
    for pattern in ad_patterns:
        html = re.sub(pattern, '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove elements with ad-related classes more aggressively
    html = re.sub(r'<div[^>]*(?:ad-|ads-|advert|banner|promo|popup|modal|cookie|newsletter)[^>]*>.*?</div>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<section[^>]*(?:ad-|ads-|advert|banner|promo|popup|modal)[^>]*>.*?</section>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove header, footer, nav (navigation bloat)
    html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # === REMOVE UNNECESSARY ATTRIBUTES ===
    
    # Keep only essential attributes: href, src, id, name, type, placeholder, value, alt, action, method
    html = re.sub(r'\s+style="[^"]*"', '', html)
    html = re.sub(r"\s+style='[^']*'", '', html)
    html = re.sub(r'\s+class="[^"]*"', '', html)
    html = re.sub(r"\s+class='[^']*'", '', html)
    html = re.sub(r'\s+data-[a-zA-Z0-9-]+="[^"]*"', '', html)
    html = re.sub(r'\s+aria-[a-zA-Z0-9-]+="[^"]*"', '', html)
    html = re.sub(r'\s+role="[^"]*"', '', html)
    html = re.sub(r'\s+tabindex="[^"]*"', '', html)
    html = re.sub(r'\s+on[a-z]+="[^"]*"', '', html)
    html = re.sub(r'\s+target="[^"]*"', '', html)
    html = re.sub(r'\s+rel="[^"]*"', '', html)
    html = re.sub(r'\s+loading="[^"]*"', '', html)
    html = re.sub(r'\s+srcset="[^"]*"', '', html)
    html = re.sub(r'\s+sizes="[^"]*"', '', html)
    html = re.sub(r'\s+width="[^"]*"', '', html)
    html = re.sub(r'\s+height="[^"]*"', '', html)
    
    # === SIMPLIFY IMAGES ===
    # Replace img tags with just their alt text
    def simplify_img(match):
        alt = re.search(r'alt="([^"]*)"', match.group(0))
        if alt and alt.group(1).strip():
            return f'[IMG:{alt.group(1)[:30]}]'
        return ''
    html = re.sub(r'<img[^>]*>', simplify_img, html, flags=re.IGNORECASE)
    
    # === CLEAN UP EMPTY AND REDUNDANT ELEMENTS ===
    
    # Remove empty tags (multiple passes)
    for _ in range(3):
        html = re.sub(r'<(div|span|p|section|article|ul|ol|li|table|tr|td|th)[^>]*>\s*</\1>', '', html, flags=re.IGNORECASE)
    
    # Remove br tags
    html = re.sub(r'<br\s*/?>', ' ', html, flags=re.IGNORECASE)
    
    # === NORMALIZE WHITESPACE ===
    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'>\s+<', '><', html)
    html = re.sub(r'\s+>', '>', html)
    html = re.sub(r'<\s+', '<', html)
    
    # === TRUNCATE IF STILL TOO LONG ===
    if len(html) > max_length:
        truncated = html[:max_length]
        # Try to truncate at a tag boundary
        last_close = truncated.rfind('>')
        if last_close > max_length * 0.8:
            truncated = truncated[:last_close + 1]
        return truncated + "\n[...TRUNCATED...]"
    
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
                print(f"🔍 Attempting to connect to Chrome at {self.CDP_URL}...")
                self._browser = await self._playwright.chromium.connect_over_cdp(self.CDP_URL)
                self._connected_to_existing = True
                # Get existing context or create new one, then open a new tab
                contexts = self._browser.contexts
                if contexts:
                    context = contexts[0]
                    print(f"📂 Using existing browser context")
                else:
                    context = await self._browser.new_context()
                    print(f"📂 Created new browser context")
                self._page = await context.new_page()
                print("✅ Connected to existing Chrome browser - opened new tab")
            except Exception as e:
                # Fall back to launching a new browser
                print(f"❌ Failed to connect to existing Chrome: {e}")
                print("ℹ️  Launching new browser instead...")
                print("💡 Tip: To use your existing Chrome browser (avoids CAPTCHAs):")
                print("   1. Close ALL Chrome windows completely")
                print("   2. Run this command to start Chrome:")
                if sys.platform == "win32":
                    print('      start chrome --remote-debugging-port=9222')
                else:
                    print('      google-chrome --remote-debugging-port=9222')
                print("   3. Then run the browsing agent again")
                
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
        # Try to get just the main content area first
        html = await self._get_main_content()
        return clean_and_truncate_html(html)

    async def _get_main_content(self) -> str:
        """Try to extract just the main content, falling back to full page."""
        # Try common main content selectors
        main_selectors = ['main', 'article', '[role="main"]', '#main', '#content', '.main-content', '.content']
        
        for selector in main_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element:
                    html = await element.inner_html()
                    if len(html) > 500:  # Make sure we got substantial content
                        return html
            except:
                pass
        
        # Fall back to body content
        try:
            body = await self.page.query_selector('body')
            if body:
                return await body.inner_html()
        except:
            pass
        
        return await self.page.content()

    async def click(self, selector: str, wait_for_enabled: bool = True) -> str:
        """
        Click an element. Handles hidden and disabled elements by:
        1. First trying normal click
        2. If element not visible, try scrolling into view
        3. If element is disabled, wait for it to become enabled
        4. If still not working, use JavaScript click
        """
        try:
            # First, try normal click with reduced timeout
            await self.page.click(selector, timeout=5000)
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if element is disabled
            if "not enabled" in error_msg or "disabled" in error_msg:
                if wait_for_enabled:
                    try:
                        # Wait for element to become enabled (max 10 seconds)
                        await self.page.wait_for_selector(
                            f"{selector}:not([disabled])", 
                            timeout=10000,
                            state="attached"
                        )
                        await asyncio.sleep(0.3)
                        await self.page.click(selector, timeout=5000)
                    except Exception as wait_e:
                        raise Exception(
                            f"Button '{selector}' is DISABLED. This usually means:\n"
                            f"1. A required field hasn't been filled correctly\n"
                            f"2. Form validation hasn't passed\n"
                            f"3. You need to complete a previous step first\n"
                            f"Try filling all required fields before clicking this button."
                        )
                else:
                    raise Exception(f"Element is disabled: {selector}")
            
            elif "not visible" in error_msg or "timeout" in error_msg:
                # Element exists but not visible - try alternative approaches
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        await element.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        
                        try:
                            await self.page.click(selector, timeout=5000)
                        except Exception:
                            # Last resort: JavaScript click
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
                    raise Exception(f"Click failed: {inner_e}")
            else:
                raise e
        
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

    async def fill(self, selector: str, text: str, press_enter: bool = False) -> str:
        """
        Fill an input field with text. Triggers proper events for form validation.
        Optionally press Enter after filling (useful for search boxes).
        Automatically handles radio buttons and checkboxes by clicking them instead.
        """
        element = await self.page.query_selector(selector)
        if not element:
            raise Exception(f"Input field not found: {selector}")
        
        # Check the input type
        input_type = await element.get_attribute("type") or "text"
        input_type = input_type.lower()
        
        # Handle radio buttons and checkboxes - click instead of fill
        if input_type in ["radio", "checkbox"]:
            await element.click()
            await asyncio.sleep(0.3)
            html = await self.page.content()
            return clean_and_truncate_html(html)
        
        # Clear existing value first
        await self.page.fill(selector, "")
        
        # Type the text character by character for better compatibility
        await self.page.type(selector, text, delay=30)
        
        # Trigger blur event to activate any validation
        await self.page.evaluate(
            """(selector) => {
                const el = document.querySelector(selector);
                if (el) {
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }""",
            selector
        )
        
        # Press Enter if requested (for search boxes)
        if press_enter:
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(1)  # Wait for search results
        else:
            await asyncio.sleep(0.3)
        
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def press(self, key: str) -> str:
        await self.page.keyboard.press(key)
        await asyncio.sleep(0.5)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def select_option(self, selector: str, value: str) -> str:
        """
        Select an option from a dropdown, radio button group, or checkbox.
        
        For SELECT dropdowns: selector should target the <select> element.
        For radio buttons: selector can target the specific radio input, or use value to find it.
        For checkboxes: selector should target the checkbox input.
        
        The value can be the option value, visible text, or label.
        """
        element = await self.page.query_selector(selector)
        
        if not element:
            # Try to find by value in radio buttons or checkboxes
            radio_selector = f'input[type="radio"][value="{value}"]'
            element = await self.page.query_selector(radio_selector)
            if element:
                await element.click()
                await asyncio.sleep(0.3)
                html = await self.page.content()
                return clean_and_truncate_html(html)
            
            raise Exception(f"Element not found: {selector}")
        
        tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
        input_type = await element.get_attribute("type") or ""
        
        # Handle SELECT dropdown
        if tag_name == "select":
            try:
                # Try selecting by value first
                await self.page.select_option(selector, value=value)
            except:
                try:
                    # Try selecting by label/text
                    await self.page.select_option(selector, label=value)
                except:
                    raise Exception(f"Could not select option '{value}' in dropdown")
        
        # Handle radio button
        elif input_type.lower() == "radio":
            await element.click()
        
        # Handle checkbox
        elif input_type.lower() == "checkbox":
            # Check if we need to check or uncheck
            is_checked = await element.is_checked()
            if value.lower() in ["true", "yes", "check", "1", "on"] and not is_checked:
                await element.click()
            elif value.lower() in ["false", "no", "uncheck", "0", "off"] and is_checked:
                await element.click()
            else:
                # Toggle if no specific value given
                await element.click()
        
        # Handle clicking a label or button that acts as a selector
        else:
            await element.click()
        
        await asyncio.sleep(0.3)
        html = await self.page.content()
        return clean_and_truncate_html(html)

    async def search(self, selector: str, query: str) -> str:
        """
        Fill a search box and press Enter to submit.
        Tries the given selector first, then common search box patterns.
        If on a blank page, automatically navigates to Google first.
        """
        # Check if we're on a blank or empty page
        current_url = self.page.url
        if current_url in ["about:blank", "", "chrome://newtab/", "edge://newtab/"]:
            # Navigate to Google first
            await self.page.goto("https://www.google.com")
            await asyncio.sleep(1)  # Wait for page to load
        
        # Common search box selectors to try
        search_selectors = [
            selector,  # User-provided selector first
            'input[type="search"]',
            'input[name="q"]',
            'input[name="query"]',
            'input[name="search"]',
            'input[name="keyword"]',
            'input[name="keywords"]',
            'input[name="searchText"]',
            'textarea[name="q"]',  # Google sometimes uses textarea
            'input[placeholder*="search" i]',
            'input[placeholder*="find" i]',
            'input[placeholder*="looking" i]',
            'input[aria-label*="search" i]',
            'textarea[aria-label*="search" i]',  # Google textarea
            '#search',
            '#search-input',
            '#searchInput',
            '#search-box',
            '#searchbox',
            '#q',
            '.search-input',
            '.search-box',
            '.searchbox',
            '[data-testid*="search"]',
            'input[autocomplete="off"]',  # Many search boxes disable autocomplete
            'input[title*="search" i]',
            'textarea[title*="search" i]',
        ]
        
        element = None
        used_selector = None
        
        # Wait a bit for dynamic content to load
        await asyncio.sleep(0.5)
        
        for sel in search_selectors:
            if not sel:
                continue
            try:
                element = await self.page.query_selector(sel)
                if element:
                    # Verify it's visible and an input or textarea
                    is_visible = await element.is_visible()
                    tag = await element.evaluate("el => el.tagName.toLowerCase()")
                    if is_visible and tag in ["input", "textarea"]:
                        used_selector = sel
                        break
            except:
                pass
        
        if not element or not used_selector:
            # If still not found, try navigating to Google and searching there
            if "google.com" not in current_url:
                await self.page.goto("https://www.google.com")
                await asyncio.sleep(1)
                
                # Try again with Google-specific selectors
                google_selectors = ['textarea[name="q"]', 'input[name="q"]', 'textarea[aria-label*="Search"]']
                for sel in google_selectors:
                    try:
                        element = await self.page.query_selector(sel)
                        if element and await element.is_visible():
                            used_selector = sel
                            break
                    except:
                        pass
        
        if not element or not used_selector:
            # Get available inputs to help the agent
            available = await self._get_available_inputs()
            raise Exception(
                f"Search box not found with selector: {selector}\n"
                f"Current URL: {self.page.url}\n"
                f"Available input fields on page:\n{available}\n"
                f"Tip: First use navigate('https://www.google.com'), then use search()"
            )
        
        await self.page.fill(used_selector, "")
        await self.page.type(used_selector, query, delay=30)
        await self.page.keyboard.press("Enter")
        
        # Wait for search results to load
        await asyncio.sleep(2)
        
        html = await self._get_main_content()
        return clean_and_truncate_html(html)

    async def _get_available_inputs(self) -> str:
        """Get a summary of available input fields."""
        try:
            inputs_info = await self.page.evaluate("""() => {
                const inputs = document.querySelectorAll('input:not([type="hidden"])');
                const results = [];
                inputs.forEach((inp, i) => {
                    if (i < 10) {  // Limit to first 10
                        const type = inp.type || 'text';
                        const name = inp.name || '';
                        const placeholder = inp.placeholder || '';
                        const id = inp.id || '';
                        results.push(`- type="${type}" name="${name}" placeholder="${placeholder}" id="${id}"`);
                    }
                });
                return results.join('\\n');
            }""")
            return inputs_info or "No visible inputs found"
        except:
            return "Could not retrieve inputs"

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

    async def goto_link_by_text(self, text: str) -> str:
        """
        Find a link containing the text and navigate directly to its URL.
        Bypasses overlay/click issues by extracting href and navigating.
        """
        href = await self.page.evaluate(
            """(text) => {
                // Find links containing the text
                const links = document.querySelectorAll('a[href]');
                for (const link of links) {
                    const linkText = link.textContent || '';
                    const ariaLabel = link.getAttribute('aria-label') || '';
                    const title = link.getAttribute('title') || '';
                    
                    if (linkText.toLowerCase().includes(text.toLowerCase()) ||
                        ariaLabel.toLowerCase().includes(text.toLowerCase()) ||
                        title.toLowerCase().includes(text.toLowerCase())) {
                        return link.href;
                    }
                }
                return null;
            }""",
            text
        )
        
        if href:
            await self.page.goto(href)
            await asyncio.sleep(1)
            html = await self._get_main_content()
            return clean_and_truncate_html(html)
        else:
            raise Exception(f"No link found containing text: '{text}'")

    async def get_form_fields(self) -> str:
        """
        Get a list of all form input fields on the page.
        Returns their type, placeholder, name/id, and suggested selectors.
        """
        fields_info = []
        
        # Get all input fields
        inputs = await self.page.query_selector_all("input:not([type='hidden']):not([type='submit']):not([type='button'])")
        for inp in inputs[:30]:
            try:
                inp_type = await inp.get_attribute("type") or "text"
                inp_id = await inp.get_attribute("id")
                inp_name = await inp.get_attribute("name")
                inp_placeholder = await inp.get_attribute("placeholder") or ""
                inp_value = await inp.get_attribute("value") or ""
                
                selector = f"#{inp_id}" if inp_id else (f"input[name='{inp_name}']" if inp_name else f"input[placeholder='{inp_placeholder}']")
                
                fields_info.append(
                    f"INPUT ({inp_type}): placeholder=\"{inp_placeholder}\" "
                    f"name=\"{inp_name}\" current_value=\"{inp_value}\" -> selector: {selector}"
                )
            except:
                pass
        
        # Get all textarea fields
        textareas = await self.page.query_selector_all("textarea")
        for ta in textareas[:10]:
            try:
                ta_id = await ta.get_attribute("id")
                ta_name = await ta.get_attribute("name")
                ta_placeholder = await ta.get_attribute("placeholder") or ""
                selector = f"#{ta_id}" if ta_id else f"textarea[name='{ta_name}']"
                fields_info.append(f"TEXTAREA: placeholder=\"{ta_placeholder}\" -> selector: {selector}")
            except:
                pass
        
        # Get select dropdowns
        selects = await self.page.query_selector_all("select")
        for sel in selects[:10]:
            try:
                sel_id = await sel.get_attribute("id")
                sel_name = await sel.get_attribute("name")
                selector = f"#{sel_id}" if sel_id else f"select[name='{sel_name}']"
                fields_info.append(f"SELECT: name=\"{sel_name}\" -> selector: {selector}")
            except:
                pass
        
        if not fields_info:
            return "No form fields found on the page."
        
        return "FORM FIELDS ON PAGE:\n" + "\n".join(fields_info)

    async def get_clickable_elements(self) -> str:
        """
        Get a list of all clickable elements (links and buttons) on the page.
        Returns their text, href/type, and suggested selectors.
        """
        elements_info = []
        
        # Get all links
        links = await self.page.query_selector_all("a[href]")
        for i, link in enumerate(links[:50]):  # Limit to first 50 links
            try:
                text = (await link.inner_text()).strip()[:50]  # Truncate long text
                href = await link.get_attribute("href")
                if text or href:
                    elements_info.append(f"LINK: \"{text}\" -> href=\"{href}\"")
            except:
                pass
        
        # Get all buttons
        buttons = await self.page.query_selector_all("button")
        for i, btn in enumerate(buttons[:20]):  # Limit to first 20 buttons
            try:
                text = (await btn.inner_text()).strip()[:50]
                btn_type = await btn.get_attribute("type") or "button"
                btn_id = await btn.get_attribute("id")
                selector_hint = f"#{btn_id}" if btn_id else f"button:has-text(\"{text}\")"
                elements_info.append(f"BUTTON: \"{text}\" (type={btn_type}) -> selector: {selector_hint}")
            except:
                pass
        
        # Get clickable inputs (submit buttons)
        inputs = await self.page.query_selector_all("input[type='submit'], input[type='button']")
        for inp in inputs[:10]:
            try:
                value = await inp.get_attribute("value") or ""
                inp_id = await inp.get_attribute("id")
                selector_hint = f"#{inp_id}" if inp_id else f"input[value=\"{value}\"]"
                elements_info.append(f"INPUT: \"{value}\" -> selector: {selector_hint}")
            except:
                pass
        
        if not elements_info:
            return "No clickable elements found on the page."
        
        return "CLICKABLE ELEMENTS ON PAGE:\n" + "\n".join(elements_info)

    async def click_by_text(self, text: str) -> str:
        """
        Click an element by its visible text content.
        Handles overlays and intercepted clicks using force mode and JS fallback.
        """
        selectors_to_try = [
            f"a:has-text(\"{text}\")",
            f"button:has-text(\"{text}\")",
            f"[aria-label*=\"{text}\" i]",
            f"text=\"{text}\"",
        ]
        
        last_error = None
        
        for selector in selectors_to_try:
            # First try normal click
            try:
                await self.page.click(selector, timeout=3000)
                await asyncio.sleep(0.5)
                html = await self._get_main_content()
                return clean_and_truncate_html(html)
            except Exception as e:
                error_msg = str(e).lower()
                
                # If intercepted by overlay, try force click
                if "intercepts pointer" in error_msg or "overlay" in error_msg:
                    try:
                        await self.page.click(selector, timeout=3000, force=True)
                        await asyncio.sleep(0.5)
                        html = await self._get_main_content()
                        return clean_and_truncate_html(html)
                    except:
                        pass
                
                last_error = e
        
        # Last resort: JavaScript click on any element containing the text
        try:
            clicked = await self.page.evaluate(
                """(text) => {
                    // Find all elements containing the text
                    const xpath = `//*[contains(text(), '${text}')]`;
                    const result = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                    
                    for (let i = 0; i < result.snapshotLength; i++) {
                        const el = result.snapshotItem(i);
                        // Try to find a clickable parent (a, button) or click the element itself
                        const clickable = el.closest('a, button') || el;
                        if (clickable) {
                            clickable.click();
                            return true;
                        }
                    }
                    
                    // Also try by href containing the text
                    const links = document.querySelectorAll('a[href]');
                    for (const link of links) {
                        if (link.textContent.includes(text) || link.getAttribute('aria-label')?.includes(text)) {
                            link.click();
                            return true;
                        }
                    }
                    
                    return false;
                }""",
                text
            )
            
            if clicked:
                await asyncio.sleep(1)
                html = await self._get_main_content()
                return clean_and_truncate_html(html)
        except:
            pass
        
        raise Exception(
            f"Could not click element with text: '{text}'.\n"
            f"Tip: Try using click() with a direct href selector like: click('a[href*=\"product-name\"]')\n"
            f"Or navigate directly to the product URL if available."
        )


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



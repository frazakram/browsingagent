from __future__ import annotations

from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

from .browser_tool import BrowserController
from .config import settings


# Global browser controller instance - maintains session across tool calls
_browser_controller = BrowserController()


def get_browser_controller() -> BrowserController:
    """Get the global browser controller instance."""
    return _browser_controller


@tool
async def navigate(url: str) -> str:
    """Navigate the browser to the given URL and return the current page HTML."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.navigate(url)


@tool
async def click(selector: str) -> str:
    """Click an element given a CSS selector and return updated page HTML. 
    Handles hidden elements automatically by scrolling or using JavaScript click."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.click(selector)


@tool
async def hover(selector: str) -> str:
    """Hover over an element to reveal dropdown menus or tooltips. 
    Use this before clicking on menu items that are hidden until hover."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.hover(selector)


@tool
async def fill(selector: str, text: str) -> str:
    """Fill an input element located by CSS selector with the provided text.
    For search boxes, use the 'search' tool instead which also presses Enter.
    Note: For radio buttons and checkboxes, use select_option() instead."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.fill(selector, text)


@tool
async def select_option(selector: str, value: str) -> str:
    """Select an option from dropdowns, radio buttons, or checkboxes.
    
    For SIZE selection (radio buttons): select_option('input[value=\"EURO-44\"]', 'click')
    For DROPDOWN: select_option('select#size', 'Large')
    For CHECKBOX: select_option('input[type=\"checkbox\"]#agree', 'check')
    
    Can also find radio by value: select_option('input[type=\"radio\"]', 'EURO-44')
    """
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.select_option(selector, value)


@tool
async def search(selector: str, query: str) -> str:
    """Fill a search box and press Enter to submit. Use this for search boxes!
    Example: search('input[type=\"search\"]', 'cold medicine')
    The selector can be a CSS selector for the search input field."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.search(selector, query)


@tool
async def scroll(amount: int = 1000) -> str:
    """Scroll the page vertically by the given pixel amount and return updated HTML."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.scroll(amount=amount)


@tool
async def extract_text(selector: str) -> str:
    """Extract inner text from all elements matching the CSS selector."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.extract_text(selector)


@tool
async def get_clickable_elements() -> str:
    """Get a list of all clickable elements (links and buttons) on the current page.
    Use this to discover what elements are available before trying to click.
    Returns text content, href attributes, and suggested selectors."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.get_clickable_elements()


@tool
async def get_form_fields() -> str:
    """Get a list of all form input fields on the current page.
    Use this to discover input fields, their types, and selectors before filling forms.
    Essential for finding the correct selector for phone numbers, emails, etc."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.get_form_fields()


@tool
async def click_by_text(text: str) -> str:
    """Click an element by its visible text content. Handles overlay elements.
    Example: click_by_text('Veg Pizzas') or click_by_text('Book Now')
    If this fails due to overlays, try goto_link_by_text instead."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.click_by_text(text)


@tool
async def goto_link_by_text(text: str) -> str:
    """Find a link containing the text and navigate directly to its URL.
    BEST for product cards, images with overlays, or any link that's hard to click.
    Example: goto_link_by_text('Nike Court Vision Low')
    This bypasses click issues by extracting the href and navigating directly."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.goto_link_by_text(text)


def build_agent(
    openai_key: str | None = None,
    openai_model: str | None = None,
) -> AgentExecutor:
    # Validate and fix model name
    model = openai_model or settings.openai_model
    valid_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4"]

    if model not in valid_models:
        import logging

        logger = logging.getLogger("browsing-agent")
        logger.warning(
            f"Invalid model '{model}' specified. Defaulting to 'gpt-4o-mini'"
        )
        model = "gpt-4o-mini"

    api_key = openai_key or settings.openai_api_key

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.1,
        timeout=60.0,  # 60 second timeout for API calls
        max_retries=3,  # Retry up to 3 times on connection errors
    )

    tools = [navigate, search, click, click_by_text, goto_link_by_text, hover, fill, select_option, scroll, extract_text, get_clickable_elements, get_form_fields]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a web-browsing assistant controlling a real browser.\n\n"
            "KEY TOOLS:\n"
            "- navigate(url): Go to a specific URL\n"
            "- search(selector, query): For SEARCH BOXES - fills and presses Enter. Auto-navigates to Google if on blank page.\n"
            "- goto_link_by_text(text): BEST for product cards/images - navigates directly\n"
            "- click_by_text(text): Click by visible text - handles overlays\n"
            "- fill(selector, text): Fill TEXT input fields (phone, email, etc.)\n"
            "- select_option(selector, value): For RADIO BUTTONS, CHECKBOXES, DROPDOWNS (size selectors, etc.)\n"
            "- get_form_fields(): Discover input fields before filling\n"
            "- get_clickable_elements(): Discover links/buttons before clicking\n\n"
            "IMPORTANT - SELECTING SIZES/OPTIONS:\n"
            "- For SIZE radio buttons: click_by_text('EURO-44') or select_option('input[value=\"EURO-44\"]', 'click')\n"
            "- For dropdown menus: select_option('select#size', 'Large')\n"
            "- NEVER use fill() on radio buttons or checkboxes!\n\n"
            "WORKFLOW:\n"
            "1. ALWAYS START by navigating to a website first!\n"
            "2. For SEARCH: Use search() tool - it auto-presses Enter.\n"
            "3. For PRODUCT CARDS: Use goto_link_by_text() - bypasses image overlays.\n"
            "4. For SIZE/COLOR selection: Use click_by_text() or select_option() - NOT fill()!\n"
            "5. For TEXT inputs: Use fill() for name, email, phone, etc.\n"
            "6. For MENUS: Use hover() first to reveal hidden items.\n\n"
            "EXAMPLE - Add product to cart:\n"
            "  Step 1: navigate('https://www.westside.com')\n"
            "  Step 2: search('input[type=\"search\"]', 'soleplay sleepers')\n"
            "  Step 3: goto_link_by_text('product name')\n"
            "  Step 4: click_by_text('EURO-44')  <-- For size selection\n"
            "  Step 5: click_by_text('Add to Cart')\n\n"
            "Stay safe: never finalize payments without user confirmation."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.max_steps,
        verbose=True,
    )
    return executor


async def run_agent(
    user_query: str,
    openai_key: str | None = None,
    openai_model: str | None = None,
) -> str:
    """Run the agent with a user query, ensuring browser session is properly managed."""
    import logging
    import traceback
    
    logger = logging.getLogger("browsing-agent")
    controller = get_browser_controller()
    
    try:
        await controller.ensure_session()
        agent = build_agent(openai_key=openai_key, openai_model=openai_model)
        result: Dict[str, Any] = await agent.ainvoke({"input": user_query})
        return result["output"]
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Agent error - Type: {error_type}, Message: {error_msg}")
        logger.debug(f"Full traceback: {error_trace}")
        
        # Check for specific error types
        if "ConnectionError" in error_type or "ConnectTimeout" in error_type or "httpx" in error_msg.lower():
            return (
                f"❌ Connection Error\n\n"
                f"Unable to connect to OpenAI API. Please check:\n"
                f"1. Your internet connection is working\n"
                f"2. OpenAI API service is accessible (check status.openai.com)\n"
                f"3. No firewall/proxy blocking the connection\n\n"
                f"Error Type: {error_type}\n"
                f"Details: {error_msg}"
            )
        elif "AuthenticationError" in error_type or "InvalidAPIKey" in error_type or "api key" in error_msg.lower():
            return (
                f"❌ Authentication Error\n\n"
                f"OpenAI API key issue. Please verify:\n"
                f"1. OPENAI_API_KEY is set in .env file\n"
                f"2. The API key starts with 'sk-' and is complete\n"
                f"3. The API key is valid and has credits\n"
                f"4. Check your OpenAI account at platform.openai.com\n\n"
                f"Error Type: {error_type}\n"
                f"Details: {error_msg}"
            )
        elif "RateLimitError" in error_type or "rate limit" in error_msg.lower():
            return (
                f"❌ Rate Limit Error\n\n"
                f"You've exceeded your OpenAI API rate limit.\n"
                f"Please wait a moment and try again, or upgrade your plan.\n\n"
                f"Error Type: {error_type}\n"
                f"Details: {error_msg}"
            )
        elif "ModelNotFound" in error_type or "model" in error_msg.lower() and "not found" in error_msg.lower():
            return (
                f"❌ Model Error\n\n"
                f"The specified OpenAI model is not available.\n"
                f"Current model in .env: {settings.openai_model}\n"
                f"Try changing OPENAI_MODEL to: gpt-4o-mini or gpt-4o\n\n"
                f"Error Type: {error_type}\n"
                f"Details: {error_msg}"
            )
        elif "context_length_exceeded" in error_msg.lower() or "maximum context length" in error_msg.lower():
            return (
                f"❌ Context Length Exceeded\n\n"
                f"The conversation exceeded the model's token limit.\n"
                f"This can happen when browsing pages with very large content.\n\n"
                f"Try:\n"
                f"1. Reduce MAX_CONTENT_LENGTH in .env (current: {settings.max_content_length})\n"
                f"2. Use simpler queries that require fewer page interactions\n"
                f"3. Break complex tasks into smaller, separate queries\n\n"
                f"Error Type: {error_type}\n"
                f"Details: {error_msg}"
            )
        else:
            return (
                f"❌ Error: {error_type}\n\n"
                f"Details: {error_msg}\n\n"
                f"Full error:\n{error_trace}"
            )
    finally:
        await controller.close_session()



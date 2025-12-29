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
    """Click an element given a CSS selector and return updated page HTML."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.click(selector)


@tool
async def fill(selector: str, text: str) -> str:
    """Fill an input element located by CSS selector with the provided text."""
    controller = get_browser_controller()
    await controller.ensure_session()
    return await controller.fill(selector, text)


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

    tools = [navigate, click, fill, scroll, extract_text]

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a web-browsing assistant that can use tools to control a real "
            "browser. Break the user task into steps and choose appropriate tools "
            "to navigate, click, fill forms, scroll, and read content. "
            "Stay safe: do not perform irreversible actions like finalizing payments "
            "or deleting data unless the user explicitly confirms. Explain what you "
            "are doing at a high level in your thoughts and then act."
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



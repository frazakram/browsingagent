from __future__ import annotations

from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage

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
        return _handle_agent_error(e)
    finally:
        await controller.close_session()


async def run_agent_stream(
    user_query: str,
    openai_key: str | None = None,
    openai_model: str | None = None,
):
    """
    Run the agent and yield events for UI streaming.
    Yields dicts with 'type' and 'content' keys.
    """
    import logging
    import traceback
    
    logger = logging.getLogger("browsing-agent")
    controller = get_browser_controller()
    
    try:
        await controller.ensure_session()
        agent = build_agent(openai_key=openai_key, openai_model=openai_model)
        
        # Stream events from the agent
        async for event in agent.astream_events(
            {"input": user_query},
            version="v1",
        ):
            kind = event["event"]
            
            # 1. Tool Start (agent decides to use a tool)
            if kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                yield {
                    "type": "log", 
                    "content": f"🛠️  Using tool: **{tool_name}**\nInput: `{tool_input}`"
                }

            # 2. Tool End (tool execution finished)
            elif kind == "on_tool_end":
                tool_name = event["name"]
                # output can be long (HTML), so we might want to truncate or just notify completion
                yield {
                    "type": "log",
                    "content": f"✅  Tool **{tool_name}** completed."
                }

            # 3. Agent Step (thought process - sometimes captured in on_chain_start/end or on_chat_model_stream)
            # For OpenAI tools agent, 'on_chat_model_stream' gives tokens.
            # We'll stick to high-level tool events and final output for now to keep it clean.
            
            # 4. Final Output (chain finish)
            elif kind == "on_chain_end":
                # Check if this is the top-level agent chain ending
                if event["name"] == "AgentExecutor":
                    output = event["data"].get("output")
                    if output and "output" in output:
                        yield {
                            "type": "result", 
                            "content": output["output"]
                        }

    except Exception as e:
        error_msg = _handle_agent_error(e)
        yield {"type": "error", "content": error_msg}
    finally:
        await controller.close_session()


def _handle_agent_error(e: Exception) -> str:
    """Helper to format error messages."""
    import logging
    import traceback
    
    logger = logging.getLogger("browsing-agent")
    error_type = type(e).__name__
    error_msg = str(e)
    error_trace = traceback.format_exc()
    
    logger.error(f"Agent error - Type: {error_type}, Message: {error_msg}")
    
    if "ConnectionError" in error_type or "ConnectTimeout" in error_type or "httpx" in error_msg.lower():
        return (
            f"❌ Connection Error\n\n"
            f"Unable to connect to OpenAI API. Please check internet and firewall."
        )
    elif "AuthenticationError" in error_type:
        return (
            f"❌ Authentication Error\n\n"
            f"OpenAI API key issue. Please check your .env file or settings."
        )
    else:
        return f"❌ Error: {error_type}\n{error_msg}"




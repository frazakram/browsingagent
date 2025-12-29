import asyncio
import logging
import sys
from typing import NoReturn

# Windows requires ProactorEventLoop for subprocess support (needed by Playwright)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from src.agent.config import settings  # noqa: F401  # trigger settings load early
from src.agent.langchain_agent import run_agent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("browsing-agent")


def main() -> NoReturn:
    if len(sys.argv) < 2:
        print("Usage: python -m src.main \"your natural language request\"")
        raise SystemExit(1)

    query = sys.argv[1]
    logger.info("Starting agent for query: %s", query)

    try:
        output = asyncio.run(run_agent(query))
        print("\n=== AGENT RESULT ===\n")
        print(output)
    except Exception as exc:  # pragma: no cover - top-level safety net
        logger.exception("Agent run failed: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()



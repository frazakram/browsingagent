# AI Browsing Agent 🤖

A production-ready web browsing agent built with **OpenAI**, **LangChain**, and **Playwright**. This agent can understand natural language requests and autonomously browse the web to complete tasks like finding information, ordering food, checking flights, and more.

## Architecture

The agent follows a clean architecture:

- **LLM (OpenAI)** → **LangChain Agent** → **Browser Controller (Playwright)**
- Tools available: Navigate, Click, Fill forms, Scroll, Extract text

## Features

- 🌐 **Natural Language Interface**: Just describe what you want in plain English
- 🤖 **Autonomous Web Browsing**: Automatically navigates, clicks, fills forms, and extracts information
- 🎨 **Modern Web UI**: Beautiful, responsive interface for easy interaction
- 🔒 **Production Ready**: Proper error handling, logging, and configuration management
- 🚀 **Fast & Efficient**: Single browser session maintained across tool calls

## Setup

### 1. Install Dependencies

```bash
cd /Users/harshit.tated/Desktop/browsingagent
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
BROWSER_HEADLESS=false
AGENT_MAX_STEPS=20
```

**⚠️ Important**: Never commit your `.env` file or share your API key publicly!

### 3. Start the Web Server

```bash
python3 start_server.py
```

The server will start on `http://localhost:8000`

### 4. Use the Agent

Open your browser and go to `http://localhost:8000`

Enter natural language queries like:
- "Go to Domino's website and show me 3 popular pizza options"
- "Search for cold medicines on a pharmacy website and list prices"
- "Find flights from Bangalore to Mumbai for tomorrow"
- "Search YouTube for Python tutorials and list top 5 results"

## CLI Usage (Alternative)

You can also use the agent from the command line:

```bash
python3 -m src.main "Your natural language request here"
```

## Project Structure

```
browsingagent/
├── src/
│   ├── agent/
│   │   ├── config.py          # Configuration management
│   │   ├── browser_tool.py    # Playwright browser controller
│   │   └── langchain_agent.py # LangChain agent implementation
│   ├── main.py                # CLI entry point
│   └── web_server.py          # FastAPI web server with UI
├── start_server.py            # Server startup script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How It Works

1. **User Request**: You provide a natural language query
2. **LangChain Agent**: Processes the request and plans actions
3. **Tool Selection**: Agent selects appropriate browser tools
4. **Browser Control**: Playwright controls a real Chromium browser
5. **Web Interaction**: Navigates, clicks, fills forms, scrolls, extracts content
6. **Result Processing**: Agent processes the data and returns a final answer

## Safety & Limitations

- The agent is designed to be **safe** and will avoid irreversible actions like finalizing payments without explicit confirmation
- Some websites may have CAPTCHAs or anti-bot protections that require manual intervention
- Complex multi-step workflows (like full checkout processes) may need site-specific customization
- The agent works best with well-structured websites

## Troubleshooting

### Server won't start
- Check that all dependencies are installed: `python3 -m pip install -r requirements.txt`
- Verify your `.env` file exists and has a valid `OPENAI_API_KEY`
- Make sure port 8000 is not already in use

### Browser issues
- Ensure Playwright browsers are installed: `python3 -m playwright install chromium`
- If you see SSL errors, you may need to configure your network settings

### Import errors
- Make sure you're running commands from the project root directory
- Verify Python version is 3.10 or higher: `python3 --version`

## License

This project is provided as-is for educational and development purposes.

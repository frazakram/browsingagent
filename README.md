# AI Browsing Agent 🤖

A production-ready web browsing agent built with **OpenAI**, **LangChain**, and **Playwright**. This agent can understand natural language requests and autonomously browse the web to complete tasks like finding information, ordering food, checking flights, and more.

## 🎬 Demo

![AI Browsing Agent Demo](https://img.shields.io/badge/AI-Browsing%20Agent-blue?style=for-the-badge&logo=openai)

## 🏗️ Architecture

The agent follows a clean architecture:

```
User Request → LLM (OpenAI) → LangChain Agent → Browser Controller (Playwright) → Web
```

**Tools available**: Navigate, Click, Fill forms, Scroll, Extract text

## ✨ Features

- 🌐 **Natural Language Interface**: Just describe what you want in plain English
- 🤖 **Autonomous Web Browsing**: Automatically navigates, clicks, fills forms, and extracts information
- 🎨 **Modern Web UI**: Beautiful, responsive interface for easy interaction
- 🔒 **Production Ready**: Proper error handling, logging, and configuration management
- 🚀 **Fast & Efficient**: Single browser session maintained across tool calls
- 🖥️ **Cross-Platform**: Works on Windows, macOS, and Linux

## 📋 Prerequisites

- Python 3.10 or higher
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/browsingagent.git
cd browsingagent
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 4. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
```

Your `.env` file should contain:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
BROWSER_HEADLESS=false
AGENT_MAX_STEPS=20
```

**⚠️ Important**: Never commit your `.env` file or share your API key publicly!

### 5. Start the Web Server

```bash
python start_server.py
```

The server will start on `http://localhost:8000`

### 6. Use the Agent

Open your browser and go to `http://localhost:8000`

Enter natural language queries like:
- "Go to Domino's website and show me 3 popular pizza options"
- "Search for cold medicines on a pharmacy website and list prices"
- "Find flights from Bangalore to Mumbai for tomorrow"
- "Search YouTube for Python tutorials and list top 5 results"

## 💻 CLI Usage (Alternative)

You can also use the agent from the command line:

```bash
python -m src.main "Your natural language request here"
```

## 📁 Project Structure

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
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## ⚙️ Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-4o-mini` |
| `BROWSER_HEADLESS` | Run browser without GUI | `false` |
| `AGENT_MAX_STEPS` | Max actions per request | `20` |

## 🔧 How It Works

1. **User Request**: You provide a natural language query
2. **LangChain Agent**: Processes the request and plans actions
3. **Tool Selection**: Agent selects appropriate browser tools
4. **Browser Control**: Playwright controls a real Chromium browser
5. **Web Interaction**: Navigates, clicks, fills forms, scrolls, extracts content
6. **Result Processing**: Agent processes the data and returns a final answer

## ⚠️ Safety & Limitations

- The agent is designed to be **safe** and will avoid irreversible actions like finalizing payments without explicit confirmation
- Some websites may have CAPTCHAs or anti-bot protections that require manual intervention
- Complex multi-step workflows (like full checkout processes) may need site-specific customization
- The agent works best with well-structured websites

## 🔍 Troubleshooting

### Server won't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify your `.env` file exists and has a valid `OPENAI_API_KEY`
- Make sure port 8000 is not already in use

### Browser issues
- Ensure Playwright browsers are installed: `playwright install chromium`
- If you see SSL errors, you may need to configure your network settings

### Import errors
- Make sure you're running commands from the project root directory
- Verify Python version is 3.10 or higher: `python --version`

### Windows-specific issues
- Use PowerShell for best compatibility
- If activation fails, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for the GPT models
- [LangChain](https://www.langchain.com/) for the agent framework
- [Playwright](https://playwright.dev/) for browser automation
- [FastAPI](https://fastapi.tiangolo.com/) for the web server

---

Made with ❤️ by Harshit Tated

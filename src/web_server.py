"""
FastAPI web server for the browsing agent with a modern UI.
"""
# Windows event loop policy MUST be set before any other imports that use asyncio
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.config import settings  # noqa: F401  # trigger settings load early
from src.agent.langchain_agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("browsing-agent-web")

app = FastAPI(title="AI Browsing Agent", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    provider: Optional[str] = "openai"  # "openai", "anthropic", "gemini"
    # Optional per-provider API keys and models (never logged)
    openai_key: Optional[str] = None
    openai_model: Optional[str] = None
    anthropic_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    gemini_key: Optional[str] = None
    gemini_model: Optional[str] = None


class QueryResponse(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Browsing Agent</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .page-wrapper {
            display: flex;
            gap: 20px;
            width: 100%;
            max-width: 1200px;
        }

        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
            width: 320px;
            padding: 24px;
            margin-right: 20px;
        }

        .sidebar h2 {
            font-size: 1.4em;
            margin-bottom: 16px;
            color: #333;
        }

        .provider-group {
            margin-bottom: 16px;
        }

        .provider-options {
            display: flex;
            flex-direction: column;
            gap: 4px;
            font-size: 0.95em;
        }

        .settings-section {
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid #eee;
        }

        .settings-section h3 {
            font-size: 1em;
            margin-bottom: 8px;
            color: #444;
        }

        .settings-section label {
            font-size: 0.85em;
            margin-top: 6px;
            margin-bottom: 4px;
        }

        .settings-section input[type="text"],
        .settings-section input[type="password"] {
            width: 100%;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
            font-size: 0.9em;
            margin-bottom: 4px;
        }

        .settings-hint {
            font-size: 0.8em;
            color: #777;
            margin-top: 10px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }

        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .input-group {
            margin-bottom: 25px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
            font-size: 1em;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
        }

        button {
            flex: 1;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .examples {
            margin-bottom: 25px;
        }

        .examples h3 {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .example-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .example-chip {
            padding: 8px 16px;
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            border-radius: 20px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s;
        }

        .example-chip:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .result-container {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            display: none;
        }

        .result-container.show {
            display: block;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .result-title {
            font-weight: 600;
            color: #333;
            font-size: 1.1em;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-content {
            color: #333;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .error {
            color: #e74c3c;
            background: #fee;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-indicator.active {
            background: #2ecc71;
        }

        .status-indicator.inactive {
            background: #95a5a6;
        }
    </style>
</head>
<body>
    <div class="page-wrapper">
        <div class="sidebar">
            <h2>⚙️ Settings</h2>
            <div class="provider-group">
                <label>Provider</label>
                <div class="provider-options">
                    <label><input type="radio" name="provider" value="openai" checked> OpenAI</label>
                    <label><input type="radio" name="provider" value="anthropic"> Anthropic</label>
                    <label><input type="radio" name="provider" value="gemini"> Gemini</label>
                </div>
            </div>

            <div class="settings-section">
                <h3>OpenAI</h3>
                <label for="openaiKey">API Key</label>
                <input type="password" id="openaiKey" placeholder="sk-..." />
                <label for="openaiModel">Model</label>
                <input type="text" id="openaiModel" placeholder="gpt-4o-mini" />
            </div>

            <div class="settings-section">
                <h3>Anthropic</h3>
                <label for="anthropicKey">API Key</label>
                <input type="password" id="anthropicKey" placeholder="anthropic-key" />
                <label for="anthropicModel">Model</label>
                <input type="text" id="anthropicModel" placeholder="claude-3.5-sonnet" />
            </div>

            <div class="settings-section">
                <h3>Gemini</h3>
                <label for="geminiKey">API Key</label>
                <input type="password" id="geminiKey" placeholder="gemini-key" />
                <label for="geminiModel">Model</label>
                <input type="text" id="geminiModel" placeholder="gemini-1.5-pro" />
            </div>
            <p class="settings-hint">Keys are used only for this session on your machine and are not logged.</p>
        </div>

        <div class="container">
        <h1>🤖 AI Browsing Agent</h1>
        <p class="subtitle">Automate web tasks with natural language</p>

        <div class="input-group">
            <label for="query">Enter your request:</label>
            <textarea 
                id="query" 
                placeholder="Example: Go to Domino's website and show me 3 popular pizza options..."
                rows="4"
            ></textarea>
        </div>

        <div class="button-group">
            <button id="submitBtn" onclick="submitQuery()">🚀 Execute Task</button>
            <button id="clearBtn" onclick="clearResult()" style="background: #95a5a6;">Clear</button>
        </div>

        <div class="examples">
            <h3>Try these examples:</h3>
            <div class="example-chips">
                <div class="example-chip" onclick="setExample('Go to the official Domino\\'s India website and list 3 popular pizza options with brief descriptions.')">
                    🍕 Find Pizza Options
                </div>
                <div class="example-chip" onclick="setExample('Search for over-the-counter cold medicines on a major pharmacy website and summarize 3 options with prices.')">
                    💊 Find Medicine Info
                </div>
                <div class="example-chip" onclick="setExample('Go to Indigo Airlines website and show me tomorrow\\'s flights from Bangalore to Mumbai with departure times.')">
                    ✈️ Check Flights
                </div>
                <div class="example-chip" onclick="setExample('Search for Python programming tutorials on YouTube and list the top 5 results.')">
                    📚 Search Tutorials
                </div>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Agent is working... This may take a minute.</p>
        </div>

        <div class="result-container" id="resultContainer">
            <div class="result-header">
                <div class="result-title">
                    <span class="status-indicator active"></span>
                    Result
                </div>
            </div>
            <div class="result-content" id="resultContent"></div>
        </div>
    </div>

    <script>
        function setExample(text) {
            document.getElementById('query').value = text;
        }

        async function submitQuery() {
            const query = document.getElementById('query').value.trim();
            if (!query) {
                alert('Please enter a query');
                return;
            }

            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('resultContainer');
            const resultContent = document.getElementById('resultContent');

            submitBtn.disabled = true;
            loading.classList.add('show');
            resultContainer.classList.remove('show');

            try {
                const provider = document.querySelector('input[name="provider"]:checked')?.value || 'openai';

                const openaiKey = document.getElementById('openaiKey').value.trim();
                const openaiModel = document.getElementById('openaiModel').value.trim();
                const anthropicKey = document.getElementById('anthropicKey').value.trim();
                const anthropicModel = document.getElementById('anthropicModel').value.trim();
                const geminiKey = document.getElementById('geminiKey').value.trim();
                const geminiModel = document.getElementById('geminiModel').value.trim();

                const payload = {
                    query: query,
                    provider: provider,
                    openai_key: openaiKey || null,
                    openai_model: openaiModel || null,
                    anthropic_key: anthropicKey || null,
                    anthropic_model: anthropicModel || null,
                    gemini_key: geminiKey || null,
                    gemini_model: geminiModel || null,
                };

                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload),
                });

                const data = await response.json();

                if (data.success) {
                    resultContent.textContent = data.result;
                    resultContent.className = 'result-content';
                } else {
                    resultContent.innerHTML = `<div class="error">Error: ${data.error || 'Unknown error'}</div>`;
                    resultContent.className = 'result-content';
                }

                resultContainer.classList.add('show');
            } catch (error) {
                resultContent.innerHTML = `<div class="error">Network error: ${error.message}</div>`;
                resultContent.className = 'result-content';
                resultContainer.classList.add('show');
            } finally {
                submitBtn.disabled = false;
                loading.classList.remove('show');
            }
        }

        function clearResult() {
            document.getElementById('resultContainer').classList.remove('show');
            document.getElementById('query').value = '';
        }

        // Allow Enter key to submit (Ctrl+Enter or Cmd+Enter)
        document.getElementById('query').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                submitQuery();
            }
        });
    </script>
</body>
</html>
    """


@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle a user query and return the agent's result."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Log only the query, never API keys
    logger.info("Received query: %s", request.query)

    try:
        # Currently only OpenAI is implemented; other providers are placeholders
        if request.provider and request.provider.lower() != "openai":
            return QueryResponse(
                result=(
                    f"Provider '{request.provider}' is not yet implemented. "
                    f"Currently only OpenAI is supported. The sidebar collects keys/models "
                    f"for Anthropic and Gemini for future use."
                ),
                success=True,
            )

        # Override OpenAI key/model for this request if provided
        result = await run_agent(
            request.query,
            openai_key=request.openai_key,
            openai_model=request.openai_model,
        )
        return QueryResponse(result=result, success=True)
    except Exception as exc:
        logger.exception("Agent execution failed: %s", exc)
        return QueryResponse(
            result="",
            success=False,
            error=str(exc),
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "browsing-agent"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


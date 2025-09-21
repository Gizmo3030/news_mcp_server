# News MCP Server

An MCP server that helps you:
- Search the latest news (Google News RSS)
- Fetch and extract article content
- Discuss articles with an AI via your choice of LLM provider: OpenAI, Gemini, or a local Ollama

Designed to work with Open WebUI as the front-end MCP client.

## Features
- Tools:
	- `search_news(query, max_results?, lang?, country?)`
	- `get_article(url)`
	- `discuss(messages[], provider?, model?)`
- Provider selection via env or CLI: `openai`, `gemini`, `ollama`

## Setup
Requires Python 3.10–3.12.

1) Install with Poetry (recommended)
	 - Ensure Poetry is installed
	 - From repo root:
		 - `poetry install`
		 - `cp .env.example .env` and fill in keys

2) Or with pip/uv
	 - `python -m venv .venv && source .venv/bin/activate`
	 - `pip install -e .`
	 - Copy `.env.example` to `.env`

Environment variables:
- LLM_PROVIDER: openai | gemini | ollama
- LLM_MODEL: optional default model name
- OPENAI_API_KEY, GEMINI_API_KEY as needed
- OLLAMA_BASE_URL (default http://localhost:11434). Example: http://ollama.gizmosdomain.com

## Run
This is an MCP server speaking over stdio.
- With Poetry: `poetry run news-mcp-server`
- With pip: `python -m news_mcp_server.server`

Optional flags:
- `--provider openai|gemini|ollama`
- `--model <model-name>`
- `--ollama-url <http(s)://host:port>`

## Run via mcpo (OpenAPI proxy)
Expose this MCP server as an OpenAPI HTTP service using mcpo.

- Quick (no install) using uv:

```bash
uvx mcpo --port 8000 --api-key "top-secret" -- news-mcp-server \
	--provider ollama --model llama3.1:8b
```

- Or install and run with pip:

```bash
pip install mcpo
mcpo --port 8000 --api-key "top-secret" -- news-mcp-server \
	--provider openai --model gpt-4o-mini
```

- Using this repo's Poetry environment:

```bash
poetry run python -m pip install mcpo
poetry run mcpo --port 8000 --api-key "top-secret" -- \
	poetry run news-mcp-server --provider ollama --model llama3.1:8b
```

Then open http://localhost:8000/docs for auto-generated Swagger UI, or point any OpenAPI-compatible client to `http://localhost:8000` with the provided API key.

Notes:
- The server still loads your `.env` (LLM_PROVIDER, LLM_MODEL, keys) as usual; flags override env when provided.
- Omit `--api-key` for local/dev if you don't need auth. See mcpo docs for more options.

## Open WebUI integration
In Open WebUI, add an MCP server configuration pointing to this binary.

Example (systemd/env) command:
```
news-mcp-server --provider ollama --model llama3.1:8b --ollama-url http://ollama.gizmosdomain.com
```

Open WebUI should detect tools:
- search_news
- get_article
- discuss

Usage tips:
- First call `search_news` with your topic.
- Then `get_article` on a chosen URL.
- Finally use `discuss` passing messages like:
	[{"role":"system","content":"You are a helpful analyst."},{"role":"user","content":"Summarize this article: ..."}]

## Notes
- News search uses Google News RSS. Results depend on region/language.
- Article extraction uses Trafilatura; some sites may block scraping or require JS.
- Respect website terms and robots.txt when fetching content.

## License
MIT

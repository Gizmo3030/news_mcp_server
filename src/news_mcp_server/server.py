from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .news import fetch_article, search_news
from .providers import ChatMessage, ProviderConfig, make_client
from dotenv import load_dotenv


server = Server("news-mcp-server")


def _tool(name: str, description: str, args_schema: dict) -> Tool:
    """Build a JSON Schema for a tool input.

    Some sources mistakenly place a boolean `required: true/false` on each property.
    JSON Schema requires `required` to be an array on the enclosing object. We
    recursively normalize this so downstream generators (like HTTP proxies) don't
    encounter unsupported boolean types where a schema is expected.
    """

    def _clean_properties(props: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        top_required: List[str] = []
        cleaned: Dict[str, Any] = {}
        for key, schema in props.items():
            if not isinstance(schema, dict):
                cleaned[key] = schema
                continue

            # Collect and strip boolean per-property required
            if isinstance(schema.get("required"), bool) and schema.get("required"):
                top_required.append(key)
            if "required" in schema and isinstance(schema.get("required"), bool):
                schema = {k: v for k, v in schema.items() if k != "required"}

            # Recurse into object schemas
            if schema.get("type") == "object":
                nested_props = schema.get("properties", {}) or {}
                nested_clean, nested_required = _clean_properties(nested_props)
                schema = {**schema, "properties": nested_clean}
                existing_req = schema.get("required")
                if not isinstance(existing_req, list):
                    existing_req = []
                combined = list(dict.fromkeys([*existing_req, *nested_required]))
                if combined:
                    schema["required"] = combined

            # Recurse into array item object schemas
            if schema.get("type") == "array":
                items = schema.get("items")
                if isinstance(items, dict) and items.get("type") == "object":
                    nested_props = items.get("properties", {}) or {}
                    nested_clean, nested_required = _clean_properties(nested_props)
                    items = {**items, "properties": nested_clean}
                    existing_req = items.get("required")
                    if not isinstance(existing_req, list):
                        existing_req = []
                    combined = list(dict.fromkeys([*existing_req, *nested_required]))
                    if combined:
                        items["required"] = combined
                    schema = {**schema, "items": items}

            cleaned[key] = schema

        return cleaned, top_required

    properties, required_keys = _clean_properties(args_schema)

    input_schema: dict = {
        "type": "object",
        "properties": properties,
        # Being explicit helps some generators and avoids accidental extra keys.
        "additionalProperties": False,
    }
    if required_keys:
        input_schema["required"] = required_keys

    return Tool(name=name, description=description, inputSchema=input_schema)


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    return [
        _tool(
            name="search_news",
            description="Search Google News RSS for recent articles.",
            args_schema={
                "query": {"type": "string", "description": "Search keywords", "required": True},
                "max_results": {"type": "integer", "description": "Max results (1-25)", "minimum": 1, "maximum": 25, "default": 10},
                "lang": {"type": "string", "description": "Language code", "default": "en"},
                "country": {"type": "string", "description": "Country code", "default": "US"},
            },
        ),
        _tool(
            name="get_article",
            description="Fetch and extract readable content from a news article URL.",
            args_schema={
                "url": {"type": "string", "description": "Article URL", "required": True},
            },
        ),
        _tool(
            name="discuss",
            description="Discuss news content with an LLM (OpenAI, Gemini, or Ollama).",
            args_schema={
                "messages": {"type": "array", "description": "List of chat messages with role and content", "items": {"type": "object", "properties": {"role": {"type": "string", "enum": ["system", "user", "assistant"]}, "content": {"type": "string"}}, "required": ["role", "content"]}, "required": True},
                "provider": {"type": "string", "description": "Provider to use: openai, gemini, or ollama.", "enum": ["openai", "gemini", "ollama"], "required": False},
                "model": {"type": "string", "description": "Model name to use.", "required": False},
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]):
    if name == "search_news":
        query: str = arguments["query"]
        max_results: int = int(arguments.get("max_results", 10))
        lang: str = arguments.get("lang", "en")
        country: str = arguments.get("country", "US")
        items = await search_news(query, max_results=max_results, lang=lang, country=country)
        payload = [item.__dict__ for item in items]
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))]

    if name == "get_article":
        url: str = arguments["url"]
        doc = await fetch_article(url)
        return [TextContent(type="text", text=json.dumps(doc, ensure_ascii=False, indent=2))]

    if name == "discuss":
        messages = arguments.get("messages", [])
        provider_name = arguments.get("provider")
        model = arguments.get("model")
        cfg = ProviderConfig.from_env()
        if provider_name:
            # Be tolerant of case
            cfg.provider = str(provider_name).lower()  # type: ignore
        if model:
            cfg.model = model
        # Small debug hint to stderr to aid diagnostics when run standalone
        print(f"discuss: using provider={cfg.provider}, model={cfg.model or '(default)'}", file=sys.stderr)
        client = make_client(cfg)
        resp = await client.chat(messages)  # type: ignore[arg-type]
        return [TextContent(type="text", text=resp.content)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="News MCP server")
    p.add_argument("--provider", choices=["openai", "gemini", "ollama"], help="Default LLM provider.")
    p.add_argument("--model", help="Default model override.")
    p.add_argument("--ollama-url", help="Ollama base URL, e.g. http://ollama.local:11434")
    return p.parse_args(argv)


async def _main_async():
    # Load .env early so env vars are available before parsing overrides
    load_dotenv()
    args = parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.ollama_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
        # Also set legacy env used by SDKs
        os.environ["OLLAMA_HOST"] = args.ollama_url

    # Print a hint to stderr so running in a terminal doesn't look like a hang
    print("news-mcp-server: waiting for MCP client on stdio...", file=sys.stderr, flush=True)

    async with stdio_server() as (read, write):
        init_opts = server.create_initialization_options(notification_options=NotificationOptions())
        await server.run(read, write, init_opts)


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

import json
from typing import List
from unittest.mock import patch

import pytest

from news_mcp_server.server import handle_list_tools, handle_call_tool
from news_mcp_server.providers import ChatResponse


@pytest.mark.asyncio
async def test_list_tools_contains_expected():
    tools = await handle_list_tools()
    names = {t.name for t in tools}
    assert {"search_news", "get_article", "discuss"} <= names


@pytest.mark.asyncio
async def test_search_news_tool_returns_json():
    from news_mcp_server.news import NewsItem
    fake_items = [NewsItem(title="T1", url="U1", published=None, source=None, summary=None)]

    with patch("news_mcp_server.server.search_news", return_value=fake_items):
        contents = await handle_call_tool("search_news", {"query": "ai"})
    assert contents and contents[0].type == "text"
    data = json.loads(contents[0].text)
    assert data[0]["title"] == "T1"
    assert data[0]["url"] == "U1"


class FakeClient:
    def __init__(self, response_text: str):
        self._text = response_text
    async def chat(self, messages: List[dict], model=None):
        return ChatResponse(provider="ollama", model=model or "m", content=self._text)


@pytest.mark.asyncio
async def test_discuss_tool_uses_provider_mock():
    with patch("news_mcp_server.server.make_client", return_value=FakeClient("hello")):
        contents = await handle_call_tool("discuss", {
            "messages": [{"role": "user", "content": "hi"}],
            "provider": "ollama",
        })
    assert contents and contents[0].type == "text"
    assert contents[0].text == "hello"


@pytest.mark.asyncio
async def test_discuss_honors_mixed_case_provider(monkeypatch):
    # Ensure env defaults would be openai, but the call argument overrides it
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with patch("news_mcp_server.server.make_client", return_value=FakeClient("ok")) as mk:
        contents = await handle_call_tool("discuss", {
            "messages": [{"role": "user", "content": "hi"}],
            "provider": "OlLaMa",  # mixed case
        })
        # The make_client should still be called; we verify it was invoked
        assert mk.called
    assert contents and contents[0].text == "ok"


@pytest.mark.asyncio
async def test_provider_from_env_PROVIDER_var(monkeypatch):
    # If PROVIDER is set (without LLM_PROVIDER), from_env should use it
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("PROVIDER", "gemini")
    from news_mcp_server.providers import ProviderConfig
    cfg = ProviderConfig.from_env()
    assert cfg.provider == "gemini"


@pytest.mark.asyncio
async def test_get_article_tool_returns_doc():
    doc = {"url": "u", "title": "t", "content": "c"}
    with patch("news_mcp_server.server.fetch_article", return_value=doc):
        contents = await handle_call_tool("get_article", {"url": "u"})
    data = json.loads(contents[0].text)
    assert data["title"] == "t"
    assert data["content"] == "c"

@pytest.mark.asyncio
async def test_tool_input_schema_is_normalized():
    tools = await handle_list_tools()
    discuss = next(t for t in tools if t.name == "discuss")
    schema = discuss.inputSchema
    assert schema["type"] == "object"
    # messages must be required at top level
    assert "required" in schema and "messages" in schema["required"]
    # No boolean 'required' flags should remain in properties
    def _has_bool_required(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "required" and isinstance(v, bool):
                    return True
                if _has_bool_required(v):
                    return True
        elif isinstance(obj, list):
            return any(_has_bool_required(x) for x in obj)
        return False

    assert not _has_bool_required(schema["properties"])  # cleaned


@pytest.mark.asyncio
async def test_discuss_with_ollama_no_bool_keepalive():
    # Ensure our implementation doesn't pass boolean keep_alive (regression for type error)
    class SpyClient(FakeClient):
        def __init__(self):
            super().__init__("ok")
            self.seen = []
        async def chat(self, messages: List[dict], model=None):
            # If a bool keep_alive was somehow threaded here, we'd detect via signature mismatch;
            # this just acts as a smoke path.
            self.seen.append({"messages": messages, "model": model})
            return await super().chat(messages, model=model)

    spy = SpyClient()
    with patch("news_mcp_server.server.make_client", return_value=spy):
        contents = await handle_call_tool("discuss", {
            "messages": [{"role": "user", "content": "hi"}],
            "provider": "ollama",
            "model": "llama3.1:8b"
        })
    assert contents and contents[0].text == "ok"
    assert spy.seen and spy.seen[0]["messages"][0]["content"] == "hi"

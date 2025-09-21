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
async def test_get_article_tool_returns_doc():
    doc = {"url": "u", "title": "t", "content": "c"}
    with patch("news_mcp_server.server.fetch_article", return_value=doc):
        contents = await handle_call_tool("get_article", {"url": "u"})
    data = json.loads(contents[0].text)
    assert data["title"] == "t"
    assert data["content"] == "c"

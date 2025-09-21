import asyncio
from unittest.mock import patch

import pytest

from news_mcp_server.news import search_news, fetch_article, NewsItem


@pytest.mark.asyncio
async def test_search_news_parses_results():
    sample_rss = """
        <rss version="2.0"><channel>
          <item>
            <title>Example Title</title>
            <link>https://example.com/article</link>
            <pubDate>Sat, 21 Sep 2025 10:00:00 GMT</pubDate>
            <source url="https://example.com">Example</source>
            <description>Summary</description>
          </item>
        </channel></rss>
    """.strip()

    class FakeResp:
        status_code = 200
        text = sample_rss
        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResp()

    with patch("httpx.AsyncClient.get", side_effect=fake_get):
        items = await search_news("query", max_results=5)
    assert len(items) == 1
    assert items[0].title == "Example Title"
    assert items[0].url == "https://example.com/article"


@pytest.mark.asyncio
async def test_fetch_article_uses_trafilatura():
    sample_html = """
        <html><head><title>My Article</title></head>
        <body><article><p>Hello world content.</p></article></body></html>
    """.strip()

    class FakeResp:
        status_code = 200
        text = sample_html
        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResp()

    with patch("httpx.AsyncClient.get", side_effect=fake_get), \
         patch("trafilatura.extract", return_value="Hello world content."):
        doc = await fetch_article("https://example.com/article")
    assert doc["title"] == "My Article"
    assert "Hello world content." in doc["content"]

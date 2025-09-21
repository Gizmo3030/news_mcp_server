from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional

import feedparser
import httpx
import trafilatura


@dataclass
class NewsItem:
    title: str
    url: str
    published: Optional[str] = None
    source: Optional[str] = None
    summary: Optional[str] = None


def google_news_rss_url(query: str, lang: str = "en", country: str = "US") -> str:
    # Google News RSS format
    from urllib.parse import quote_plus

    q = quote_plus(query)
    hl = f"{lang}-{country}"
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={country}&ceid={country}:{lang}"


async def search_news(query: str, max_results: int = 10, lang: str = "en", country: str = "US") -> List[NewsItem]:
    url = google_news_rss_url(query, lang=lang, country=country)
    # feedparser is sync; fetch RSS ourselves to control timeouts
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.text
    feed = feedparser.parse(data)

    items: List[NewsItem] = []
    for e in feed.entries[:max_results]:
        link = getattr(e, "link", "")
        title = getattr(e, "title", "")
        published = getattr(e, "published", None)
        source = getattr(getattr(e, "source", None), "title", None)
        summary = getattr(e, "summary", None)
        items.append(NewsItem(title=title, url=link, published=published, source=source, summary=summary))
    return items


async def fetch_article(url: str) -> dict:
    # Use trafilatura to extract main content
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "news-mcp/0.1"}) as client:
        r = await client.get(url)
        r.raise_for_status()
        html = r.text
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False, favor_recall=True)
    clean = extracted or ""
    # Attempt to get title via regex fallback if trafilatura didn't include it
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else None
    return {"url": url, "title": title, "content": clean}

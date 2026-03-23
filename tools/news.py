"""
News tool -- fetches financial news from multiple sources.

Source A: MarketAux (global financial news with sentiment)
Source B: NewsData.io (India-specific business news)
Source C: Google News RSS (free, no API key, broad coverage)
Source D: Economic Times RSS (free, India's top business news)

Results are merged and deduplicated by headline similarity.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import requests
from langchain_core.tools import tool

from config import MARKETAUX_API_KEY, NEWSDATA_API_KEY, NEWS_KEYWORDS

RSS_FEEDS = [
    {
        "url": "https://news.google.com/rss/search?q=India+stock+market+economy&hl=en-IN&gl=IN&ceid=IN:en",
        "provider": "google_news",
    },
    {
        "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        "provider": "economic_times",
    },
]


def _fetch_marketaux(keywords: list[str] | None = None) -> list[dict]:
    """
    Fetch financial news from MarketAux API.

    Uses the 'search' parameter with OR logic to find articles
    matching any of the provided keywords.
    Free tier: 100 requests/day, 3 articles/request.
    """
    if not MARKETAUX_API_KEY:
        return []

    search_query = " | ".join(keywords or NEWS_KEYWORDS)

    yesterday = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")

    params = {
        "api_token": MARKETAUX_API_KEY,
        "search": search_query,
        "language": "en",
        "published_after": yesterday,
        "sort": "published_at",
        "limit": 3,
    }

    try:
        resp = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    articles = []
    for item in data.get("data", []):
        sentiment_scores = [
            e.get("sentiment_score", 0)
            for e in item.get("entities", [])
            if e.get("sentiment_score") is not None
        ]
        avg_sentiment = (
            round(sum(sentiment_scores) / len(sentiment_scores), 2)
            if sentiment_scores
            else None
        )

        articles.append({
            "title": item.get("title", ""),
            "summary": item.get("description") or item.get("snippet", ""),
            "source": item.get("source", ""),
            "published": item.get("published_at", ""),
            "url": item.get("url", ""),
            "sentiment": avg_sentiment,
            "provider": "marketaux",
        })

    return articles


def _fetch_newsdata() -> list[dict]:
    """
    Fetch India-specific business news from NewsData.io.

    Filters by country=in, category=business, language=en.
    Free tier: 200 credits/day, 10 articles/request, 12h delay.
    """
    if not NEWSDATA_API_KEY:
        return []

    params = {
        "apikey": NEWSDATA_API_KEY,
        "country": "in",
        "category": "business",
        "language": "en",
        "timeframe": 24,
    }

    try:
        resp = requests.get(
            "https://newsdata.io/api/1/latest",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    articles = []
    for item in data.get("results", []):
        articles.append({
            "title": item.get("title", ""),
            "summary": item.get("description") or "",
            "source": item.get("source_id", ""),
            "published": item.get("pubDate", ""),
            "url": item.get("link", ""),
            "sentiment": None,
            "provider": "newsdata",
        })

    return articles


def _deduplicate(articles: list[dict], threshold: float = 0.7) -> list[dict]:
    """
    Remove near-duplicate articles by comparing titles.

    Uses SequenceMatcher to compute similarity between headlines.
    If two titles are >70% similar, the duplicate is dropped
    (keeps the first occurrence, which is from MarketAux since
    it has sentiment data).
    """
    unique = []
    for article in articles:
        is_dup = False
        for kept in unique:
            similarity = SequenceMatcher(
                None,
                article["title"].lower(),
                kept["title"].lower(),
            ).ratio()
            if similarity >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(article)
    return unique


def _fetch_rss() -> list[dict]:
    """
    Fetch news from RSS feeds (Google News India + Economic Times).

    Free, no API key needed, no rate limits. Returns up to 10 articles per feed.
    """
    articles = []
    for feed in RSS_FEEDS:
        try:
            resp = requests.get(feed["url"], timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")[:10]
            for item in items:
                title = item.findtext("title", "")
                desc = item.findtext("description", "")
                link = item.findtext("link", "")
                pub_date = item.findtext("pubDate", "")
                source_elem = item.find("source")
                source = source_elem.text if source_elem is not None else feed["provider"]

                articles.append({
                    "title": title,
                    "summary": desc,
                    "source": source,
                    "published": pub_date,
                    "url": link,
                    "sentiment": None,
                    "provider": feed["provider"],
                })
        except Exception:
            continue

    return articles


@tool
def fetch_news() -> list[dict]:
    """Fetch latest financial news relevant to Indian markets from multiple sources."""
    marketaux_articles = _fetch_marketaux()
    newsdata_articles = _fetch_newsdata()
    rss_articles = _fetch_rss()

    combined = marketaux_articles + newsdata_articles + rss_articles
    deduplicated = _deduplicate(combined)

    return deduplicated

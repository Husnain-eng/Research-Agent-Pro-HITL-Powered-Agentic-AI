"""
DuckDuckGo web search tool.

Returns a concise summary of top search results.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from config.settings import settings


def _fetch_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return raw DuckDuckGo results (imported lazily to keep startup fast)."""
    try:
        from ddgs import DDGS  # new package name (ddgs >= 9.x)
    except ImportError:
        from duckduckgo_search import DDGS  # legacy fallback

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return summarised results.

    Args:
        query: The search query string.

    Returns:
        A formatted string with the top search results including titles,
        URLs, and snippets.
    """
    try:
        results = _fetch_results(query, settings.duckduckgo_max_results)
        if not results:
            return f"No results found for: {query}"

        lines: list[str] = [f"Web Search Results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "")
            body = r.get("body", "No snippet available")
            lines.append(f"{i}. {title}\n   URL: {url}\n   {body}\n")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"DuckDuckGo search failed: {exc}"

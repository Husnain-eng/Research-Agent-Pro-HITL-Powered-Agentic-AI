"""
Wikipedia topic summary tool.

Fetches clean, concise summaries from Wikipedia.
"""

from __future__ import annotations

from langchain_core.tools import tool

from config.settings import settings


@tool
def wikipedia_search(query: str) -> str:
    """
    Fetch a topic summary from Wikipedia.

    Args:
        query: The topic or concept to look up on Wikipedia.

    Returns:
        A concise summary of the Wikipedia article for the topic.
    """
    try:
        import wikipedia  # type: ignore[import]

        wikipedia.set_lang("en")

        # Try direct lookup first, fall back to search
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(
                query,
                sentences=settings.wikipedia_sentences,
                auto_suggest=True,
            )
            return (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.url}\n\n"
                f"{summary}"
            )
        except wikipedia.DisambiguationError as e:
            # Take the first disambiguation option
            first_option = e.options[0]
            summary = wikipedia.summary(first_option, sentences=settings.wikipedia_sentences)
            return (
                f"Wikipedia (disambiguation → '{first_option}'):\n\n"
                f"{summary}"
            )
        except wikipedia.PageError:
            # Fall back to search results
            results = wikipedia.search(query, results=3)
            if not results:
                return f"No Wikipedia article found for: {query}"
            summary = wikipedia.summary(results[0], sentences=settings.wikipedia_sentences)
            return f"Wikipedia (closest match: '{results[0]}'):\n\n{summary}"

    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia search failed: {exc}"

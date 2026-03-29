"""
Arxiv research paper search tool.

Fetches paper titles, authors, and abstracts from arxiv.org.
"""

from __future__ import annotations

from langchain_core.tools import tool

from config.settings import settings


@tool
def arxiv_search(query: str) -> str:
    """
    Search Arxiv for research papers matching the query.

    Args:
        query: Topic or keywords to search for on Arxiv.

    Returns:
        A formatted string with paper titles, authors, and abstracts.
    """
    try:
        import arxiv  # type: ignore[import]

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=settings.arxiv_max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))

        if not results:
            return f"No Arxiv papers found for: {query}"

        lines: list[str] = [f"Arxiv Papers for: '{query}'\n"]
        for i, paper in enumerate(results, 1):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = paper.summary.replace("\n", " ")[:500]
            lines.append(
                f"{i}. {paper.title}\n"
                f"   Authors: {authors}\n"
                f"   Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"   URL: {paper.entry_id}\n"
                f"   Abstract: {abstract}...\n"
            )

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Arxiv search failed: {exc}"

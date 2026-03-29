"""Tool registry — single source of truth for all available tools."""

from .arxiv_tool import arxiv_search
from .duckduckgo import duckduckgo_search
from .wikipedia_tool import wikipedia_search

ALL_TOOLS = [duckduckgo_search, arxiv_search, wikipedia_search]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

__all__ = ["ALL_TOOLS", "TOOL_MAP", "arxiv_search", "duckduckgo_search", "wikipedia_search"]

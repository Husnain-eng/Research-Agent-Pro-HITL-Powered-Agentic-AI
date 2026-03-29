"""
Ollama LLM setup with tool-calling support.

Provides a pre-configured ChatOllama instance bound to all research tools.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from agent.tools import ALL_TOOLS
from config.settings import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    Return a cached, tool-bound ChatOllama instance.

    Uses low temperature for deterministic, reproducible outputs.
    """
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_predict=settings.max_tokens,
    )
    return llm.bind_tools(ALL_TOOLS)


def get_llm_no_tools() -> ChatOllama:
    """Return a plain LLM without tool bindings (used for final answer synthesis)."""
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_predict=settings.max_tokens,
    )

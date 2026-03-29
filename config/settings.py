"""
Global configuration settings for Research Agent Pro.
"""

from dataclasses import dataclass


@dataclass
class Settings:
    """Central configuration for the research agent."""

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 2048

    # Tools
    duckduckgo_max_results: int = 5
    arxiv_max_results: int = 3
    wikipedia_sentences: int = 10

    # Agent
    max_iterations: int = 10
    checkpoint_db: str = "checkpoints.db"

    # CLI
    stream_delay: float = 0.02


settings = Settings()

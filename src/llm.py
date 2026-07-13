"""
LLM wrapper. Uses Groq's free API tier via langchain-groq - fast inference,
no cost, no local GPU required.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from config import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None) -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and add it to your .env file or Streamlit secrets."
        )
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model_name,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )

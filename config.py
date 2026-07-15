"""
Central configuration for the Disneyland Review RAG app.

Uses pydantic-settings so every config value is validated and can be
overridden via environment variables or a .env file without touching code.
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Groq LLM ---
    groq_api_key: str = ""
    llm_model_name: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    # --- Hugging Face (optional but recommended - avoids unauthenticated rate limits,
    # which can cause the embedding model download to stall on shared cloud IPs) ---
    hf_token: str = ""

    # --- Embeddings / Reranking (all free, local, no API cost) ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Data paths ---
    raw_data_path: Path = Path("data/raw/DisneylandReviews.csv")
    processed_data_path: Path = Path("data/processed/cleaned_disneylandreviews.csv")
    chroma_persist_dir: Path = Path("data/processed/chroma_db")
    bm25_corpus_path: Path = Path("data/processed/bm25_corpus.pkl")

    # --- Ingestion ---
    # Keep the deployed demo light on a free-tier host; set to None to use the full dataset.
    sample_size: int | None = 4000
    random_seed: int = 42

    # --- Retrieval ---
    dense_k: int = 10          # candidates pulled from Chroma
    bm25_k: int = 10           # candidates pulled from BM25
    ensemble_weights: tuple[float, float] = (0.5, 0.5)  # (dense, sparse)
    rerank_top_n: int = 4      # final docs sent to the LLM after reranking
    simple_top_k: int = 4      # final docs for the "simple RAG" path

    # --- Sentiment ---
    sentiment_positive_threshold: float = 0.05
    sentiment_negative_threshold: float = -0.05


settings = Settings()

# Type alias used across the app for the pipeline toggle
RagPipeline = Literal["simple", "hybrid_rerank"]

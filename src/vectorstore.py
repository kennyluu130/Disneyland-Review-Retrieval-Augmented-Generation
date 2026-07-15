"""
Vector store construction:
  - Dense index: Chroma, embedded with a free local sentence-transformers model
  - Sparse index: BM25 over the same documents (for hybrid search)

Both are persisted to disk so the Streamlit app can load them instantly
instead of re-embedding on every restart.
"""
from __future__ import annotations

import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

try:
    # LangChain 0.2.9+ moved Chroma into its own dedicated package
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


def _configure_hf_auth() -> None:
    """
    Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN so huggingface_hub authenticates its
    requests. Unauthenticated requests share a much lower rate limit, and on a
    shared-IP host like Streamlit Community Cloud that limit can be hit by
    other apps' traffic too - which manifests as a download that silently
    stalls rather than a clean error. A free token from
    https://huggingface.co/settings/tokens fixes this.
    """
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Free, local embedding model - no API key required, but an HF token is
    recommended (see _configure_hf_auth) to avoid rate-limited/stalled downloads."""
    _configure_hf_auth()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


def build_chroma_index(documents: list[Document], persist_dir: str | None = None) -> Chroma:
    persist_dir = persist_dir or str(settings.chroma_persist_dir)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_function(),
        persist_directory=persist_dir,
        collection_name="disneyland_reviews",
    )
    # Note: Chroma auto-persists as of 0.4.x - no manual .persist() call needed/supported.
    logger.info("Built Chroma index with %d documents at %s", len(documents), persist_dir)
    return vectordb


def load_chroma_index(persist_dir: str | None = None) -> Chroma:
    persist_dir = persist_dir or str(settings.chroma_persist_dir)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_function(),
        collection_name="disneyland_reviews",
    )


def build_bm25_retriever(documents: list[Document], k: int | None = None) -> BM25Retriever:
    k = k or settings.bm25_k
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def save_bm25_corpus(documents: list[Document], path: str | None = None) -> None:
    path = path or str(settings.bm25_corpus_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(documents, f)
    logger.info("Saved BM25 corpus (%d docs) to %s", len(documents), path)


def load_bm25_corpus(path: str | None = None) -> list[Document]:
    path = path or str(settings.bm25_corpus_path)
    with open(path, "rb") as f:
        return pickle.load(f)

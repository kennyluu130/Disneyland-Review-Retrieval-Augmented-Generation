"""
Retrieval strategies. This is the core of the "toggle" the user sees in the UI:

  - "simple":        pure dense (Chroma) similarity search, top_k results
  - "hybrid_rerank":  BM25 (sparse) + Chroma (dense) candidates merged via an
                       EnsembleRetriever, then reranked with a cross-encoder
                       and truncated to the final top_n
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    # LangChain 1.0+ moved legacy retrievers (including EnsembleRetriever) into
    # the separate langchain-classic package.
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        # Pre-1.0 LangChain
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import RagPipeline, settings
from src.rerank import rerank_documents
from src.vectorstore import build_bm25_retriever

logger = logging.getLogger(__name__)


def _apply_branch_filter(vectordb: Chroma, k: int, branch_filter: Optional[list[str]]) -> dict:
    search_kwargs = {"k": k}
    if branch_filter:
        search_kwargs["filter"] = {"branch": {"$in": branch_filter}}
    return search_kwargs


def simple_retrieve(
    query: str,
    vectordb: Chroma,
    top_k: int | None = None,
    branch_filter: Optional[list[str]] = None,
) -> list[Document]:
    """Plain dense similarity search - the 'simple RAG' baseline."""
    top_k = top_k or settings.simple_top_k
    search_kwargs = _apply_branch_filter(vectordb, top_k, branch_filter)
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def hybrid_rerank_retrieve(
    query: str,
    vectordb: Chroma,
    bm25_documents: list[Document],
    top_n: int | None = None,
    branch_filter: Optional[list[str]] = None,
) -> list[Document]:
    """BM25 + dense ensemble retrieval, then cross-encoder reranking."""
    top_n = top_n or settings.rerank_top_n

    dense_kwargs = _apply_branch_filter(vectordb, settings.dense_k, branch_filter)
    dense_retriever = vectordb.as_retriever(search_kwargs=dense_kwargs)

    bm25_retriever = build_bm25_retriever(bm25_documents, k=settings.bm25_k)

    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=list(settings.ensemble_weights),
    )
    candidates = ensemble.invoke(query)

    if branch_filter:
        candidates = [d for d in candidates if d.metadata.get("branch") in branch_filter]

    return rerank_documents(query, candidates, top_n=top_n)


def retrieve(
    pipeline: RagPipeline,
    query: str,
    vectordb: Chroma,
    bm25_documents: list[Document],
    top_k: int | None = None,
    branch_filter: Optional[list[str]] = None,
) -> list[Document]:
    """Single entry point the app calls - dispatches on the pipeline toggle."""
    if pipeline == "simple":
        return simple_retrieve(query, vectordb, top_k=top_k, branch_filter=branch_filter)
    elif pipeline == "hybrid_rerank":
        return hybrid_rerank_retrieve(query, vectordb, bm25_documents, top_n=top_k, branch_filter=branch_filter)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

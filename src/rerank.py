"""
Cross-encoder reranking step for the hybrid RAG pipeline.

A cross-encoder scores (query, document) pairs jointly, which is much more
accurate than the cosine-similarity used for initial retrieval - but too slow
to run over the whole corpus. So the pattern is: retrieve a broad candidate
set cheaply (dense + BM25), then rerank only those candidates.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from src.vectorstore import _configure_hf_auth

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    _configure_hf_auth()
    return CrossEncoder(settings.cross_encoder_model_name)


def rerank_documents(query: str, documents: list[Document], top_n: int | None = None) -> list[Document]:
    """Score each candidate document against the query and return the top_n, sorted."""
    if not documents:
        return []

    top_n = top_n or settings.rerank_top_n
    cross_encoder = get_cross_encoder()

    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for doc, score in scored_docs[:top_n]:
        doc = Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": float(score)})
        reranked.append(doc)

    logger.debug("Reranked %d candidates down to top %d", len(documents), len(reranked))
    return reranked

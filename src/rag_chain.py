"""
End-to-end RAG chain: question -> retrieve -> grounded prompt -> LLM answer.

Returns a Pydantic-validated ChatResponse so the UI layer never has to trust
an unstructured dict coming back from the model.
"""
from __future__ import annotations

import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import RagPipeline
from src.llm import get_llm
from src.retrieval import retrieve
from src.schemas import ChatResponse, RetrievedDocument

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Disneyland park \
visitor reviews (Disneyland California, Paris, and Hong Kong).

Answer ONLY using the review excerpts provided in the context below. Each excerpt is \
tagged with its branch, star rating, and sentiment. If the context doesn't contain enough \
information to answer, say so honestly rather than guessing.

When relevant, mention which park(s) the observation applies to and whether it reflects \
positive or negative sentiment. Keep answers concise (3-6 sentences) unless asked for detail."""


def _format_context(documents: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        blocks.append(
            f"[Review {i}] Branch: {meta.get('branch')} | Rating: {meta.get('rating')}/5 "
            f"| Sentiment: {meta.get('sentiment_label')} ({meta.get('sentiment', 0):.2f})\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def _to_retrieved_documents(documents: list[Document]) -> list[RetrievedDocument]:
    results = []
    for doc in documents:
        meta = doc.metadata
        score = meta.get("rerank_score", meta.get("score"))
        results.append(
            RetrievedDocument(
                review_id=meta.get("review_id", -1),
                branch=meta.get("branch", "unknown"),
                rating=meta.get("rating", 0),
                sentiment_label=meta.get("sentiment_label", "Neutral"),
                snippet=doc.page_content[:280],
                score=score,
            )
        )
    return results


def answer_question(
    question: str,
    pipeline: RagPipeline,
    vectordb: Chroma,
    bm25_documents: list[Document],
    top_k: int | None = None,
    branch_filter: list[str] | None = None,
) -> ChatResponse:
    start = time.perf_counter()

    documents = retrieve(
        pipeline=pipeline,
        query=question,
        vectordb=vectordb,
        bm25_documents=bm25_documents,
        top_k=top_k,
        branch_filter=branch_filter,
    )

    context = _format_context(documents)
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)

    latency = time.perf_counter() - start

    return ChatResponse(
        answer=response.content,
        pipeline=pipeline,
        sources=_to_retrieved_documents(documents),
        latency_seconds=latency,
    )

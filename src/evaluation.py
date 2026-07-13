"""
Evaluation harness comparing the "simple" and "hybrid_rerank" pipelines on a
small curated question set, using two complementary metrics:

  1. keyword_recall  - cheap, deterministic proxy: fraction of expected
                        keywords that show up in the generated answer.
  2. LLM-judge score - Groq LLM rates faithfulness (1-5, is the answer
                        grounded in the retrieved reviews) and relevance
                        (1-5, does it address the question), returned as
                        validated structured JSON via Pydantic.

Results feed the "Evaluation & Comparison" tab in the Streamlit app.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from config import RagPipeline
from src.llm import get_llm
from src.rag_chain import answer_question
from src.schemas import EvalJudgeScore, EvalResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are grading a RAG chatbot's answer about Disneyland reviews.

Score the answer on two dimensions, each from 1 (worst) to 5 (best):
- faithfulness: does the answer avoid making claims that aren't supported by the given context?
- relevance: does the answer actually address the question asked?

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "justification": "<one short sentence>"}"""


def load_eval_questions(path: str = "eval/eval_questions.json") -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def llm_judge(question: str, answer: str, context: str) -> EvalJudgeScore:
    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {question}\n\nContext used:\n{context}\n\nAnswer to grade:\n{answer}"
        ),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()
    # Models occasionally wrap JSON in markdown fences - strip defensively
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        return EvalJudgeScore(**data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Judge output failed to parse (%s); defaulting to neutral scores", exc)
        return EvalJudgeScore(faithfulness=3, relevance=3, justification="Judge output unparseable.")


def run_evaluation(
    vectordb: Chroma,
    bm25_documents: list[Document],
    pipelines: list[RagPipeline] = ("simple", "hybrid_rerank"),
    questions_path: str = "eval/eval_questions.json",
) -> pd.DataFrame:
    """Run every question through every pipeline and score the results."""
    questions = load_eval_questions(questions_path)
    rows: list[EvalResult] = []

    for item in questions:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        for pipeline in pipelines:
            response = answer_question(
                question=question,
                pipeline=pipeline,
                vectordb=vectordb,
                bm25_documents=bm25_documents,
            )
            context = "\n".join(s.snippet for s in response.sources)
            judge = llm_judge(question, response.answer, context)
            recall = keyword_recall(response.answer, expected_keywords)

            rows.append(
                EvalResult(
                    question=question,
                    pipeline=pipeline,
                    answer=response.answer,
                    latency_seconds=response.latency_seconds,
                    keyword_recall=recall,
                    faithfulness=judge.faithfulness,
                    relevance=judge.relevance,
                )
            )

    return pd.DataFrame([r.model_dump() for r in rows])


def summarize_evaluation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean metrics per pipeline for the comparison chart."""
    return (
        results_df.groupby("pipeline")[["latency_seconds", "keyword_recall", "faithfulness", "relevance"]]
        .mean()
        .round(3)
        .reset_index()
    )

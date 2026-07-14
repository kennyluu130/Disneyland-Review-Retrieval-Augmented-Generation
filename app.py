"""
Streamlit app for the Disneyland Review RAG chatbot.

Tabs:
  - Chat:            ask questions, toggle Simple RAG vs Hybrid Search + Reranking
  - Evaluation:       run the curated eval set through both pipelines and compare
  - Dataset Insights: EDA charts reproduced from the source notebooks
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src.evaluation import run_evaluation, summarize_evaluation
from src.ingestion import load_processed_data
from src.rag_chain import answer_question
from src.schemas import Branch
from src.sentiment import top_words_by_sentiment
from src.vectorstore import load_bm25_corpus, load_chroma_index

st.set_page_config(page_title="Disneyland Review RAG", page_icon="🎢", layout="wide")


# ----------------------------------------------------------------------------
# Cached resource loading (index + data load once per session, not per query)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vector index...")
def _load_vectordb():
    return load_chroma_index()


@st.cache_resource(show_spinner="Loading BM25 corpus...")
def _load_bm25():
    return load_bm25_corpus()


@st.cache_data(show_spinner="Loading processed reviews...")
def _load_processed_df() -> pd.DataFrame:
    return load_processed_data()


@st.cache_data(show_spinner=False)
def _cached_top_words_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Uncached, this reruns NLTK tokenization over the whole corpus on every
    # Streamlit script rerun (which happens on every chat message, since
    # st.tabs() executes all tab bodies unconditionally on every rerun).
    return top_words_by_sentiment(df)


def _indexes_available() -> bool:
    return settings.chroma_persist_dir.exists() and settings.bm25_corpus_path.exists()


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("🎢 Disneyland Review RAG")
st.sidebar.caption("Hybrid search + reranking vs. simple RAG, side by side.")

pipeline_label = st.sidebar.radio(
    "Retrieval pipeline",
    options=["Hybrid Search + Reranking", "Simple RAG"],
    help=(
        "Simple RAG: dense (Chroma) similarity search only.\n\n"
        "Hybrid + Reranking: BM25 + dense candidates merged, then reranked "
        "with a cross-encoder before being sent to the LLM."
    ),
)
pipeline = "hybrid_rerank" if pipeline_label == "Hybrid Search + Reranking" else "simple"

branch_options = [b.value for b in Branch]
branch_filter = st.sidebar.multiselect("Filter by park", options=branch_options, default=[])

top_k = st.sidebar.slider("Number of reviews to retrieve", min_value=2, max_value=8, value=4)

st.sidebar.divider()
st.sidebar.markdown(
    "**Stack:** pandas · NLTK VADER · Pydantic · Chroma · BM25 · "
    "sentence-transformers cross-encoder · LangChain · Groq · Streamlit"
)

if not settings.groq_api_key:
    st.sidebar.warning("No GROQ_API_KEY found. Add one to `.env` or Streamlit secrets to enable the chatbot.")

if not _indexes_available():
    st.error(
        "No index found yet. Run `python scripts/build_index.py` locally first "
        "(after placing the Kaggle CSV in `data/raw/DisneylandReviews.csv`)."
    )
    st.stop()

chat_tab, eval_tab, insights_tab = st.tabs(["💬 Chat", "📊 Evaluation & Comparison", "🔎 Dataset Insights"])

# ----------------------------------------------------------------------------
# Chat tab
# ----------------------------------------------------------------------------
with chat_tab:
    st.subheader("Ask a question about Disneyland reviews")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(f"Pipeline: {turn['pipeline']} · {turn['latency']:.2f}s")
            with st.expander(f"View {len(turn['sources'])} retrieved reviews"):
                for src in turn["sources"]:
                    score_str = f" · score: {src.score:.3f}" if src.score is not None else ""
                    st.markdown(
                        f"**{src.branch}** · {src.rating}⭐ · {src.sentiment_label}{score_str}\n\n"
                        f"> {src.snippet}..."
                    )

    question = st.chat_input("e.g. What do people say about wait times at Disneyland Paris?")

    if question:
        vectordb = _load_vectordb()
        bm25_documents = _load_bm25()

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving with {pipeline_label}..."):
                try:
                    response = answer_question(
                        question=question,
                        pipeline=pipeline,
                        vectordb=vectordb,
                        bm25_documents=bm25_documents,
                        top_k=top_k,
                        branch_filter=branch_filter or None,
                    )
                    st.write(response.answer)
                    st.caption(f"Pipeline: {response.pipeline} · {response.latency_seconds:.2f}s")
                    with st.expander(f"View {len(response.sources)} retrieved reviews"):
                        for src in response.sources:
                            score_str = f" · score: {src.score:.3f}" if src.score is not None else ""
                            st.markdown(
                                f"**{src.branch}** · {src.rating}⭐ · {src.sentiment_label}{score_str}\n\n"
                                f"> {src.snippet}..."
                            )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": response.answer,
                            "pipeline": response.pipeline,
                            "latency": response.latency_seconds,
                            "sources": response.sources,
                        }
                    )
                except RuntimeError as exc:
                    st.error(str(exc))

# ----------------------------------------------------------------------------
# Evaluation tab
# ----------------------------------------------------------------------------
with eval_tab:
    st.subheader("Simple RAG vs. Hybrid Search + Reranking")
    st.caption(
        "Runs a curated set of Disneyland questions through both pipelines and scores each "
        "answer on keyword recall (cheap heuristic) and an LLM-judge's faithfulness + "
        "relevance ratings (1-5)."
    )

    if st.button("Run evaluation", type="primary"):
        if not settings.groq_api_key:
            st.error("GROQ_API_KEY required to run evaluation (the judge uses the LLM too).")
        else:
            vectordb = _load_vectordb()
            bm25_documents = _load_bm25()
            with st.spinner("Running both pipelines over the eval set... this takes a minute."):
                results_df = run_evaluation(vectordb, bm25_documents)
                st.session_state.eval_results = results_df

    if "eval_results" in st.session_state:
        results_df = st.session_state.eval_results
        summary_df = summarize_evaluation(results_df)

        st.markdown("#### Summary (averaged across all questions)")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(summary_df.set_index("pipeline")[["faithfulness", "relevance"]])
            st.caption("LLM-judge scores (1-5, higher is better)")
        with col2:
            st.bar_chart(summary_df.set_index("pipeline")[["latency_seconds"]])
            st.caption("Average latency in seconds (lower is better)")

        st.bar_chart(summary_df.set_index("pipeline")[["keyword_recall"]])
        st.caption("Keyword recall proxy (higher is better)")

        st.markdown("#### Per-question detail")
        st.dataframe(results_df, use_container_width=True)

# ----------------------------------------------------------------------------
# Dataset Insights tab (reproduces the source notebooks' EDA)
# ----------------------------------------------------------------------------
with insights_tab:
    st.subheader("Dataset insights")
    df = _load_processed_df()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total reviews indexed", len(df))
    col2.metric("Average rating", f"{df['Rating'].mean():.2f} / 5")
    col3.metric("Average sentiment", f"{df['Sentiment'].mean():.2f}")

    st.markdown("#### Rating distribution by park")
    rating_by_branch = df.groupby(["Branch", "Rating"]).size().unstack(fill_value=0)
    st.bar_chart(rating_by_branch.T)

    st.markdown("#### Sentiment label distribution by park")
    sentiment_by_branch = df.groupby(["Branch", "Sentiment_Label"]).size().unstack(fill_value=0)
    st.bar_chart(sentiment_by_branch)

    st.markdown("#### Most common words: high vs. low sentiment reviews")
    with st.spinner("Computing word frequencies..."):
        word_df = _cached_top_words_by_sentiment(df)
    wcol1, wcol2 = st.columns(2)
    with wcol1:
        st.markdown("**High sentiment**")
        st.dataframe(word_df[word_df["Sentiment"] == "High"][["Word", "Count"]], hide_index=True)
    with wcol2:
        st.markdown("**Low sentiment**")
        st.dataframe(word_df[word_df["Sentiment"] == "Low"][["Word", "Count"]], hide_index=True)

# 🎢 Disneyland Review RAG Chatbot

An end-to-end RAG application over the [Disneyland Reviews dataset](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews)
(42k+ reviews across the California, Paris, and Hong Kong parks), with a live toggle between
**Simple RAG** and **Hybrid Search + Reranking**, plus an evaluation dashboard comparing them.

Runs entirely on free tiers: local embeddings + reranker (sentence-transformers), Chroma for
vector storage, and Groq's free LLM API for generation.

## Architecture

```
Raw CSV (Kaggle)
   │
   ▼
Preprocessing (dedupe, recover missing dates, cast types)
   │
   ▼
Pydantic validation (schema-checked rows dropped if malformed)
   │
   ▼
Sentiment scoring (NLTK VADER)
   │
   ▼
Documents (LangChain)
   │
   ├──► Chroma dense index (sentence-transformers embeddings)
   └──► BM25 sparse index                                        
              │
              ▼
   Retrieval toggle: Simple (dense only) vs. Hybrid (BM25+dense ensemble → cross-encoder rerank)
              │                                                    
              ▼
   Grounded prompt → Groq LLM → validated ChatResponse            
              │
              ▼
   Streamlit UI: Chat / Evaluation & Comparison / Dataset Insights
```

Evaluation (`src/evaluation.py`) runs a curated question set through both pipelines and scores
each answer with keyword recall + an LLM-judge (faithfulness & relevance, 1-5), so the two
retrieval strategies can be compared quantitatively

## Setup

```bash
git clone <this-repo>
cd disneyland-rag
python -m venv venv && source venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

Get a **free** Groq API key at https://console.groq.com/keys, then:

```bash
cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

### Get the data

Option A - via kagglehub (needs a Kaggle account + API token, see https://www.kaggle.com/docs/api):
```bash
python scripts/download_data.py
```

Option B - manual: download `DisneylandReviews.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/arushchillar/disneyland-reviews) and place
it at `data/raw/DisneylandReviews.csv`.

### Build the indexes (one time)

```bash
python scripts/build_index.py
```

This cleans the data, scores sentiment, and builds/persists the Chroma + BM25 indexes to
`data/processed/`. Re-run it any time you change `config.py`'s `sample_size` or embedding model.

### Run the app

```bash
streamlit run app.py
```

## Deployment

1. Push this repo to GitHub.
2. Run `scripts/build_index.py` locally and commit the resulting `data/processed/` folder
   (Chroma index + BM25 pickle + cleaned CSV) - Streamlit Community Cloud won't run the build
   script for you, so the prebuilt index needs to ship with the repo.
3. Deploy on [Streamlit Community Cloud](https://share.streamlit.io) pointing at `app.py`.
4. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

## Notes

- `config.py`'s `sample_size` (default 4000, stratified by park) keeps the deployed demo's
  memory footprint small. Set it to `None` to index the full ~40k reviews if running locally
  with more RAM.
- Embeddings (`all-MiniLM-L6-v2`) and the reranker (`ms-marco-MiniLM-L-6-v2`) both run locally
  via `sentence-transformers` - no API calls, no cost, no rate limits.
- Only the LLM calls (chat answers + eval judge) hit an external API (Groq), which has a
  generous free tier.

## Project structure

See the module docstrings in `src/` for details on each stage of the pipeline - each file
maps to one stage in the architecture diagram above.

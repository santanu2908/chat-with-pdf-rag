# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Minimal RAG (Retrieval-Augmented Generation) service: upload a PDF via API, ask questions, get answers grounded in the document with cited source chunks. Single-user, single-document, in-memory — a demo/interview project, not production.

## Commands

```bash
uv sync                              # install deps into .venv
cp .env.example .env                  # then set your API key
uv run uvicorn app.main:app --reload  # start dev server on :8000
uv add <package>                      # add a dependency
```

```bash
uv run pytest tests/test_eval.py -v   # run 19-question eval harness
```

## Architecture

Request flow: **FastAPI endpoint → PDF extraction → chunking → embedding → FAISS + BM25 index → hybrid retrieval (RRF) → cross-encoder rerank → LLM generation → response** (blocking or SSE-streamed)

Six modules in `app/`:

- **`main.py`** — FastAPI app with four endpoints (`/health`, `/upload`, `/query`, `/query/stream`). Loads `.env` at import time before other app modules. Owns the system prompt and user-prompt formatting. Orchestrates the full pipeline.
- **`pdf_loader.py`** — Extracts text per page from PDF bytes via `pypdf`. Returns `List[Tuple[str, page_number]]`. Writes to `/tmp` for pypdf compatibility.
- **`rag.py`** — Chunking (500-char window, 50-char overlap, per-page) and embedding (`all-MiniLM-L6-v2` via sentence-transformers, 384-dim, normalized). The `Chunk` dataclass lives here. Embedding model is lazy-loaded as a module-level singleton.
- **`store.py`** — Hybrid retrieval store combining FAISS `IndexFlatIP` (dense) and BM25 (sparse). Uses Reciprocal Rank Fusion (RRF, k=60) to merge ranked lists. Global singleton `store`. `reset()` clears both indices (single-doc design).
- **`reranker.py`** — Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`). Uses guaranteed-slots strategy: top `top_k - 1` first-stage results are preserved, cross-encoder picks the last slot from remaining candidates. Prevents CE from burying dense table chunks.
- **`llm.py`** — Abstract `LLMClient` with two methods (`generate` for blocking, `stream` for SSE) and three implementations: `GroqClient`, `OpenAIClient`, `AnthropicClient`. Factory `get_llm_client()` reads `LLM_PROVIDER` env var. All use `temperature=0.2, max_tokens=800`.

## Key Design Decisions

- **LLM provider is swappable** via `LLM_PROVIDER` env var (`groq`|`openai`|`anthropic`). Only the chosen provider's API key is needed.
- **`load_dotenv()` runs in `main.py` before other app imports** — module-level env reads in `llm.py` depend on this ordering.
- **Embeddings are local** (no API call) — `all-MiniLM-L6-v2` runs on CPU. First call downloads ~80MB.
- **Hybrid retrieval** via BM25 (`rank-bm25`) + FAISS, fused with RRF. Catches keyword matches that pure vector search misses (e.g. "CEO", "MTBF").
- **Reranker uses guaranteed slots** — cross-encoder (`ms-marco-MiniLM-L-6-v2`) trained on MS MARCO struggles with dense table chunks. First-stage top results are preserved; CE only fills the last slot. Over-fetches `top_k * 2` candidates.
- **Eval harness** — 19 questions across 6 categories (single-hop, specific-numbers, tables, multi-hop, negative, tricky-retrieval) in `tests/eval_questions.json`. Tests hit a real LLM — need API key and network.
- **Streaming via SSE** — `/query/stream` yields `data: {"token": "..."}` events followed by a final `data: {"sources": [...]}` event. Each LLM client implements `stream()` using its SDK's native streaming API.
- **No persistence** — FAISS index, BM25 index, and chunks are in-memory only.

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

No test suite or linter configured yet. Dev deps group in `pyproject.toml` is empty.

## Architecture

Request flow: **FastAPI endpoint → PDF extraction → chunking → embedding → FAISS index → retrieval → LLM generation → response**

Four modules in `app/`:

- **`main.py`** — FastAPI app with three endpoints (`/health`, `/upload`, `/query`). Loads `.env` at import time before other app modules. Owns the system prompt and user-prompt formatting. Orchestrates the full pipeline.
- **`pdf_loader.py`** — Extracts text per page from PDF bytes via `pypdf`. Returns `List[Tuple[str, page_number]]`. Writes to `/tmp` for pypdf compatibility.
- **`rag.py`** — Chunking (500-char window, 50-char overlap, per-page) and embedding (`all-MiniLM-L6-v2` via sentence-transformers, 384-dim, normalized). The `Chunk` dataclass lives here. Embedding model is lazy-loaded as a module-level singleton.
- **`store.py`** — FAISS `IndexFlatIP` wrapper. Global singleton `store`. Inner product = cosine similarity because embeddings are normalized. `reset()` clears and replaces the index (single-doc design).
- **`llm.py`** — Abstract `LLMClient` with three implementations: `GroqClient`, `OpenAIClient`, `AnthropicClient`. Factory `get_llm_client()` reads `LLM_PROVIDER` env var. All use `temperature=0.2, max_tokens=800`.

## Key Design Decisions

- **LLM provider is swappable** via `LLM_PROVIDER` env var (`groq`|`openai`|`anthropic`). Only the chosen provider's API key is needed.
- **`load_dotenv()` runs in `main.py` before other app imports** — module-level env reads in `llm.py` depend on this ordering.
- **Embeddings are local** (no API call) — `all-MiniLM-L6-v2` runs on CPU. First call downloads ~80MB.
- **No streaming** — v1 intentional simplification.
- **No persistence** — FAISS index and chunks are in-memory only.

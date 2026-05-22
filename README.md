# Chat with PDF (RAG Lite)

A minimal Retrieval-Augmented Generation (RAG) service. Upload a PDF, ask
questions, get answers grounded in the document with cited source chunks.

> Status: Day 1 — core pipeline working. Day 2 will add polish, error handling,
> and a stretch evaluation set.

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Type-safe, async, auto Swagger UI |
| PDF | pypdf | Pure Python, no system deps |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, local, 384-dim, CPU-friendly |
| Vector store | FAISS (`IndexFlatIP`) | In-memory exact search, no external DB |
| LLM | Groq / OpenAI / Anthropic | Swappable via `LLM_PROVIDER` env var |

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync                            # creates .venv and installs deps from pyproject.toml
cp .env.example .env               # then edit .env and set your API key
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs and:
1. Use **POST /upload** to send a PDF
2. Use **POST /query** with `{"question": "...", "top_k": 3}`

> Adding a dependency later: `uv add <package>`. Running any script: `uv run <cmd>`.

## API

- `GET /health` — sanity check + how many chunks are indexed
- `POST /upload` — multipart `file` (PDF). Replaces any previous index.
- `POST /query` — `{question, top_k}` → `{answer, sources[]}`

## Design notes (the interesting bits)

**Chunking.** 500 characters with 50-char overlap, chunked within each page
so every chunk has an unambiguous page number. Character-based (not token-based)
for v1 simplicity; production would use a tokenizer.

**Embeddings.** `all-MiniLM-L6-v2` is normalized so we can use FAISS
`IndexFlatIP` (inner product) as cosine similarity. Tradeoff vs OpenAI
embeddings: lower quality but zero API cost and works offline.

**Retrieval.** Top-k semantic search only. No reranker, no hybrid (BM25 +
dense), no query rewriting. Would add a reranker first if accuracy matters.

**Grounding.** System prompt instructs the model to answer only from context
and explicitly say "I couldn't find that in the document." otherwise.
Sources are returned to the user separately, not formatted into the answer —
this keeps the LLM from hallucinating citations.

**LLM swap.** Single `LLMClient.generate(system, user)` interface. Three
implementations behind `LLM_PROVIDER=groq|openai|anthropic`. No streaming in
v1 (streaming is where SDKs diverge sharply).

## Known limitations (and what I'd do next)

- Scanned PDFs aren't OCR'd
- One PDF at a time (re-uploading replaces the index)
- No persistence — index is in-memory
- No retrieval evaluation set yet *(Day 2 stretch)*
- No conversation history (each query is independent)

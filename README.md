# Chat with PDF (RAG Lite)

A minimal Retrieval-Augmented Generation (RAG) service. Upload a PDF, ask
questions, get answers grounded in the document with cited source chunks.

Read the full write-up: [I built a RAG pipeline from scratch — no LangChain, just FastAPI + FAISS](https://dev.to/santanu_mohanta_29/i-built-a-rag-pipeline-from-scratch-no-langchain-just-fastapi-faiss-28ke)

## How it works

```
PDF ─► extract text (pypdf) ─► chunk (500 char, 50 overlap) ─► embed (MiniLM-L6-v2)
                                                                        │
                                                                        ▼
question ─► embed ─► FAISS + BM25 hybrid search (RRF) ─► build prompt with chunks ─► LLM ─► answer + sources
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Type-safe, async, auto Swagger UI |
| PDF | pypdf | Pure Python, no system deps |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, local, 384-dim, CPU-friendly |
| Vector store | FAISS (`IndexFlatIP`) + BM25 (`rank-bm25`) | Hybrid retrieval with Reciprocal Rank Fusion |
| LLM | Groq / OpenAI / Anthropic | Swappable via `LLM_PROVIDER` env var |

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
uv sync                            # creates .venv and installs deps from pyproject.toml
cp .env.example .env               # then edit .env — see env vars below
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs and:
1. Use **POST /upload** to send a PDF (a sample is included at `data/sample_test_file.pdf`)
2. Use **POST /query** with `{"question": "...", "top_k": 3}`

> Adding a dependency later: `uv add <package>`. Running any script: `uv run <cmd>`.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | Yes | `groq`, `openai`, or `anthropic` |
| `GROQ_API_KEY` | If provider = groq | API key from [Groq console](https://console.groq.com) |
| `OPENAI_API_KEY` | If provider = openai | API key from OpenAI |
| `ANTHROPIC_API_KEY` | If provider = anthropic | API key from Anthropic |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |

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

**Retrieval.** Hybrid search combining FAISS (dense/semantic) and BM25
(sparse/keyword), fused with Reciprocal Rank Fusion (RRF, k=60). This
catches exact keyword matches (e.g. "CEO") that pure vector search misses.

**Grounding.** System prompt instructs the model to answer only from context
and explicitly say "I couldn't find that in the document." otherwise.
Sources are returned to the user separately, not formatted into the answer —
this keeps the LLM from hallucinating citations.

**LLM swap.** Single `LLMClient.generate(system, user)` interface. Three
implementations behind `LLM_PROVIDER=groq|openai|anthropic`. No streaming in
v1 (streaming is where SDKs diverge sharply).

## Example

**Request:**
```json
POST /query
{"question": "What is the list price of the Magpie-7?", "top_k": 3}
```

**Response:**
```json
{
  "answer": "The list price of the Magpie-7 is €68,400 per unit.",
  "sources": [
    {"chunk_id": 4, "text": "...", "page": 2, "score": 0.7542}
  ]
}
```

## Sample Q&A results

Tested against the included `data/sample_test_file.pdf` — a fictional 5-page company/product document. Results are grouped by difficulty.

### Single-hop (direct lookup)

| Question | Expected answer | Result |
|---|---|---|
| What is the list price of the Magpie-7? | €68,400 per unit (p2) | Correct |
| What is the IP rating of the Magpie-7? | IP54 (p2) | Correct |
| When was the Magpie-7 released? | 14 September 2025 (p2, p5) | Correct |
| What's the response SLA on the Enterprise tier? | 1 hour, 24/7 (p3) | Correct |
| Who is the CEO of Zentara Robotics? | Iris Kallas (p1) | Correct (fixed in v2 with hybrid retrieval) |

### Specific numbers (hallucination check)

| Question | Expected answer | Result |
|---|---|---|
| How long does it take to charge the Magpie-7 from 0 to 80%? | 42 minutes (p2) | Correct |
| What's the maximum payload per arm? | 22 kg / 40 kg total (p2) | Correct |
| What's the MTBF of the Magpie-7? | 4,200 hours (p5) | Correct |
| How many employees does Zentara have? | 287 as of 1 March 2026 (p1) | Correct with top_k=5 (fixed in v2 with hybrid retrieval) |

### Tables

| Question | Expected answer | Result |
|---|---|---|
| What's included in the Standard tier? | €2,150/month, 4-hour SLA, 750k picks/month (p3) | Correct |
| List all company history milestones from 2020 onwards | 2020 Series A, 2022 Voorhuis deployment, 2024 Series B + Kraków, 2025 Magpie-7 launch + 1M picks, 2026 Munich office (p5) | Correct |

### Multi-hop / synthesis

| Question | Expected answer | Result |
|---|---|---|
| If I want 1-hour SLA support, what will it cost per month? | €4,800/month — Enterprise tier (p3) | Correct |
| Which languages will the voice interface support after the June 2026 update? | 7 total: EN, DE, ET, PL, JA + ES and FR (p4) | Correct |
| What happens if I want to cancel my subscription early? | 60 days' notice; early termination fee = 2 months of contracted rate (p3) | Correct |

### Negative tests (should say "not in document")

| Question | Expected answer | Result |
|---|---|---|
| Who is Zentara's Chief Financial Officer? | Not stated — only CEO, CTO, and Head of People are named | Correct |
| Does the Magpie-7 support Mandarin? | Not supported and not on the roadmap | Correct |
| What's Zentara's stock ticker? | Not in the document (company appears private) | Correct |

### Tricky retrieval (info spread across pages)

| Question | Expected answer | Result |
|---|---|---|
| What forms or certifications do operators need? | Zentara Safety Certification (4-hr course), Form ZR-INSP-22, ISO 10218-2, ISO/TS 15066 (p4) | Correct |
| Compare the Starter and Enterprise tiers | Starter: €1,200/mo, next-biz-day SLA, 200k picks. Enterprise: €4,800/mo, 1-hr 24/7 SLA, unlimited picks (p3) | Correct |

### v1 → v2: hybrid retrieval fixed the failures

In v1 (pure FAISS), "Who is the CEO?" and "How many employees?" both failed because page 1's dense "Company snapshot" table produced muddy embeddings. In v2, BM25 catches the exact keyword matches and RRF fuses them with FAISS results. The CEO question now passes at default `top_k=3`. The employee count question passes at `top_k=5` — the chunk still ranks lower due to fact density, but hybrid retrieval brings it within reach.

## Known limitations

- Scanned PDFs aren't OCR'd
- One PDF at a time (re-uploading replaces the index)
- No persistence — index is in-memory
- No conversation history (each query is independent)
- Dense fact-packed chunks can still rank low even with hybrid retrieval (may need higher top_k)

## What I'd add next

- ~~Hybrid retrieval (BM25 + vector) to catch keyword matches~~ ✓ Added in v2
- Reranker (e.g. cross-encoder) for better precision
- Retrieval evaluation set to measure accuracy
- Streaming responses
- Conversation memory

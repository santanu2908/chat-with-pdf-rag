# RAG Pipeline - Complete Visual Workflow

## Example: A 10-page PDF about "Company Policies"

---

## PHASE 1: UPLOAD (`POST /upload`)

### Step 1: PDF Upload & Text Extraction (`pdf_loader.py`)

```
                        +------------------+
   User uploads         |   FastAPI         |
   company.pdf  ------> |   /upload         |
   (10 pages)           +--------+---------+
                                 |
                                 v
                        +------------------+
                        |   pypdf           |
                        |   PdfReader       |
                        +--------+---------+
                                 |
                                 v
              Extracts text from each page (1-indexed)
              Skips empty pages (e.g., page 5 is a scan with no text)

   Output: List of (text, page_number) tuples
   +---------------------------------------------------------+
   | Page 1: "Welcome to Company Policies. This document..." |  ~2000 chars
   | Page 2: "Section 1: Leave Policy. Employees are..."     |  ~1800 chars
   | Page 3: "...entitled to 20 days annual leave..."         |  ~1500 chars
   | Page 4: "Section 2: Remote Work. Employees may..."      |  ~2200 chars
   | Page 5: (skipped - scanned image, no text)              |
   | Page 6: "Section 3: Travel Policy. All business..."     |  ~1900 chars
   | Page 7: "...receipts must be submitted within 30..."     |  ~1700 chars
   | Page 8: "Section 4: Code of Conduct. Employees..."      |  ~2100 chars
   | Page 9: "...disciplinary actions may include..."         |  ~1400 chars
   | Page 10: "Appendix A: Contact HR at hr@company.com..."  |  ~800 chars
   +---------------------------------------------------------+
   Result: 9 pages with text (page 5 dropped)
```

---

### Step 2: Chunking (`rag.py → chunk_pages()`)

```
   Settings: CHUNK_SIZE = 500 chars, CHUNK_OVERLAP = 50 chars

   Each page is chunked INDEPENDENTLY (so page number stays unambiguous)

   Example: Page 1 has ~2000 chars
   +================================================================+
   |                        Page 1 text (2000 chars)                 |
   +================================================================+

   Chunk 0: chars [0 ──────── 500]                     page=1
   Chunk 1:            chars [450 ──────── 950]         page=1   ← 50-char overlap
   Chunk 2:                       chars [900 ──────── 1400]      page=1
   Chunk 3:                                  chars [1350 ─── 1850]  page=1
   Chunk 4:                                             chars [1800 ── 2000]  page=1

                  ◄─50─►
                  overlap

   Page 1 → 5 chunks (IDs 0-4)
   Page 2 → 4 chunks (IDs 5-8)
   Page 3 → 3 chunks (IDs 9-11)
   Page 4 → 5 chunks (IDs 12-16)
   Page 6 → 4 chunks (IDs 17-20)
   Page 7 → 4 chunks (IDs 21-24)
   Page 8 → 5 chunks (IDs 25-29)
   Page 9 → 3 chunks (IDs 30-32)
   Page 10 → 2 chunks (IDs 33-34)

   TOTAL: ~35 chunks
```

**Why overlap?** So context isn't lost at boundaries. If a sentence is split across two chunks, the overlap ensures the full sentence appears in at least one chunk.

```
   Each Chunk is a dataclass:
   +-----------------------------------+
   | Chunk                             |
   |   chunk_id: 7                     |
   |   text: "Employees are entitled   |
   |          to 20 days of annual..." |
   |   page: 2                         |
   +-----------------------------------+
```

---

### Step 3: Embedding (`rag.py → embed_texts()`)

```
   Model: all-MiniLM-L6-v2 (runs locally on CPU, 384 dimensions)

   Each chunk's text → 384-dimensional vector (normalized to unit length)

   Chunk 0: "Welcome to Company Policies..."  →  [0.023, -0.041, 0.089, ..., 0.015]  (384 floats)
   Chunk 1: "...document outlines the key..."  →  [0.051, -0.018, 0.033, ..., -0.027] (384 floats)
   Chunk 2: "...regulations that govern..."    →  [-0.012, 0.067, 0.044, ..., 0.039]  (384 floats)
     ...
   Chunk 34: "Contact HR at hr@company..."     →  [0.038, -0.055, 0.021, ..., 0.011]  (384 floats)

   Output shape: numpy array (35, 384) — 35 chunks x 384 dimensions

   +--------- vectors (35 x 384) ---------+
   | [ 0.023, -0.041, 0.089, ..., 0.015]  |  ← chunk 0
   | [ 0.051, -0.018, 0.033, ..., -0.027] |  ← chunk 1
   | [-0.012,  0.067, 0.044, ...,  0.039] |  ← chunk 2
   |           ...                         |
   | [ 0.038, -0.055, 0.021, ...,  0.011] |  ← chunk 34
   +-----------------------------------------+

   Vectors are NORMALIZED → ||v|| = 1.0
   This means: inner product = cosine similarity (no extra math needed)
```

---

### Step 4: Indexing in FAISS (`store.py → VectorStore`)

```
   store.reset()   ← clears any previous PDF's data
   store.add(vectors, chunks)

   +==============================================================+
   |                    VectorStore (in-memory)                    |
   |                                                               |
   |  FAISS IndexFlatIP (384-dim)          chunks[] metadata       |
   |  +----------------------------+      +--------------------+  |
   |  | index 0: [0.023, -0.041..] | ---> | Chunk(id=0, pg=1)  |  |
   |  | index 1: [0.051, -0.018..] | ---> | Chunk(id=1, pg=1)  |  |
   |  | index 2: [-0.012, 0.067..] | ---> | Chunk(id=2, pg=1)  |  |
   |  | index 3: [0.044, 0.012..]  | ---> | Chunk(id=3, pg=1)  |  |
   |  | index 4: [0.091, -0.033..] | ---> | Chunk(id=4, pg=1)  |  |
   |  | index 5: [-0.008, 0.055..] | ---> | Chunk(id=5, pg=2)  |  |
   |  |          ...                |      |       ...           |  |
   |  | index 34: [0.038, -0.055..]| ---> | Chunk(id=34, pg=10)|  |
   |  +----------------------------+      +--------------------+  |
   |                                                               |
   |  IndexFlatIP = brute-force exact search using inner product   |
   |  Good for up to ~100k vectors. No approximation.             |
   +==============================================================+

   The FAISS index and chunks[] list are parallel arrays:
   - FAISS index position 0 ↔ chunks[0]
   - FAISS index position 1 ↔ chunks[1]
   - etc.
```

---

## PHASE 2: QUERY (`POST /query`)

### Example Query: "How many days of annual leave do employees get?"

```
   +-------------------------------------------------------+
   |  POST /query                                           |
   |  {                                                     |
   |    "question": "How many days of annual leave?",       |
   |    "top_k": 3                                          |
   |  }                                                     |
   +-------------------------------------------------------+
```

### Step 5: Embed the Question (same model)

```
   "How many days of annual leave?" 
       │
       ▼
   all-MiniLM-L6-v2
       │
       ▼
   query_vec = [0.032, -0.028, 0.071, ..., 0.019]  (384 floats, normalized)
```

### Step 6: Vector Search (`store.search()`)

```
   FAISS computes cosine similarity between query_vec and ALL 35 stored vectors

   query_vec ●──────────────────────────────────────────────────┐
             │                                                   │
             │  cosine_sim(query, chunk_0)  = 0.312              │
             │  cosine_sim(query, chunk_1)  = 0.287              │
             │  cosine_sim(query, chunk_2)  = 0.198              │
             │  cosine_sim(query, chunk_3)  = 0.245              │
             │  cosine_sim(query, chunk_4)  = 0.156              │
             │  cosine_sim(query, chunk_5)  = 0.534   ★ leave policy content
             │  cosine_sim(query, chunk_6)  = 0.421              │
             │  cosine_sim(query, chunk_7)  = 0.189              │
             │  cosine_sim(query, chunk_8)  = 0.267              │
             │  cosine_sim(query, chunk_9)  = 0.612   ★ "entitled to 20 days"
             │  cosine_sim(query, chunk_10) = 0.478   ★ annual leave details
             │  cosine_sim(query, chunk_11) = 0.145              │
             │          ...                                      │
             │  cosine_sim(query, chunk_34) = 0.089              │
             └───────────────────────────────────────────────────┘

   Sort by score, return top_k=3:
   +------+----------+------+-------+------------------------------------------+
   | Rank | Chunk ID | Page | Score | Text (preview)                           |
   +------+----------+------+-------+------------------------------------------+
   |  1   |    9     |  3   | 0.612 | "...entitled to 20 days annual leave..." |
   |  2   |    5     |  2   | 0.534 | "Section 1: Leave Policy. Employees..."  |
   |  3   |    10    |  3   | 0.478 | "...carry forward up to 5 days..."       |
   +------+----------+------+-------+------------------------------------------+
```

### Step 7: Build Prompt & Call LLM (`main.py`)

```
   build_user_prompt() assembles:

   ┌─────────────────── SYSTEM PROMPT ───────────────────┐
   │ You are a careful assistant that answers questions   │
   │ strictly from the provided document context.        │
   │ Rules:                                              │
   │ - Use ONLY the context below...                     │
   │ - If not in context, say "I couldn't find that..."  │
   └─────────────────────────────────────────────────────┘

   ┌─────────────────── USER PROMPT ────────────────────┐
   │ Context:                                            │
   │                                                     │
   │ [Chunk 9 | Page 3]                                  │
   │ ...entitled to 20 days annual leave per year.       │
   │ New employees receive prorated leave...             │
   │                                                     │
   │ ---                                                 │
   │                                                     │
   │ [Chunk 5 | Page 2]                                  │
   │ Section 1: Leave Policy. Employees are eligible     │
   │ for various types of leave including...             │
   │                                                     │
   │ ---                                                 │
   │                                                     │
   │ [Chunk 10 | Page 3]                                 │
   │ ...carry forward up to 5 days of unused leave       │
   │ to the next calendar year...                        │
   │                                                     │
   │ Question: How many days of annual leave?            │
   └─────────────────────────────────────────────────────┘

                         │
                         ▼
              ┌─────────────────────┐
              │   LLM (e.g. Groq    │
              │   llama-3.3-70b)    │
              │   temp=0.2          │
              │   max_tokens=800    │
              └──────────┬──────────┘
                         │
                         ▼

   "Employees are entitled to 20 days of annual leave per year.
    New employees receive prorated leave based on their start date."
```

### Step 8: Return Response

```json
{
  "answer": "Employees are entitled to 20 days of annual leave per year. New employees receive prorated leave based on their start date.",
  "sources": [
    {
      "chunk_id": 9,
      "text": "...entitled to 20 days annual leave per year. New employees receive prorated leave...",
      "page": 3,
      "score": 0.612
    },
    {
      "chunk_id": 5,
      "text": "Section 1: Leave Policy. Employees are eligible for various types of leave including...",
      "page": 2,
      "score": 0.534
    },
    {
      "chunk_id": 10,
      "text": "...carry forward up to 5 days of unused leave to the next calendar year...",
      "page": 3,
      "score": 0.478
    }
  ]
}
```

---

## END-TO-END SUMMARY

```
 UPLOAD PHASE                                    QUERY PHASE
 ============                                    ===========

 PDF (10 pages)                                  User Question
      │                                               │
      ▼                                               ▼
 ┌──────────┐                                   ┌──────────┐
 │  pypdf    │  extract text per page            │ embed    │  same model
 └────┬─────┘                                   └────┬─────┘
      │ 9 pages with text                             │ query vector (384-dim)
      ▼                                               ▼
 ┌──────────┐                                   ┌──────────────┐
 │  chunk   │  500 chars, 50 overlap             │ FAISS search │  cosine similarity
 └────┬─────┘                                   └──────┬───────┘
      │ ~35 chunks                                     │ top-K chunks
      ▼                                               ▼
 ┌──────────┐                                   ┌──────────────┐
 │  embed   │  all-MiniLM-L6-v2                  │ build prompt │  chunks + question
 └────┬─────┘                                   └──────┬───────┘
      │ 35 vectors (384-dim each)                      │
      ▼                                               ▼
 ┌──────────┐                                   ┌──────────────┐
 │  FAISS   │  IndexFlatIP (in-memory)           │  LLM call    │  Groq/OpenAI/Anthropic
 │  store   │                                    └──────┬───────┘
 └──────────┘                                          │
                                                       ▼
                                                  JSON response
                                                  (answer + sources)
```

---

## Key Numbers at a Glance

| Parameter             | Value                          |
|----------------------|--------------------------------|
| Chunk size           | 500 characters                 |
| Chunk overlap        | 50 characters                  |
| Embedding model      | all-MiniLM-L6-v2 (local, CPU) |
| Vector dimensions    | 384                            |
| FAISS index type     | IndexFlatIP (exact, brute-force) |
| Similarity metric    | Cosine (via normalized inner product) |
| Default top_k        | 3 (configurable 1-10)          |
| LLM temperature      | 0.2 (low, for grounded answers) |
| LLM max tokens       | 800                            |

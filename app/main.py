"""FastAPI app exposing /upload and /query.

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs for the Swagger UI.
"""
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field


load_dotenv()  # must happen before importing modules that read env

from app.pdf_loader import load_pdf
from app.rag import chunk_pages, embed_texts
from app.store import store
from app.llm import get_llm_client

app = FastAPI(
    title="Chat with PDF (RAG Lite)",
    description="Upload a PDF, ask grounded questions, get cited answers.",
    version="0.1.0",
)


# ---------- Schemas ----------

class UploadResponse(BaseModel):
    chunks_indexed: int
    pages_processed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: int
    text: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# ---------- Prompt ----------

SYSTEM_PROMPT = """You are a careful assistant that answers questions strictly from the provided document context.

Rules:
- Use ONLY the context below. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Be concise. 1-3 sentences unless the question requires more.
- Do not invent page numbers or citations — the user will see source chunks separately."""


def build_user_prompt(question: str, retrieved_chunks: list) -> str:
    """Format the retrieved chunks + question into a single user message."""
    context_blocks = []
    for r in retrieved_chunks:
        context_blocks.append(
            f"[Chunk {r.chunk.chunk_id} | Page {r.chunk.page}]\n{r.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


# ---------- Endpoints ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks_indexed": store.size()}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a .pdf")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    pages = load_pdf(file_bytes)
    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No extractable text. Is this a scanned PDF? OCR is not supported in v1.",
        )

    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced")

    vectors = embed_texts([c.text for c in chunks])

    # Replace any previous index — v1 is single-doc.
    store.reset()
    store.add(vectors, chunks)

    return UploadResponse(chunks_indexed=len(chunks), pages_processed=len(pages))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if store.size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document indexed. POST a PDF to /upload first.",
        )

    query_vec = embed_texts([req.question])[0]
    retrieved = store.search(query_vec, top_k=req.top_k, query_text=req.question)

    if not retrieved:
        return QueryResponse(
            answer="I couldn't find that in the document.",
            sources=[],
        )

    llm = get_llm_client()
    user_prompt = build_user_prompt(req.question, retrieved)
    answer = llm.generate(system=SYSTEM_PROMPT, user=user_prompt).strip()

    sources = [
        Source(
            chunk_id=r.chunk.chunk_id,
            text=r.chunk.text,
            page=r.chunk.page,
            score=round(r.score, 4),
        )
        for r in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources)

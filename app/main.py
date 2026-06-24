"""FastAPI app exposing /upload, /query, and /query/stream.

Run locally:
    uvicorn app.main:app --reload
Then open http://localhost:8000/docs for the Swagger UI.
"""
import json
import uuid
from typing import Dict, Iterator, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


load_dotenv()  # must happen before importing modules that read env

from app.pdf_loader import load_pdf
from app.rag import chunk_pages, embed_texts
from app.store import store
from app.llm import get_llm_client
from app.reranker import rerank

app = FastAPI(
    title="Chat with PDF (RAG Lite)",
    description="Upload a PDF, ask grounded questions, get cited answers.",
    version="0.1.0",
)


# ---------- Conversation memory ----------

MAX_HISTORY_TURNS = 5  # keep last N exchanges (user + assistant pairs)

# session_id → list of {"role": "user"|"assistant", "content": "..."} messages
_sessions: Dict[str, List[Dict[str, str]]] = {}


def _get_history(session_id: Optional[str]) -> List[Dict[str, str]]:
    """Return conversation history for a session (empty list if no session)."""
    if session_id and session_id in _sessions:
        return list(_sessions[session_id])
    return []


def _save_turn(session_id: Optional[str], user_msg: str, assistant_msg: str) -> str:
    """Append a turn to the session and return the session_id."""
    if not session_id:
        session_id = uuid.uuid4().hex
    
    # Append the new turn to the session history, trimming to last N turns
    # if session_id is not in _sessions, create a new empty list for it
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # Trim to last N turns (each turn = 2 messages)
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        _sessions[session_id] = history[-max_messages:]

    print(_sessions)  # Debug: print the current session history
    return session_id


# ---------- Schemas ----------

class UploadResponse(BaseModel):
    chunks_indexed: int
    pages_processed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory. Omit to start a new session.",
    )


class Source(BaseModel):
    chunk_id: int
    text: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


# ---------- Prompt ----------

SYSTEM_PROMPT = """You are a careful assistant that answers questions strictly from the provided document context.

Rules:
- Use ONLY the context below. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in the document."
- Be concise. 1-3 sentences unless the question requires more.
- Do not invent page numbers or citations — the user will see source chunks separately.
- You may receive prior conversation turns. Use them to resolve follow-up references (e.g. "it", "that", "the same tier") but still answer from the retrieved context, not from prior answers."""


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

    # Replace any previous index and clear conversation history — single-doc design.
    store.reset()
    _sessions.clear()
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
    # Over-fetch candidates for reranking, then keep top_k
    fetch_k = req.top_k * 2
    candidates = store.search(query_vec, top_k=fetch_k, query_text=req.question)
    retrieved = rerank(req.question, candidates, top_k=req.top_k)

    if not retrieved:
        sid = _save_turn(req.session_id, req.question, "I couldn't find that in the document.")
        return QueryResponse(
            answer="I couldn't find that in the document.",
            sources=[],
            session_id=sid,
        )

    llm = get_llm_client()
    user_prompt = build_user_prompt(req.question, retrieved)

    # Build messages: prior history + current user prompt
    history = _get_history(req.session_id)
    messages = history + [{"role": "user", "content": user_prompt}]

    answer = llm.generate(system=SYSTEM_PROMPT, messages=messages).strip()

    sid = _save_turn(req.session_id, req.question, answer)

    sources = [
        Source(
            chunk_id=r.chunk.chunk_id,
            text=r.chunk.text,
            page=r.chunk.page,
            score=round(r.score, 4),
        )
        for r in retrieved
    ]

    return QueryResponse(answer=answer, sources=sources, session_id=sid)


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """Streaming version of /query. Returns SSE events:
    - data: {"token": "..."} for each text chunk
    - data: {"sources": [...], "session_id": "..."} as the final event
    """
    if store.size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No document indexed. POST a PDF to /upload first.",
        )

    query_vec = embed_texts([req.question])[0]
    fetch_k = req.top_k * 2
    candidates = store.search(query_vec, top_k=fetch_k, query_text=req.question)
    retrieved = rerank(req.question, candidates, top_k=req.top_k)

    if not retrieved:
        sid = _save_turn(req.session_id, req.question, "I couldn't find that in the document.")
        def empty_stream() -> Iterator[str]:
            yield f"data: {json.dumps({'token': 'I couldn\'t find that in the document.'})}\n\n"
            yield f"data: {json.dumps({'sources': [], 'session_id': sid})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    sources = [
        {
            "chunk_id": r.chunk.chunk_id,
            "text": r.chunk.text,
            "page": r.chunk.page,
            "score": round(r.score, 4),
        }
        for r in retrieved
    ]

    llm = get_llm_client()
    user_prompt = build_user_prompt(req.question, retrieved)

    # Build messages: prior history + current user prompt
    history = _get_history(req.session_id)
    messages = history + [{"role": "user", "content": user_prompt}]

    def event_stream() -> Iterator[str]:
        tokens: list[str] = []
        for token in llm.stream(system=SYSTEM_PROMPT, messages=messages):
            tokens.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
        answer = "".join(tokens)
        sid = _save_turn(req.session_id, req.question, answer)
        yield f"data: {json.dumps({'sources': sources, 'session_id': sid})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

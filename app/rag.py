"""Chunking + embedding.

Chunking strategy: ~500 chars per chunk with 50-char overlap, split on
sentence-ish boundaries. We use character counts (not tokens) for v1 because:
- simpler, no tokenizer dependency
- sentence-transformers has a 256-token limit anyway; ~500 chars ≈ 100-150 tokens
- in interviews you should be able to say: "I'd switch to token-aware chunking
    with tiktoken or the model's own tokenizer for production"

Embedding model: all-MiniLM-L6-v2 — 384-dim, fast, runs on CPU, free.
Tradeoff vs OpenAI text-embedding-3-small: lower quality but no API cost
and works offline. Mention this in the README.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class Chunk:
    chunk_id: int
    text: str
    page: int


_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model. First call downloads ~80MB."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def chunk_pages(pages: List[Tuple[str, int]]) -> List[Chunk]:
    """Split each page's text into overlapping chunks.

    We chunk within a page so every chunk has an unambiguous page number.
    For docs with content flowing across pages this loses some context;
    acceptable for v1.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    for text, page_num in pages:
        # Slide a window of CHUNK_SIZE across the page text.
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, page=page_num))
                chunk_id += 1
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP

    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, EMBED_DIM) float32 array.

    Normalized so we can use inner-product as cosine similarity in FAISS.
    """
    model = get_embed_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype("float32")

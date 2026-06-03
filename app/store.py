"""In-memory FAISS + BM25 hybrid store with chunk metadata.

Retrieval uses Reciprocal Rank Fusion (RRF) to combine dense (FAISS)
and sparse (BM25) search results. This catches both semantic matches
and exact keyword matches that pure vector search misses.

V1 holds one PDF's vectors at a time. Re-uploading replaces the index.
"""
import re
from dataclasses import dataclass
from typing import List
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.rag import Chunk, EMBED_DIM

RRF_K = 60  # standard constant from Cormack et al.


def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Retrieval:
    chunk: Chunk
    score: float  # RRF score (higher = more relevant)


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index = faiss.IndexFlatIP(EMBED_DIM)
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.chunks = []
        self.bm25 = None

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("vectors and chunks length mismatch")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        # Rebuild BM25 index from all chunks
        # Precomputes DF/IDF, doc lengths, and term statistics for fast query-time scoring
        tokenized = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self, query_vector: np.ndarray, top_k: int = 3, query_text: str = ""
    ) -> List[Retrieval]:
        if self.index.ntotal == 0:
            return []

        # Fetch more candidates than needed so RRF has enough to work with
        top_k_fetch = min(top_k * 3, self.index.ntotal)

        # --- FAISS (dense) search ---
        q = query_vector.reshape(1, -1).astype("float32")
        _, faiss_indices = self.index.search(q, top_k_fetch)
        faiss_ranking = [
            int(idx) for idx in faiss_indices[0] if idx != -1
        ]

        # --- BM25 (sparse) search ---
        bm25_ranking: List[int] = []
        if self.bm25 and query_text:
            bm25_scores = self.bm25.get_scores(_tokenize(query_text))
            bm25_ranking = np.argsort(bm25_scores)[::-1][:top_k_fetch].tolist()

        # --- Reciprocal Rank Fusion ---
        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(faiss_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        # Sort by fused score, take top_k
        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        return [
            Retrieval(chunk=self.chunks[idx], score=round(rrf_scores[idx], 4))
            for idx in sorted_indices
        ]

    def size(self) -> int:
        return self.index.ntotal


# Single global store (single-user, single-doc demo).
store = VectorStore()

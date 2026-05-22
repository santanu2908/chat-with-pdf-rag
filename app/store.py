"""In-memory FAISS vector store with chunk metadata.

V1 holds one PDF's vectors at a time. Re-uploading replaces the index.
For multi-doc support you'd add a doc_id field on each chunk and either
namespace indices per doc or filter on retrieval.
"""
from dataclasses import dataclass
from typing import List
import faiss
import numpy as np

from app.rag import Chunk, EMBED_DIM


@dataclass
class Retrieval:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1], higher = more similar


class VectorStore:
    def __init__(self) -> None:
        # IndexFlatIP = exact inner-product search. Since we normalize embeddings,
        # inner product == cosine similarity. Brute force, fine up to ~100k vectors.
        self.index: faiss.Index = faiss.IndexFlatIP(EMBED_DIM)
        self.chunks: List[Chunk] = []

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks: List[Chunk]) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("vectors and chunks length mismatch")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Retrieval]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        # query_vector arrives as (EMBED_DIM,); FAISS wants (1, EMBED_DIM)
        q = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(q, top_k)
        results: List[Retrieval] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(Retrieval(chunk=self.chunks[idx], score=float(score)))
        return results

    def size(self) -> int:
        return self.index.ntotal


# Single global store for v1 (single-user, single-doc demo).
# For multi-user you'd key this by session_id or user_id.
store = VectorStore()

"""Cross-encoder reranker for second-stage retrieval.

After hybrid search (FAISS + BM25) returns broad candidates, the cross-encoder
scores each (query, passage) pair jointly for higher precision.

Strategy: the top first-stage results are guaranteed (they already passed
keyword + vector matching), and the cross-encoder fills the remaining slots
from the broader candidate pool. This prevents the CE from burying chunks
that matched well on keywords but look unlike natural prose (e.g. dense
fact-table rows).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — ~80MB, CPU-friendly, trained
on MS MARCO passage ranking.
"""
import math
from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.store import Retrieval

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder. First call downloads ~80MB."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, retrievals: List[Retrieval], top_k: int) -> List[Retrieval]:
    """Rerank with guaranteed slots for top first-stage results.

    - Top ceil(top_k / 2) from first-stage are kept (guaranteed slots).
    - The cross-encoder picks the remaining slots from all other candidates.
    """
    if not retrievals:
        return []
    if top_k >= len(retrievals):
        return retrievals

    n_guaranteed = top_k - 1
    n_ce_slots = 1

    guaranteed = retrievals[:n_guaranteed]
    remaining = retrievals[n_guaranteed:]

    if n_ce_slots > 0 and remaining:
        model = get_reranker()
        pairs = [[query, r.chunk.text] for r in remaining]
        scores = model.predict(pairs)
        for r, score in zip(remaining, scores):
            r.score = round(float(score), 4)
        remaining.sort(key=lambda r: r.score, reverse=True)

    return guaranteed + remaining[:n_ce_slots]

"""
Retrieval configurations: BM25, dense (BGE-small), hybrid RRF.
Paper: How Chunking, Embedding, and Reranking Interact in Long-Document RAG
Author: Praveen Kumar Varkala
"""

from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Type alias
RankedList = List[Tuple[int, float]]   # [(chunk_idx, score), ...]


def build_bm25(chunks: List[str]) -> BM25Okapi:
    tokenized = [c.split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25: BM25Okapi, top_k: int = 50) -> RankedList:
    scores = bm25.get_scores(query.split())
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]


def encode_chunks(chunks: List[str], model: SentenceTransformer,
                  batch_size: int = 64) -> np.ndarray:
    return model.encode(
        chunks, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=False
    )


def dense_retrieve(query: str, chunk_embeddings: np.ndarray,
                   model: SentenceTransformer, top_k: int = 50) -> RankedList:
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = chunk_embeddings @ q_emb
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]


def rrf_fuse(ranked_lists: List[RankedList], k: int = 60) -> RankedList:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank(query: str, chunks: List[str], candidates: RankedList,
           cross_encoder: CrossEncoder, top_k: int = 10) -> RankedList:
    """Cross-encoder reranking over top-50 candidates."""
    pool = candidates[:50]
    pairs = [(query, chunks[i]) for i, _ in pool]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(
        zip([i for i, _ in pool], ce_scores),
        key=lambda x: x[1], reverse=True
    )
    return reranked[:top_k]

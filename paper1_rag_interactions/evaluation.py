"""
Evaluation metrics: Recall@k, MRR@k.
Paper: How Chunking, Embedding, and Reranking Interact in Long-Document RAG
Author: Praveen Kumar Varkala
"""

from typing import List, Tuple, Set

RankedList = List[Tuple[int, float]]


def recall_at_k(retrieved: RankedList, gold_ids: Set[int], k: int) -> float:
    """1.0 if any gold chunk appears in top-k, else 0.0."""
    top_k = {i for i, _ in retrieved[:k]}
    return 1.0 if top_k & gold_ids else 0.0


def mrr_at_k(retrieved: RankedList, gold_ids: Set[int], k: int) -> float:
    """Mean Reciprocal Rank at cutoff k."""
    for rank, (i, _) in enumerate(retrieved[:k], 1):
        if i in gold_ids:
            return 1.0 / rank
    return 0.0


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(results: List[dict]) -> None:
    """Pretty-print a results table."""
    header = f"{'Config':<52} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'N':>5}"
    print(header)
    print('-' * len(header))
    for r in results:
        label = f"{r['dataset']} | {r['config']}"
        print(f"{label:<52} {r['recall@1']:>6.3f} {r['recall@5']:>6.3f} "
              f"{r['recall@10']:>6.3f} {r['mrr@10']:>6.3f} {r['n_questions']:>5}")

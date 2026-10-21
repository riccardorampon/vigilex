"""Binary-relevance IR metrics over one ranked list of article ids."""

import math
from statistics import fmean

METRICS = ("recall@5", "recall@10", "recall@20", "ndcg@10", "mrr")


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    return next((1 / i for i, a in enumerate(ranked, start=1) if a in relevant), 0.0)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, a in enumerate(ranked[:k], start=1) if a in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def query_metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(ranked, relevant, 5),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "recall@20": recall_at_k(ranked, relevant, 20),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
        "mrr": reciprocal_rank(ranked, relevant),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {m: fmean(r[m] for r in rows) for m in METRICS}

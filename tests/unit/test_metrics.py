import math

import pytest
from vigilex_rag.metrics import (
    METRICS,
    mean_metrics,
    ndcg_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 2) == pytest.approx(1 / 3)
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == pytest.approx(2 / 3)
    assert recall_at_k([], {1}, 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 6, 7], {7}) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 6, 7], {1}) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 1 and 3, two relevant docs in total
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([1, 9, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([1, 2], {1, 2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([9, 8], {1}, 10) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # 5 relevant docs, k=2, both top-2 relevant: perfect score
    assert ndcg_at_k([1, 2, 9], {1, 2, 3, 4, 5}, 2) == pytest.approx(1.0)


def test_query_and_mean_metrics():
    row = query_metrics([1, 2, 3], {3})
    assert set(row) == set(METRICS)
    assert row["mrr"] == pytest.approx(1 / 3)
    good = {"mrr": 1.0, "recall@5": 1.0, "recall@10": 1.0, "recall@20": 1.0, "ndcg@10": 1.0}
    bad = {"mrr": 0.0, "recall@5": 0.0, "recall@10": 0.5, "recall@20": 0.5, "ndcg@10": 0.0}
    mean = mean_metrics([{**good, "retrieved": [1]}, {**bad, "retrieved": [2]}])
    assert mean["mrr"] == pytest.approx(0.5) and mean["recall@10"] == pytest.approx(0.75)
    assert set(mean) == set(METRICS)

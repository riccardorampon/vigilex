from vigilex_rag.retrieval import RetrieverConfig, rrf


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == 1 / 61 + 1 / 62
    assert scores[3] == 1 / 63 + 1 / 61
    assert scores[2] == 1 / 62
    assert [a for a, _ in fused] == [1, 3, 2]


def test_rrf_breaks_ties_by_id_and_handles_empty():
    assert [a for a, _ in rrf([[5], [4]], k=60)] == [4, 5]
    assert rrf([], k=60) == []


def test_config_name_and_hash():
    a = RetrieverConfig()
    assert a.name() == "hybrid"
    assert RetrieverConfig(rerank=True).name() == "hybrid+rerank"
    assert a.config_hash() == RetrieverConfig().config_hash()
    assert a.config_hash() != RetrieverConfig(top_k=20).config_hash()
    assert len(a.config_hash()) == 16

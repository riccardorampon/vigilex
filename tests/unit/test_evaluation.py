import pytest
from vigilex_rag.evaluation import EvalRun, parse_configs, report
from vigilex_rag.retrieval import RetrieverConfig


def test_parse_configs():
    cfgs = parse_configs("bm25,hybrid+rerank", "e", rerank_available=True)
    assert [c.name() for c in cfgs] == ["bm25", "hybrid+rerank"]
    assert all(c.embed_model_id == "e" for c in cfgs)


def test_parse_configs_skips_rerank_without_reranker():
    cfgs = parse_configs("hybrid,hybrid+rerank", "e", rerank_available=False)
    assert [c.name() for c in cfgs] == ["hybrid"]


def test_parse_configs_rejects_unknown():
    with pytest.raises(ValueError, match="bogus"):
        parse_configs("hybrid,bogus", "e", rerank_available=True)


def test_report_has_one_row_per_run():
    m = {"recall@5": 0.5, "recall@10": 0.6, "recall@20": 0.7, "ndcg@10": 0.4, "mrr": 0.3}
    md = report(
        [
            EvalRun(RetrieverConfig(mode="bm25"), "validation", m, {"q": {}}),
            EvalRun(RetrieverConfig(), "validation", m, {"q": {}}),
        ]
    )
    lines = md.strip().splitlines()
    assert lines[0].startswith("| config |") and len(lines) == 4
    assert "| bm25 |" in lines[2] and "0.600" in lines[2]

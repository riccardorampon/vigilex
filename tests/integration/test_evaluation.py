import pytest
from helpers import article, question
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.evaluation import evaluate, save_run
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


def small(conn, tmp_path, write_release):
    release = write_release(
        tmp_path,
        [article(1, "adeguata verifica della clientela"), article(2, "segnalazione sospetta")],
        [question("adeguata verifica", [1]), question("segnalazione", [2])],
    )
    ingest(conn, load_release(release), FakeEmbedder())


def test_eval_bm25_perfect_on_trivial_data(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    run = evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None)
    assert run.metrics["recall@10"] == 1.0 and run.metrics["mrr"] == 1.0
    assert set(run.per_query) == {"validation-0", "validation-1"}
    assert run.per_query["validation-0"]["relevant"] == [1]
    assert run.config == RetrieverConfig(mode="bm25")  # stored config is not the depth-20 one


def test_save_run_round_trips(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    run = evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None)
    run_id = save_run(conn, run, "abc123")
    row = conn.execute(
        "SELECT config_name, config_hash, split, n_questions, metrics->>'mrr', git_sha "
        "FROM eval_runs WHERE id = %s",
        (run_id,),
    ).fetchone()
    assert row == ("bm25", run.config.config_hash(), "validation", 2, "1.0", "abc123")


def test_eval_empty_split_raises(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    with pytest.raises(ValueError, match="train"):
        evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None, split="train")

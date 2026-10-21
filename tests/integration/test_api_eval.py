from helpers import article, login, question
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.evaluation import evaluate, save_run
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


def two_runs(conn, tmp_path, write_release):
    release = write_release(
        tmp_path,
        [article(1, "adeguata verifica"), article(2, "segnalazione sospetta")],
        [question("adeguata verifica", [1]), question("di il la", [2])],
    )
    ingest(conn, load_release(release), FakeEmbedder())
    live = save_run(conn, evaluate(conn, RetrieverConfig(), FakeEmbedder(), None), "sha1")
    bm25 = RetrieverConfig(mode="bm25")
    other = save_run(conn, evaluate(conn, bm25, FakeEmbedder(), None), "sha1")
    return live, other


def test_runs_scope_current_and_all(client, conn, tmp_path, write_release):
    live, other = two_runs(conn, tmp_path, write_release)
    login(client)
    current = client.get("/api/eval/runs").json()
    assert [r["id"] for r in current] == [live]
    assert {r["id"] for r in client.get("/api/eval/runs?scope=all").json()} == {live, other}
    assert "per_query" not in current[0] and current[0]["metrics"]["recall@10"] >= 0


def test_run_detail(client, conn, tmp_path, write_release):
    live, _ = two_runs(conn, tmp_path, write_release)
    login(client)
    body = client.get(f"/api/eval/runs/{live}").json()
    assert set(body["per_query"]) == {"validation-0", "validation-1"}
    assert client.get("/api/eval/runs/999999").status_code == 404


def test_eval_page(client, conn, tmp_path, write_release):
    two_runs(conn, tmp_path, write_release)
    login(client)
    page = client.get("/eval").text
    assert "hybrid" in page and "bm25" in page and 'class="current"' in page
    assert "di il la" in page  # stopword-only: retrieves nothing, so the live config misses it


def test_eval_page_without_runs(client):
    login(client)
    assert "Nessuna valutazione" in client.get("/eval").text

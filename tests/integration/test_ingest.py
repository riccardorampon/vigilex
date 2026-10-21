from helpers import article, question
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def counts(conn) -> dict[str, int]:
    tables = ("articles", "article_refs", "article_chunks", "questions", "qrels")
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}


def small_release(tmp_path, write_release, text1="Art. 1 obblighi di verifica"):
    return load_release(
        write_release(
            tmp_path,
            [article(1, text1, refs=[2]), article(2, "Art. 2 segnalazione operazioni sospette")],
            [question("Quando si segnala?", [2])],
        )
    )


def test_ingest_writes_all_tables(conn, tmp_path, write_release):
    stats = ingest(conn, small_release(tmp_path, write_release), FakeEmbedder())
    assert counts(conn) == {
        "articles": 2,
        "article_refs": 1,
        "article_chunks": 2,
        "questions": 1,
        "qrels": 1,
    }
    assert stats.chunks == 2


def test_ingest_is_idempotent(conn, tmp_path, write_release):
    release = small_release(tmp_path, write_release)
    ingest(conn, release, FakeEmbedder())
    before = counts(conn)
    ingest(conn, release, FakeEmbedder())
    assert counts(conn) == before


def test_reingest_replaces_stale_chunks(conn, tmp_path, write_release):
    long_text = " ".join(f"parola{i}" for i in range(1000))  # several chunks
    ingest(conn, small_release(tmp_path / "a", write_release, text1=long_text), FakeEmbedder())
    assert (
        conn.execute("SELECT count(*) FROM article_chunks WHERE article_id = 1").fetchone()[0] > 1
    )
    ingest(conn, small_release(tmp_path / "b", write_release, text1="breve"), FakeEmbedder())
    rows = conn.execute("SELECT ord, text FROM article_chunks WHERE article_id = 1").fetchall()
    assert rows == [(0, "breve")]


def test_ingest_survives_giant_token(conn, tmp_path, write_release):
    release = load_release(write_release(tmp_path, [article(1, "tabella " + "x" * 5000)]))
    stats = ingest(conn, release, FakeEmbedder())
    assert stats.chunks >= 3


def test_empty_article_is_kept_without_chunks(conn, tmp_path, write_release):
    ingest(conn, load_release(write_release(tmp_path, [article(1, "   ")])), FakeEmbedder())
    assert counts(conn)["articles"] == 1 and counts(conn)["article_chunks"] == 0

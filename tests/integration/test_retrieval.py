from helpers import article
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import bm25_search, vector_search


def load(conn, tmp_path, write_release, corpus):
    ingest(conn, load_release(write_release(tmp_path, corpus)), FakeEmbedder())


def test_bm25_uses_or_semantics(conn, tmp_path, write_release):
    load(
        conn,
        tmp_path,
        write_release,
        [
            article(1, "Obblighi di adeguata verifica della clientela"),
            article(2, "Segnalazione di operazioni sospette"),
        ],
    )
    # "bitcoin" appears nowhere: AND semantics would return nothing
    hits = bm25_search(conn, "Quali obblighi di verifica per bitcoin?", 10)
    assert [a for a, _ in hits] == [1]


def test_bm25_ranks_more_matching_terms_higher(conn, tmp_path, write_release):
    load(
        conn,
        tmp_path,
        write_release,
        [article(1, "verifica"), article(2, "adeguata verifica della clientela")],
    )
    assert [a for a, _ in bm25_search(conn, "adeguata verifica clientela", 10)] == [2, 1]


def test_bm25_stopword_only_query_returns_nothing(conn, tmp_path, write_release):
    load(conn, tmp_path, write_release, [article(1, "Obblighi di verifica")])
    assert bm25_search(conn, "di il la", 10) == []
    assert bm25_search(conn, "", 10) == []


def test_vector_search_scores_article_by_best_chunk(conn, tmp_path, write_release):
    long_text = " ".join(f"parola{i}" for i in range(1000))  # several chunks
    load(conn, tmp_path, write_release, [article(1, long_text), article(2, "breve")])
    # query vector equal to article 2's only chunk: similarity 1.0, ranked first
    (q,) = FakeEmbedder().embed(["breve"], "search_document")
    hits = vector_search(conn, q, 10)
    assert [a for a, _ in hits] == [2, 1]  # one row per article despite many chunks
    assert abs(hits[0][1] - 1.0) < 1e-5

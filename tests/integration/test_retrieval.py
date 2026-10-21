from pathlib import Path

import pytest
from helpers import article
from vigilex_core.embeddings import CachedEmbedder, FakeEmbedder
from vigilex_core.settings import Settings
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig, bm25_search, retrieve, vector_search
from vigilex_rag.seed import load_seed

SEED = Path(__file__).resolve().parents[2] / "data" / "seed"


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


class ReverseReranker:
    """Puts the fused order upside down, so the test can see rerank was applied."""

    def __init__(self):
        self.documents: list[str] = []

    def rerank(self, query, documents, top_n):
        self.documents = list(documents)
        n = len(documents)
        return [(i, float(i)) for i in reversed(range(n))][:top_n]


def three_articles(conn, tmp_path, write_release):
    load(
        conn,
        tmp_path,
        write_release,
        [
            article(1, "adeguata verifica della clientela"),
            article(2, "verifica"),
            article(3, "segnalazione sospetta"),
        ],
    )


def test_retrieve_hybrid_returns_hits_with_text(conn, tmp_path, write_release):
    three_articles(conn, tmp_path, write_release)
    hits = retrieve(conn, "adeguata verifica", RetrieverConfig(top_k=2), FakeEmbedder())
    assert len(hits) == 2
    assert hits[0].id == 1 and hits[0].text == "adeguata verifica della clientela"
    assert hits[0].source_act == "act" and hits[0].path == "article_1"


def test_retrieve_bm25_only_never_embeds(conn, tmp_path, write_release):
    three_articles(conn, tmp_path, write_release)

    class NoEmbed(FakeEmbedder):
        def embed(self, texts, input_type):
            raise AssertionError("bm25 mode must not embed")

    hits = retrieve(conn, "verifica", RetrieverConfig(mode="bm25"), NoEmbed())
    assert {h.id for h in hits} == {1, 2}


def test_retrieve_applies_reranker_order(conn, tmp_path, write_release):
    three_articles(conn, tmp_path, write_release)
    rr = ReverseReranker()
    cfg = RetrieverConfig(rerank=True, top_k=3)
    fused = [
        h.id for h in retrieve(conn, "adeguata verifica", RetrieverConfig(top_k=3), FakeEmbedder())
    ]
    hits = retrieve(conn, "adeguata verifica", cfg, FakeEmbedder(), rr)
    assert [h.id for h in hits] == list(reversed(fused))
    assert len(rr.documents) == len(fused)


def test_retrieve_rerank_without_reranker_is_an_error(conn):
    with pytest.raises(ValueError, match="reranker"):
        retrieve(conn, "x", RetrieverConfig(rerank=True), FakeEmbedder())


def test_retrieve_nothing_found(conn, tmp_path, write_release):
    three_articles(conn, tmp_path, write_release)
    assert retrieve(conn, "di il", RetrieverConfig(mode="bm25"), FakeEmbedder()) == []


def test_seed_vector_search_returns_full_candidate_set(conn):
    model = Settings.from_env().embed_model_id
    cached = CachedEmbedder(conn, model, FakeEmbedder(), store=False)
    load_seed(conn, SEED, cached)
    question = conn.execute("SELECT text FROM questions ORDER BY id LIMIT 1").fetchone()[0]
    conn.execute(
        "SET enable_seqscan = off"
    )  # small seed: force the HNSW index, as on the full corpus
    hits = retrieve(conn, question, RetrieverConfig(mode="vector", top_k=50), cached)
    assert len(hits) == 50  # not capped by hnsw.ef_search = 40

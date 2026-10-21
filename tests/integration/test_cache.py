import numpy as np
from vigilex_core.embeddings import CachedEmbedder, FakeEmbedder, cache_key


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, input_type):
        self.seen.extend(texts)
        return super().embed(texts, input_type)


def count(conn) -> int:
    return conn.execute("SELECT count(*) FROM embedding_cache").fetchone()[0]


def test_misses_are_embedded_once_and_stored(conn):
    inner = CountingEmbedder()
    cached = CachedEmbedder(conn, "m", inner, store=True)
    first = cached.embed(["a", "b", "a"], "search_document")
    conn.commit()
    second = cached.embed(["b", "a"], "search_document")
    assert inner.seen == ["a", "b"]  # duplicates deduped, second call all hits
    assert count(conn) == 2
    assert first[0] == first[2]
    assert np.allclose(second, [first[1], first[0]], atol=1e-6)  # float32 round-trip


def test_input_type_is_part_of_the_key(conn):
    inner = CountingEmbedder()
    cached = CachedEmbedder(conn, "m", inner, store=True)
    cached.embed(["a"], "search_document")
    cached.embed(["a"], "search_query")
    assert inner.seen == ["a", "a"]
    assert cache_key("m", "search_document", "a") != cache_key("m", "search_query", "a")


def test_store_false_never_writes(conn):
    cached = CachedEmbedder(conn, "m", FakeEmbedder(), store=False)
    cached.embed(["a", "b"], "search_query")
    conn.commit()
    assert count(conn) == 0


def test_store_false_still_reads_hits(conn):
    CachedEmbedder(conn, "m", CountingEmbedder(), store=True).embed(["a"], "search_query")
    inner = CountingEmbedder()
    CachedEmbedder(conn, "m", inner, store=False).embed(["a"], "search_query")
    assert inner.seen == []

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import psycopg
from vigilex_core.embeddings import Embedder
from vigilex_core.rerank import Reranker

Mode = Literal["bm25", "vector", "hybrid"]
CHUNKS_PER_ARTICLE = 4  # chunks read per wanted article, so max-per-article still fills n

# OR of the query's Italian lexemes: plainto_tsquery ANDs them, which natural-language
# questions almost never satisfy. ts_rank with normalization 1 (divide by 1 + log length):
# on the seed eval it scored recall@10 0.120 vs 0.034 for ts_rank_cd. Neither has IDF, so
# this is BM25-like, not BM25 (an Okapi-in-SQL prototype reached 0.159).
BM25_SQL = """
SELECT a.id, ts_rank(a.tsv, t.q, 1) AS score
FROM articles a,
     (SELECT replace(plainto_tsquery('italian', %(q)s)::text, ' & ', ' | ')::tsquery AS q) t
WHERE a.tsv @@ t.q
ORDER BY score DESC, a.id
LIMIT %(n)s
"""

VECTOR_SQL = """
WITH c AS (
    SELECT article_id, embedding <=> %(v)s AS dist
    FROM article_chunks
    ORDER BY embedding <=> %(v)s
    LIMIT %(chunks)s
)
SELECT article_id, 1 - min(dist) AS score
FROM c
GROUP BY article_id
ORDER BY score DESC, article_id
LIMIT %(n)s
"""


@dataclass(frozen=True)
class RetrieverConfig:
    mode: Mode = "hybrid"
    candidates: int = 50
    top_k: int = 10
    rerank: bool = False
    rrf_k: int = 60
    embed_model_id: str = "cohere.embed-multilingual-v3"

    def name(self) -> str:
        return self.mode + ("+rerank" if self.rerank else "")

    def config_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, article_id in enumerate(ranking, start=1):
            scores[article_id] = scores.get(article_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def bm25_search(conn: psycopg.Connection, query: str, n: int) -> list[tuple[int, float]]:
    rows = conn.execute(BM25_SQL, {"q": query, "n": n}).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def has_content_words(conn: psycopg.Connection, query: str) -> bool:
    """False for empty or stopword-only questions (no Italian lexemes left)."""
    row = conn.execute("SELECT numnode(plainto_tsquery('italian', %s)) > 0", (query,)).fetchone()
    return bool(row and row[0])


def vector_search(
    conn: psycopg.Connection, query_vector: Sequence[float], n: int
) -> list[tuple[int, float]]:
    chunks = n * CHUNKS_PER_ARTICLE
    with conn.transaction():
        # HNSW returns at most ef_search rows (default 40); raise it for this transaction only
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(40, chunks)),))
        rows = conn.execute(
            VECTOR_SQL,
            {"v": np.asarray(query_vector, dtype=np.float32), "chunks": chunks, "n": n},
        ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


RERANK_DOC_CHARS = 4000  # Cohere Rerank 3.5 reads ~4k tokens per document; keep requests small


@dataclass(frozen=True)
class ArticleHit:
    id: int
    score: float
    source_act: str
    path: str
    text: str


def retrieve(
    conn: psycopg.Connection,
    query: str,
    cfg: RetrieverConfig,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> list[ArticleHit]:
    if cfg.rerank and reranker is None:
        raise ValueError(f"config {cfg.name()} needs a reranker")
    if not has_content_words(conn, query):
        return []  # vector search always has neighbours; an empty question must not get answers
    rankings: list[list[int]] = []
    if cfg.mode in ("bm25", "hybrid"):
        rankings.append([a for a, _ in bm25_search(conn, query, cfg.candidates)])
    if cfg.mode in ("vector", "hybrid"):
        (qv,) = embedder.embed([query], "search_query")
        rankings.append([a for a, _ in vector_search(conn, qv, cfg.candidates)])
    fused = rrf(rankings, cfg.rrf_k)[: cfg.candidates]
    if not fused:
        return []

    ids = [a for a, _ in fused]
    rows = conn.execute(
        "SELECT id, source_act, path, text FROM articles WHERE id = ANY(%s)", (ids,)
    ).fetchall()
    by_id = {r[0]: r[1:] for r in rows}
    if reranker is not None and cfg.rerank:
        ranked = reranker.rerank(query, [by_id[a][2][:RERANK_DOC_CHARS] for a in ids], cfg.top_k)
        top = [(ids[i], score) for i, score in ranked]
    else:
        top = fused[: cfg.top_k]
    return [ArticleHit(a, score, *by_id[a]) for a, score in top]

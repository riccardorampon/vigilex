import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import psycopg

Mode = Literal["bm25", "vector", "hybrid"]
CHUNKS_PER_ARTICLE = 4  # chunks read per wanted article, so max-per-article still fills n

# OR of the query's Italian lexemes: plainto_tsquery ANDs them, which natural-language
# questions almost never satisfy. ts_rank_cd still rewards articles matching more terms.
BM25_SQL = """
SELECT a.id, ts_rank_cd(a.tsv, t.q) AS score
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

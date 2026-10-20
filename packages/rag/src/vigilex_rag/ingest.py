import logging
from dataclasses import dataclass

import numpy as np
import psycopg
from vigilex_core.embeddings import Embedder

from vigilex_rag.chunking import chunk_text
from vigilex_rag.release import Release

log = logging.getLogger(__name__)
BATCH = 200


@dataclass
class IngestStats:
    articles: int
    refs: int
    questions: int
    chunks: int


def ingest(conn: psycopg.Connection, release: Release, embedder: Embedder) -> IngestStats:
    """Upsert a release. Safe to re-run: rows upsert, each article's chunks are replaced."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO articles (id, source_act, path, text) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET source_act = EXCLUDED.source_act, "
            "path = EXCLUDED.path, text = EXCLUDED.text",
            [(a.id, a.source_act, a.path, a.text) for a in release.articles],
        )
        cur.executemany(
            "INSERT INTO article_refs (src_id, dst_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            release.refs,
        )
        cur.executemany(
            "INSERT INTO questions (id, split, text, domain, topics) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET split = EXCLUDED.split, text = EXCLUDED.text, "
            "domain = EXCLUDED.domain, topics = EXCLUDED.topics",
            [(q.id, q.split, q.text, q.domain, q.topics) for q in release.questions],
        )
        cur.executemany(
            "INSERT INTO qrels (question_id, article_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(q.id, a) for q in release.questions for a in q.relevant],
        )
    conn.commit()

    n_chunks = 0
    for i in range(0, len(release.articles), BATCH):
        batch = release.articles[i : i + BATCH]
        rows = [(a.id, j, c) for a in batch for j, c in enumerate(chunk_text(a.text))]
        vectors = embedder.embed([c for _, _, c in rows], "search_document") if rows else []
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM article_chunks WHERE article_id = ANY(%s)", ([a.id for a in batch],)
            )
            cur.executemany(
                "INSERT INTO article_chunks (article_id, ord, text, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [
                    (aid, j, c, np.asarray(v, dtype=np.float32))
                    for (aid, j, c), v in zip(rows, vectors, strict=True)
                ],
            )
        conn.commit()  # a crash loses at most one batch; embeddings already paid stay cached
        n_chunks += len(rows)
        log.info("ingested %d/%d articles", i + len(batch), len(release.articles))

    return IngestStats(len(release.articles), len(release.refs), len(release.questions), n_chunks)

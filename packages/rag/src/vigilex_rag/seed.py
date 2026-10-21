import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from vigilex_core.embeddings import Embedder, cache_key

from vigilex_rag.chunking import chunk_text
from vigilex_rag.ingest import IngestStats, ingest
from vigilex_rag.release import Release, load_release

SEED_SPLIT = "validation"
SEED_DOMAINS = ("Antiriciclaggio e Antiterrorismo", "Privacy")
HOP_SAMPLE = 500
DISTRACTOR_SAMPLE = 1000
DISTRACTOR_MAX_CHARS = 12000


def select_seed(release: Release, rng: random.Random) -> Release:
    """Validation AML/privacy questions + gold + sampled one-hop + same-act distractors."""
    qs = [q for q in release.questions if q.split == SEED_SPLIT and q.domain in SEED_DOMAINS]
    by_id = {a.id: a for a in release.articles}
    gold = {a for q in qs for a in q.relevant}
    out_refs: dict[int, set[int]] = defaultdict(set)
    for s, d in release.refs:
        out_refs[s].add(d)
    hops = sorted({d for g in gold for d in out_refs[g]} - gold)
    hop = set(rng.sample(hops, min(HOP_SAMPLE, len(hops))))
    acts = {by_id[g].source_act for g in gold}
    pool = sorted(
        a.id
        for a in release.articles
        if a.source_act in acts and a.id not in gold | hop and len(a.text) <= DISTRACTOR_MAX_CHARS
    )
    keep = gold | hop | set(rng.sample(pool, min(DISTRACTOR_SAMPLE, len(pool))))
    return Release(
        [a for a in release.articles if a.id in keep],
        [(s, d) for s, d in release.refs if s in keep and d in keep],
        qs,
    )


def write_seed(release: Release, out: Path, embedder: Embedder, model_id: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    refs: dict[int, list[int]] = defaultdict(list)
    for s, d in release.refs:
        refs[s].append(d)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": a.id,
                    "file_name": a.source_act,
                    "path": a.path,
                    "content": a.text,
                    "reference": refs[a.id],
                }
                for a in release.articles
            ],
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("file_name", pa.string()),
                    ("path", pa.string()),
                    ("content", pa.string()),
                    ("reference", pa.list_(pa.int64())),
                ]
            ),
        ),
        out / "corpus.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": q.id,
                    "question": q.text,
                    "topics": q.topics,
                    "normative_area": q.domain,
                    "relevant_doc_ids": list(q.relevant),
                }
                for q in release.questions
            ]
        ),
        out / "validation.parquet",
    )
    docs = sorted({c for a in release.articles for c in chunk_text(a.text)})
    queries = sorted({q.text for q in release.questions})
    keys = [cache_key(model_id, "search_document", t) for t in docs]
    keys += [cache_key(model_id, "search_query", t) for t in queries]
    vecs = embedder.embed(docs, "search_document") + embedder.embed(queries, "search_query")
    emb = np.asarray(vecs, dtype=np.float16)
    pq.write_table(
        pa.table(
            {
                "key": pa.array(keys),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), emb.shape[1]),
            }
        ),
        out / "embeddings.parquet",
    )


def load_seed(conn: psycopg.Connection, seed_dir: Path, embedder: Embedder) -> IngestStats:
    t = pq.read_table(seed_dir / "embeddings.parquet")
    keys = t.column("key").to_pylist()
    emb = t.column("embedding").combine_chunks().flatten().to_numpy().astype(np.float32)
    emb = emb.reshape(len(keys), -1)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO embedding_cache (key, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            list(zip(keys, emb, strict=True)),
        )
    conn.commit()
    return ingest(conn, load_release(seed_dir), embedder)

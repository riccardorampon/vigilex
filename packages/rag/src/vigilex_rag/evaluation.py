import logging
from dataclasses import asdict, dataclass, replace

import psycopg
from psycopg.types.json import Jsonb
from vigilex_core.embeddings import Embedder
from vigilex_core.rerank import Reranker

from vigilex_rag.metrics import METRICS, mean_metrics, query_metrics
from vigilex_rag.retrieval import RetrieverConfig, retrieve

log = logging.getLogger(__name__)

EVAL_DEPTH = 20  # deepest cutoff in METRICS
CONFIG_NAMES = ("bm25", "vector", "hybrid", "hybrid+rerank")


@dataclass
class EvalRun:
    config: RetrieverConfig
    split: str
    metrics: dict[str, float]
    per_query: dict[str, dict]


def parse_configs(spec: str, embed_model_id: str, rerank_available: bool) -> list[RetrieverConfig]:
    configs: list[RetrieverConfig] = []
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        if name not in CONFIG_NAMES:
            raise ValueError(f"unknown config {name!r}; choose from {CONFIG_NAMES}")
        mode, _, rerank = name.partition("+")
        if rerank and not rerank_available:
            log.warning("skipping %s: no reranker in this mode (VIGILEX_LLM=fake?)", name)
            continue
        configs.append(
            RetrieverConfig(mode=mode, rerank=bool(rerank), embed_model_id=embed_model_id)  # type: ignore[arg-type]
        )
    return configs


def evaluate(
    conn: psycopg.Connection,
    cfg: RetrieverConfig,
    embedder: Embedder,
    reranker: Reranker | None,
    split: str = "validation",
) -> EvalRun:
    questions = conn.execute(
        "SELECT q.id, q.text, array_agg(r.article_id) FROM questions q "
        "JOIN qrels r ON r.question_id = q.id WHERE q.split = %s "
        "GROUP BY q.id, q.text ORDER BY q.id",
        (split,),
    ).fetchall()
    if not questions:
        raise ValueError(f"no questions with relevant articles in split {split!r}")
    deep = replace(cfg, top_k=EVAL_DEPTH)
    per_query: dict[str, dict] = {}
    for qid, text, relevant in questions:
        ranked = [h.id for h in retrieve(conn, text, deep, embedder, reranker)]
        per_query[qid] = {
            **query_metrics(ranked, set(relevant)),
            "retrieved": ranked[:10],
            "relevant": sorted(relevant),
        }
    return EvalRun(cfg, split, mean_metrics(list(per_query.values())), per_query)


def save_run(conn: psycopg.Connection, run: EvalRun, git_sha: str | None) -> int:
    row = conn.execute(
        "INSERT INTO eval_runs (git_sha, config_name, config_hash, retriever_config, split, "
        "n_questions, metrics, per_query) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            git_sha,
            run.config.name(),
            run.config.config_hash(),
            Jsonb(asdict(run.config)),
            run.split,
            len(run.per_query),
            Jsonb(run.metrics),
            Jsonb(run.per_query),
        ),
    ).fetchone()
    conn.commit()
    assert row is not None
    return row[0]


def report(runs: list[EvalRun]) -> str:
    lines = [
        "| config | questions | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for r in runs:
        cells = " | ".join(f"{r.metrics[m]:.3f}" for m in METRICS)
        lines.append(f"| {r.config.name()} | {len(r.per_query)} | {cells} |")
    return "\n".join(lines) + "\n"

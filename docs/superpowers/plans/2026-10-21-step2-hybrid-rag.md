# Step 2 — Hybrid RAG + Evaluation Layer 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vigilex ask "<domanda>"` answers from retrieved JuriFindIT articles with checked `[art:ID]` citations, and `vigilex eval` measures BM25 / vector / hybrid / hybrid+rerank retrieval (recall@5/10/20, nDCG@10, MRR) into an `eval_runs` table, with a recall@10 gate in CI.

**Architecture:** `vigilex_core` gains a shared Bedrock client, a Converse-based `Chat` and an InvokeModel-based `Reranker` (each with a fake or absent counterpart for tests). `vigilex_rag` gains `retrieval.py` (keyword search, vector search, RRF, rerank), `answer.py` (prompt + citation post-check), `metrics.py` (pure IR metrics) and `evaluation.py` (runs configs over the gold questions, persists and reports). CI evaluates on the committed seed with its cached real embeddings, so no Bedrock calls.

**Tech Stack:** as step 1 (Python 3.12, uv workspace, psycopg 3, pgvector, boto3, pytest). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-20-vigilex-umbrella-design.md` (§4, step 2 of §12).

**Branch:** `step2/hybrid-rag`, created from `step1/foundations` (step 1 is not yet merged into `main`; rebase onto `main` after it merges).

## Global Constraints

- Everything from the step 1 plan's Global Constraints still holds (Python 3.12, tests never call Bedrock, `VIGILEX_LLM=fake` in tests, ruff 0.17.1 with `docs/` excluded, commit trailer).
- Defaults: chat model `eu.anthropic.claude-sonnet-4-6`, rerank model `cohere.rerank-v3-5:0`, region `eu-central-1`. Overridable via `VIGILEX_CHAT_MODEL_ID`, `VIGILEX_RERANK_MODEL_ID`.
- All Bedrock traffic goes through `bedrock-runtime` (Converse for chat, InvokeModel for embed and rerank): one VPC endpoint in step 3.
- RRF `k = 60`; candidates per retriever 50; answer and default `top_k` 10; eval retrieval depth 20.
- Every Bedrock chat call has a fixed `max_tokens` and `temperature = 0`.

## Spec deltas (update the spec in Task 7)

- **Chat models:** this AWS account gets `AccessDeniedException` on Claude 5.x EU profiles (verified 2026-10-21). Defaults are `eu.anthropic.claude-sonnet-4-6` (answers) and, for later steps, `eu.anthropic.claude-haiku-4-5-20251001-v1:0` (routing/judge). Switch by env var once access is granted.
- **Rerank** uses `InvokeModel` on `bedrock-runtime` (body `{"query", "documents", "top_n", "api_version": 2}`), not the Rerank API on `bedrock-agent-runtime`: same client and VPC endpoint as embeddings.
- **Keyword search uses OR semantics:** the query's Italian lexemes are OR-ed (`plainto_tsquery` with `&` replaced by `|`), ranked by `ts_rank_cd`. `websearch_to_tsquery` ANDs every term, so natural-language questions would rarely match anything.
- **Vector search** raises `hnsw.ef_search` per query (pgvector caps HNSW results at `ef_search`, default 40) and reads `4 × candidates` chunks to obtain `candidates` distinct articles.
- **Metrics** are ~30 lines of stdlib code instead of `ranx` (which pulls numba/llvmlite into the step 3 Lambda image), each pinned by hand-computed unit tests.
- **`RetrieverConfig`** is a frozen dataclass (Pydantic arrives with FastAPI in step 3); `config_hash` = first 16 hex chars of sha256 of its sorted JSON.
- **Eval** retrieves to depth 20 for recall@20 but stores the evaluated config unchanged, so its hash matches the deployed config. In fake mode, rerank configs are skipped with a warning (no reranker), never scored with a fake.
- **Seed metrics are not comparable with the JuriFindIT paper** (37 questions over 1.9k articles vs 179 over 23k). `make ingest-full` (optional, ≈$1.5 of embeddings) loads the full corpus for a like-for-like comparison with the paper's BM25 recall@5 = 43.02.

## Review Focus

1. **Natural-language question containing words absent from the corpus:** keyword search still returns the articles matching the other words. Test: Task 2 `test_bm25_uses_or_semantics`.
2. **Stopword-only or empty question:** retrieval returns `[]` and `answer` returns the not-found message without calling the LLM. Tests: Task 2 `test_bm25_stopword_only_query_returns_nothing`, Task 4 `test_answer_without_hits_skips_llm`.
3. **HNSW `ef_search` cap:** vector search over the seed returns 50 distinct articles, not ≤ 40 chunks' worth. Test: Task 3 `test_seed_vector_search_returns_full_candidate_set`.
4. **LLM cites an article that was not retrieved (or invents an id):** it is reported as invalid, never as a citation. Test: Task 4 `test_citations_outside_hits_are_invalid`.
5. **Eval misuse:** an empty split fails loudly; rerank configs in fake mode are skipped, not scored. Tests: Task 6 `test_eval_empty_split_raises`, `test_parse_configs_skips_rerank_without_reranker`.

---

## File map

```
packages/core/src/vigilex_core/settings.py      # + chat_model_id, rerank_model_id
packages/core/src/vigilex_core/bedrock.py       # runtime_client(settings) shared by embed/chat/rerank
packages/core/src/vigilex_core/embeddings.py    # bedrock_embedder uses runtime_client
packages/core/src/vigilex_core/llm.py           # Chat, BedrockChat, FakeChat, make_chat
packages/core/src/vigilex_core/rerank.py        # Reranker, BedrockReranker, make_reranker
packages/rag/src/vigilex_rag/retrieval.py       # RetrieverConfig, ArticleHit, rrf, bm25_search, vector_search, retrieve
packages/rag/src/vigilex_rag/answer.py          # Answer, build_prompt, check_citations, answer
packages/rag/src/vigilex_rag/metrics.py         # recall_at_k, ndcg_at_k, reciprocal_rank, query_metrics, mean_metrics
packages/rag/src/vigilex_rag/evaluation.py      # parse_configs, evaluate, save_run, report
packages/rag/src/vigilex_rag/cli.py             # + ask, eval
migrations/0002_eval_runs.sql
tests/unit/test_settings.py, test_llm.py, test_rerank.py, test_retrieval.py, test_answer.py, test_metrics.py, test_evaluation.py
tests/integration/conftest.py (+eval_runs), test_retrieval.py, test_answer.py, test_evaluation.py
Makefile, .github/workflows/ci.yml, README.md, CLAUDE.md, spec
```

---

### Task 1: Shared Bedrock client, Chat and Reranker adapters

**Files:**
- Create: `packages/core/src/vigilex_core/bedrock.py`, `packages/core/src/vigilex_core/llm.py`, `packages/core/src/vigilex_core/rerank.py`
- Modify: `packages/core/src/vigilex_core/settings.py`, `packages/core/src/vigilex_core/embeddings.py` (`bedrock_embedder`), `tests/unit/test_settings.py`, `tests/unit/test_embeddings.py` (`Settings(...)` gains two args)
- Test: `tests/unit/test_llm.py`, `tests/unit/test_rerank.py`

**Interfaces:**
- Produces:
  - `Settings` gains `chat_model_id: str` (default `eu.anthropic.claude-sonnet-4-6`, env `VIGILEX_CHAT_MODEL_ID`) and `rerank_model_id: str` (default `cohere.rerank-v3-5:0`, env `VIGILEX_RERANK_MODEL_ID`), appended after `embed_model_id`.
  - `vigilex_core.bedrock.runtime_client(settings) -> Any` (boto3 `bedrock-runtime`, adaptive retries).
  - `vigilex_core.llm`: `class Chat(Protocol): def complete(self, system: str, user: str, max_tokens: int) -> str`; `BedrockChat(client, model_id)`; `FakeChat()` (replies `"Risposta di prova [art:N]"` citing the first `[art:N]` in `user`, or `"Risposta di prova."`); `make_chat(settings) -> Chat`.
  - `vigilex_core.rerank`: `class Reranker(Protocol): def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]` (document index, score; best first); `BedrockReranker(client, model_id)`; `make_reranker(settings) -> Reranker | None` (`None` in fake mode).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_llm.py`:
```python
from vigilex_core.llm import BedrockChat, FakeChat


class StubConverse:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 12, "outputTokens": 3},
            "stopReason": "end_turn",
        }


def test_bedrock_chat_sends_system_user_and_fixed_inference_config():
    stub = StubConverse("Roma [art:5]")
    out = BedrockChat(stub, "eu.anthropic.claude-sonnet-4-6").complete("sys", "domanda", 300)
    assert out == "Roma [art:5]"
    (call,) = stub.calls
    assert call["modelId"] == "eu.anthropic.claude-sonnet-4-6"
    assert call["system"] == [{"text": "sys"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "domanda"}]}]
    assert call["inferenceConfig"] == {"maxTokens": 300, "temperature": 0}


def test_fake_chat_cites_first_article_in_prompt():
    assert FakeChat().complete("s", "[art:7] testo\n[art:9] altro", 10) == "Risposta di prova [art:7]"
    assert FakeChat().complete("s", "nessun articolo", 10) == "Risposta di prova."
```

`tests/unit/test_rerank.py`:
```python
import io
import json

from vigilex_core.rerank import BedrockReranker, make_reranker
from vigilex_core.settings import Settings


class StubInvoke:
    def __init__(self):
        self.bodies: list[dict] = []

    def invoke_model(self, *, modelId, body, contentType, accept):
        self.bodies.append({"modelId": modelId, **json.loads(body)})
        results = [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        return {"body": io.BytesIO(json.dumps({"results": results}).encode())}


def test_bedrock_reranker_request_and_order():
    stub = StubInvoke()
    out = BedrockReranker(stub, "cohere.rerank-v3-5:0").rerank("q", ["a", "b", "c"], 5)
    assert out == [(2, 0.9), (0, 0.1)]
    (body,) = stub.bodies
    assert body == {
        "modelId": "cohere.rerank-v3-5:0",
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_n": 3,  # clamped to the number of documents
        "api_version": 2,
    }


def test_no_documents_makes_no_call():
    stub = StubInvoke()
    assert BedrockReranker(stub, "m").rerank("q", [], 5) == []
    assert stub.bodies == []


def test_fake_mode_has_no_reranker():
    s = Settings("u", "fake", "eu-central-1", "e", "c", "r")
    assert make_reranker(s) is None
```

Also update the two existing `Settings(...)` constructions: in `tests/unit/test_embeddings.py::test_fake_mode_never_stores` use
`Settings("postgresql://unused", "fake", "eu-central-1", "cohere.embed-multilingual-v3", "c", "r")`, and extend `tests/unit/test_settings.py::test_defaults` (delete the two new env vars in its loop and assert):
```python
    assert s.chat_model_id == "eu.anthropic.claude-sonnet-4-6"
    assert s.rerank_model_id == "cohere.rerank-v3-5:0"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vigilex_core.llm'` / `'vigilex_core.rerank'`, and `test_defaults` fails on `chat_model_id`.

- [ ] **Step 3: Implement**

`packages/core/src/vigilex_core/settings.py` — add the two fields after `embed_model_id` and in `from_env`:
```python
    chat_model_id: str
    rerank_model_id: str
```
```python
            chat_model_id=os.environ.get("VIGILEX_CHAT_MODEL_ID", "eu.anthropic.claude-sonnet-4-6"),
            rerank_model_id=os.environ.get("VIGILEX_RERANK_MODEL_ID", "cohere.rerank-v3-5:0"),
```

`packages/core/src/vigilex_core/bedrock.py`:
```python
from typing import Any

import boto3
from botocore.config import Config

from vigilex_core.settings import Settings


def runtime_client(settings: Settings) -> Any:
    """One bedrock-runtime client for embed, chat and rerank (one VPC endpoint in AWS)."""
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )
```

In `embeddings.py`, replace the body of `bedrock_embedder` with `return BedrockEmbedder(runtime_client(settings), settings.embed_model_id)`, import `from vigilex_core.bedrock import runtime_client`, and drop the now-unused `boto3` / `Config` imports.

`packages/core/src/vigilex_core/llm.py`:
```python
import logging
import re
import time
from typing import Any, Protocol

from vigilex_core.bedrock import runtime_client
from vigilex_core.settings import Settings

log = logging.getLogger(__name__)
_FIRST_CITATION = re.compile(r"\[art:\d+\]")


class Chat(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class BedrockChat:
    def __init__(self, client: Any, model_id: str):
        self.client = client
        self.model_id = model_id

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        start = time.monotonic()
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = resp["usage"]
        log.info(
            "chat model=%s input_tokens=%d output_tokens=%d stop=%s latency_ms=%d",
            self.model_id,
            usage["inputTokens"],
            usage["outputTokens"],
            resp["stopReason"],
            (time.monotonic() - start) * 1000,
        )
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


class FakeChat:
    """Deterministic stand-in: cites the first article tag it was given."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        m = _FIRST_CITATION.search(user)
        return f"Risposta di prova {m.group(0)}" if m else "Risposta di prova."


def make_chat(settings: Settings) -> Chat:
    if settings.llm == "fake":
        return FakeChat()
    return BedrockChat(runtime_client(settings), settings.chat_model_id)
```

`packages/core/src/vigilex_core/rerank.py`:
```python
import json
from collections.abc import Sequence
from typing import Any, Protocol

from vigilex_core.bedrock import runtime_client
from vigilex_core.settings import Settings


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]: ...


class BedrockReranker:
    """Cohere Rerank via InvokeModel on bedrock-runtime (same endpoint as embeddings)."""

    def __init__(self, client: Any, model_id: str):
        self.client = client
        self.model_id = model_id

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        body = json.dumps(
            {
                "query": query,
                "documents": list(documents),
                "top_n": min(top_n, len(documents)),
                "api_version": 2,
            }
        )
        resp = self.client.invoke_model(
            modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
        )
        results = json.loads(resp["body"].read())["results"]
        return [(r["index"], r["relevance_score"]) for r in results]


def make_reranker(settings: Settings) -> Reranker | None:
    """None in fake mode: rerank quality cannot be faked, so callers must skip it."""
    if settings.llm == "fake":
        return None
    return BedrockReranker(runtime_client(settings), settings.rerank_model_id)
```

- [ ] **Step 4: Run tests and tooling**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pytest tests/unit -q && uv run pyright`
Expected: all pass; pyright 0 errors.

- [ ] **Step 5: Commit**

```bash
git add packages/core tests/unit
git commit -m "feat: bedrock chat and rerank adapters with shared runtime client

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Retrieval primitives (config, RRF, keyword and vector search)

**Files:**
- Create: `packages/rag/src/vigilex_rag/retrieval.py`
- Test: `tests/unit/test_retrieval.py`, `tests/integration/test_retrieval.py`

**Interfaces:**
- Consumes: `connect`, `conn` fixture, `helpers.article`, `write_release` (step 1), `ingest`, `load_release`, `FakeEmbedder`.
- Produces (in `vigilex_rag.retrieval`):
  - `Mode = Literal["bm25", "vector", "hybrid"]`
  - `RetrieverConfig` frozen dataclass: `mode: Mode = "hybrid"`, `candidates: int = 50`, `top_k: int = 10`, `rerank: bool = False`, `rrf_k: int = 60`, `embed_model_id: str = "cohere.embed-multilingual-v3"`; methods `name() -> str` (`"hybrid"`, `"hybrid+rerank"`, …) and `config_hash() -> str`.
  - `rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]` — best first, ties by id.
  - `bm25_search(conn, query: str, n: int) -> list[tuple[int, float]]`
  - `vector_search(conn, query_vector: Sequence[float], n: int) -> list[tuple[int, float]]` (score = cosine similarity of the article's best chunk)

- [ ] **Step 1: Write the failing unit tests** — `tests/unit/test_retrieval.py`

```python
from vigilex_rag.retrieval import RetrieverConfig, rrf


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == 1 / 61 + 1 / 62
    assert scores[3] == 1 / 63 + 1 / 61
    assert scores[2] == 1 / 62
    assert [a for a, _ in fused] == [1, 3, 2]


def test_rrf_breaks_ties_by_id_and_handles_empty():
    assert [a for a, _ in rrf([[5], [4]], k=60)] == [4, 5]
    assert rrf([], k=60) == []


def test_config_name_and_hash():
    a = RetrieverConfig()
    assert a.name() == "hybrid"
    assert RetrieverConfig(rerank=True).name() == "hybrid+rerank"
    assert a.config_hash() == RetrieverConfig().config_hash()
    assert a.config_hash() != RetrieverConfig(top_k=20).config_hash()
    assert len(a.config_hash()) == 16
```

- [ ] **Step 2: Write the failing integration tests** — `tests/integration/test_retrieval.py`

```python
from helpers import article
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import bm25_search, vector_search


def load(conn, tmp_path, write_release, corpus):
    ingest(conn, load_release(write_release(tmp_path, corpus)), FakeEmbedder())


def test_bm25_uses_or_semantics(conn, tmp_path, write_release):
    load(conn, tmp_path, write_release, [
        article(1, "Obblighi di adeguata verifica della clientela"),
        article(2, "Segnalazione di operazioni sospette"),
    ])
    # "bitcoin" appears nowhere: AND semantics would return nothing
    hits = bm25_search(conn, "Quali obblighi di verifica per bitcoin?", 10)
    assert [a for a, _ in hits] == [1]


def test_bm25_ranks_more_matching_terms_higher(conn, tmp_path, write_release):
    load(conn, tmp_path, write_release, [
        article(1, "verifica"),
        article(2, "adeguata verifica della clientela"),
    ])
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_retrieval.py -q; VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_retrieval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.retrieval'`

- [ ] **Step 4: Implement** — `packages/rag/src/vigilex_rag/retrieval.py`

```python
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit -q && VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration -q && uv run pyright`
Expected: all pass. If `test_bm25_stopword_only_query_returns_nothing` errors on the empty tsquery cast, wrap the replaced text with `nullif(..., '')` and compare with `coalesce`, and record the ruling.

- [ ] **Step 6: Commit**

```bash
git add packages/rag/src/vigilex_rag/retrieval.py tests/unit/test_retrieval.py tests/integration/test_retrieval.py
git commit -m "feat: keyword (OR) and vector retrieval with RRF

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `retrieve()` — fusion, rerank, article hits

**Files:**
- Modify: `packages/rag/src/vigilex_rag/retrieval.py`
- Test: `tests/integration/test_retrieval.py` (append)

**Interfaces:**
- Consumes: Task 2; `Embedder` (step 1); `Reranker` (Task 1).
- Produces:
  - `ArticleHit` frozen dataclass: `id: int`, `score: float`, `source_act: str`, `path: str`, `text: str`
  - `retrieve(conn, query: str, cfg: RetrieverConfig, embedder: Embedder, reranker: Reranker | None = None) -> list[ArticleHit]` — at most `cfg.top_k` hits; `ValueError` if `cfg.rerank` and `reranker is None`.
  - `RERANK_DOC_CHARS = 4000`

- [ ] **Step 1: Write the failing tests** (append to `tests/integration/test_retrieval.py`; move the new imports to the top of the file and merge them with the existing ones — ruff's E402 rejects imports below code)

```python
from pathlib import Path

import pytest
from vigilex_core.embeddings import CachedEmbedder
from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig, retrieve
from vigilex_rag.seed import load_seed

SEED = Path(__file__).resolve().parents[2] / "data" / "seed"


class ReverseReranker:
    """Puts the fused order upside down, so the test can see rerank was applied."""

    def __init__(self):
        self.documents: list[str] = []

    def rerank(self, query, documents, top_n):
        self.documents = list(documents)
        n = len(documents)
        return [(i, float(i)) for i in reversed(range(n))][:top_n]


def three_articles(conn, tmp_path, write_release):
    load(conn, tmp_path, write_release, [
        article(1, "adeguata verifica della clientela"),
        article(2, "verifica"),
        article(3, "segnalazione sospetta"),
    ])


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
    fused = [h.id for h in retrieve(conn, "adeguata verifica", RetrieverConfig(top_k=3), FakeEmbedder())]
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
    hits = retrieve(conn, question, RetrieverConfig(mode="vector", top_k=50), cached)
    assert len(hits) == 50  # not capped by hnsw.ef_search = 40
```

- [ ] **Step 2: Run them to verify they fail**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_retrieval.py -q`
Expected: FAIL with `ImportError: cannot import name 'retrieve'`

- [ ] **Step 3: Implement** (append to `retrieval.py`; add imports `from vigilex_core.embeddings import Embedder` and `from vigilex_core.rerank import Reranker`)

```python
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
```

- [ ] **Step 4: Run tests**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -q && uv run pyright`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/rag/src/vigilex_rag/retrieval.py tests/integration/test_retrieval.py
git commit -m "feat: retrieve() with hybrid fusion and optional rerank

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Answering with checked citations, `vigilex ask`

**Files:**
- Create: `packages/rag/src/vigilex_rag/answer.py`
- Modify: `packages/rag/src/vigilex_rag/cli.py`
- Test: `tests/unit/test_answer.py`, `tests/integration/test_answer.py`

**Interfaces:**
- Consumes: `retrieve`, `ArticleHit`, `RetrieverConfig` (Tasks 2–3); `Chat`, `make_chat`, `make_reranker` (Task 1).
- Produces (in `vigilex_rag.answer`):
  - `Answer` frozen dataclass: `text: str`, `citations: list[int]`, `invalid_citations: list[int]`, `hits: list[ArticleHit]`
  - `NOT_FOUND: str`, `SYSTEM_PROMPT: str`, `MAX_ANSWER_TOKENS = 1024`, `PROMPT_DOC_CHARS = 4000`
  - `build_prompt(query: str, hits: list[ArticleHit]) -> str`
  - `check_citations(text: str, hits: list[ArticleHit]) -> tuple[list[int], list[int]]` — (valid, invalid), each in first-cited order, deduplicated
  - `answer(conn, query, cfg, embedder, chat, reranker=None) -> Answer`
  - CLI: `vigilex ask "<question>" [--mode bm25|vector|hybrid] [--rerank]`

- [ ] **Step 1: Write the failing unit tests** — `tests/unit/test_answer.py`

```python
from vigilex_rag.answer import build_prompt, check_citations
from vigilex_rag.retrieval import ArticleHit


def hit(i: int, text: str = "testo") -> ArticleHit:
    return ArticleHit(i, 1.0, "act", f"article_{i}", text)


def test_citations_outside_hits_are_invalid():
    text = "Vale l'art. [art:2] e anche [art:99]; ribadito in [art:2] e [art:3]."
    valid, invalid = check_citations(text, [hit(2), hit(3)])
    assert valid == [2, 3]
    assert invalid == [99]


def test_no_citations():
    assert check_citations("Nessuna fonte.", [hit(1)]) == ([], [])


def test_prompt_tags_every_article_and_truncates_long_text():
    prompt = build_prompt("Chi verifica?", [hit(1, "breve"), hit(2, "x" * 10_000)])
    assert "[art:1]" in prompt and "[art:2]" in prompt
    assert "x" * 4000 in prompt and "x" * 4001 not in prompt
    assert prompt.rstrip().endswith("Domanda: Chi verifica?")
```

- [ ] **Step 2: Write the failing integration tests** — `tests/integration/test_answer.py`

```python
from helpers import article
from vigilex_core.embeddings import FakeEmbedder
from vigilex_core.llm import FakeChat
from vigilex_rag.answer import NOT_FOUND, answer
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


class RecordingChat(FakeChat):
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return super().complete(system, user, max_tokens)


def test_answer_cites_retrieved_article(conn, tmp_path, write_release):
    ingest(conn, load_release(write_release(tmp_path, [article(1, "adeguata verifica")])), FakeEmbedder())
    a = answer(conn, "adeguata verifica?", RetrieverConfig(), FakeEmbedder(), FakeChat())
    assert a.citations == [1] and a.invalid_citations == []
    assert [h.id for h in a.hits] == [1]


def test_answer_without_hits_skips_llm(conn, tmp_path, write_release):
    ingest(conn, load_release(write_release(tmp_path, [article(1, "adeguata verifica")])), FakeEmbedder())
    chat = RecordingChat()
    a = answer(conn, "di il la", RetrieverConfig(mode="bm25"), FakeEmbedder(), chat)
    assert a.text == NOT_FOUND and a.hits == [] and chat.calls == 0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_answer.py -q; VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_answer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.answer'`

- [ ] **Step 4: Implement** — `packages/rag/src/vigilex_rag/answer.py`

```python
import logging
import re
from dataclasses import dataclass

import psycopg

from vigilex_core.embeddings import Embedder
from vigilex_core.llm import Chat
from vigilex_core.rerank import Reranker
from vigilex_rag.retrieval import ArticleHit, RetrieverConfig, retrieve

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[art:(\d+)\]")
MAX_ANSWER_TOKENS = 1024
PROMPT_DOC_CHARS = 4000
NOT_FOUND = "Non ho trovato fonti pertinenti per rispondere a questa domanda."
SYSTEM_PROMPT = (
    "Sei un assistente per la compliance antiriciclaggio e privacy (diritto italiano ed europeo). "
    "Rispondi in italiano usando SOLO gli articoli forniti. "
    "Dopo ogni affermazione cita l'articolo da cui proviene con il suo tag, ad esempio [art:123]. "
    "Se gli articoli non bastano per rispondere, dillo esplicitamente. "
    "Non citare articoli che non sono nell'elenco. "
    "Le risposte sono informative e non costituiscono consulenza legale."
)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[int]
    invalid_citations: list[int]
    hits: list[ArticleHit]


def build_prompt(query: str, hits: list[ArticleHit]) -> str:
    docs = "\n\n".join(
        f"[art:{h.id}] ({h.source_act} — {h.path})\n{h.text[:PROMPT_DOC_CHARS]}" for h in hits
    )
    return f"Articoli:\n\n{docs}\n\nDomanda: {query}\n"


def check_citations(text: str, hits: list[ArticleHit]) -> tuple[list[int], list[int]]:
    allowed = {h.id for h in hits}
    cited = list(dict.fromkeys(int(m) for m in CITATION.findall(text)))
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def answer(
    conn: psycopg.Connection,
    query: str,
    cfg: RetrieverConfig,
    embedder: Embedder,
    chat: Chat,
    reranker: Reranker | None = None,
) -> Answer:
    hits = retrieve(conn, query, cfg, embedder, reranker)
    if not hits:
        return Answer(NOT_FOUND, [], [], [])
    text = chat.complete(SYSTEM_PROMPT, build_prompt(query, hits), MAX_ANSWER_TOKENS)
    valid, invalid = check_citations(text, hits)
    if invalid:
        log.warning("answer cited articles that were not retrieved: %s", invalid)
    return Answer(text, valid, invalid, hits)
```

Add to `cli.py` (parser and dispatch; imports `from vigilex_core.llm import make_chat`, `from vigilex_core.rerank import make_reranker`, `from vigilex_rag.answer import answer`, `from vigilex_rag.retrieval import RetrieverConfig`):
```python
    a = sub.add_parser("ask", help="answer a question with cited articles")
    a.add_argument("question")
    a.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    a.add_argument("--rerank", action="store_true")
```
```python
    elif args.cmd == "ask":
        cfg = RetrieverConfig(mode=args.mode, rerank=args.rerank, embed_model_id=settings.embed_model_id)
        with connect(settings.database_url) as conn:
            result = answer(
                conn, args.question, cfg, make_embedder(settings, conn),
                make_chat(settings), make_reranker(settings),
            )
        print(result.text, "\n")
        for h in result.hits:
            mark = "*" if h.id in result.citations else " "
            print(f"{mark} [art:{h.id}] {h.score:.4f} {h.source_act} {h.path}")
        if result.invalid_citations:
            print("invalid citations (not retrieved):", result.invalid_citations)
```

- [ ] **Step 5: Run tests and a smoke run**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -q && uv run pyright`
Expected: all pass.

Run (compose DB seeded by `make seed`): `VIGILEX_LLM=fake uv run vigilex ask "Quali sono gli obblighi di adeguata verifica della clientela?"`
Expected: `Risposta di prova [art:N]` followed by 10 hit lines, the first marked `*`.

- [ ] **Step 6: Commit**

```bash
git add packages/rag/src/vigilex_rag/answer.py packages/rag/src/vigilex_rag/cli.py tests/unit/test_answer.py tests/integration/test_answer.py
git commit -m "feat: cited answers with citation post-check and vigilex ask

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: IR metrics

**Files:**
- Create: `packages/rag/src/vigilex_rag/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces (in `vigilex_rag.metrics`):
  - `recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float`
  - `ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float` (binary gains)
  - `reciprocal_rank(ranked: list[int], relevant: set[int]) -> float`
  - `METRICS = ("recall@5", "recall@10", "recall@20", "ndcg@10", "mrr")`
  - `query_metrics(ranked, relevant) -> dict[str, float]` (keys = `METRICS`)
  - `mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]` (keys = `METRICS`, ignores other keys)

- [ ] **Step 1: Write the failing test** — `tests/unit/test_metrics.py`

```python
import math

import pytest
from vigilex_rag.metrics import (
    METRICS,
    mean_metrics,
    ndcg_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 2) == pytest.approx(1 / 3)
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == pytest.approx(2 / 3)
    assert recall_at_k([], {1}, 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 6, 7], {7}) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 6, 7], {1}) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 1 and 3, two relevant docs in total
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([1, 9, 2], {1, 2}, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([1, 2], {1, 2}, 10) == pytest.approx(1.0)
    assert ndcg_at_k([9, 8], {1}, 10) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # 5 relevant docs, k=2, both top-2 relevant: perfect score
    assert ndcg_at_k([1, 2, 9], {1, 2, 3, 4, 5}, 2) == pytest.approx(1.0)


def test_query_and_mean_metrics():
    row = query_metrics([1, 2, 3], {3})
    assert set(row) == set(METRICS)
    assert row["mrr"] == pytest.approx(1 / 3)
    mean = mean_metrics([{"mrr": 1.0, "recall@5": 1.0, "recall@10": 1.0, "recall@20": 1.0,
                          "ndcg@10": 1.0, "retrieved": [1]},
                         {"mrr": 0.0, "recall@5": 0.0, "recall@10": 0.5, "recall@20": 0.5,
                          "ndcg@10": 0.0, "retrieved": [2]}])
    assert mean["mrr"] == pytest.approx(0.5) and mean["recall@10"] == pytest.approx(0.75)
    assert set(mean) == set(METRICS)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.metrics'`

- [ ] **Step 3: Implement** — `packages/rag/src/vigilex_rag/metrics.py`

```python
"""Binary-relevance IR metrics over one ranked list of article ids."""

import math
from statistics import fmean

METRICS = ("recall@5", "recall@10", "recall@20", "ndcg@10", "mrr")


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    return next((1 / i for i, a in enumerate(ranked, start=1) if a in relevant), 0.0)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, a in enumerate(ranked[:k], start=1) if a in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def query_metrics(ranked: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(ranked, relevant, 5),
        "recall@10": recall_at_k(ranked, relevant, 10),
        "recall@20": recall_at_k(ranked, relevant, 20),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
        "mrr": reciprocal_rank(ranked, relevant),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {m: fmean(r[m] for r in rows) for m in METRICS}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_metrics.py -q && uv run pyright`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/rag/src/vigilex_rag/metrics.py tests/unit/test_metrics.py
git commit -m "feat: recall, nDCG and MRR metrics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Evaluation runner, `eval_runs`, `vigilex eval`

**Files:**
- Create: `migrations/0002_eval_runs.sql`, `packages/rag/src/vigilex_rag/evaluation.py`
- Modify: `packages/rag/src/vigilex_rag/cli.py`, `tests/integration/conftest.py` (`TABLES` += `eval_runs`)
- Test: `tests/unit/test_evaluation.py`, `tests/integration/test_evaluation.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces (in `vigilex_rag.evaluation`):
  - `EVAL_DEPTH = 20`; `CONFIG_NAMES = ("bm25", "vector", "hybrid", "hybrid+rerank")`
  - `EvalRun` dataclass: `config: RetrieverConfig`, `split: str`, `metrics: dict[str, float]`, `per_query: dict[str, dict]`
  - `parse_configs(spec: str, embed_model_id: str, rerank_available: bool) -> list[RetrieverConfig]` — comma-separated names from `CONFIG_NAMES`; unknown → `ValueError`; `+rerank` configs dropped with a warning when `rerank_available` is false
  - `evaluate(conn, cfg, embedder, reranker, split: str = "validation") -> EvalRun` — `ValueError` when the split has no questions
  - `save_run(conn, run: EvalRun, git_sha: str | None) -> int` (commits, returns id)
  - `report(runs: list[EvalRun]) -> str` (markdown table)
  - CLI `vigilex eval [--configs bm25,vector,hybrid,hybrid+rerank] [--split validation] [--min-recall-at-10 X] [--report FILE]` — gate applies to the `hybrid` run; exit code 1 when below.

- [ ] **Step 1: Write the failing unit tests** — `tests/unit/test_evaluation.py`

```python
import pytest
from vigilex_rag.evaluation import EvalRun, parse_configs, report
from vigilex_rag.retrieval import RetrieverConfig


def test_parse_configs():
    cfgs = parse_configs("bm25,hybrid+rerank", "e", rerank_available=True)
    assert [c.name() for c in cfgs] == ["bm25", "hybrid+rerank"]
    assert all(c.embed_model_id == "e" for c in cfgs)


def test_parse_configs_skips_rerank_without_reranker():
    cfgs = parse_configs("hybrid,hybrid+rerank", "e", rerank_available=False)
    assert [c.name() for c in cfgs] == ["hybrid"]


def test_parse_configs_rejects_unknown():
    with pytest.raises(ValueError, match="bogus"):
        parse_configs("hybrid,bogus", "e", rerank_available=True)


def test_report_has_one_row_per_run():
    m = {"recall@5": 0.5, "recall@10": 0.6, "recall@20": 0.7, "ndcg@10": 0.4, "mrr": 0.3}
    md = report([EvalRun(RetrieverConfig(mode="bm25"), "validation", m, {"q": {}}),
                 EvalRun(RetrieverConfig(), "validation", m, {"q": {}})])
    lines = md.strip().splitlines()
    assert lines[0].startswith("| config |") and len(lines) == 4
    assert "| bm25 |" in lines[2] and "0.600" in lines[2]
```

- [ ] **Step 2: Write the failing integration tests** — `tests/integration/test_evaluation.py`

```python
import pytest
from helpers import article, question
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.evaluation import evaluate, save_run
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


def small(conn, tmp_path, write_release):
    release = write_release(
        tmp_path,
        [article(1, "adeguata verifica della clientela"), article(2, "segnalazione sospetta")],
        [question("adeguata verifica", [1]), question("segnalazione", [2])],
    )
    ingest(conn, load_release(release), FakeEmbedder())


def test_eval_bm25_perfect_on_trivial_data(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    run = evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None)
    assert run.metrics["recall@10"] == 1.0 and run.metrics["mrr"] == 1.0
    assert set(run.per_query) == {"validation-0", "validation-1"}
    assert run.per_query["validation-0"]["relevant"] == [1]
    assert run.config == RetrieverConfig(mode="bm25")  # stored config is not the depth-20 one


def test_save_run_round_trips(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    run = evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None)
    run_id = save_run(conn, run, "abc123")
    row = conn.execute(
        "SELECT config_name, config_hash, split, n_questions, metrics->>'mrr', git_sha "
        "FROM eval_runs WHERE id = %s", (run_id,)
    ).fetchone()
    assert row == ("bm25", run.config.config_hash(), "validation", 2, "1.0", "abc123")


def test_eval_empty_split_raises(conn, tmp_path, write_release):
    small(conn, tmp_path, write_release)
    with pytest.raises(ValueError, match="train"):
        evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None, split="train")
```

Add `eval_runs` to `TABLES` in `tests/integration/conftest.py`.

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation.py -q; VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_evaluation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.evaluation'` (the integration conftest also fails to truncate the missing `eval_runs` table until the migration exists).

- [ ] **Step 4: Implement**

`migrations/0002_eval_runs.sql`:
```sql
CREATE TABLE eval_runs (
    id                bigserial PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    git_sha           text,
    config_name       text NOT NULL,
    config_hash       text NOT NULL,
    retriever_config  jsonb NOT NULL,
    split             text NOT NULL,
    n_questions       int NOT NULL,
    metrics           jsonb NOT NULL,
    per_query         jsonb NOT NULL,
    judge             boolean NOT NULL DEFAULT false
);
CREATE INDEX eval_runs_config_hash_idx ON eval_runs (config_hash, created_at DESC);
```

`packages/rag/src/vigilex_rag/evaluation.py`:
```python
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
```

Add to `cli.py` (imports `import os`, `import subprocess`, `import sys`, `from vigilex_rag.evaluation import evaluate, parse_configs, report, save_run`):
```python
    e = sub.add_parser("eval", help="evaluate retrieval configs on the gold questions")
    e.add_argument("--configs", default="bm25,vector,hybrid,hybrid+rerank")
    e.add_argument("--split", default="validation")
    e.add_argument("--min-recall-at-10", type=float, help="fail if hybrid recall@10 is below")
    e.add_argument("--report", type=Path, help="append the markdown table to this file")
```
```python
    elif args.cmd == "eval":
        reranker = make_reranker(settings)
        try:
            configs = parse_configs(args.configs, settings.embed_model_id, reranker is not None)
        except ValueError as exc:
            p.error(str(exc))
        with connect(settings.database_url) as conn:
            embedder = make_embedder(settings, conn)
            runs = [evaluate(conn, c, embedder, reranker, args.split) for c in configs]
            for run in runs:
                save_run(conn, run, _git_sha())
        table = report(runs)
        print(table)
        if args.report:
            with args.report.open("a") as f:
                f.write("## Retrieval eval (" + args.split + ")\n\n" + table)
        if args.min_recall_at_10 is not None:
            hybrid = next((r for r in runs if r.config.name() == "hybrid"), None)
            if hybrid is None:
                sys.exit("--min-recall-at-10 needs the hybrid config in --configs")
            got = hybrid.metrics["recall@10"]
            if got < args.min_recall_at_10:
                sys.exit(f"hybrid recall@10 {got:.3f} < threshold {args.min_recall_at_10:.3f}")
```
and at module level:
```python
def _git_sha() -> str | None:
    if sha := os.environ.get("VIGILEX_GIT_SHA"):
        return sha
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
```

- [ ] **Step 5: Run tests**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -q && uv run pyright`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add migrations/0002_eval_runs.sql packages/rag tests
git commit -m "feat: retrieval evaluation runner, eval_runs table, vigilex eval

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Calibrate the CI gate, workflow, Makefile, docs, spec

**Files:**
- Modify: `Makefile`, `.github/workflows/ci.yml`, `README.md`, `CLAUDE.md`, `docs/superpowers/specs/2026-10-20-vigilex-umbrella-design.md`

- [ ] **Step 1: Measure on the seed (no Bedrock: fake mode, cached real query embeddings)**

Run: `make up && make seed && VIGILEX_LLM=fake uv run vigilex eval --configs bm25,vector,hybrid`
Expected: a 3-row table; vector and hybrid clearly above bm25 is the usual shape, but record whatever comes out. Set `GATE = floor((hybrid recall@10 − 0.05) × 100) / 100` (a 5-point margin for HNSW's non-deterministic graph build).

- [ ] **Step 2: Makefile** (append)

```makefile
GATE ?= <value from step 1>

eval:
	uv run vigilex eval

eval-ci:
	VIGILEX_LLM=fake uv run vigilex eval --configs bm25,vector,hybrid --min-recall-at-10 $(GATE)

# Optional, needs AWS: embeds the full corpus (~36k chunks, ≈$1.5) into the local DB.
ingest-full: download migrate
	uv run vigilex ingest --source data/raw

.PHONY: eval eval-ci ingest-full
```

- [ ] **Step 3: CI** — append to `.github/workflows/ci.yml` steps:

```yaml
      - run: uv run vigilex migrate && uv run vigilex load-seed
      - run: make eval-ci
        env:
          VIGILEX_GIT_SHA: ${{ github.sha }}
      - run: VIGILEX_LLM=fake uv run vigilex eval --configs bm25,vector,hybrid --report "$GITHUB_STEP_SUMMARY" > /dev/null
```
Ruling to make if it reads better: fold the summary into `eval-ci` via `--report "$${GITHUB_STEP_SUMMARY:-/dev/null}"` so the eval runs once.

- [ ] **Step 4: Verify**

Run: `make lint && make test && make eval-ci`
Expected: all green; `eval-ci` prints the table and exits 0.

- [ ] **Step 5: Docs and spec**

- README "Local development": add `make eval` (needs AWS for rerank), `make eval-ci` (no AWS), `VIGILEX_LLM=fake uv run vigilex ask "..."`, and the seed eval table from step 1.
- CLAUDE.md: add `vigilex ask` / `vigilex eval` to Commands; add `retrieval.py`, `answer.py`, `metrics.py`, `evaluation.py` and `vigilex_core.llm` / `rerank` to Architecture.
- Spec: apply the "Spec deltas" above (§3 LLM line, §4.1 keyword/vector details and rerank transport, §4.3 metrics library and fake-mode rule, §13 model access finding).

- [ ] **Step 6: Commit**

```bash
git add Makefile .github README.md CLAUDE.md docs/superpowers/specs
git commit -m "chore: eval gate in CI, eval make targets, docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7 (optional, human, needs AWS): the four-way comparison**

`make eval` on the seed with real Bedrock (rerank ≈ $0.10) gives the hybrid+rerank row. `make ingest-full` then `make eval` gives numbers comparable with the paper's BM25 baseline. Paste the tables into the PR description.

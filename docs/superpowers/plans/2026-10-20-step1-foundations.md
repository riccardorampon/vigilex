# Step 1 — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A uv-workspace monorepo where `docker compose up -d db && make migrate && make seed` gives a local Postgres holding a JuriFindIT subset with real embeddings, with lint, type-check, unit and integration tests running in GitHub Actions.

**Architecture:** Two workspace packages: `vigilex_core` (settings, DB connection, SQL migration runner, embedders) and `vigilex_rag` (chunking, release loader, ingestion, seed subset, `vigilex` CLI). Embeddings always go through a Postgres-backed cache, so the committed seed subset loads with zero Bedrock calls, locally and in CI.

**Tech Stack:** Python 3.12, uv workspace, hatchling, psycopg 3, pgvector, numpy, pyarrow, boto3, pytest, ruff, pyright, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-20-vigilex-umbrella-design.md` (step 1 of §12).

## Global Constraints

- Python `>=3.12,<3.13`; every package is a uv workspace member under `packages/`.
- Region `eu-central-1`; embedding model id default `cohere.embed-multilingual-v3`; embedding dimension 1024.
- Database: Postgres 16 + pgvector (`pgvector/pgvector:pg16` locally and in CI).
- Tests never call Bedrock. `VIGILEX_LLM=fake` selects fakes; it is the only test-specific setting.
- Dataset: `jurifindit/JuriFindIT` on Hugging Face, pinned revision `d765a61998a0dc962a9c06394ab546b5a5ee0aa5`, license CC BY-NC-SA 4.0.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Spec deltas (found by inspecting the real dataset; the spec is updated in Task 1)

- The data has no article number/title/domain: `articles` holds `id`, `source_act` (= `file_name`), `path`, `text`. Domain lives on questions (`normative_area`).
- Expert questions have no id: id is `"{split}-{row_index}"` in the original release file (the seed keeps the original id).
- Only expert questions are ingested (synthetic ones are unused until a later step); `questions` has `split` and `topics`, no `kind`.
- 159 rows with content `<< file root node >>` are document roots, not articles; they are skipped and references to them dropped.
- Bedrock's Cohere embed accepts at most 2048 characters per text: chunking is by characters (max 1800, overlap 200), hard-splitting any single token longer than that (the corpus has a 1.1M-character table article).
- Cache key is `sha256(model_id \0 input_type \0 text)`: query and document embeddings of the same text differ.
- Migrations: ~20-line plain-SQL runner instead of Alembic (no SQLAlchemy in the stack; psycopg is used directly). `eval_runs` and `session_quota` arrive with steps 2 and 3 as new migration files.
- Seed subset: the 37 `validation`-split AML + privacy questions, their 432 gold articles, up to 500 sampled one-hop neighbors, up to 1000 sampled distractors from the same acts (≈1.9k articles).
- `docker-compose.yml` has only `db` in this step; the API service arrives with step 3.

## Review Focus

1. **Giant article** (1.1M chars, one token > 2047 chars): chunker hard-splits the token, ingestion succeeds. Tests: Task 3 `test_giant_token_is_split`, Task 6 `test_ingest_survives_giant_token`.
2. **Fake vectors leaking into the real cache:** running with `VIGILEX_LLM=fake` must never write to `embedding_cache`. Test: Task 4 `test_store_false_never_writes`.
3. **Re-running ingestion** (after a crash or twice): no duplicate rows, same counts. Test: Task 6 `test_ingest_is_idempotent`.
4. **Article text changed between runs:** stale chunks (higher `ord`) disappear. Test: Task 6 `test_reingest_replaces_stale_chunks`.
5. **Relevant ids pointing at missing/skipped articles:** dropped from qrels, and a question left with none is skipped instead of failing on the FK. Test: Task 5 `test_dangling_relevant_ids`.

---

## File map

```
pyproject.toml                         # workspace root, dev tools, ruff/pyright/pytest config
packages/core/pyproject.toml
packages/core/src/vigilex_core/__init__.py
packages/core/src/vigilex_core/settings.py      # Settings.from_env()
packages/core/src/vigilex_core/db.py            # connect() with pgvector registered
packages/core/src/vigilex_core/migrate.py       # plain-SQL migration runner
packages/core/src/vigilex_core/embeddings.py    # Bedrock/Fake/Cached embedders, make_embedder
packages/rag/pyproject.toml
packages/rag/src/vigilex_rag/__init__.py
packages/rag/src/vigilex_rag/chunking.py        # chunk_text()
packages/rag/src/vigilex_rag/release.py         # load_release() → Release
packages/rag/src/vigilex_rag/ingest.py          # ingest()
packages/rag/src/vigilex_rag/seed.py            # select_seed(), write_seed(), load_seed()
packages/rag/src/vigilex_rag/cli.py             # `vigilex` command
migrations/0001_init.sql
docker-compose.yml
Makefile
.pre-commit-config.yaml
.github/workflows/ci.yml
data/LICENSE, data/README.md, data/seed/{corpus,validation,embeddings}.parquet
tests/helpers.py, tests/conftest.py             # JuriFindIT-shaped parquet writer + fixture
tests/unit/test_settings.py, test_chunking.py, test_embeddings.py, test_release.py, test_seed.py
tests/integration/conftest.py, test_migrate.py, test_cache.py, test_ingest.py, test_seed_data.py
```

---

### Task 1: Workspace skeleton, tooling, settings

**Files:**
- Create: `pyproject.toml`, `packages/core/pyproject.toml`, `packages/core/src/vigilex_core/__init__.py`, `packages/core/src/vigilex_core/settings.py`, `packages/rag/pyproject.toml`, `packages/rag/src/vigilex_rag/__init__.py`, `.pre-commit-config.yaml`
- Modify: `docs/superpowers/specs/2026-10-20-vigilex-umbrella-design.md` (apply the spec deltas above to §2.1, §2.2, §2.3)
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `vigilex_core.settings.Settings` (frozen dataclass: `database_url: str`, `llm: Literal["bedrock","fake"]`, `aws_region: str`, `embed_model_id: str`) and `Settings.from_env() -> Settings`.

- [ ] **Step 1: Create workspace files**

`pyproject.toml`:
```toml
[project]
name = "vigilex"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["vigilex-core", "vigilex-rag"]

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
vigilex-core = { workspace = true }
vigilex-rag = { workspace = true }

[dependency-groups]
dev = ["pytest>=8.3", "ruff>=0.7", "pyright>=1.1.390", "pre-commit>=4.0"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
include = ["packages"]
pythonVersion = "3.12"
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`packages/core/pyproject.toml`:
```toml
[project]
name = "vigilex-core"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["psycopg[binary]>=3.2", "pgvector>=0.3.6", "numpy>=2.0", "boto3>=1.35"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vigilex_core"]
```

`packages/rag/pyproject.toml`:
```toml
[project]
name = "vigilex-rag"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["vigilex-core", "pyarrow>=17"]

[project.scripts]
vigilex = "vigilex_rag.cli:main"

[tool.uv.sources]
vigilex-core = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vigilex_rag"]
```

Both `__init__.py` files are empty.

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Note: `vigilex_rag.cli` does not exist until Task 6; the script entry is harmless until then.

- [ ] **Step 2: Write the failing test** — `tests/unit/test_settings.py`

```python
import pytest

from vigilex_core.settings import Settings


def test_defaults(monkeypatch):
    for k in ("VIGILEX_DATABASE_URL", "VIGILEX_LLM", "VIGILEX_AWS_REGION", "VIGILEX_EMBED_MODEL_ID"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.database_url == "postgresql://vigilex:vigilex@localhost:5432/vigilex"
    assert s.llm == "bedrock"
    assert s.aws_region == "eu-central-1"
    assert s.embed_model_id == "cohere.embed-multilingual-v3"


def test_invalid_llm_rejected(monkeypatch):
    monkeypatch.setenv("VIGILEX_LLM", "openai")
    with pytest.raises(ValueError, match="VIGILEX_LLM"):
        Settings.from_env()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv sync && uv run pytest tests/unit/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_core.settings'`

- [ ] **Step 4: Implement** — `packages/core/src/vigilex_core/settings.py`

```python
import os
from dataclasses import dataclass
from typing import Literal, cast

LLM_MODES = ("bedrock", "fake")


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm: Literal["bedrock", "fake"]
    aws_region: str
    embed_model_id: str

    @classmethod
    def from_env(cls) -> "Settings":
        llm = os.environ.get("VIGILEX_LLM", "bedrock")
        if llm not in LLM_MODES:
            raise ValueError(f"VIGILEX_LLM must be one of {LLM_MODES}, got {llm!r}")
        return cls(
            database_url=os.environ.get(
                "VIGILEX_DATABASE_URL", "postgresql://vigilex:vigilex@localhost:5432/vigilex"
            ),
            llm=cast(Literal["bedrock", "fake"], llm),
            aws_region=os.environ.get("VIGILEX_AWS_REGION", "eu-central-1"),
            embed_model_id=os.environ.get("VIGILEX_EMBED_MODEL_ID", "cohere.embed-multilingual-v3"),
        )
```

- [ ] **Step 5: Run tests and tooling**

Run: `uv run pytest tests/unit -v && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: 2 passed; ruff and pyright report no errors.

- [ ] **Step 6: Update the spec** with the "Spec deltas" list above (rewrite the §2.1 table rows for `articles`, `questions`; replace "Alembic with plain SQL" with the plain-SQL runner; §2.2 chunking by characters and cache key; §2.3 seed numbers).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock packages .pre-commit-config.yaml tests/unit/test_settings.py docs/superpowers/specs
git commit -m "feat: uv workspace skeleton and settings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Local Postgres, schema, migration runner

**Files:**
- Create: `docker-compose.yml`, `migrations/0001_init.sql`, `packages/core/src/vigilex_core/db.py`, `packages/core/src/vigilex_core/migrate.py`, `tests/integration/conftest.py`
- Test: `tests/integration/test_migrate.py`

**Interfaces:**
- Produces: `vigilex_core.db.connect(url: str) -> psycopg.Connection` (pgvector types registered; requires migrations applied); `vigilex_core.migrate.migrate(url: str, migrations_dir: Path) -> list[str]` (returns versions applied this call); pytest fixtures `db_url` (session) and `conn` (per test, tables truncated).

- [ ] **Step 1: Compose file** — `docker-compose.yml`

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: vigilex
      POSTGRES_PASSWORD: vigilex
      POSTGRES_DB: vigilex
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U vigilex"]
      interval: 2s
      retries: 15

volumes:
  pgdata: {}
```

Run: `docker compose up -d --wait db`
Expected: `Container vigilex-db-1 Healthy`

- [ ] **Step 2: Integration fixtures** — `tests/integration/conftest.py`

```python
import os
from pathlib import Path

import psycopg
import pytest

from vigilex_core.db import connect
from vigilex_core.migrate import migrate

REPO = Path(__file__).resolve().parents[2]
TEST_DB = "vigilex_test"
TABLES = "articles, article_refs, article_chunks, questions, qrels, embedding_cache"


@pytest.fixture(scope="session")
def db_url() -> str:
    server = os.environ.get("VIGILEX_TEST_DATABASE_URL")
    if not server:
        pytest.skip("VIGILEX_TEST_DATABASE_URL not set")
    with psycopg.connect(server, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {TEST_DB}")
    url = server.rsplit("/", 1)[0] + f"/{TEST_DB}"
    migrate(url, REPO / "migrations")
    return url


@pytest.fixture
def conn(db_url):
    with connect(db_url) as c:
        c.execute(f"TRUNCATE {TABLES} CASCADE")
        c.commit()
        yield c
```

- [ ] **Step 3: Write the failing test** — `tests/integration/test_migrate.py`

```python
from pathlib import Path

from vigilex_core.migrate import migrate

REPO = Path(__file__).resolve().parents[2]


def test_migrate_is_idempotent(db_url):
    assert migrate(db_url, REPO / "migrations") == []


def test_italian_stemming_matches_inflections(conn):
    conn.execute(
        "INSERT INTO articles (id, source_act, path, text) VALUES (1, 'act', 'article_1', %s)",
        ("Gli obblighi di adeguata verifica della clientela",),
    )
    hit = conn.execute(
        "SELECT id FROM articles WHERE tsv @@ websearch_to_tsquery('italian', 'obbligo verifiche')"
    ).fetchall()
    assert hit == [(1,)]


def test_vector_index_exists(conn):
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'article_chunks' AND indexdef LIKE '%hnsw%'"
    ).fetchall()
    assert len(idx) == 1
```

- [ ] **Step 4: Run it to verify it fails**

Run: `VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_migrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_core.db'`

- [ ] **Step 5: Schema** — `migrations/0001_init.sql`

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE articles (
    id          bigint PRIMARY KEY,
    source_act  text NOT NULL,
    path        text NOT NULL,
    text        text NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('italian', text)) STORED
);
CREATE INDEX articles_tsv_idx ON articles USING gin (tsv);

CREATE TABLE article_chunks (
    article_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    ord         int NOT NULL,
    text        text NOT NULL,
    embedding   vector(1024) NOT NULL,
    PRIMARY KEY (article_id, ord)
);
CREATE INDEX article_chunks_embedding_idx ON article_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE article_refs (
    src_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    dst_id  bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    PRIMARY KEY (src_id, dst_id)
);
CREATE INDEX article_refs_dst_idx ON article_refs (dst_id);

CREATE TABLE questions (
    id      text PRIMARY KEY,
    split   text NOT NULL,
    text    text NOT NULL,
    domain  text NOT NULL,
    topics  text NOT NULL
);

CREATE TABLE qrels (
    question_id  text NOT NULL REFERENCES questions (id) ON DELETE CASCADE,
    article_id   bigint NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    PRIMARY KEY (question_id, article_id)
);

CREATE TABLE embedding_cache (
    key        text PRIMARY KEY,
    embedding  vector(1024) NOT NULL
);
```

- [ ] **Step 6: Runner and connection**

`packages/core/src/vigilex_core/migrate.py`:
```python
from pathlib import Path

import psycopg


def migrate(url: str, migrations_dir: Path) -> list[str]:
    """Apply *.sql files in name order, each in its own transaction, once."""
    applied: list[str] = []
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.stem in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text())  # type: ignore[arg-type]  # file content, not user input
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,))
            applied.append(path.stem)
    return applied
```

`packages/core/src/vigilex_core/db.py`:
```python
import psycopg
from pgvector.psycopg import register_vector


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn
```

- [ ] **Step 7: Run tests**

Run: `VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration -v && uv run pyright`
Expected: 3 passed; pyright clean.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml migrations packages/core tests/integration
git commit -m "feat: postgres schema, migration runner, integration fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Chunking

**Files:**
- Create: `packages/rag/src/vigilex_rag/chunking.py`
- Test: `tests/unit/test_chunking.py`

**Interfaces:**
- Produces: `vigilex_rag.chunking.chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]`; `MAX_CHARS = 1800`. Empty/whitespace text → `[]`. Every chunk ≤ `max_chars`. Whitespace is normalized to single spaces.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_chunking.py`

```python
from vigilex_rag.chunking import chunk_text


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("  \n\t ") == []


def test_short_text_is_one_normalized_chunk():
    assert chunk_text("Art. 1\nObblighi   di verifica") == ["Art. 1 Obblighi di verifica"]


def test_long_text_chunks_are_bounded_overlapping_and_complete():
    words = [f"parola{i}" for i in range(2000)]
    chunks = chunk_text(" ".join(words), max_chars=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-1] in b.split()  # consecutive chunks overlap
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_giant_token_is_split():
    chunks = chunk_text("inizio " + "x" * 5000 + " fine", max_chars=1800, overlap=200)
    assert all(len(c) <= 1800 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks).count("x") >= 5000
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.chunking'`

- [ ] **Step 3: Implement** — `packages/rag/src/vigilex_rag/chunking.py`

```python
# Bedrock's Cohere embed rejects texts over 2048 characters; keep a safety margin.
MAX_CHARS = 1800
OVERLAP = 200


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Greedy word-packing into chunks of at most max_chars, with ~overlap chars repeated."""
    words: list[str] = []
    for w in text.split():
        words.extend(w[i : i + max_chars] for i in range(0, len(w), max_chars))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, size = start + 1, len(words[start])
        while end < len(words) and size + 1 + len(words[end]) <= max_chars:
            size += 1 + len(words[end])
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        back, kept = end, 0
        while back - 1 > start and kept + 1 + len(words[back - 1]) <= overlap:
            back -= 1
            kept += 1 + len(words[back])
        start = back
    return chunks
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/rag/src/vigilex_rag/chunking.py tests/unit/test_chunking.py
git commit -m "feat: character-bounded chunking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Embedders and the embedding cache

**Files:**
- Create: `packages/core/src/vigilex_core/embeddings.py`
- Test: `tests/unit/test_embeddings.py`, `tests/integration/test_cache.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `connect` (Task 2), table `embedding_cache`.
- Produces (all in `vigilex_core.embeddings`):
  - `EMBED_DIM = 1024`; `InputType = Literal["search_document", "search_query"]`
  - `class Embedder(Protocol): def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]`
  - `BedrockEmbedder(client, model_id: str)` — batches of 96
  - `FakeEmbedder()` — deterministic, unit-norm, depends on `(input_type, text)`
  - `cache_key(model_id: str, input_type: InputType, text: str) -> str`
  - `CachedEmbedder(conn, model_id: str, inner: Embedder, store: bool)` — does not commit; caller commits
  - `bedrock_embedder(settings: Settings) -> BedrockEmbedder`
  - `make_embedder(settings: Settings, conn) -> CachedEmbedder` — fake mode → `store=False`

- [ ] **Step 1: Write the failing unit tests** — `tests/unit/test_embeddings.py`

```python
import io
import json
import math

from vigilex_core.embeddings import EMBED_DIM, BedrockEmbedder, FakeEmbedder


class StubBedrock:
    def __init__(self):
        self.calls: list[dict] = []

    def invoke_model(self, *, modelId, body, contentType, accept):
        req = json.loads(body)
        self.calls.append({"modelId": modelId, **req})
        vecs = [[float(len(t))] * EMBED_DIM for t in req["texts"]]
        return {"body": io.BytesIO(json.dumps({"embeddings": vecs}).encode())}


def test_bedrock_embedder_batches_by_96_and_keeps_order():
    stub = StubBedrock()
    texts = ["a" * (i % 7 + 1) for i in range(200)]
    out = BedrockEmbedder(stub, "cohere.embed-multilingual-v3").embed(texts, "search_document")
    assert [len(c["texts"]) for c in stub.calls] == [96, 96, 8]
    assert all(c["input_type"] == "search_document" and c["truncate"] == "END" for c in stub.calls)
    assert [v[0] for v in out] == [float(len(t)) for t in texts]


def test_bedrock_embedder_empty_input_makes_no_call():
    stub = StubBedrock()
    assert BedrockEmbedder(stub, "m").embed([], "search_query") == []
    assert stub.calls == []


def test_fake_embedder_is_deterministic_unit_norm_and_input_type_aware():
    f = FakeEmbedder()
    a, b = f.embed(["ciao", "ciao"], "search_query")
    (c,) = f.embed(["ciao"], "search_document")
    assert a == b and a != c
    assert len(a) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
```

- [ ] **Step 2: Write the failing integration tests** — `tests/integration/test_cache.py`

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_embeddings.py -v; VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_core.embeddings'`

- [ ] **Step 4: Implement** — `packages/core/src/vigilex_core/embeddings.py`

```python
import hashlib
import json
import math
import random
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import boto3
import numpy as np
import psycopg
from botocore.config import Config

from vigilex_core.settings import Settings

EMBED_DIM = 1024
InputType = Literal["search_document", "search_query"]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]: ...


class BedrockEmbedder:
    BATCH = 96  # Cohere embed limit per request

    def __init__(self, client: Any, model_id: str):
        self.client = client
        self.model_id = model_id

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            body = json.dumps(
                {"texts": list(texts[i : i + self.BATCH]), "input_type": input_type, "truncate": "END"}
            )
            resp = self.client.invoke_model(
                modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
            )
            out.extend(json.loads(resp["body"].read())["embeddings"])
        return out


class FakeEmbedder:
    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        return [_fake_vector(f"{input_type}\0{t}") for t in texts]


def _fake_vector(seed: str) -> list[float]:
    rng = random.Random(hashlib.sha256(seed.encode()).digest())
    v = [rng.gauss(0.0, 1.0) for _ in range(EMBED_DIM)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def cache_key(model_id: str, input_type: InputType, text: str) -> str:
    return hashlib.sha256(f"{model_id}\0{input_type}\0{text}".encode()).hexdigest()


class CachedEmbedder:
    """Postgres-backed cache in front of an embedder. store=False never writes (fake mode)."""

    def __init__(self, conn: psycopg.Connection, model_id: str, inner: Embedder, store: bool):
        self.conn = conn
        self.model_id = model_id
        self.inner = inner
        self.store = store

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        keys = [cache_key(self.model_id, input_type, t) for t in texts]
        rows = self.conn.execute(
            "SELECT key, embedding FROM embedding_cache WHERE key = ANY(%s)", (list(set(keys)),)
        ).fetchall()
        found: dict[str, list[float]] = {k: e.tolist() for k, e in rows}
        missing = {k: t for k, t in zip(keys, texts) if k not in found}
        if missing:
            fresh = dict(zip(missing, self.inner.embed(list(missing.values()), input_type)))
            if self.store:
                with self.conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO embedding_cache (key, embedding) VALUES (%s, %s) "
                        "ON CONFLICT DO NOTHING",
                        [(k, np.asarray(v, dtype=np.float32)) for k, v in fresh.items()],
                    )
            found.update(fresh)
        return [found[k] for k in keys]


def bedrock_embedder(settings: Settings) -> BedrockEmbedder:
    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )
    return BedrockEmbedder(client, settings.embed_model_id)


def make_embedder(settings: Settings, conn: psycopg.Connection) -> CachedEmbedder:
    if settings.llm == "fake":
        return CachedEmbedder(conn, settings.embed_model_id, FakeEmbedder(), store=False)
    return CachedEmbedder(conn, settings.embed_model_id, bedrock_embedder(settings), store=True)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit -v && VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration -v && uv run pyright`
Expected: all pass; pyright clean.

- [ ] **Step 6: Commit**

```bash
git add packages/core/src/vigilex_core/embeddings.py tests/unit/test_embeddings.py tests/integration/test_cache.py
git commit -m "feat: bedrock, fake and postgres-cached embedders

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Release loader

**Files:**
- Create: `packages/rag/src/vigilex_rag/release.py`, `tests/helpers.py`, `tests/conftest.py`
- Test: `tests/unit/test_release.py`

**Interfaces:**
- Produces (in `vigilex_rag.release`):
  - `Article(id: int, source_act: str, path: str, text: str)` frozen dataclass
  - `Question(id: str, split: str, text: str, domain: str, topics: str, relevant: tuple[int, ...])` frozen dataclass
  - `Release(articles: list[Article], refs: list[tuple[int, int]], questions: list[Question])` dataclass
  - `load_release(d: Path) -> Release` — reads `d/corpus.parquet` and, if present, `d/train.parquet`, `d/validation.parquet` (JuriFindIT column names; an optional `id` column on question files overrides the generated id)
  - `ROOT_PLACEHOLDER = "<< file root node >>"`
- Produces (in `tests/helpers.py`, importable as `helpers` from any test): `article(id, content, refs=None, act="act") -> dict`, `question(text, relevant, area="Privacy") -> dict`, `write_release(d: Path, corpus: list[dict], validation: list[dict] | None = None) -> Path` writing JuriFindIT-shaped parquet; `tests/conftest.py` exposes `write_release` as a fixture. (A plain module, not `conftest`, because `tests/integration/conftest.py` also exists and `import conftest` would be ambiguous.)

- [ ] **Step 1: Shared test helper** — `tests/helpers.py`

```python
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CORPUS_SCHEMA = pa.schema(
    [("id", pa.int64()), ("file_name", pa.string()), ("path", pa.string()),
     ("content", pa.string()), ("reference", pa.list_(pa.int64()))]
)
QUESTION_SCHEMA = pa.schema(
    [("question", pa.string()), ("topics", pa.string()), ("normative_area", pa.string()),
     ("relevant_doc_ids", pa.list_(pa.int64()))]
)


def article(id: int, content: str, refs: list[int] | None = None, act: str = "act") -> dict:
    return {"id": id, "file_name": act, "path": f"article_{id}", "content": content,
            "reference": refs or []}


def question(text: str, relevant: list[int], area: str = "Privacy") -> dict:
    return {"question": text, "topics": "t", "normative_area": area, "relevant_doc_ids": relevant}


def write_release(d: Path, corpus: list[dict], validation: list[dict] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(corpus, schema=CORPUS_SCHEMA), d / "corpus.parquet")
    if validation is not None:
        pq.write_table(
            pa.Table.from_pylist(validation, schema=QUESTION_SCHEMA), d / "validation.parquet"
        )
    return d
```

`tests/conftest.py`:
```python
import helpers
import pytest


@pytest.fixture
def write_release():
    return helpers.write_release
```

- [ ] **Step 2: Write the failing test** — `tests/unit/test_release.py`

```python
from helpers import article, question
from vigilex_rag.release import ROOT_PLACEHOLDER, load_release


def test_loads_articles_refs_and_questions(tmp_path, write_release):
    d = write_release(
        tmp_path,
        [article(0, ROOT_PLACEHOLDER), article(1, "Art. 1", refs=[2, 0]), article(2, "Art. 2")],
        [question("Chi verifica?", [1, 2])],
    )
    r = load_release(d)
    assert [a.id for a in r.articles] == [1, 2]  # root placeholder skipped
    assert r.refs == [(1, 2)]  # ref to skipped root dropped
    assert len(r.questions) == 1
    q = r.questions[0]
    assert (q.id, q.split, q.domain, q.relevant) == ("validation-0", "validation", "Privacy", (1, 2))


def test_dangling_relevant_ids(tmp_path, write_release):
    d = write_release(
        tmp_path,
        [article(1, "Art. 1")],
        [question("partly known", [1, 99]), question("unknown", [99])],
    )
    r = load_release(d)
    assert [(q.id, q.relevant) for q in r.questions] == [("validation-0", (1,))]


def test_missing_question_files_are_fine(tmp_path, write_release):
    assert load_release(write_release(tmp_path, [article(1, "x")])).questions == []
```

`tests/` is not a package, so add to root `pyproject.toml` under `[tool.pytest.ini_options]`: `pythonpath = ["tests"]` (makes `import helpers` work from unit and integration tests).

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_release.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.release'`

- [ ] **Step 4: Implement** — `packages/rag/src/vigilex_rag/release.py`

```python
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT_PLACEHOLDER = "<< file root node >>"
QUESTION_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class Article:
    id: int
    source_act: str
    path: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    text: str
    domain: str
    topics: str
    relevant: tuple[int, ...]


@dataclass
class Release:
    articles: list[Article]
    refs: list[tuple[int, int]]
    questions: list[Question]


def load_release(d: Path) -> Release:
    """Read a JuriFindIT-shaped directory: corpus.parquet [+ train/validation.parquet]."""
    rows = [
        r for r in pq.read_table(d / "corpus.parquet").to_pylist()
        if r["content"].strip() != ROOT_PLACEHOLDER
    ]
    ids = {r["id"] for r in rows}
    articles = [Article(r["id"], r["file_name"], r["path"], r["content"]) for r in rows]
    refs = sorted({(r["id"], dst) for r in rows for dst in r["reference"] or [] if dst in ids})

    questions: list[Question] = []
    for split in QUESTION_SPLITS:
        path = d / f"{split}.parquet"
        if not path.exists():
            continue
        for i, q in enumerate(pq.read_table(path).to_pylist()):
            relevant = tuple(x for x in q["relevant_doc_ids"] if x in ids)
            if not relevant:
                continue  # cannot be evaluated against this corpus
            questions.append(
                Question(
                    id=q.get("id") or f"{split}-{i}",
                    split=split,
                    text=q["question"],
                    domain=q["normative_area"],
                    topics=q["topics"] or "",
                    relevant=relevant,
                )
            )
    return Release(articles, refs, questions)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit -v && uv run pyright`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml packages/rag/src/vigilex_rag/release.py tests/helpers.py tests/conftest.py tests/unit/test_release.py
git commit -m "feat: JuriFindIT release loader

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Ingestion and the `vigilex` CLI

**Files:**
- Create: `packages/rag/src/vigilex_rag/ingest.py`, `packages/rag/src/vigilex_rag/cli.py`
- Test: `tests/integration/test_ingest.py`

**Interfaces:**
- Consumes: `Release`, `load_release` (Task 5), `chunk_text` (Task 3), `Embedder`, `make_embedder`, `bedrock_embedder` (Task 4), `connect`, `migrate` (Task 2), `Settings` (Task 1).
- Produces:
  - `vigilex_rag.ingest.IngestStats(articles: int, refs: int, questions: int, chunks: int)` dataclass
  - `vigilex_rag.ingest.ingest(conn, release: Release, embedder: Embedder) -> IngestStats` — idempotent, commits per batch of 200 articles
  - CLI `vigilex migrate [--dir migrations]`, `vigilex ingest --source DIR`, `vigilex check-bedrock`; `vigilex_rag.cli.main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing test** — `tests/integration/test_ingest.py`

```python
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
        "articles": 2, "article_refs": 1, "article_chunks": 2, "questions": 1, "qrels": 1
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
    assert conn.execute("SELECT count(*) FROM article_chunks WHERE article_id = 1").fetchone()[0] > 1
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.ingest'`

- [ ] **Step 3: Implement** — `packages/rag/src/vigilex_rag/ingest.py`

```python
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
                [(aid, j, c, np.asarray(v, dtype=np.float32)) for (aid, j, c), v in zip(rows, vectors)],
            )
        conn.commit()  # a crash loses at most one batch; embeddings already paid stay cached
        n_chunks += len(rows)
        log.info("ingested %d/%d articles", i + len(batch), len(release.articles))

    return IngestStats(len(release.articles), len(release.refs), len(release.questions), n_chunks)
```

`packages/rag/src/vigilex_rag/cli.py`:
```python
import argparse
import logging
from pathlib import Path

from vigilex_core.db import connect
from vigilex_core.embeddings import EMBED_DIM, bedrock_embedder, make_embedder
from vigilex_core.migrate import migrate
from vigilex_core.settings import Settings
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="vigilex")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="apply SQL migrations")
    m.add_argument("--dir", type=Path, default=Path("migrations"))
    i = sub.add_parser("ingest", help="ingest a JuriFindIT-shaped directory")
    i.add_argument("--source", type=Path, required=True)
    sub.add_parser("check-bedrock", help="embed one text to verify Bedrock access")
    args = p.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "migrate":
        print("applied:", migrate(settings.database_url, args.dir) or "nothing")
    elif args.cmd == "ingest":
        with connect(settings.database_url) as conn:
            print(ingest(conn, load_release(args.source), make_embedder(settings, conn)))
    elif args.cmd == "check-bedrock":
        (v,) = bedrock_embedder(settings).embed(["prova"], "search_query")
        assert len(v) == EMBED_DIM, f"expected {EMBED_DIM} dims, got {len(v)}"
        print(f"ok: {settings.embed_model_id} in {settings.aws_region} returned {len(v)} dims")
```

- [ ] **Step 4: Run tests**

Run: `VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -v && uv run pyright`
Expected: all pass.

- [ ] **Step 5: Smoke the CLI against compose**

Run: `uv run vigilex migrate && uv run vigilex migrate`
Expected: first prints `applied: ['0001_init']`, second `applied: nothing`.

- [ ] **Step 6: Commit**

```bash
git add packages/rag/src/vigilex_rag/ingest.py packages/rag/src/vigilex_rag/cli.py tests/integration/test_ingest.py
git commit -m "feat: idempotent ingestion and vigilex CLI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Seed subset (build, commit, load)

**Files:**
- Create: `packages/rag/src/vigilex_rag/seed.py`, `data/LICENSE`, `data/README.md`, `data/seed/*.parquet` (generated), `Makefile`
- Modify: `packages/rag/src/vigilex_rag/cli.py` (add `build-seed`, `load-seed`), `.gitignore` (add `data/raw/`)
- Test: `tests/unit/test_seed.py`, `tests/integration/test_seed_data.py`

**Interfaces:**
- Consumes: `Release`, `load_release` (Task 5), `chunk_text` (Task 3), `Embedder`, `cache_key` (Task 4), `ingest` (Task 6).
- Produces (in `vigilex_rag.seed`):
  - `select_seed(release: Release, rng: random.Random) -> Release`
  - `write_seed(release: Release, out: Path, embedder: Embedder, model_id: str) -> None` — writes `corpus.parquet`, `validation.parquet` (with `id` column), `embeddings.parquet` (`key: string`, `embedding: list<float16>`) covering every chunk (`search_document`) and question (`search_query`)
  - `load_seed(conn, seed_dir: Path, embedder: Embedder) -> IngestStats` — loads `embeddings.parquet` into `embedding_cache` then ingests
  - Constants `SEED_DOMAINS = ("Antiriciclaggio e Antiterrorismo", "Privacy")`, `HOP_SAMPLE = 500`, `DISTRACTOR_SAMPLE = 1000`, `DISTRACTOR_MAX_CHARS = 12000`

- [ ] **Step 1: Write the failing unit test** — `tests/unit/test_seed.py`

```python
import random

from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.release import Article, Question, Release, load_release
from vigilex_rag.seed import select_seed, write_seed


def make_release() -> Release:
    arts = [Article(i, "aml" if i < 50 else "civile", f"a{i}", f"testo {i}") for i in range(100)]
    qs = [
        Question("validation-0", "validation", "q aml", "Antiriciclaggio e Antiterrorismo", "t", (1,)),
        Question("validation-1", "validation", "q civ", "Codice civile", "t", (60,)),
        Question("train-0", "train", "q train", "Privacy", "t", (2,)),
    ]
    return Release(arts, [(1, 3), (3, 4), (60, 61)], qs)


def test_select_seed_keeps_gold_hops_and_same_act_distractors():
    seed = select_seed(make_release(), random.Random(0))
    ids = {a.id for a in seed.articles}
    assert [q.id for q in seed.questions] == ["validation-0"]  # validation AML/privacy only
    assert {1, 3} <= ids  # gold + one-hop neighbor
    assert all(a.source_act == "aml" for a in seed.articles)  # distractors from the gold acts
    assert all(s in ids and d in ids for s, d in seed.refs)


def test_write_seed_round_trips_and_covers_all_embeddings(tmp_path):
    seed = select_seed(make_release(), random.Random(0))
    write_seed(seed, tmp_path, FakeEmbedder(), "m")
    back = load_release(tmp_path)
    assert [a.id for a in back.articles] == [a.id for a in seed.articles]
    assert [q.id for q in back.questions] == ["validation-0"]
    import pyarrow.parquet as pq

    n = pq.read_table(tmp_path / "embeddings.parquet").num_rows
    assert n == len(seed.articles) + len(seed.questions)  # one short chunk per article + questions
```

- [ ] **Step 2: Write the failing integration test** — `tests/integration/test_seed_data.py`

```python
from pathlib import Path

import pytest
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.seed import load_seed

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "seed"


class NoCalls(FakeEmbedder):
    def embed(self, texts, input_type):
        if texts:
            raise AssertionError(f"seed must be fully cached, {len(texts)} misses")
        return []


def test_committed_seed_loads_without_embedding_calls(conn):
    if not (SEED / "embeddings.parquet").exists():
        pytest.fail("data/seed not built yet — run Task 7 step 6")
    from vigilex_core.embeddings import CachedEmbedder
    from vigilex_core.settings import Settings

    model = Settings.from_env().embed_model_id
    stats = load_seed(conn, Path(SEED), CachedEmbedder(conn, model, NoCalls(), store=False))
    assert stats.questions == 37
    assert 1500 <= stats.articles <= 2000
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_seed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_rag.seed'`

- [ ] **Step 4: Implement** — `packages/rag/src/vigilex_rag/seed.py`

```python
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
        a.id for a in release.articles
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
            [{"id": a.id, "file_name": a.source_act, "path": a.path, "content": a.text,
              "reference": refs[a.id]} for a in release.articles],
            schema=pa.schema([("id", pa.int64()), ("file_name", pa.string()), ("path", pa.string()),
                              ("content", pa.string()), ("reference", pa.list_(pa.int64()))]),
        ),
        out / "corpus.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": q.id, "question": q.text, "topics": q.topics, "normative_area": q.domain,
              "relevant_doc_ids": list(q.relevant)} for q in release.questions]
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
        pa.table({
            "key": pa.array(keys),
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), emb.shape[1]),
        }),
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
            list(zip(keys, emb)),
        )
    conn.commit()
    return ingest(conn, load_release(seed_dir), embedder)
```

Add to `cli.py` (subparsers and dispatch):
```python
    b = sub.add_parser("build-seed", help="build data/seed from a full release (calls Bedrock)")
    b.add_argument("--release", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/seed"))
    s = sub.add_parser("load-seed", help="load the committed seed subset (no Bedrock calls)")
    s.add_argument("--dir", type=Path, default=Path("data/seed"))
```
```python
    elif args.cmd == "build-seed":
        seed = select_seed(load_release(args.release), random.Random(0))
        write_seed(seed, args.out, bedrock_embedder(settings), settings.embed_model_id)
        print(f"seed: {len(seed.articles)} articles, {len(seed.questions)} questions -> {args.out}")
    elif args.cmd == "load-seed":
        with connect(settings.database_url) as conn:
            print(load_seed(conn, args.dir, make_embedder(settings, conn)))
```
with imports `import random` and `from vigilex_rag.seed import load_seed, select_seed, write_seed`.

- [ ] **Step 5: Run the unit tests**

Run: `uv run pytest tests/unit -v && uv run pyright`
Expected: all pass (`test_seed_data.py` is integration; it runs after step 6).

- [ ] **Step 6: HUMAN STEP — build the seed with real Bedrock** (one-off, costs cents)

`Makefile` (create now; Task 8 adds more targets):
```makefile
REV := d765a61998a0dc962a9c06394ab546b5a5ee0aa5
HF  := https://huggingface.co/datasets/jurifindit/JuriFindIT/resolve/$(REV)

data/raw/corpus.parquet:
	mkdir -p data/raw
	curl -fsSL -o data/raw/corpus.parquet $(HF)/corpus/corpus.parquet
	curl -fsSL -o data/raw/train.parquet $(HF)/questions/train.parquet
	curl -fsSL -o data/raw/validation.parquet $(HF)/questions/validation.parquet

download: data/raw/corpus.parquet

build-seed: download
	uv run vigilex build-seed --release data/raw --out data/seed

.PHONY: download build-seed
```

Add `data/raw/` to `.gitignore`.

Run (human, with AWS credentials):
```bash
aws login
uv run vigilex check-bedrock          # expect: ok: cohere.embed-multilingual-v3 in eu-central-1 returned 1024 dims
make build-seed                        # expect: seed: ~1900 articles, 37 questions -> data/seed
du -sh data/seed                       # expect: well under 50M
```
If `check-bedrock` fails with `AccessDeniedException`, enable the Cohere Embed Multilingual model in the Bedrock console (eu-central-1 → Model access) and retry. Also run, for spec §13:
```bash
aws bedrock list-foundation-models --region eu-central-1 --query "modelSummaries[?contains(modelId,'cohere')].modelId"
aws bedrock list-inference-profiles --region eu-central-1 --query "inferenceProfileSummaries[?contains(inferenceProfileId,'anthropic')].inferenceProfileId"
```
and paste the output into the PR description for step 2.

- [ ] **Step 7: License files**

`data/README.md`:
```markdown
# Data

`seed/` is a subset of **JuriFindIT** (Dalla Noce, Colla, Farhang Doust, De Mattei, Bacciu — EACL 2026 Findings),
https://huggingface.co/datasets/jurifindit/JuriFindIT, revision d765a61998a0dc962a9c06394ab546b5a5ee0aa5,
licensed **CC BY-NC-SA 4.0** (see `LICENSE`). Non-commercial use only.

Changes: subset selection (validation-split AML and privacy questions, their relevant articles, sampled
neighbors and distractors), added `id` column on questions, added Cohere Embed Multilingual v3 embeddings
(float16). These derived files are shared under the same license.

The code in this repository is licensed separately.
```
`data/LICENSE`: the CC BY-NC-SA 4.0 legal code, downloaded with
`curl -fsSL -o data/LICENSE https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt`.

- [ ] **Step 8: Run the integration test with the committed seed**

Run: `VIGILEX_TEST_DATABASE_URL=postgresql://vigilex:vigilex@localhost:5432/postgres VIGILEX_LLM=fake uv run pytest tests/integration -v`
Expected: all pass, including `test_committed_seed_loads_without_embedding_calls`.

- [ ] **Step 9: Commit**

```bash
git add packages/rag Makefile .gitignore data tests/unit/test_seed.py tests/integration/test_seed_data.py
git commit -m "feat: JuriFindIT seed subset with cached embeddings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Developer workflow, CI, docs

**Files:**
- Modify: `Makefile`, `README.md`, `CLAUDE.md`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: everything above.
- Produces: `make up | migrate | seed | test-unit | test-integration | test | lint | fmt`; CI workflow `ci`.

- [ ] **Step 1: Extend the Makefile** (append; keep Task 7 targets)

```makefile
export VIGILEX_TEST_DATABASE_URL ?= postgresql://vigilex:vigilex@localhost:5432/postgres

up:
	docker compose up -d --wait db

migrate:
	uv run vigilex migrate

seed: migrate
	uv run vigilex load-seed

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test: test-unit test-integration

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

fmt:
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: up migrate seed test-unit test-integration test lint fmt
```

- [ ] **Step 2: Verify the fresh-clone flow**

Run: `docker compose down -v && make up && make seed && make lint && VIGILEX_LLM=fake make test`
Expected: seed prints `IngestStats(articles=~1900, ..., questions=37, ...)` without any AWS credentials needed; lint clean; all tests pass.

- [ ] **Step 3: CI workflow** — `.github/workflows/ci.yml`

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  check:
    runs-on: ubuntu-24.04
    services:
      db:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: vigilex
          POSTGRES_PASSWORD: vigilex
          POSTGRES_DB: vigilex
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U vigilex"
          --health-interval 2s
          --health-retries 15
    env:
      VIGILEX_TEST_DATABASE_URL: postgresql://vigilex:vigilex@localhost:5432/postgres
      VIGILEX_LLM: fake
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest tests/unit
      - run: uv run pytest tests/integration
```

- [ ] **Step 4: Docs**

Append to `README.md`:
```markdown
## Local development

Requirements: Docker, [uv](https://docs.astral.sh/uv/).

    make up        # Postgres 16 + pgvector
    make seed      # migrations + JuriFindIT seed subset (no AWS needed)
    make test      # unit + integration (set VIGILEX_LLM=fake)
    make lint

Data: JuriFindIT, CC BY-NC-SA 4.0 — see `data/README.md`. Non-commercial use only.
```

Replace the "Current state" section of `CLAUDE.md` with:
```markdown
## Commands

- `make up` / `make seed` — local Postgres and seed data
- `make test`, `make lint`, `make fmt`
- Single test: `uv run pytest tests/unit/test_chunking.py::test_giant_token_is_split -v`
- Integration tests need `VIGILEX_TEST_DATABASE_URL` (Makefile default points at compose) and skip without it.
- Tests run with `VIGILEX_LLM=fake`; they never call Bedrock.

## Architecture

uv workspace. `packages/core` (`vigilex_core`): settings, DB, SQL migration runner, embedders with a
Postgres embedding cache. `packages/rag` (`vigilex_rag`): chunking, JuriFindIT loader, ingestion, seed
subset, `vigilex` CLI. Design: `docs/superpowers/specs/`; plans: `docs/superpowers/plans/`.
```

- [ ] **Step 5: Commit**

```bash
git add Makefile .github README.md CLAUDE.md
git commit -m "chore: make targets, CI workflow, dev docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Push the branch and confirm CI is green** (`gh pr create` after the human's go-ahead).

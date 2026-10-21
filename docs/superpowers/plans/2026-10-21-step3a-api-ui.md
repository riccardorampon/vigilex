# Step 3a — API + UI (local) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make app` runs vigilex locally in Docker: a passcode-protected web app where you ask a question, watch the retrieved articles and the cited answer stream in, click a citation to read the article, and open `/eval` to see the retrieval metrics of the live config. Same image as the step 3b Lambda.

**Architecture:** New workspace package `packages/api` (`vigilex_api`): FastAPI app factory with a psycopg connection pool; JSON + SSE endpoints (UI-agnostic, so the later React SPA reuses them); server-rendered Jinja pages (`/login`, `/`, `/eval`) with Pico CSS and ~70 lines of vanilla JS that consume the SSE stream via `EventSource`. HMAC-signed session cookie after a shared passcode; per-session and global daily quotas in Postgres. One Dockerfile (uvicorn + AWS Lambda Web Adapter) used by compose now and Lambda in 3b.

**Tech Stack:** FastAPI 0.156, uvicorn 0.54, Jinja2 3.1, psycopg-pool 3.3, Pico CSS 2.1.1 (vendored), AWS Lambda Web Adapter 1.1.0; tests: FastAPI TestClient (httpx), pytest-playwright for e2e.

**Spec:** `docs/superpowers/specs/2026-10-20-vigilex-umbrella-design.md` (§7, §10, §11; step 3 of §12, API/UI half).

**Branch:** `step3/api-ui`, created from `step2/hybrid-rag` (rebase onto `main` as earlier steps merge).

## Global Constraints

- Everything from the step 1 and 2 plans' Global Constraints still holds (tests never call Bedrock; `VIGILEX_LLM=fake`; ruff 0.17.1; commit trailer).
- The API returns only JSON or `text/event-stream`. HTML lives only in `/login`, `/`, `/eval`.
- The app refuses to start without `VIGILEX_PASSCODE` and `VIGILEX_SECRET_KEY` (fail closed).
- Live retriever config from env: `VIGILEX_RETRIEVER_MODE` (default `hybrid`), `VIGILEX_RERANK` (default `true`). With `VIGILEX_LLM=fake` the reranker is `None`, so `VIGILEX_RERANK=true` is a startup error with a clear message.
- Quotas (env, defaults): `VIGILEX_DAILY_QUOTA=30` per session, `VIGILEX_GLOBAL_DAILY_QUOTA=300`; questions longer than 1000 characters are rejected.
- Session cookie `vigilex_session`: `HttpOnly`, `Secure`, `SameSite=Lax`, 7 days. Browsers accept `Secure` cookies on `http://localhost`; tests use `https://testserver`.

## Spec deltas (update the spec in Task 8)

- **No htmx.** htmx's SSE extension swaps server-rendered HTML fragments, so it would need an HTML twin of every JSON endpoint. The chat page uses `EventSource` and renders the JSON events itself (~70 lines of JS). Pages stay Jinja + Pico CSS.
- **The answer stream is `GET /api/rag/answer?q=…`**, not POST: `EventSource` can only GET. `Cache-Control: no-store`; length-capped; quota-checked before streaming.
- **Session cookie** is signed with stdlib `hmac` (no itsdangerous); needs `VIGILEX_SECRET_KEY` besides the passcode.
- **Quotas:** per session (30/day) **and** global (300/day), because re-entering the passcode mints a new session. Attempts are counted, including refused ones.
- **Answers render as plain text** (`white-space: pre-wrap`) with `[art:N]` turned into links; no Markdown renderer.
- **New endpoint** `GET /api/articles/{id}` backs the citation dialog.
- **`answer()` is split** into `retrieve()` + `generate()` so the stream can send hits before the LLM call.
- **New dependencies:** fastapi, uvicorn, jinja2, psycopg-pool (runtime); httpx, pytest-playwright (dev).
- **Compose defaults to fake mode** (`VIGILEX_LLM=fake`, `VIGILEX_RERANK=false`) so a fresh clone runs without AWS; real mode is `VIGILEX_LLM=bedrock VIGILEX_RERANK=true make app` with `~/.aws` mounted read-only.

## Review Focus

1. **`EventSource` reconnects automatically when a stream ends**, which would re-run (and re-bill) the question. The server ends the stream after `answer` or `error`; the client closes on `answer`, `error` and `onerror`. Tests: Task 5 `test_stream_ends_with_answer_or_error`, Task 8 e2e `test_one_question_costs_one_quota_unit`.
2. **Missing or tampered session cookie:** API → 401 JSON, pages → redirect to `/login`, `/healthz` stays public. Tests: Task 2 `test_tampered_cookie_is_rejected`, Task 4 `test_api_requires_session`, `test_pages_redirect_to_login`.
3. **Quota exhausted:** 429 before any embedding or LLM call. Test: Task 5 `test_quota_exhausted_is_429_without_llm_call`.
4. **Overlong or empty question:** overlong → 422; empty → a `not found` answer event without an LLM call. Tests: Task 5 `test_overlong_question_is_422`, `test_empty_question_streams_not_found`.
5. **Failure mid-stream** (Bedrock error, DB drop): the client gets an `error` event and the stream closes; no stack trace in the response. Test: Task 5 `test_failure_mid_stream_sends_error_event`.

---

## File map

```
packages/rag/src/vigilex_rag/answer.py         # + generate(); answer() = retrieve + generate
packages/api/pyproject.toml
packages/api/src/vigilex_api/__init__.py
packages/api/src/vigilex_api/config.py          # ApiConfig.from_env
packages/api/src/vigilex_api/auth.py            # sign/verify cookie, passcode check, new session id
packages/api/src/vigilex_api/sse.py             # event(name, data)
packages/api/src/vigilex_api/quota.py           # consume()
packages/api/src/vigilex_api/app.py             # create_app(), AppState, pool lifespan
packages/api/src/vigilex_api/deps.py            # state(), require_session(), page_session()
packages/api/src/vigilex_api/api.py             # /healthz, /api/*
packages/api/src/vigilex_api/pages.py           # /login, /logout, /, /eval, sparkline()
packages/api/src/vigilex_api/templates/{base,login,chat,eval}.html
packages/api/src/vigilex_api/static/{pico.min.css,app.js,app.css}
migrations/0003_session_quota.sql
Dockerfile, .dockerignore, docker-compose.yml (+api), Makefile (+app, e2e)
tests/unit/test_api_auth.py, test_api_config.py, test_sse.py, test_sparkline.py, test_generate.py
tests/integration/conftest.py (+session_quota, client fixture), test_quota.py, test_api.py, test_api_answer.py, test_api_eval.py, test_pages.py
tests/e2e/conftest.py, tests/e2e/test_ui.py
.github/workflows/ci.yml (+e2e job), README.md, CLAUDE.md, spec
```

---

### Task 1: Split `answer()` into `retrieve()` + `generate()`

**Files:**
- Modify: `packages/rag/src/vigilex_rag/answer.py`
- Test: `tests/unit/test_generate.py`

**Interfaces:**
- Produces: `vigilex_rag.answer.generate(query: str, hits: list[ArticleHit], chat: Chat) -> Answer` — returns the `NOT_FOUND` answer without calling `chat` when `hits` is empty. `answer(...)` keeps its signature and becomes `generate(query, retrieve(...), chat)`.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_generate.py`

```python
from vigilex_core.llm import Completion, FakeChat
from vigilex_rag.answer import NOT_FOUND, generate
from vigilex_rag.retrieval import ArticleHit


class CountingChat(FakeChat):
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return Completion("Vedi [art:4] e [art:99]", truncated=False)


def test_generate_checks_citations_against_hits():
    hits = [ArticleHit(4, 1.0, "act", "p", "testo")]
    a = generate("domanda", hits, CountingChat())
    assert a.citations == [4] and a.invalid_citations == [99] and a.hits == hits


def test_generate_without_hits_skips_llm():
    chat = CountingChat()
    a = generate("domanda", [], chat)
    assert a.text == NOT_FOUND and chat.calls == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_generate.py -q`
Expected: FAIL with `ImportError: cannot import name 'generate'`

- [ ] **Step 3: Implement** — in `answer.py` replace the body of `answer()`:

```python
def generate(query: str, hits: list[ArticleHit], chat: Chat) -> Answer:
    if not hits:
        return Answer(NOT_FOUND, [], [], [])
    completion = chat.complete(SYSTEM_PROMPT, build_prompt(query, hits), MAX_ANSWER_TOKENS)
    valid, invalid = check_citations(completion.text, hits)
    if invalid:
        log.warning("answer cited articles that were not retrieved: %s", invalid)
    return Answer(completion.text, valid, invalid, hits, completion.truncated)


def answer(
    conn: psycopg.Connection,
    query: str,
    cfg: RetrieverConfig,
    embedder: Embedder,
    chat: Chat,
    reranker: Reranker | None = None,
) -> Answer:
    return generate(query, retrieve(conn, query, cfg, embedder, reranker), chat)
```

- [ ] **Step 4: Run tests**

Run: `make test`
Expected: all pass (existing `answer` tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add packages/rag/src/vigilex_rag/answer.py tests/unit/test_generate.py
git commit -m "refactor: split answer into retrieve + generate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: API package, config, cookie signing, SSE formatting

**Files:**
- Create: `packages/api/pyproject.toml`, `packages/api/src/vigilex_api/__init__.py`, `config.py`, `auth.py`, `sse.py`
- Modify: root `pyproject.toml` (dependency + source + dev deps)
- Test: `tests/unit/test_api_config.py`, `tests/unit/test_api_auth.py`, `tests/unit/test_sse.py`

**Interfaces:**
- Produces:
  - `vigilex_api.config.ApiConfig` frozen dataclass: `passcode: str`, `secret_key: bytes`, `retriever: RetrieverConfig`, `daily_quota: int = 30`, `global_daily_quota: int = 300`, `max_question_chars: int = 1000`; `ApiConfig.from_env(settings: Settings) -> ApiConfig` (raises `RuntimeError` naming the missing variables; `ValueError` for an unknown mode).
  - `vigilex_api.auth`: `COOKIE = "vigilex_session"`, `SESSION_MAX_AGE = 7 * 24 * 3600`, `new_session_id() -> str`, `sign(session_id: str, key: bytes) -> str`, `verify(cookie: str | None, key: bytes) -> str | None`, `check_passcode(given: str, expected: str) -> bool`.
  - `vigilex_api.sse.event(name: str, data: object) -> str` — one SSE frame, JSON on a single `data:` line.

- [ ] **Step 1: Package files**

`packages/api/pyproject.toml`:
```toml
[project]
name = "vigilex-api"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "vigilex-core",
    "vigilex-rag",
    "fastapi>=0.156",
    "uvicorn>=0.54",
    "jinja2>=3.1",
    "psycopg-pool>=3.3",
]

[tool.uv.sources]
vigilex-core = { workspace = true }
vigilex-rag = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vigilex_api"]
```

Root `pyproject.toml`: add `"vigilex-api"` to `dependencies`, `vigilex-api = { workspace = true }` to `[tool.uv.sources]`, and `"httpx>=0.28"`, `"pytest-playwright>=0.7"` to the `dev` group. `__init__.py` is empty. Run `uv sync`.

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_api_config.py`:
```python
import pytest
from vigilex_api.config import ApiConfig
from vigilex_core.settings import Settings

ENV = ("VIGILEX_PASSCODE", "VIGILEX_SECRET_KEY", "VIGILEX_RETRIEVER_MODE", "VIGILEX_RERANK",
       "VIGILEX_DAILY_QUOTA", "VIGILEX_GLOBAL_DAILY_QUOTA")
SETTINGS = Settings("u", "fake", "eu-central-1", "emb", "c", "r")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def test_fails_closed_without_secrets():
    with pytest.raises(RuntimeError, match="VIGILEX_PASSCODE.*VIGILEX_SECRET_KEY"):
        ApiConfig.from_env(SETTINGS)


def test_defaults(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", "pw")
    monkeypatch.setenv("VIGILEX_SECRET_KEY", "k")
    c = ApiConfig.from_env(SETTINGS)
    assert (c.passcode, c.secret_key, c.daily_quota, c.global_daily_quota) == ("pw", b"k", 30, 300)
    assert c.retriever.name() == "hybrid+rerank" and c.retriever.embed_model_id == "emb"


def test_overrides_and_bad_mode(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", "pw")
    monkeypatch.setenv("VIGILEX_SECRET_KEY", "k")
    monkeypatch.setenv("VIGILEX_RETRIEVER_MODE", "vector")
    monkeypatch.setenv("VIGILEX_RERANK", "false")
    monkeypatch.setenv("VIGILEX_DAILY_QUOTA", "5")
    c = ApiConfig.from_env(SETTINGS)
    assert c.retriever.name() == "vector" and c.daily_quota == 5
    monkeypatch.setenv("VIGILEX_RETRIEVER_MODE", "magic")
    with pytest.raises(ValueError, match="magic"):
        ApiConfig.from_env(SETTINGS)
```

`tests/unit/test_api_auth.py`:
```python
from vigilex_api.auth import check_passcode, new_session_id, sign, verify


def test_sign_verify_round_trip():
    sid = new_session_id()
    assert verify(sign(sid, b"k"), b"k") == sid


def test_tampered_cookie_is_rejected():
    cookie = sign("abc", b"k")
    assert verify(cookie, b"other-key") is None
    assert verify("xyz" + cookie[3:], b"k") is None  # different session id, old signature
    assert verify(cookie[:-1] + ("0" if cookie[-1] != "0" else "1"), b"k") is None
    assert verify(None, b"k") is None and verify("no-dot", b"k") is None


def test_session_ids_are_unique_and_unguessable():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100 and all(len(i) >= 20 for i in ids)


def test_check_passcode():
    assert check_passcode("pw", "pw")
    assert not check_passcode("pw ", "pw") and not check_passcode("", "pw")
```

`tests/unit/test_sse.py`:
```python
import json

from vigilex_api.sse import event


def test_event_is_one_frame_with_single_line_json():
    frame = event("answer", {"text": "riga 1\nriga 2", "n": 1})
    assert frame.startswith("event: answer\ndata: ") and frame.endswith("\n\n")
    data_line = frame.split("\n")[1]
    assert json.loads(data_line.removeprefix("data: ")) == {"text": "riga 1\nriga 2", "n": 1}
    assert frame.count("\n") == 3  # newlines inside the text are JSON-escaped


def test_event_keeps_unicode():
    assert "perché" in event("x", {"t": "perché"})
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_api_config.py tests/unit/test_api_auth.py tests/unit/test_sse.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_api.config'` (and `.auth`, `.sse`).

- [ ] **Step 4: Implement**

`packages/api/src/vigilex_api/config.py`:
```python
import os
from dataclasses import dataclass, field

from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig

MODES = ("bm25", "vector", "hybrid")


@dataclass(frozen=True)
class ApiConfig:
    passcode: str
    secret_key: bytes
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    daily_quota: int = 30
    global_daily_quota: int = 300
    max_question_chars: int = 1000

    @classmethod
    def from_env(cls, settings: Settings) -> "ApiConfig":
        missing = [k for k in ("VIGILEX_PASSCODE", "VIGILEX_SECRET_KEY") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"refusing to start without {', '.join(missing)}")
        mode = os.environ.get("VIGILEX_RETRIEVER_MODE", "hybrid")
        if mode not in MODES:
            raise ValueError(f"VIGILEX_RETRIEVER_MODE must be one of {MODES}, got {mode!r}")
        rerank = os.environ.get("VIGILEX_RERANK", "true").lower() in ("1", "true", "yes")
        return cls(
            passcode=os.environ["VIGILEX_PASSCODE"],
            secret_key=os.environ["VIGILEX_SECRET_KEY"].encode(),
            retriever=RetrieverConfig(
                mode=mode,  # type: ignore[arg-type]
                rerank=rerank,
                embed_model_id=settings.embed_model_id,
            ),
            daily_quota=int(os.environ.get("VIGILEX_DAILY_QUOTA", "30")),
            global_daily_quota=int(os.environ.get("VIGILEX_GLOBAL_DAILY_QUOTA", "300")),
        )
```

`packages/api/src/vigilex_api/auth.py`:
```python
import hashlib
import hmac
import secrets

COOKIE = "vigilex_session"
SESSION_MAX_AGE = 7 * 24 * 3600


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


def _mac(session_id: str, key: bytes) -> str:
    return hmac.new(key, session_id.encode(), hashlib.sha256).hexdigest()


def sign(session_id: str, key: bytes) -> str:
    return f"{session_id}.{_mac(session_id, key)}"


def verify(cookie: str | None, key: bytes) -> str | None:
    """The session id if the cookie carries a valid signature, else None."""
    if not cookie or "." not in cookie:
        return None
    session_id, mac = cookie.rsplit(".", 1)
    return session_id if hmac.compare_digest(mac, _mac(session_id, key)) else None


def check_passcode(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())
```

`packages/api/src/vigilex_api/sse.py`:
```python
import json


def event(name: str, data: object) -> str:
    """One Server-Sent Events frame. json.dumps never emits raw newlines, so data stays one line."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

- [ ] **Step 5: Run tests**

Run: `uv run ruff format . && uv run ruff check --fix . && uv run pytest tests/unit -q && uv run pyright`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock packages/api tests/unit/test_api_config.py tests/unit/test_api_auth.py tests/unit/test_sse.py
git commit -m "feat: api package with fail-closed config, signed session cookie, SSE frames

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `session_quota` table and `consume()`

**Files:**
- Create: `migrations/0003_session_quota.sql`, `packages/api/src/vigilex_api/quota.py`
- Modify: `tests/integration/conftest.py` (`TABLES` += `session_quota`)
- Test: `tests/integration/test_quota.py`

**Interfaces:**
- Produces: `vigilex_api.quota.GLOBAL = "__global__"`; `consume(conn, session_id: str, limit: int, global_limit: int) -> bool` — atomically counts one attempt for the session and for `GLOBAL` (today, DB date), commits, returns whether both counts are within their limits.

- [ ] **Step 1: Write the failing test** — `tests/integration/test_quota.py`

```python
from vigilex_api.quota import GLOBAL, consume


def count(conn, sid):
    row = conn.execute(
        "SELECT count FROM session_quota WHERE session_id = %s AND day = current_date", (sid,)
    ).fetchone()
    return row[0] if row else 0


def test_session_limit(conn):
    assert [consume(conn, "s1", 2, 100) for _ in range(3)] == [True, True, False]
    assert count(conn, "s1") == 3  # refused attempts are counted too


def test_global_limit_spans_sessions(conn):
    assert consume(conn, "a", 10, 2) and consume(conn, "b", 10, 2)
    assert not consume(conn, "c", 10, 2)
    assert count(conn, GLOBAL) == 3
```

Add `session_quota` to `TABLES` in `tests/integration/conftest.py`.

- [ ] **Step 2: Run it to verify it fails**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_quota.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vigilex_api.quota'` (the conftest TRUNCATE also fails until the table exists).

- [ ] **Step 3: Implement**

`migrations/0003_session_quota.sql`:
```sql
CREATE TABLE session_quota (
    session_id  text NOT NULL,
    day         date NOT NULL,
    count       int NOT NULL,
    PRIMARY KEY (session_id, day)
);
```

`packages/api/src/vigilex_api/quota.py`:
```python
import psycopg

GLOBAL = "__global__"  # one row per day across all sessions: a new login cannot reset it

_BUMP = (
    "INSERT INTO session_quota (session_id, day, count) VALUES (%s, current_date, 1) "
    "ON CONFLICT (session_id, day) DO UPDATE SET count = session_quota.count + 1 "
    "RETURNING count"
)


def consume(conn: psycopg.Connection, session_id: str, limit: int, global_limit: int) -> bool:
    own = conn.execute(_BUMP, (session_id,)).fetchone()
    total = conn.execute(_BUMP, (GLOBAL,)).fetchone()
    conn.commit()
    assert own is not None and total is not None
    return own[0] <= limit and total[0] <= global_limit
```

- [ ] **Step 4: Run tests**

Run: `make test && uv run pyright`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add migrations/0003_session_quota.sql packages/api/src/vigilex_api/quota.py tests/integration
git commit -m "feat: per-session and global daily quotas

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: App factory, pool, auth flow, health/config/article endpoints

**Files:**
- Create: `packages/api/src/vigilex_api/app.py`, `deps.py`, `api.py`, `pages.py` (login/logout only in this task), `templates/base.html`, `templates/login.html`, `static/pico.min.css`, `static/app.css`
- Modify: `tests/integration/conftest.py` (+ `client` fixture)
- Test: `tests/integration/test_api.py`

**Interfaces:**
- Consumes: Tasks 2–3; `Settings`, `make_chat`, `make_reranker`.
- Produces:
  - `vigilex_api.app.AppState` dataclass: `settings`, `config`, `pool: ConnectionPool`, `chat: Chat`, `reranker: Reranker | None`; `create_app(config: ApiConfig | None = None, settings: Settings | None = None) -> FastAPI` (pool opened/closed by lifespan; `RuntimeError` if `config.retriever.rerank` and no reranker).
  - `vigilex_api.deps`: `state(request) -> AppState`; `require_session(request) -> str` (401 JSON); `page_session(request) -> str | None`.
  - Endpoints: `GET /healthz` (public; 200 `{"status":"ok","db":"ok"}` or 503 `{"status":"degraded","db":"unavailable"}`), `GET /login`, `POST /login` (urlencoded `passcode`; 303 to `/` + cookie, or 401 page), `POST /logout` (303 to `/login`, cookie cleared), `GET /api/config`, `GET /api/articles/{id}` (404 if missing).
  - Test fixtures in `tests/integration/conftest.py`: `client` (TestClient on `https://testserver`, app bound to the test DB, passcode `pw`, rerank off, quotas 3/100) and `login(client)` helper in `tests/helpers.py`.

- [ ] **Step 1: Vendor Pico CSS**

```bash
mkdir -p packages/api/src/vigilex_api/static packages/api/src/vigilex_api/templates
curl -fsSL -o packages/api/src/vigilex_api/static/pico.min.css \
  https://cdn.jsdelivr.net/npm/@picocss/pico@2.1.1/css/pico.min.css
head -c 120 packages/api/src/vigilex_api/static/pico.min.css   # expect: @charset "UTF-8";/*! * Pico CSS ✨ v2.1.1
```

- [ ] **Step 2: Test fixtures**

Append to `tests/helpers.py`:
```python
def login(client, passcode: str = "pw"):
    r = client.post("/login", data={"passcode": passcode}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r
```

Append to `tests/integration/conftest.py`:
```python
from fastapi.testclient import TestClient
from vigilex_api.app import create_app
from vigilex_api.config import ApiConfig
from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig


@pytest.fixture
def api_config() -> ApiConfig:
    return ApiConfig(
        passcode="pw",
        secret_key=b"test-key",
        retriever=RetrieverConfig(),  # hybrid, no rerank: fake mode has no reranker
        daily_quota=3,
        global_daily_quota=100,
    )


@pytest.fixture
def client(db_url, conn, api_config):
    settings = Settings(db_url, "fake", "eu-central-1", "cohere.embed-multilingual-v3", "c", "r")
    with TestClient(create_app(api_config, settings), base_url="https://testserver") as c:
        yield c
```
(`conn` is requested so tables are truncated first; move the new imports to the top of the file.)

- [ ] **Step 3: Write the failing tests** — `tests/integration/test_api.py`

```python
from helpers import article, login
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def test_healthz_is_public(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "db": "ok"}


def test_api_requires_session(client):
    for path in ("/api/config", "/api/articles/1", "/api/eval/runs", "/api/rag/answer?q=x"):
        r = client.get(path)
        assert r.status_code == 401 and r.json() == {"detail": "login required"}, path


def test_wrong_passcode_is_401_and_sets_no_cookie(client):
    r = client.post("/login", data={"passcode": "nope"}, follow_redirects=False)
    assert r.status_code == 401 and "vigilex_session" not in r.cookies
    assert "Passcode errato" in r.text


def test_login_sets_secure_httponly_cookie_and_logout_clears_it(client):
    r = login(client)
    cookie = r.headers["set-cookie"].lower()
    assert "vigilex_session=" in cookie and "httponly" in cookie and "secure" in cookie
    assert "samesite=lax" in cookie and r.headers["location"] == "/"
    assert client.get("/api/config").status_code == 200
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/api/config").status_code == 401


def test_config_reports_live_retriever(client):
    login(client)
    body = client.get("/api/config").json()
    assert body["name"] == "hybrid" and len(body["hash"]) == 16 and body["top_k"] == 10


def test_article_endpoint(client, conn, tmp_path, write_release):
    ingest(conn, load_release(write_release(tmp_path, [article(7, "Art. 7 testo")])), FakeEmbedder())
    login(client)
    assert client.get("/api/articles/7").json() == {
        "id": 7, "source_act": "act", "path": "article_7", "text": "Art. 7 testo"
    }
    assert client.get("/api/articles/8").status_code == 404
```

- [ ] **Step 4: Run them to verify they fail**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_api.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vigilex_api.app'`

- [ ] **Step 5: Implement**

`packages/api/src/vigilex_api/app.py`:
```python
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from vigilex_api import api, pages
from vigilex_api.config import ApiConfig
from vigilex_core.llm import Chat, make_chat
from vigilex_core.rerank import Reranker, make_reranker
from vigilex_core.settings import Settings

HERE = Path(__file__).parent


@dataclass
class AppState:
    settings: Settings
    config: ApiConfig
    pool: ConnectionPool
    chat: Chat
    reranker: Reranker | None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()  # register_vector queries the catalog; the pool wants idle connections back


def create_app(config: ApiConfig | None = None, settings: Settings | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or Settings.from_env()
    config = config or ApiConfig.from_env(settings)
    reranker = make_reranker(settings)
    if config.retriever.rerank and reranker is None:
        raise RuntimeError("VIGILEX_RERANK=true needs a reranker: set VIGILEX_RERANK=false with VIGILEX_LLM=fake")
    # ponytail: max 4 connections; Lambda reserved concurrency is 2, local dev is one user
    pool = ConnectionPool(
        settings.database_url, min_size=0, max_size=4, open=False, configure=_configure, timeout=30
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool.open()
        yield
        pool.close()

    app = FastAPI(title="vigilex", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.state.v = AppState(settings, config, pool, make_chat(settings), reranker)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(api.router)
    app.include_router(pages.router)
    return app
```

`packages/api/src/vigilex_api/deps.py`:
```python
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from vigilex_api.auth import COOKIE, verify

if TYPE_CHECKING:
    from vigilex_api.app import AppState


def state(request: Request) -> "AppState":
    return request.app.state.v


def page_session(request: Request) -> str | None:
    return verify(request.cookies.get(COOKIE), state(request).config.secret_key)


def require_session(request: Request) -> str:
    session_id = page_session(request)
    if session_id is None:
        raise HTTPException(status_code=401, detail="login required")
    return session_id
```

`packages/api/src/vigilex_api/api.py` (this task: healthz, config, articles; Tasks 5–6 append):
```python
import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from vigilex_api.deps import require_session, state

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    try:
        with state(request).pool.connection(timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception:
        log.exception("health check: database unavailable")
        return JSONResponse({"status": "degraded", "db": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok", "db": "ok"})


@router.get("/api/config", dependencies=[Depends(require_session)])
def live_config(request: Request) -> dict:
    cfg = state(request).config.retriever
    return {"name": cfg.name(), "hash": cfg.config_hash(), **asdict(cfg)}


@router.get("/api/articles/{article_id}", dependencies=[Depends(require_session)])
def get_article(article_id: int, request: Request) -> dict:
    with state(request).pool.connection() as conn:
        row = conn.execute(
            "SELECT id, source_act, path, text FROM articles WHERE id = %s", (article_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="article not found")
    return dict(zip(("id", "source_act", "path", "text"), row, strict=True))
```

`test_api_requires_session` also covers `/api/eval/runs` and `/api/rag/answer`, which Tasks 5 and 6 add (404 until then). Tasks 4 and 5 therefore run with `-k "not requires_session"`; Task 6 runs the whole file.

`packages/api/src/vigilex_api/pages.py` (this task: login/logout; Tasks 6–7 append):
```python
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from vigilex_api.auth import COOKIE, SESSION_MAX_AGE, check_passcode, new_session_id, sign
from vigilex_api.deps import state

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> Response:
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login(request: Request) -> Response:
    # plain urlencoded form parsed with the stdlib: no python-multipart dependency
    form = parse_qs((await request.body()).decode())
    given = form.get("passcode", [""])[0]
    st = state(request)
    if not check_passcode(given, st.config.passcode):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Passcode errato."}, status_code=401
        )
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        COOKIE,
        sign(new_session_id(), st.config.secret_key),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@router.post("/logout")
def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE)
    return response
```

`templates/base.html`:
```html
<!doctype html>
<html lang="it" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}vigilex{% endblock %}</title>
  <link rel="stylesheet" href="/static/pico.min.css">
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header class="container">
    <nav>
      <ul><li><strong>vigilex</strong></li></ul>
      {% if logged_in %}
      <ul>
        <li><a href="/">Chat</a></li>
        <li><a href="/eval">Eval</a></li>
        <li><form method="post" action="/logout" class="inline"><button class="secondary outline">Esci</button></form></li>
      </ul>
      {% endif %}
    </nav>
  </header>
  <main class="container">{% block content %}{% endblock %}</main>
  <footer class="container"><small>Dati: JuriFindIT (CC BY-NC-SA 4.0). Risposte informative, non consulenza legale.</small></footer>
  {% block scripts %}{% endblock %}
</body>
</html>
```

`templates/login.html`:
```html
{% extends "base.html" %}
{% block title %}Accesso — vigilex{% endblock %}
{% block content %}
<article class="narrow">
  <h2>Accesso demo</h2>
  <form method="post" action="/login">
    <label>Passcode <input type="password" name="passcode" autocomplete="current-password" required autofocus></label>
    {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
    <button type="submit">Entra</button>
  </form>
</article>
{% endblock %}
```

`static/app.css`:
```css
.narrow { max-width: 28rem; margin: 2rem auto; }
.error { color: var(--pico-del-color); }
form.inline { margin: 0; }
form.inline button { margin: 0; padding: 0.25rem 0.75rem; }
.answer { white-space: pre-wrap; }
.layout { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
@media (max-width: 768px) { .layout { grid-template-columns: 1fr; } }
.hits li { font-size: 0.85rem; }
.cited { font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.current { background: var(--pico-mark-background-color); }
svg.spark { vertical-align: middle; }
dialog article { max-width: 60rem; }
dialog pre { white-space: pre-wrap; }
```

- [ ] **Step 6: Run tests**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_api.py -q -k "not requires_session" && uv run pyright`
Expected: 5 passed; pyright clean.

- [ ] **Step 7: Commit**

```bash
git add packages/api tests/helpers.py tests/integration
git commit -m "feat: FastAPI app factory, pooled DB, passcode login, health/config/article endpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Streaming answer endpoint

**Files:**
- Modify: `packages/api/src/vigilex_api/api.py`
- Test: `tests/integration/test_api_answer.py`

**Interfaces:**
- Consumes: `retrieve`, `generate`, `make_embedder`, `consume`, `event`.
- Produces: `GET /api/rag/answer?q=…` → 422 (`q` longer than `max_question_chars`), 429 (quota), else `text/event-stream` with frames `hits` (list of `{id, score, source_act, path, snippet}`; snippet = first 300 chars), then `answer` (`{text, citations, invalid_citations, truncated}`), or `error` (`{message}`) as the last frame.

- [ ] **Step 1: Write the failing tests** — `tests/integration/test_api_answer.py`

```python
import json

from helpers import article, login
from vigilex_core.embeddings import FakeEmbedder
from vigilex_core.llm import FakeChat
from vigilex_rag.answer import NOT_FOUND
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def frames(text: str) -> list[tuple[str, object]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def seed_two(conn, tmp_path, write_release):
    corpus = [article(1, "adeguata verifica della clientela"), article(2, "segnalazione sospetta")]
    ingest(conn, load_release(write_release(tmp_path, corpus)), FakeEmbedder())


def test_stream_ends_with_answer_or_error(client, conn, tmp_path, write_release):
    seed_two(conn, tmp_path, write_release)
    login(client)
    r = client.get("/api/rag/answer", params={"q": "adeguata verifica?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    events = frames(r.text)
    assert [n for n, _ in events] == ["hits", "answer"]
    hits, answer = events[0][1], events[1][1]
    assert hits[0]["id"] == 1 and set(hits[0]) == {"id", "score", "source_act", "path", "snippet"}
    assert answer["citations"] == [hits[0]["id"]] and answer["truncated"] is False


def test_empty_question_streams_not_found(client):
    login(client)
    events = frames(client.get("/api/rag/answer", params={"q": "  "}).text)
    assert events == [("hits", []), ("answer", {
        "text": NOT_FOUND, "citations": [], "invalid_citations": [], "truncated": False
    })]


def test_overlong_question_is_422(client):
    login(client)
    assert client.get("/api/rag/answer", params={"q": "x" * 1001}).status_code == 422


def test_quota_exhausted_is_429_without_llm_call(client, conn, tmp_path, write_release):
    seed_two(conn, tmp_path, write_release)
    login(client)
    calls = []

    class Counting(FakeChat):
        def complete(self, system, user, max_tokens):
            calls.append(1)
            return super().complete(system, user, max_tokens)

    client.app.state.v.chat = Counting()  # type: ignore[attr-defined]
    for _ in range(3):  # quota is 3 in the test config
        assert client.get("/api/rag/answer", params={"q": "verifica"}).status_code == 200
    r = client.get("/api/rag/answer", params={"q": "verifica"})
    assert r.status_code == 429 and len(calls) == 3


def test_failure_mid_stream_sends_error_event(client, conn, tmp_path, write_release):
    seed_two(conn, tmp_path, write_release)
    login(client)

    class Broken(FakeChat):
        def complete(self, system, user, max_tokens):
            raise RuntimeError("bedrock exploded: secret internals")

    client.app.state.v.chat = Broken()  # type: ignore[attr-defined]
    events = frames(client.get("/api/rag/answer", params={"q": "verifica"}).text)
    assert [n for n, _ in events] == ["hits", "error"]
    assert "secret internals" not in json.dumps(events)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_api_answer.py -q`
Expected: FAIL — 404 on `/api/rag/answer`.

- [ ] **Step 3: Implement** (append to `api.py`; imports `from collections.abc import Iterator`, `from fastapi.responses import StreamingResponse`, `from vigilex_api.quota import consume`, `from vigilex_api.sse import event`, `from vigilex_core.embeddings import make_embedder`, `from vigilex_rag.answer import generate`, `from vigilex_rag.retrieval import retrieve`)

```python
SNIPPET_CHARS = 300
STREAM_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


@router.get("/api/rag/answer")
def rag_answer(q: str, request: Request, session_id: str = Depends(require_session)) -> StreamingResponse:
    st = state(request)
    if len(q) > st.config.max_question_chars:
        raise HTTPException(status_code=422, detail=f"question longer than {st.config.max_question_chars} characters")
    with st.pool.connection() as conn:
        allowed = consume(conn, session_id, st.config.daily_quota, st.config.global_daily_quota)
    if not allowed:
        raise HTTPException(status_code=429, detail="daily question quota reached")

    def stream() -> Iterator[str]:
        # EventSource reconnects when a stream ends; the client closes on answer/error, and
        # every path below ends with exactly one of them.
        try:
            with st.pool.connection() as conn:
                embedder = make_embedder(st.settings, conn)
                hits = retrieve(conn, q, st.config.retriever, embedder, st.reranker)
                conn.commit()  # release locks (cache inserts) before the slow LLM call
            yield event("hits", [
                {"id": h.id, "score": h.score, "source_act": h.source_act, "path": h.path,
                 "snippet": h.text[:SNIPPET_CHARS]}
                for h in hits
            ])
            result = generate(q, hits, st.chat)
            yield event("answer", {
                "text": result.text,
                "citations": result.citations,
                "invalid_citations": result.invalid_citations,
                "truncated": result.truncated,
            })
        except Exception:
            log.exception("answer stream failed")
            yield event("error", {"message": "Errore interno: riprova tra poco."})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=STREAM_HEADERS)
```

- [ ] **Step 4: Run tests**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_api_answer.py tests/integration/test_api.py -q -k "not requires_session" && uv run pyright`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/api/src/vigilex_api/api.py tests/integration/test_api_answer.py
git commit -m "feat: streaming answer endpoint with quota, length cap and error events

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Eval API and `/eval` page

**Files:**
- Modify: `packages/api/src/vigilex_api/api.py`, `pages.py`
- Create: `templates/eval.html`
- Test: `tests/unit/test_sparkline.py`, `tests/integration/test_api_eval.py`

**Interfaces:**
- Produces:
  - `GET /api/eval/runs?scope=current|all` (default `current`: runs whose `config_hash` is the live config's) → list, newest first, max 50, each `{id, created_at, git_sha, config_name, config_hash, split, n_questions, metrics}`.
  - `GET /api/eval/runs/{id}` → the same plus `per_query`; 404 if missing.
  - `vigilex_api.pages.sparkline(values: list[float], width: int = 120, height: int = 24) -> str` — SVG polyline `points`, oldest left; `""` for fewer than 2 values.
  - `GET /eval` page: the latest `validation` run per config name (live config highlighted), recall@10 history sparkline of the live config, and the questions the live config's latest run misses entirely (recall@20 = 0).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_sparkline.py`:
```python
from vigilex_api.pages import sparkline


def test_sparkline_scales_to_box():
    assert sparkline([0.0, 0.5, 1.0], width=100, height=10) == "0.0,10.0 50.0,5.0 100.0,0.0"


def test_flat_and_short_series():
    assert sparkline([0.3, 0.3], width=10, height=10) == "0.0,5.0 10.0,5.0"
    assert sparkline([0.3]) == "" and sparkline([]) == ""
```

`tests/integration/test_api_eval.py`:
```python
from helpers import article, login, question
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.evaluation import evaluate, save_run
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


def two_runs(conn, tmp_path, write_release):
    release = write_release(
        tmp_path,
        [article(1, "adeguata verifica"), article(2, "segnalazione sospetta")],
        [question("adeguata verifica", [1]), question("zzz", [2])],
    )
    ingest(conn, load_release(release), FakeEmbedder())
    live = save_run(conn, evaluate(conn, RetrieverConfig(), FakeEmbedder(), None), "sha1")
    other = save_run(conn, evaluate(conn, RetrieverConfig(mode="bm25"), FakeEmbedder(), None), "sha1")
    return live, other


def test_runs_scope_current_and_all(client, conn, tmp_path, write_release):
    live, other = two_runs(conn, tmp_path, write_release)
    login(client)
    current = client.get("/api/eval/runs").json()
    assert [r["id"] for r in current] == [live]
    assert {r["id"] for r in client.get("/api/eval/runs?scope=all").json()} == {live, other}
    assert "per_query" not in current[0] and current[0]["metrics"]["recall@10"] >= 0


def test_run_detail(client, conn, tmp_path, write_release):
    live, _ = two_runs(conn, tmp_path, write_release)
    login(client)
    body = client.get(f"/api/eval/runs/{live}").json()
    assert set(body["per_query"]) == {"validation-0", "validation-1"}
    assert client.get("/api/eval/runs/999999").status_code == 404


def test_eval_page(client, conn, tmp_path, write_release):
    two_runs(conn, tmp_path, write_release)
    login(client)
    page = client.get("/eval").text
    assert "hybrid" in page and "bm25" in page and 'class="current"' in page
    assert "zzz" in page  # a question the live config misses entirely


def test_eval_page_without_runs(client):
    login(client)
    assert "Nessuna valutazione" in client.get("/eval").text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_sparkline.py -q; VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_api_eval.py -q`
Expected: FAIL — `ImportError: cannot import name 'sparkline'`; 404 on `/api/eval/runs`.

- [ ] **Step 3: Implement**

Append to `api.py`:
```python
RUN_COLUMNS = ("id", "created_at", "git_sha", "config_name", "config_hash", "split", "n_questions", "metrics")
RUN_SELECT = "SELECT " + ", ".join(RUN_COLUMNS)


@router.get("/api/eval/runs", dependencies=[Depends(require_session)])
def eval_runs(request: Request, scope: str = "current") -> list[dict]:
    st = state(request)
    if scope not in ("current", "all"):
        raise HTTPException(status_code=422, detail="scope must be current or all")
    where, params = ("WHERE config_hash = %s", (st.config.retriever.config_hash(),)) if scope == "current" else ("", ())
    with st.pool.connection() as conn:
        rows = conn.execute(f"{RUN_SELECT} FROM eval_runs {where} ORDER BY created_at DESC, id DESC LIMIT 50", params).fetchall()  # type: ignore[arg-type]
    return [dict(zip(RUN_COLUMNS, r, strict=True)) for r in rows]


@router.get("/api/eval/runs/{run_id}", dependencies=[Depends(require_session)])
def eval_run(run_id: int, request: Request) -> dict:
    with state(request).pool.connection() as conn:
        row = conn.execute(f"{RUN_SELECT}, per_query FROM eval_runs WHERE id = %s", (run_id,)).fetchone()  # type: ignore[arg-type]
    if row is None:
        raise HTTPException(status_code=404, detail="eval run not found")
    return dict(zip((*RUN_COLUMNS, "per_query"), row, strict=True))
```
(`where` and the column list are constants chosen in code, never user input.)

Append to `pages.py` (imports `from fastapi import Depends`, `from vigilex_api.deps import page_session`, `from vigilex_rag.evaluation import CONFIG_NAMES`, `from vigilex_rag.metrics import METRICS`):
```python
def sparkline(values: list[float], width: int = 120, height: int = 24) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = width / (len(values) - 1)
    ys = [height / 2 if hi == lo else height - (v - lo) / span * height for v in values]
    return " ".join(f"{i * step:.1f},{y:.1f}" for i, y in enumerate(ys))


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/login", status_code=303)


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request) -> Response:
    if page_session(request) is None:
        return _login_redirect()
    st = state(request)
    live = st.config.retriever
    with st.pool.connection() as conn:
        latest = conn.execute(
            "SELECT DISTINCT ON (config_name) id, config_name, config_hash, n_questions, metrics, created_at, git_sha "
            "FROM eval_runs WHERE split = 'validation' ORDER BY config_name, created_at DESC, id DESC"
        ).fetchall()
        history = conn.execute(
            "SELECT (metrics->>'recall@10')::float FROM eval_runs WHERE config_hash = %s "
            "ORDER BY created_at DESC, id DESC LIMIT 20",
            (live.config_hash(),),
        ).fetchall()
        missed: list[tuple[str, str]] = []
        live_run = next((r for r in latest if r[2] == live.config_hash()), None)
        if live_run is not None:
            row = conn.execute("SELECT per_query FROM eval_runs WHERE id = %s", (live_run[0],)).fetchone()
            per_query = row[0] if row else {}
            ids = [qid for qid, m in per_query.items() if m["recall@20"] == 0]
            missed = conn.execute(
                "SELECT id, text FROM questions WHERE id = ANY(%s) ORDER BY id", (ids,)
            ).fetchall()
    order = {name: i for i, name in enumerate(CONFIG_NAMES)}
    rows = sorted(latest, key=lambda r: order.get(r[1], len(order)))
    recall_history = [h[0] for h in reversed(history)]
    return templates.TemplateResponse(request, "eval.html", {
        "logged_in": True,
        "live": live,
        "rows": rows,
        "metrics": METRICS,
        "spark": sparkline(recall_history),
        "history_len": len(recall_history),
        "missed": missed,
    })
```

`templates/eval.html`:
```html
{% extends "base.html" %}
{% block title %}Eval — vigilex{% endblock %}
{% block content %}
<h2>Valutazione del retrieval</h2>
<p>Configurazione attiva: <strong>{{ live.name() }}</strong> <code>{{ live.config_hash() }}</code></p>
{% if not rows %}
<p>Nessuna valutazione registrata. Esegui <code>make eval-ci</code> (o <code>make eval</code> con AWS).</p>
{% else %}
<figure>
<table>
  <thead><tr><th>config</th><th class="num">domande</th>{% for m in metrics %}<th class="num">{{ m }}</th>{% endfor %}<th>quando</th></tr></thead>
  <tbody>
  {% for r in rows %}
    <tr{% if r[2] == live.config_hash() %} class="current"{% endif %}>
      <td>{{ r[1] }}</td><td class="num">{{ r[3] }}</td>
      {% for m in metrics %}<td class="num">{{ "%.3f"|format(r[4][m]) }}</td>{% endfor %}
      <td><small>{{ r[5].strftime("%Y-%m-%d %H:%M") }}{% if r[6] %} · {{ r[6][:7] }}{% endif %}</small></td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</figure>
{% if spark %}
<p>recall@10 della configurazione attiva, ultime {{ history_len }} esecuzioni:
  <svg class="spark" width="120" height="24" viewBox="-2 -2 124 28" role="img" aria-label="andamento recall@10">
    <polyline points="{{ spark }}" fill="none" stroke="currentColor" stroke-width="1.5"/>
  </svg>
</p>
{% endif %}
<h3>Domande senza alcun articolo pertinente nei primi 20</h3>
{% if missed %}
<ul>{% for qid, text in missed %}<li><code>{{ qid }}</code> {{ text }}</li>{% endfor %}</ul>
{% else %}
<p>Nessuna.</p>
{% endif %}
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Run tests** (now including the full auth test)

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -q && uv run pyright`
Expected: all pass, including `test_api_requires_session`.

- [ ] **Step 5: Commit**

```bash
git add packages/api tests/unit/test_sparkline.py tests/integration/test_api_eval.py
git commit -m "feat: eval runs API and /eval page

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Chat page and client

**Files:**
- Modify: `packages/api/src/vigilex_api/pages.py`
- Create: `templates/chat.html`, `static/app.js`
- Test: `tests/integration/test_pages.py`

**Interfaces:**
- Produces: `GET /` (redirect to `/login` without session; else chat page showing the live config name). `app.js`: on submit opens `EventSource("/api/rag/answer?q=…")`, renders `hits` into the side list, `answer` into `.answer` with `[art:N]` turned into buttons that open a `<dialog>` with `/api/articles/N`, marks cited hits, shows a truncation note, and **closes the EventSource on `answer`, `error` and `onerror`**. Quota/length errors arrive as an HTTP error on the stream request: `onerror` shows a generic message (the client cannot read the status from `EventSource`; it fetches `/api/config` to tell "logged out" from "quota/other").

- [ ] **Step 1: Write the failing tests** — `tests/integration/test_pages.py`

```python
from helpers import login


def test_pages_redirect_to_login(client):
    for path in ("/", "/eval"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login", path


def test_chat_page(client):
    login(client)
    page = client.get("/").text
    assert 'id="ask"' in page and "/static/app.js" in page and "hybrid" in page


def test_login_page_and_static_assets(client):
    assert 'name="passcode"' in client.get("/login").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/pico.min.css").headers["content-type"].startswith("text/css")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests/integration/test_pages.py -q`
Expected: FAIL — `/` is 404 and `/static/app.js` is 404.

- [ ] **Step 3: Implement**

Append to `pages.py`:
```python
@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> Response:
    if page_session(request) is None:
        return _login_redirect()
    return templates.TemplateResponse(
        request, "chat.html", {"logged_in": True, "live": state(request).config.retriever}
    )
```

`templates/chat.html`:
```html
{% extends "base.html" %}
{% block title %}Chat — vigilex{% endblock %}
{% block content %}
<form id="ask">
  <fieldset role="group">
    <input id="question" name="q" maxlength="1000" required
           placeholder="Es. Quali sono gli obblighi di adeguata verifica della clientela?">
    <button type="submit">Chiedi</button>
  </fieldset>
  <small>Retrieval: <code>{{ live.name() }}</code> · le risposte citano gli articoli come [art:ID]</small>
</form>
<div class="layout">
  <section>
    <h3>Risposta</h3>
    <p id="status" aria-live="polite"></p>
    <div class="answer" id="answer"></div>
  </section>
  <aside>
    <h3>Articoli recuperati</h3>
    <ol class="hits" id="hits"></ol>
  </aside>
</div>
<dialog id="article-dialog">
  <article>
    <header><button class="close" aria-label="Chiudi" data-close></button><strong id="article-title"></strong></header>
    <pre id="article-text"></pre>
  </article>
</dialog>
{% endblock %}
{% block scripts %}<script src="/static/app.js" defer></script>{% endblock %}
```

`static/app.js`:
```javascript
// Consumes GET /api/rag/answer (Server-Sent Events, JSON payloads).
// EventSource reconnects when a stream ends, which would re-run (and re-bill) the question:
// every terminal path below closes it.
(() => {
  const form = document.getElementById("ask");
  const input = document.getElementById("question");
  const status = document.getElementById("status");
  const answerBox = document.getElementById("answer");
  const hitsList = document.getElementById("hits");
  const dialog = document.getElementById("article-dialog");
  let source = null;

  const setBusy = (busy) => { form.querySelector("button").setAttribute("aria-busy", busy); };

  function renderAnswer(text, citations) {
    answerBox.replaceChildren();
    const parts = text.split(/(\[art:\d+\])/);
    for (const part of parts) {
      const m = part.match(/^\[art:(\d+)\]$/);
      if (m) {
        const btn = document.createElement("a");
        btn.href = "#";
        btn.textContent = part;
        btn.dataset.article = m[1];
        answerBox.append(btn);
      } else {
        answerBox.append(document.createTextNode(part));
      }
    }
    for (const li of hitsList.children) {
      li.classList.toggle("cited", citations.includes(Number(li.dataset.article)));
    }
  }

  function renderHits(hits) {
    hitsList.replaceChildren(...hits.map((h) => {
      const li = document.createElement("li");
      li.dataset.article = h.id;
      const a = document.createElement("a");
      a.href = "#";
      a.dataset.article = h.id;
      a.textContent = `[art:${h.id}]`;
      li.append(a, ` ${h.path} `);
      const small = document.createElement("small");
      small.textContent = h.source_act;
      li.append(small);
      return li;
    }));
  }

  function finish(message) {
    if (source) { source.close(); source = null; }
    setBusy(false);
    status.textContent = message;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || source) return;
    answerBox.replaceChildren();
    hitsList.replaceChildren();
    status.textContent = "Ricerca degli articoli…";
    setBusy(true);
    source = new EventSource(`/api/rag/answer?q=${encodeURIComponent(q)}`);
    source.addEventListener("hits", (ev) => {
      renderHits(JSON.parse(ev.data));
      status.textContent = "Generazione della risposta…";
    });
    source.addEventListener("answer", (ev) => {
      const a = JSON.parse(ev.data);
      renderAnswer(a.text, a.citations);
      let note = a.truncated ? "Risposta troncata (limite di lunghezza)." : "";
      if (a.invalid_citations.length) note += ` Citazioni non verificate: ${a.invalid_citations.join(", ")}.`;
      finish(note.trim());
    });
    source.addEventListener("error", (ev) => {
      if (ev.data) { finish(JSON.parse(ev.data).message); return; }  // server-sent error event
      // transport error: HTTP 401/422/429 or network. EventSource hides the status.
      finish("Richiesta non riuscita.");
      fetch("/api/config").then((r) => {
        if (r.status === 401) window.location = "/login";
        else status.textContent = "Richiesta non riuscita: quota giornaliera esaurita o domanda troppo lunga.";
      });
    });
  });

  document.addEventListener("click", async (e) => {
    const target = e.target.closest("[data-article]");
    if (target && target.tagName === "A") {
      e.preventDefault();
      const r = await fetch(`/api/articles/${target.dataset.article}`);
      if (!r.ok) return;
      const art = await r.json();
      document.getElementById("article-title").textContent = `[art:${art.id}] ${art.path}`;
      document.getElementById("article-text").textContent = `${art.source_act}\n\n${art.text}`;
      dialog.showModal();
    }
    if (e.target.matches("[data-close]")) dialog.close();
  });
})();
```

- [ ] **Step 4: Run tests**

Run: `VIGILEX_TEST_SERVER_URL=postgresql://vigilex:vigilex@localhost:5432/postgres uv run pytest tests -q && uv run pyright`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add packages/api tests/integration/test_pages.py
git commit -m "feat: chat page with streamed hits, cited answer and article dialog

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docker image, compose, e2e, CI, docs

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `tests/e2e/conftest.py`, `tests/e2e/test_ui.py`
- Modify: `docker-compose.yml`, `Makefile`, `.github/workflows/ci.yml`, `README.md`, `CLAUDE.md`, spec

**Interfaces:**
- Produces: image `vigilex` (uvicorn on 8080 + Lambda Web Adapter extension); compose service `api`; `make app` (db + migrations + seed + api), `make e2e`; CI job `e2e`.

- [ ] **Step 1: Dockerfile and .dockerignore**

`Dockerfile`:
```dockerfile
FROM python:3.12-slim
# Lambda Web Adapter: a Lambda extension that turns invocations into HTTP requests to :8080.
# Outside Lambda it is never started, so the same image runs in compose.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH PORT=8080
COPY pyproject.toml uv.lock ./
COPY packages packages
RUN uv sync --frozen --no-dev
COPY migrations migrations
EXPOSE 8080
CMD ["uvicorn", "--factory", "vigilex_api.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
```

`.dockerignore`:
```
.git
.venv
.superpowers
data
docs
tests
**/__pycache__
```

Run: `docker build -t vigilex . && docker run --rm vigilex python -c "import vigilex_api.app"`
Expected: build succeeds; import succeeds.

- [ ] **Step 2: Compose service** — add to `docker-compose.yml` under `services:`

```yaml
  api:
    build: .
    ports: ["8080:8080"]
    depends_on:
      db: { condition: service_healthy }
    environment:
      VIGILEX_DATABASE_URL: postgresql://vigilex:vigilex@db:5432/vigilex
      VIGILEX_PASSCODE: ${VIGILEX_PASSCODE:-dev}
      VIGILEX_SECRET_KEY: ${VIGILEX_SECRET_KEY:-dev-only-secret}
      VIGILEX_LLM: ${VIGILEX_LLM:-fake}
      VIGILEX_RERANK: ${VIGILEX_RERANK:-false}
      AWS_PROFILE: ${AWS_PROFILE:-default}
    volumes:
      - ./packages:/app/packages            # live code; the venv installs workspace packages editable
      - ${HOME}/.aws:/root/.aws:ro          # real Bedrock mode only
    command: ["uvicorn", "--factory", "vigilex_api.app:create_app", "--host", "0.0.0.0",
              "--port", "8080", "--reload", "--reload-dir", "/app/packages"]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')"]
      interval: 2s
      retries: 30
```

- [ ] **Step 3: Makefile** (append)

```makefile
app: up seed
	docker compose up -d --wait --build api
	@echo "vigilex on http://localhost:8080 (passcode: $${VIGILEX_PASSCODE:-dev})"

e2e:
	VIGILEX_E2E_URL=http://localhost:8080 uv run pytest tests/e2e

.PHONY: app e2e
```
Also change `up` to keep starting only `db` (`docker compose up -d --wait db`), so `make test` never needs the image.

- [ ] **Step 4: e2e tests**

`tests/e2e/conftest.py`:
```python
import os

import pytest

URL = os.environ.get("VIGILEX_E2E_URL")


@pytest.fixture(scope="session")
def base_url():
    if not URL:
        pytest.skip("VIGILEX_E2E_URL not set (run `make app` then `make e2e`)")
    return URL
```

`tests/e2e/test_ui.py`:
```python
import os
import time

import psycopg
from playwright.sync_api import Page, expect

PASSCODE = os.environ.get("VIGILEX_PASSCODE", "dev")
DB = "postgresql://vigilex:vigilex@localhost:5432/vigilex"


def login(page: Page, base_url: str) -> None:
    page.goto(base_url + "/")
    page.fill("input[name=passcode]", PASSCODE)
    page.click("button[type=submit]")
    expect(page.locator("#ask")).to_be_visible()


def session_count(page: Page) -> int:
    sid = next(c["value"] for c in page.context.cookies() if c["name"] == "vigilex_session").rsplit(".", 1)[0]
    with psycopg.connect(DB) as conn:
        row = conn.execute(
            "SELECT count FROM session_quota WHERE session_id = %s AND day = current_date", (sid,)
        ).fetchone()
    return row[0] if row else 0


def test_ask_shows_cited_answer_and_article(page: Page, base_url: str):
    login(page, base_url)
    page.fill("#question", "Quali sono gli obblighi di adeguata verifica della clientela?")
    page.click("#ask button")
    expect(page.locator("#hits li")).to_have_count(10)
    expect(page.locator("#answer")).to_contain_text("[art:")
    page.locator("#answer a").first.click()
    expect(page.locator("#article-dialog")).to_be_visible()
    expect(page.locator("#article-text")).not_to_be_empty()


def test_one_question_costs_one_quota_unit(page: Page, base_url: str):
    login(page, base_url)
    page.fill("#question", "obblighi di segnalazione")
    page.click("#ask button")
    expect(page.locator("#answer")).not_to_be_empty()
    time.sleep(4)  # an EventSource left open would reconnect within ~3 s and count again
    assert session_count(page) == 1


def test_eval_page(page: Page, base_url: str):
    login(page, base_url)
    page.goto(base_url + "/eval")
    expect(page.locator("table")).to_contain_text("hybrid")
```

- [ ] **Step 5: CI** — add a job to `.github/workflows/ci.yml`

```yaml
  e2e:
    runs-on: ubuntu-24.04
    env:
      VIGILEX_LLM: fake
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: make app
      - run: make eval-ci
      - run: uv run playwright install --with-deps chromium
      - run: make e2e
      - if: failure()
        run: docker compose logs api
```

- [ ] **Step 6: Verify locally**

Run: `make lint && make test && make app && make eval-ci && uv run playwright install chromium && make e2e`
Expected: all green; open http://localhost:8080, log in with `dev`, ask a question (fake answer), open `/eval`.
Then real mode (needs AWS): `VIGILEX_LLM=bedrock VIGILEX_RERANK=true docker compose up -d --wait api`, ask the same question: a real cited answer streams in.

- [ ] **Step 7: Docs and spec**

- README "Local development": `make app` (http://localhost:8080, passcode `dev`), real mode command, `make e2e`.
- CLAUDE.md: `packages/api` (`vigilex_api`) in Architecture; `make app`, `make e2e` in Commands; note that e2e needs the app running.
- Spec: apply the "Spec deltas" above (§3 web line, §7.1 endpoint table, §7.2 pages, §7.3 quotas/secret, §10 compose).

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml Makefile .github tests/e2e README.md CLAUDE.md docs/superpowers/specs
git commit -m "feat: docker image (uvicorn + Lambda Web Adapter), compose app, e2e tests in CI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

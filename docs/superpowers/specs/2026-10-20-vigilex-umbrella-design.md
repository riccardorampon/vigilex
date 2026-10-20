# vigilex — umbrella design

Status: draft for review · Date: 2026-10-20

## 1. Purpose and scope

vigilex is a portfolio project for AI engineering interviews: an AI assistant over Italian/EU anti-money-laundering (AML) and privacy law. Success means it is demoable live and every implementation choice can be defended in depth. It has no real users and no commercial use.

This umbrella spec fixes the cross-cutting decisions (data, stack, cloud, LLM, repo layout, testing, CI/CD) and the design of four sub-projects:

1. **Hybrid RAG + evaluation framework**
2. **Linear ReAct agent** using tools through an **MCP server**
3. **Graph agent** (LangGraph) with state, memory and multi-step logic
4. **Platform**: API/UI, infrastructure, CI/CD

Each sub-project gets its own implementation plan, built in the order of §12.

### Non-goals

- Multi-tenancy, user accounts, billing, SLAs.
- Commercial use (blocked by the dataset license).
- Legal advice: answers are informational and always cite their sources.
- A second LLM provider or an offline local LLM.
- Distributed tracing (X-Ray/OpenTelemetry), self-hosted observability (Langfuse).

### Planned later phase

- Replace the server-rendered htmx UI with a React/Vite SPA on S3 + CloudFront. The API is UI-agnostic (JSON + SSE) so that this switch does not touch the backend.

## 2. Data

**Source:** JuriFindIT (EACL 2026 Findings, Dalla Noce et al.). About 23k Italian statutory articles across civil law, criminal law, AML/counter-terrorism and privacy; 895 expert-written questions; about 169k generated questions; cross-references between articles. License: CC BY-NC-SA 4.0 (to be confirmed against the official release before ingestion).

**Licensing in the repo:** `data/LICENSE` carries the CC BY-NC-SA 4.0 text and attribution. The README states that code and data are licensed separately and the data is non-commercial.

### 2.1 Schema (Postgres 16 + pgvector)

| table | columns / indexes |
|---|---|
| `articles` | `id` PK (JuriFindIT id), `source_act` (source document `file_name`), `path` (position in the act), `text`, `tsv tsvector GENERATED ALWAYS AS (to_tsvector('italian', text)) STORED`; GIN on `tsv`. The data has no article number, title or domain; domain lives on questions |
| `article_chunks` | `article_id` FK, `ord`, `text`, `embedding vector(1024)`; PK (`article_id`, `ord`); HNSW (cosine) on `embedding` |
| `article_refs` | `src_id`, `dst_id` (both FK to `articles`); PK (`src_id`, `dst_id`); index on `dst_id` |
| `questions` | `id` PK (`"{split}-{row_index}"` in the original release file), `split`, `text`, `domain` (`normative_area`), `topics`. Only expert questions are ingested |
| `qrels` | `question_id`, `article_id`; PK on both |
| `eval_runs` | `id`, `created_at`, `git_sha`, `config_hash`, `retriever_config jsonb`, `metrics jsonb`, `per_query jsonb`, `judge bool` |
| `embedding_cache` | `key` PK = `sha256(model_id \0 input_type \0 text)`, `embedding vector(1024)` |
| `session_quota` | `session_id`, `day`, `count`; PK (`session_id`, `day`) |
| LangGraph tables | created by `langgraph-checkpoint-postgres` (`PostgresSaver`, `PostgresStore`) |

Migrations: plain `.sql` files applied in name order by a ~20-line runner (`schema_migrations` table), no Alembic/SQLAlchemy. Tables arrive with the step that needs them (`eval_runs` in step 2, `session_quota` in step 3). Every migration must be compatible with both the previous and the new code version (expand → contract), because migrations run after the code update (§9).

### 2.2 Ingestion

`vigilex ingest --source <path>` reads the JuriFindIT release and upserts into the tables above. It is idempotent and merges: rows upsert and each article's chunks are replaced, but rows missing from a newer release are never deleted (recreate the database for an exact mirror). The 159 `<< file root node >>` rows are document roots, not articles, and are skipped. The seed subset uses `vigilex build-seed` / `vigilex load-seed` (§2.3).

- **The unit of retrieval is the article**, because the gold labels (`qrels`) are articles.
- Articles are split into chunks by characters (max 1800, overlap 200), because Bedrock's Cohere embed accepts at most 2048 characters per text; a single token longer than that is hard-split (the corpus contains a 1.1M-character table). An article's vector score is the max over its chunks.
- Embedding model: **Cohere Embed Multilingual v3 on Bedrock** (1024 dimensions). Documents use `input_type=search_document`, queries `search_query`. Embeddings are cached in `embedding_cache`, so re-ingestion never pays twice. Only the 895 expert questions are embedded.

### 2.3 Committed seed subset (`data/seed/`)

For local development and CI, with no Bedrock calls needed:

- The 37 `validation`-split expert questions from the AML and privacy domains.
- Their 432 gold articles, up to 500 sampled articles that gold articles cite (outgoing references only), and up to 1000 sampled distractors from the same acts: 1,932 articles. Distractors keep local metrics from being inflated.
- Article chunk embeddings and seed question embeddings as float16 Parquet (≈11 MB in total with the corpus).
- `make seed` loads it into the local database in about a minute.

## 3. Stack and repository layout

- **Language and tooling:** Python 3.12, uv workspace, ruff (lint + format), pyright, pytest, pre-commit.
- **LLM:** Amazon Bedrock only. Claude models (via EU cross-region inference profiles) for chat; the cheapest current Claude model for routing and LLM-as-judge; Cohere Embed Multilingual v3 for embeddings; Cohere Rerank 3.5 via the Bedrock Rerank API.
- **Region:** `eu-central-1` (Frankfurt) for every resource, chosen for Bedrock model availability.
- **Database:** Postgres (Aurora Serverless v2 in AWS, `pgvector/pgvector:pg16` locally).
- **Web:** FastAPI, Jinja, htmx with the SSE extension, Pico.css. Static files are vendored.
- **Agents:** hand-written ReAct loop; LangGraph for the graph agent; official `mcp` Python SDK (FastMCP) for the MCP server.
- **Evaluation:** `ranx` for IR metrics; custom judge prompt on Bedrock.
- **IaC:** Terraform. **CI/CD:** GitHub Actions with AWS OIDC.

```
vigilex/
├── packages/
│   ├── core/          # settings, Bedrock client (chat, embed, rerank), DB session, fakes for tests
│   ├── rag/           # ingestion, retrievers, RRF, reranker, answer, eval CLI
│   ├── mcp_server/    # MCP server exposing rag tools
│   ├── agent_react/   # hand-written ReAct loop, tools via MCP client
│   ├── agent_graph/   # LangGraph agent
│   └── api/           # FastAPI: JSON + SSE endpoints, Jinja/htmx pages
├── data/              # seed subset, LICENSE
├── migrations/        # plain SQL
├── infra/
│   ├── bootstrap/     # state bucket, GitHub OIDC provider and roles
│   └── main/          # VPC, Aurora, Lambda, ECR, endpoints, S3
├── tests/{unit,integration,e2e}/
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml
└── .github/workflows/
```

Dependencies: `api` → `agent_react`, `agent_graph`, `rag`; `agent_react` → `mcp_server` (over MCP stdio); `agent_graph` → `rag` (direct Python calls); everything → `core`.

The ReAct agent uses MCP and the graph agent does not. This is a deliberate contrast: MCP gives interoperability (any MCP client can use vigilex's tools) at the cost of an extra process hop.

## 4. Sub-project 1: hybrid RAG and evaluation

### 4.1 Retrieval

`rag.retrieve(query, cfg: RetrieverConfig) -> list[ArticleHit]`

1. **Keyword:** `websearch_to_tsquery('italian', q)`, ranked by `ts_rank_cd`, top 50 articles.
2. **Vector:** embed the query; HNSW search over `article_chunks`; max score per article; top 50 articles.
3. **Fusion:** Reciprocal Rank Fusion with `k = 60`.
4. **Rerank (optional):** Cohere Rerank 3.5 reorders the fused top 50 and keeps the top 10.

`RetrieverConfig` (Pydantic) holds `mode` (`bm25` | `vector` | `hybrid`), top-k values, `rerank` flag and embedding model id. The deployed config is read from environment variables. `config_hash` = sha256 of its canonical JSON; it ties `eval_runs` to the "current" RAG.

### 4.2 Answering

`rag.answer(query)`: retrieve, then prompt Claude with the articles tagged `[art:ID]`, requiring citations in that format. A post-check keeps only citations whose IDs were retrieved and flags any others as invalid. If retrieval returns nothing relevant, the answer is "not found in sources".

### 4.3 Evaluation framework

`vigilex eval [--judge] [--sample N] [--config <json>]` writes one `eval_runs` row per config evaluated.

- **Layer 1, retrieval (deterministic, free):** recall@k (k = 5, 10, 20), nDCG@10, MRR, computed by `ranx` against `qrels` of the expert questions. Default run compares four configs: BM25, vector, hybrid, hybrid + rerank.
- **Layer 2, generation (LLM-as-judge, paid):** on a sample (default 100 questions), the cheapest Claude model scores faithfulness (claims supported by retrieved articles), citation correctness (cited IDs vs gold) and answer relevance, using a tool-use JSON schema for verdicts. The judge is validated once against a small hand-labelled sample; agreement is recorded in the repo.
- **When it runs:** layer 1 on every PR (seed subset) and after every deploy (full corpus, in AWS). Layer 2 only by manual trigger.
- **Regression gate:** CI fails if hybrid recall@10 on the seed subset drops below a threshold committed in the repo.

## 5. Sub-project 2: MCP server and ReAct agent

### 5.1 MCP server

FastMCP, **stdio transport everywhere** (locally and inside Lambda). Tools:

- `search_articles(query: str, k: int = 10)` → hits with id, title, snippet, score
- `get_article(id: str)` → full article
- `get_references(id: str, direction: "out" | "in")` → referenced / referencing article ids and titles

It can also be registered in Claude Desktop or Claude Code for a live demo.

### 5.2 ReAct agent

Hand-written loop (target about 150 lines) over the Bedrock **Converse API with native tool calling**. The model's text between tool calls is the reasoning trace. On start it spawns the MCP server as a subprocess and lists its tools.

- Limits: at most 6 steps, a total token budget, fixed `max_tokens` per call.
- A tool error becomes an observation the model sees, not a crash.
- The agent is a generator yielding events: `thought`, `tool_call`, `observation`, `answer`, `error`. The API forwards them as SSE.

## 6. Sub-project 3: graph agent (LangGraph)

### 6.1 Graph

```
route ──simple──► answer_direct (rag.answer) ──► END
  │ ──ambiguous──► clarify (interrupt) ──► route
  └─complex─► plan ──Send per sub-question──► retrieve ──► expand ──► synthesize ──► verify
                                                 ▲                                    │
                                                 └──── unsupported claims, iter < 2 ──┘
                                                                         ok ──► END
```

- **State** (`TypedDict`): `messages`, `question`, `domain`, `sub_questions`, `evidence: dict[str, Article]` (merge reducer), `visited`, `draft`, `critique`, `iteration`.
- **`route`:** one cheap LLM call (tool-use schema) classifying domain and complexity, and flagging ambiguity.
- **`clarify`:** `interrupt()` with a clarification question; resumed by the client with the same `thread_id`.
- **`plan` → `retrieve`:** decomposes into sub-questions and retrieves for each in parallel via `Send`.
- **`expand`:** follows `article_refs` from the evidence, at most 2 hops and a bounded number of nodes; the LLM prunes irrelevant neighbors.
- **`synthesize`:** cited answer, same citation post-check as §4.2.
- **`verify`:** checks each claim against cited articles; unsupported claims trigger gap queries back to `retrieve`, at most 2 iterations.

### 6.2 Memory

- **Short-term:** `PostgresSaver` checkpoints per `thread_id`: multi-turn follow-ups, resume after failure, interrupt/resume.
- **Long-term:** `PostgresStore`, namespaced by an anonymous session id (signed cookie). Stores a user profile (e.g. entity type, preferred domain), injected into `route` and `synthesize`.
- All state is in Postgres, so any Lambda instance can resume any thread.

### 6.3 Streaming

`graph.astream(stream_mode=["updates", "custom"])` → SSE events `node_start`, `node_end`, `evidence_added`, `interrupt`, `answer`, `error`.

## 7. Sub-project 4a: API and UI

### 7.1 Endpoints

| endpoint | returns |
|---|---|
| `POST /api/rag/answer` | SSE: `hits`, `answer`, `error` |
| `POST /api/agents/react` | SSE: `thought`, `tool_call`, `observation`, `answer`, `error` |
| `POST /api/agents/graph` | SSE as §6.3. Body `{thread_id?, message}` or `{thread_id, resume}` |
| `GET /api/threads/{id}` | graph thread history |
| `GET /api/eval/runs?config=current` | runs for the live config |
| `GET /api/eval/runs/{id}` | one run with per-query breakdown |
| `GET /api/config` | live `RetrieverConfig` and hash |
| `GET /healthz` | liveness and DB reachability |

### 7.2 Pages

- `/` — chat with a mode switch (RAG / ReAct / Graph); clickable `[art:ID]` citations open the article; side trace panel (ReAct steps, or graph nodes with the active one highlighted, plus clarification prompts).
- `/eval` — current config metrics, four-config comparison table, judge scores, run history with inline-SVG sparklines, per-question breakdown of failures.

### 7.3 Protection of the public endpoint

- Demo passcode: Lambda environment variable (sensitive Terraform variable from a GitHub secret, KMS-encrypted at rest). Entered once, stored in a signed cookie. Requests without it get 401.
- Per-session daily request quota (`session_quota`), input length cap, fixed `max_tokens` on every Bedrock call.
- Lambda reserved concurrency = 2.
- AWS Budgets alarms at €10 and €25, emailed.

## 8. Sub-project 4b: infrastructure (Terraform)

**`infra/bootstrap/`** (applied once, locally):

- S3 state bucket, versioned, native locking (`use_lockfile = true`).
- GitHub OIDC provider; **plan role** (read-only, usable from PRs) and **deploy role** (restricted to `main` and the `prod` GitHub environment).

**`infra/main/`** (single `prod` environment; local Docker is the dev environment):

- VPC, 2 AZs, private subnets only, no NAT gateway.
- VPC endpoints: Bedrock Runtime interface endpoint (1 AZ), S3 gateway endpoint.
- Aurora Serverless v2 PostgreSQL 16: 0–2 ACU, auto-pause after 10 minutes idle, IAM database authentication, encrypted storage, 7-day backups.
- ECR repository, lifecycle keeps the last 10 images.
- One container image, two Lambda functions:
  - `api`: FastAPI via AWS Lambda Web Adapter, Function URL with `RESPONSE_STREAM`, reserved concurrency 2, 5-minute timeout.
  - `jobs`: same image, command dispatch `migrate | ingest | eval`. Needed because Aurora is private; this is the cheapest way to run work inside the VPC.
- Private S3 bucket holding the raw dataset, read by `jobs ingest`.
- CloudWatch log groups, 14-day retention. AWS Budgets alarms.

Expected idle cost: about €10/month (Bedrock interface endpoint plus Aurora storage).

## 9. CI/CD (GitHub Actions)

- **`ci.yml`** (pull requests): `uv sync` → ruff → pyright → unit tests → integration tests (pgvector service container) → eval layer 1 on seed with the recall@10 gate (metrics in the job summary) → Playwright e2e against compose with the fake LLM → Docker build → `terraform fmt`, `validate`, `plan` (plan role), plan posted to the PR.
- **`deploy.yml`** (push to `main`, `prod` environment): build and push image tagged with the git SHA → `terraform apply -var image_tag=$SHA` → invoke `jobs migrate` → deployed e2e smoke tests → invoke `jobs eval` (layer 1, full corpus).
- **`eval-judge.yml`** (manual): layer 2 on a sample.
- **`ingest.yml`** (manual): `jobs ingest` on the full corpus.

## 10. Local development

- `docker compose up`: Postgres with pgvector, and the API (same Dockerfile as Lambda) with hot reload.
- `make seed` loads the seed subset. `make test`, `make eval`, `make lint` wrap the uv commands.
- Manual runs call real Bedrock through `AWS_PROFILE`. Tests never do.
- `VIGILEX_LLM=fake` selects the scripted FakeLLM and the hash-based fake embedder (seed questions and articles use their committed real embeddings).

## 11. Testing, error handling, observability

### 11.1 Tests

| level | where | covers |
|---|---|---|
| unit (no I/O) | every PR | RRF; max-chunk aggregation; citation post-check; ReAct loop (scripted tool sequences, step limit, tool error as observation); graph routing; `verify` stops at 2 iterations; config hash; eval metrics on toy qrels; quota logic |
| integration (real Postgres) | every PR, local compose | Italian full-text search; HNSW queries; ingestion idempotency; migrations from empty DB; LangGraph resume on a fresh graph instance; interrupt/resume; MCP server over stdio; API endpoints with httpx including SSE event order; passcode and quota |
| e2e local (Playwright, 1–2 tests) | every PR | chat page streams a trace with citations; `/eval` renders |
| e2e deployed (httpx) | after deploy | `/healthz`; 401 without passcode; one real RAG question returns ≥1 valid citation; graph agent completes a known multi-hop question |
| eval regression | every PR (seed), after deploy (full) | recall@10 threshold |

No coverage percentage target.

### 11.2 Error handling

- Bedrock throttling: botocore `adaptive` retry mode.
- Aurora resuming from pause (up to about 15 s): connection retry with backoff up to 30 s; the UI shows "waking up the database…".
- Structured LLM outputs use tool-use JSON schemas; one retry on schema violation, then an `error` event.
- Every stream ends with `answer` or `error`; streams never hang silently.

### 11.3 Observability

Structured JSON logs (stdlib `logging`) to CloudWatch. Each Bedrock call logs model id, input/output tokens, estimated cost, latency, `request_id` and `thread_id`. Graph runs are replayable from checkpoints.

## 12. Build order

Each step gets its own implementation plan and ends deployable.

1. **Foundations:** uv workspace, `core`, migrations, seed subset, ingestion, docker compose, `ci.yml` (lint/type/unit/integration).
2. **Hybrid RAG + eval layer 1:** retrievers, RRF, rerank, `answer`, `eval` CLI, `eval_runs`, CI gate.
3. **Platform:** API, htmx UI (`/`, `/eval`), passcode/quota, Terraform bootstrap and main, `deploy.yml`, full ingestion in AWS.
4. **MCP server + ReAct agent**, wired into API and UI.
5. **Graph agent** with memory and interrupts, wired into API and UI.
6. **Eval layer 2** (judge) and its validation.
7. Later: React SPA replacing the htmx UI.

## 13. To verify before or during step 1

- ~~JuriFindIT license and release format~~ — verified 2026-10-20: CC BY-NC-SA 4.0; `corpus.parquet` (`id`, `file_name`, `path`, `content`, `reference`), `questions/{train,validation}.parquet` (`question`, `topics`, `normative_area`, `relevant_doc_ids`), synthetic questions separately.
- Availability in `eu-central-1` of: the chosen Claude models through EU inference profiles, Cohere Embed Multilingual v3, Cohere Rerank 3.5.
- Aurora Serverless v2 scale-to-zero support for the chosen PostgreSQL 16 minor version.

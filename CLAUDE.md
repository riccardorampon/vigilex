# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

vigilex: an AI compliance platform for Italian and EU anti-money-laundering (AML) and privacy law. The README lists three planned building blocks:
- hybrid RAG (retrieval-augmented generation) over legal sources
- ReAct agents that use MCP tools
- graph-based legal reasoning

## Commands

- `make up` / `make seed` — local Postgres and seed data
- `make test`, `make lint`, `make fmt`
- Single test: `uv run pytest tests/unit/test_chunking.py::test_giant_token_is_split -v`
- Integration tests need `VIGILEX_TEST_SERVER_URL`: a URL to any database on the test server (Makefile default: compose's `postgres` database), not the test database itself. They skip without it.
- Tests run with `VIGILEX_LLM=fake` (the Makefile sets it); they never call Bedrock.
- `uv run vigilex ask "<domanda>" [--mode bm25|vector|hybrid] [--rerank]` — cited answer (needs Bedrock unless `VIGILEX_LLM=fake`)
- `make eval` (all four configs; rerank needs Bedrock), `make eval-ci` (no AWS; CI gate on hybrid recall@10 ≥ `GATE`)

## Architecture

uv workspace. `packages/core` (`vigilex_core`): settings, DB, SQL migration runner, embedders with a
Postgres embedding cache. `packages/rag` (`vigilex_rag`): chunking, JuriFindIT loader, ingestion, seed
subset, `vigilex` CLI; `retrieval.py` (OR keyword search, HNSW vector search, RRF, rerank), `answer.py` (prompt +
citation post-check), `metrics.py`, `evaluation.py` (`eval_runs`). `vigilex_core` also has `bedrock.py` (one
`bedrock-runtime` client), `llm.py` (`Chat`), `rerank.py` (`Reranker`; `None` in fake mode). Design: `docs/superpowers/specs/`; plans: `docs/superpowers/plans/`.

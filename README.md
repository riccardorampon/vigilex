# vigilex
AI compliance platform for Italian &amp; EU AML and privacy law: hybrid RAG, ReAct agents with MCP, graph-based legal reasoning

## Local development

Requirements: Docker, [uv](https://docs.astral.sh/uv/).

    make up        # Postgres 16 + pgvector
    make seed      # migrations + JuriFindIT seed subset (no AWS needed)
    make test      # unit + integration (sets VIGILEX_LLM=fake)
    make lint
    make eval-ci   # retrieval eval on the seed, no AWS

With AWS credentials (Bedrock in eu-central-1):

    uv run vigilex ask "Quali sono gli obblighi di adeguata verifica della clientela?" --rerank
    make eval      # adds the hybrid+rerank row

Seed retrieval eval (37 validation questions, 1,932 articles):

| config | recall@5 | recall@10 | recall@20 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.096 | 0.120 | 0.156 | 0.199 | 0.334 |
| vector | 0.143 | 0.205 | 0.343 | 0.319 | 0.504 |
| hybrid (RRF) | 0.124 | 0.153 | 0.246 | 0.251 | 0.426 |

Data: JuriFindIT, CC BY-NC-SA 4.0 — see `data/README.md`. Non-commercial use only.

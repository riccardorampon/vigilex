# vigilex
AI compliance platform for Italian &amp; EU AML and privacy law: hybrid RAG, ReAct agents with MCP, graph-based legal reasoning

## Local development

Requirements: Docker, [uv](https://docs.astral.sh/uv/).

    make up        # Postgres 16 + pgvector
    make seed      # migrations + JuriFindIT seed subset (no AWS needed)
    make test      # unit + integration (sets VIGILEX_LLM=fake)
    make lint

Data: JuriFindIT, CC BY-NC-SA 4.0 — see `data/README.md`. Non-commercial use only.

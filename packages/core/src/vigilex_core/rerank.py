import json
from collections.abc import Sequence
from typing import Any, Protocol

from vigilex_core.bedrock import runtime_client
from vigilex_core.settings import Settings


class Reranker(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]: ...


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
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        results = json.loads(resp["body"].read())["results"]
        return [(r["index"], r["relevance_score"]) for r in results]


def make_reranker(settings: Settings) -> Reranker | None:
    """None in fake mode: rerank quality cannot be faked, so callers must skip it."""
    if settings.llm == "fake":
        return None
    return BedrockReranker(runtime_client(settings), settings.rerank_model_id)

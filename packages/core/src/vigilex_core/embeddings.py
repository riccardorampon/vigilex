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
                {
                    "texts": list(texts[i : i + self.BATCH]),
                    "input_type": input_type,
                    "truncate": "END",
                }
            )
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
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
        found: dict[str, list[float]] = {k: e.to_list() for k, e in rows}
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        if missing:
            fresh = dict(
                zip(missing, self.inner.embed(list(missing.values()), input_type), strict=True)
            )
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

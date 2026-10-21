import io
import json
import math

from vigilex_core.embeddings import EMBED_DIM, BedrockEmbedder, FakeEmbedder, make_embedder
from vigilex_core.settings import Settings


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


def test_fake_mode_never_stores():
    s = Settings(
        "postgresql://unused", "fake", "eu-central-1", "cohere.embed-multilingual-v3", "c", "r"
    )
    cached = make_embedder(s, None)  # fake mode never touches conn at construction
    assert cached.store is False and isinstance(cached.inner, FakeEmbedder)

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

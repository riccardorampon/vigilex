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
    not_found = {"text": NOT_FOUND, "citations": [], "invalid_citations": [], "truncated": False}
    assert events == [("hits", []), ("answer", not_found)]


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

    client.app.state.v.chat = Counting()
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

    client.app.state.v.chat = Broken()
    events = frames(client.get("/api/rag/answer", params={"q": "verifica"}).text)
    assert [n for n, _ in events] == ["hits", "error"]
    assert "secret internals" not in json.dumps(events)

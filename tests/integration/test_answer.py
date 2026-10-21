import pytest
from helpers import article
from vigilex_core.embeddings import FakeEmbedder
from vigilex_core.llm import Completion, FakeChat
from vigilex_rag.answer import NOT_FOUND, answer
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig


class RecordingChat(FakeChat):
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return super().complete(system, user, max_tokens)


def one_article(conn, tmp_path, write_release):
    release = write_release(tmp_path, [article(1, "adeguata verifica")])
    ingest(conn, load_release(release), FakeEmbedder())


def test_answer_cites_retrieved_article(conn, tmp_path, write_release):
    one_article(conn, tmp_path, write_release)
    a = answer(conn, "adeguata verifica?", RetrieverConfig(), FakeEmbedder(), FakeChat())
    assert a.citations == [1] and a.invalid_citations == []
    assert [h.id for h in a.hits] == [1]
    assert a.truncated is False


def test_answer_reports_truncation(conn, tmp_path, write_release):
    one_article(conn, tmp_path, write_release)

    class CutChat(FakeChat):
        def complete(self, system, user, max_tokens):
            return Completion("Gli obblighi [art:1] sono", truncated=True)

    a = answer(conn, "adeguata verifica?", RetrieverConfig(), FakeEmbedder(), CutChat())
    assert a.truncated is True and a.citations == [1]


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
@pytest.mark.parametrize("question", ["", "  ", "di il la"])
def test_answer_without_hits_skips_llm(conn, tmp_path, write_release, mode, question):
    one_article(conn, tmp_path, write_release)
    chat = RecordingChat()

    class NoEmbed(FakeEmbedder):
        def embed(self, texts, input_type):
            raise AssertionError("a question with no content words must not be embedded")

    a = answer(conn, question, RetrieverConfig(mode=mode), NoEmbed(), chat)
    assert a.text == NOT_FOUND and a.hits == [] and chat.calls == 0

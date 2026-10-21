from vigilex_core.llm import Completion, FakeChat
from vigilex_rag.answer import NOT_FOUND, generate
from vigilex_rag.retrieval import ArticleHit


class CountingChat(FakeChat):
    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return Completion("Vedi [art:4] e [art:99]", truncated=False)


def test_generate_checks_citations_against_hits():
    hits = [ArticleHit(4, 1.0, "act", "p", "testo")]
    a = generate("domanda", hits, CountingChat())
    assert a.citations == [4] and a.invalid_citations == [99] and a.hits == hits


def test_generate_without_hits_skips_llm():
    chat = CountingChat()
    a = generate("domanda", [], chat)
    assert a.text == NOT_FOUND and chat.calls == 0

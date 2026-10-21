from vigilex_rag.answer import build_prompt, check_citations
from vigilex_rag.retrieval import ArticleHit


def hit(i: int, text: str = "testo") -> ArticleHit:
    return ArticleHit(i, 1.0, "act", f"article_{i}", text)


def test_citations_outside_hits_are_invalid():
    text = "Vale l'art. [art:2] e anche [art:99]; ribadito in [art:2] e [art:3]."
    valid, invalid = check_citations(text, [hit(2), hit(3)])
    assert valid == [2, 3]
    assert invalid == [99]


def test_no_citations():
    assert check_citations("Nessuna fonte.", [hit(1)]) == ([], [])


def test_prompt_tags_every_article_and_truncates_long_text():
    prompt = build_prompt("Chi verifica?", [hit(1, "breve"), hit(2, "x" * 10_000)])
    assert "[art:1]" in prompt and "[art:2]" in prompt
    assert "x" * 4000 in prompt and "x" * 4001 not in prompt
    assert prompt.rstrip().endswith("Domanda: Chi verifica?")

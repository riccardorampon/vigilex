from helpers import article, question
from vigilex_rag.release import ROOT_PLACEHOLDER, load_release


def test_loads_articles_refs_and_questions(tmp_path, write_release):
    d = write_release(
        tmp_path,
        [article(0, ROOT_PLACEHOLDER), article(1, "Art. 1", refs=[2, 0]), article(2, "Art. 2")],
        [question("Chi verifica?", [1, 2])],
    )
    r = load_release(d)
    assert [a.id for a in r.articles] == [1, 2]  # root placeholder skipped
    assert r.refs == [(1, 2)]  # ref to skipped root dropped
    assert len(r.questions) == 1
    q = r.questions[0]
    assert (q.id, q.split, q.domain, q.relevant) == (
        "validation-0",
        "validation",
        "Privacy",
        (1, 2),
    )


def test_dangling_relevant_ids(tmp_path, write_release):
    d = write_release(
        tmp_path,
        [article(1, "Art. 1")],
        [question("partly known", [1, 99]), question("unknown", [99])],
    )
    r = load_release(d)
    assert [(q.id, q.relevant) for q in r.questions] == [("validation-0", (1,))]


def test_missing_question_files_are_fine(tmp_path, write_release):
    assert load_release(write_release(tmp_path, [article(1, "x")])).questions == []

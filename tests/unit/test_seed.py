import random

from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.release import Article, Question, Release, load_release
from vigilex_rag.seed import select_seed, write_seed


def make_release() -> Release:
    arts = [Article(i, "aml" if i < 50 else "civile", f"a{i}", f"testo {i}") for i in range(100)]
    qs = [
        Question(
            "validation-0", "validation", "q aml", "Antiriciclaggio e Antiterrorismo", "t", (1,)
        ),
        Question("validation-1", "validation", "q civ", "Codice civile", "t", (60,)),
        Question("train-0", "train", "q train", "Privacy", "t", (2,)),
    ]
    return Release(arts, [(1, 3), (3, 4), (60, 61)], qs)


def test_select_seed_keeps_gold_hops_and_same_act_distractors():
    seed = select_seed(make_release(), random.Random(0))
    ids = {a.id for a in seed.articles}
    assert [q.id for q in seed.questions] == ["validation-0"]  # validation AML/privacy only
    assert {1, 3} <= ids  # gold + one-hop neighbor
    assert all(a.source_act == "aml" for a in seed.articles)  # distractors from the gold acts
    assert all(s in ids and d in ids for s, d in seed.refs)


def test_write_seed_round_trips_and_covers_all_embeddings(tmp_path):
    seed = select_seed(make_release(), random.Random(0))
    write_seed(seed, tmp_path, FakeEmbedder(), "m")
    back = load_release(tmp_path)
    assert [a.id for a in back.articles] == [a.id for a in seed.articles]
    assert [q.id for q in back.questions] == ["validation-0"]
    import pyarrow.parquet as pq

    n = pq.read_table(tmp_path / "embeddings.parquet").num_rows
    assert n == len(seed.articles) + len(seed.questions)  # one short chunk per article + questions

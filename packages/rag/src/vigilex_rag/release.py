from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT_PLACEHOLDER = "<< file root node >>"
QUESTION_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class Article:
    id: int
    source_act: str
    path: str
    text: str


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    text: str
    domain: str
    topics: str
    relevant: tuple[int, ...]


@dataclass
class Release:
    articles: list[Article]
    refs: list[tuple[int, int]]
    questions: list[Question]


def load_release(d: Path) -> Release:
    """Read a JuriFindIT-shaped directory: corpus.parquet [+ train/validation.parquet]."""
    rows = [
        r
        for r in pq.read_table(d / "corpus.parquet").to_pylist()
        if r["content"].strip() != ROOT_PLACEHOLDER
    ]
    ids = {r["id"] for r in rows}
    articles = [Article(r["id"], r["file_name"], r["path"], r["content"]) for r in rows]
    refs = sorted({(r["id"], dst) for r in rows for dst in r["reference"] or [] if dst in ids})

    questions: list[Question] = []
    for split in QUESTION_SPLITS:
        path = d / f"{split}.parquet"
        if not path.exists():
            continue
        for i, q in enumerate(pq.read_table(path).to_pylist()):
            relevant = tuple(x for x in q["relevant_doc_ids"] if x in ids)
            if not relevant:
                continue  # cannot be evaluated against this corpus
            questions.append(
                Question(
                    id=q.get("id") or f"{split}-{i}",
                    split=split,
                    text=q["question"],
                    domain=q["normative_area"],
                    topics=q["topics"] or "",
                    relevant=relevant,
                )
            )
    return Release(articles, refs, questions)

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CORPUS_SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("file_name", pa.string()),
        ("path", pa.string()),
        ("content", pa.string()),
        ("reference", pa.list_(pa.int64())),
    ]
)
QUESTION_SCHEMA = pa.schema(
    [
        ("question", pa.string()),
        ("topics", pa.string()),
        ("normative_area", pa.string()),
        ("relevant_doc_ids", pa.list_(pa.int64())),
    ]
)


def article(id: int, content: str, refs: list[int] | None = None, act: str = "act") -> dict:
    return {
        "id": id,
        "file_name": act,
        "path": f"article_{id}",
        "content": content,
        "reference": refs or [],
    }


def question(text: str, relevant: list[int], area: str = "Privacy") -> dict:
    return {"question": text, "topics": "t", "normative_area": area, "relevant_doc_ids": relevant}


def write_release(d: Path, corpus: list[dict], validation: list[dict] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(corpus, schema=CORPUS_SCHEMA), d / "corpus.parquet")
    if validation is not None:
        pq.write_table(
            pa.Table.from_pylist(validation, schema=QUESTION_SCHEMA), d / "validation.parquet"
        )
    return d


def login(client, passcode: str = "pw"):
    r = client.post("/login", data={"passcode": passcode}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r

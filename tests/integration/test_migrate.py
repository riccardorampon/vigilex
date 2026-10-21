from pathlib import Path

from vigilex_core.migrate import migrate

REPO = Path(__file__).resolve().parents[2]


def test_migrate_is_idempotent(db_url):
    assert migrate(db_url, REPO / "migrations") == []


def test_italian_stemming_matches_inflections(conn):
    conn.execute(
        "INSERT INTO articles (id, source_act, path, text) VALUES (1, 'act', 'article_1', %s)",
        ("Gli obblighi di adeguata verifica della clientela",),
    )
    hit = conn.execute(
        "SELECT id FROM articles WHERE tsv @@ websearch_to_tsquery('italian', 'obbligo verifiche')"
    ).fetchall()
    assert hit == [(1,)]


def test_vector_index_exists(conn):
    idx = conn.execute(
        "SELECT indexdef FROM pg_indexes "
        "WHERE tablename = 'article_chunks' AND indexdef LIKE '%hnsw%'"
    ).fetchall()
    assert len(idx) == 1

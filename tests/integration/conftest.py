import os
from pathlib import Path

import psycopg
import pytest
from vigilex_core.db import connect
from vigilex_core.migrate import migrate

REPO = Path(__file__).resolve().parents[2]
TEST_DB = "vigilex_test"
TABLES = "articles, article_refs, article_chunks, questions, qrels, embedding_cache"


@pytest.fixture(scope="session")
def db_url() -> str:
    server = os.environ.get("VIGILEX_TEST_SERVER_URL")
    if not server:
        pytest.skip("VIGILEX_TEST_SERVER_URL not set")
    with psycopg.connect(server, autocommit=True) as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        c.execute(f"CREATE DATABASE {TEST_DB}")
    url = server.rsplit("/", 1)[0] + f"/{TEST_DB}"
    migrate(url, REPO / "migrations")
    return url


@pytest.fixture
def conn(db_url):
    with connect(db_url) as c:
        c.execute(f"TRUNCATE {TABLES} CASCADE")
        c.commit()
        yield c

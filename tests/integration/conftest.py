import os
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from vigilex_api.app import create_app
from vigilex_api.config import ApiConfig
from vigilex_core.db import connect
from vigilex_core.migrate import migrate
from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig

REPO = Path(__file__).resolve().parents[2]
TEST_DB = "vigilex_test"
TABLES = (
    "articles, article_refs, article_chunks, questions, qrels, embedding_cache, eval_runs, "
    "session_quota"
)


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


@pytest.fixture
def api_config() -> ApiConfig:
    return ApiConfig(
        passcode="pw",
        secret_key=b"test-key",
        retriever=RetrieverConfig(),  # hybrid, no rerank: fake mode has no reranker
        daily_quota=3,
        global_daily_quota=100,
    )


@pytest.fixture
def client(db_url, conn, api_config):
    settings = Settings(db_url, "fake", "eu-central-1", "cohere.embed-multilingual-v3", "c", "r")
    with TestClient(create_app(api_config, settings), base_url="https://testserver") as c:
        yield c

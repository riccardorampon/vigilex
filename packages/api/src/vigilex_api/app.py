import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from vigilex_core.llm import Chat, make_chat
from vigilex_core.rerank import Reranker, make_reranker
from vigilex_core.settings import Settings

from vigilex_api import api, pages
from vigilex_api.config import ApiConfig

HERE = Path(__file__).parent


@dataclass
class AppState:
    settings: Settings
    config: ApiConfig
    pool: ConnectionPool
    chat: Chat
    reranker: Reranker | None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.commit()  # register_vector queries the catalog; the pool wants idle connections back


def create_app(config: ApiConfig | None = None, settings: Settings | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or Settings.from_env()
    config = config or ApiConfig.from_env(settings)
    reranker = make_reranker(settings)
    if config.retriever.rerank and reranker is None:
        raise RuntimeError(
            "VIGILEX_RERANK=true needs a reranker: set VIGILEX_RERANK=false with VIGILEX_LLM=fake"
        )
    # ponytail: max 4 connections; Lambda reserved concurrency is 2, local dev is one user
    pool = ConnectionPool(
        settings.database_url,
        connection_class=psycopg.Connection,
        min_size=0,
        max_size=4,
        open=False,
        configure=_configure,
        timeout=30,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool.open()
        yield
        pool.close()

    app = FastAPI(title="vigilex", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.state.v = AppState(settings, config, pool, make_chat(settings), reranker)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(api.router)
    app.include_router(pages.router)
    return app

import logging
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from vigilex_api.deps import require_session, state

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    try:
        with state(request).pool.connection(timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception:
        log.exception("health check: database unavailable")
        return JSONResponse({"status": "degraded", "db": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ok", "db": "ok"})


@router.get("/api/config", dependencies=[Depends(require_session)])
def live_config(request: Request) -> dict:
    cfg = state(request).config.retriever
    return {"name": cfg.name(), "hash": cfg.config_hash(), **asdict(cfg)}


@router.get("/api/articles/{article_id}", dependencies=[Depends(require_session)])
def get_article(article_id: int, request: Request) -> dict:
    with state(request).pool.connection() as conn:
        row = conn.execute(
            "SELECT id, source_act, path, text FROM articles WHERE id = %s", (article_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="article not found")
    return dict(zip(("id", "source_act", "path", "text"), row, strict=True))

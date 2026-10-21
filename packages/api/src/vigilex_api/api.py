import logging
from collections.abc import Iterator
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vigilex_core.embeddings import make_embedder
from vigilex_rag.answer import generate
from vigilex_rag.retrieval import retrieve

from vigilex_api.deps import require_session, state
from vigilex_api.quota import consume
from vigilex_api.sse import event

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


SNIPPET_CHARS = 300
STREAM_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


@router.get("/api/rag/answer")
def rag_answer(
    q: str, request: Request, session_id: str = Depends(require_session)
) -> StreamingResponse:
    st = state(request)
    if len(q) > st.config.max_question_chars:
        raise HTTPException(
            status_code=422,
            detail=f"question longer than {st.config.max_question_chars} characters",
        )
    with st.pool.connection() as conn:
        allowed = consume(conn, session_id, st.config.daily_quota, st.config.global_daily_quota)
    if not allowed:
        raise HTTPException(status_code=429, detail="daily question quota reached")

    def stream() -> Iterator[str]:
        # EventSource reconnects when a stream ends; the client closes on answer/error, and
        # every path below ends with exactly one of them.
        try:
            with st.pool.connection() as conn:
                embedder = make_embedder(st.settings, conn)
                hits = retrieve(conn, q, st.config.retriever, embedder, st.reranker)
                conn.commit()  # release locks (cache inserts) before the slow LLM call
            yield event(
                "hits",
                [
                    {
                        "id": h.id,
                        "score": h.score,
                        "source_act": h.source_act,
                        "path": h.path,
                        "snippet": h.text[:SNIPPET_CHARS],
                    }
                    for h in hits
                ],
            )
            result = generate(q, hits, st.chat)
            yield event(
                "answer",
                {
                    "text": result.text,
                    "citations": result.citations,
                    "invalid_citations": result.invalid_citations,
                    "truncated": result.truncated,
                },
            )
        except Exception:
            log.exception("answer stream failed")
            yield event("error", {"message": "Errore interno: riprova tra poco."})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=STREAM_HEADERS)

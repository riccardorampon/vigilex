from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from vigilex_rag.evaluation import CONFIG_NAMES
from vigilex_rag.metrics import METRICS

from vigilex_api.auth import COOKIE, SESSION_MAX_AGE, check_passcode, new_session_id, sign
from vigilex_api.deps import page_session, state

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> Response:
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
async def login(request: Request) -> Response:
    # plain urlencoded form parsed with the stdlib: no python-multipart dependency
    form = parse_qs((await request.body()).decode())
    given = form.get("passcode", [""])[0]
    st = state(request)
    if not check_passcode(given, st.config.passcode):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Passcode errato."}, status_code=401
        )
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        COOKIE,
        sign(new_session_id(), st.config.secret_key),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@router.post("/logout")
def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE)
    return response


def sparkline(values: list[float], width: int = 120, height: int = 24) -> str:
    """SVG polyline points, oldest value on the left; empty below two values."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = width / (len(values) - 1)
    ys = [height / 2 if hi == lo else height - (v - lo) / span * height for v in values]
    return " ".join(f"{i * step:.1f},{y:.1f}" for i, y in enumerate(ys))


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/login", status_code=303)


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request) -> Response:
    if page_session(request) is None:
        return _login_redirect()
    st = state(request)
    live = st.config.retriever
    with st.pool.connection() as conn:
        latest = conn.execute(
            "SELECT DISTINCT ON (config_name) id, config_name, config_hash, n_questions, metrics, "
            "created_at, git_sha FROM eval_runs WHERE split = 'validation' "
            "ORDER BY config_name, created_at DESC, id DESC"
        ).fetchall()
        history = conn.execute(
            "SELECT (metrics->>'recall@10')::float FROM eval_runs WHERE config_hash = %s "
            "ORDER BY created_at DESC, id DESC LIMIT 20",
            (live.config_hash(),),
        ).fetchall()
        missed: list[tuple[str, str]] = []
        live_run = next((r for r in latest if r[2] == live.config_hash()), None)
        if live_run is not None:
            row = conn.execute(
                "SELECT per_query FROM eval_runs WHERE id = %s", (live_run[0],)
            ).fetchone()
            per_query = row[0] if row else {}
            ids = [qid for qid, m in per_query.items() if m["recall@20"] == 0]
            missed = conn.execute(
                "SELECT id, text FROM questions WHERE id = ANY(%s) ORDER BY id", (ids,)
            ).fetchall()
    order = {name: i for i, name in enumerate(CONFIG_NAMES)}
    rows = sorted(latest, key=lambda r: order.get(r[1], len(order)))
    recall_history = [h[0] for h in reversed(history)]
    return templates.TemplateResponse(
        request,
        "eval.html",
        {
            "logged_in": True,
            "live": live,
            "rows": rows,
            "metrics": METRICS,
            "spark": sparkline(recall_history),
            "history_len": len(recall_history),
            "missed": missed,
        },
    )


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> Response:
    if page_session(request) is None:
        return _login_redirect()
    return templates.TemplateResponse(
        request, "chat.html", {"logged_in": True, "live": state(request).config.retriever}
    )

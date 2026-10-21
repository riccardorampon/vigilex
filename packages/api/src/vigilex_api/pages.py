from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from vigilex_api.auth import COOKIE, SESSION_MAX_AGE, check_passcode, new_session_id, sign
from vigilex_api.deps import state

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

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from vigilex_api.auth import COOKIE, verify

if TYPE_CHECKING:
    from vigilex_api.app import AppState


def state(request: Request) -> "AppState":
    return request.app.state.v


def page_session(request: Request) -> str | None:
    return verify(request.cookies.get(COOKIE), state(request).config.secret_key)


def require_session(request: Request) -> str:
    session_id = page_session(request)
    if session_id is None:
        raise HTTPException(status_code=401, detail="login required")
    return session_id

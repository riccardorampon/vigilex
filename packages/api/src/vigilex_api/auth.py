import hashlib
import hmac
import secrets

COOKIE = "vigilex_session"
SESSION_MAX_AGE = 7 * 24 * 3600


def new_session_id() -> str:
    return secrets.token_urlsafe(16)


def _mac(session_id: str, key: bytes) -> str:
    return hmac.new(key, session_id.encode(), hashlib.sha256).hexdigest()


def sign(session_id: str, key: bytes) -> str:
    return f"{session_id}.{_mac(session_id, key)}"


def verify(cookie: str | None, key: bytes) -> str | None:
    """The session id if the cookie carries a valid signature, else None."""
    if not cookie or "." not in cookie:
        return None
    session_id, mac = cookie.rsplit(".", 1)
    return session_id if hmac.compare_digest(mac, _mac(session_id, key)) else None


def check_passcode(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())

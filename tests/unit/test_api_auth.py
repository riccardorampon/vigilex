from vigilex_api.auth import check_passcode, new_session_id, sign, verify


def test_sign_verify_round_trip():
    sid = new_session_id()
    assert verify(sign(sid, b"k"), b"k") == sid


def test_tampered_cookie_is_rejected():
    cookie = sign("abc", b"k")
    assert verify(cookie, b"other-key") is None
    assert verify("xyz" + cookie[3:], b"k") is None  # different session id, old signature
    assert verify(cookie[:-1] + ("0" if cookie[-1] != "0" else "1"), b"k") is None
    assert verify(None, b"k") is None and verify("no-dot", b"k") is None


def test_session_ids_are_unique_and_unguessable():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100 and all(len(i) >= 20 for i in ids)


def test_check_passcode():
    assert check_passcode("pw", "pw")
    assert not check_passcode("pw ", "pw") and not check_passcode("", "pw")

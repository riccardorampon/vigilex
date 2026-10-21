from vigilex_api.quota import GLOBAL, consume


def count(conn, sid):
    row = conn.execute(
        "SELECT count FROM session_quota WHERE session_id = %s AND day = current_date", (sid,)
    ).fetchone()
    return row[0] if row else 0


def test_session_limit(conn):
    assert [consume(conn, "s1", 2, 100) for _ in range(3)] == [True, True, False]
    assert count(conn, "s1") == 3  # refused attempts are counted too


def test_global_limit_spans_sessions(conn):
    assert consume(conn, "a", 10, 2) and consume(conn, "b", 10, 2)
    assert not consume(conn, "c", 10, 2)
    assert count(conn, GLOBAL) == 3

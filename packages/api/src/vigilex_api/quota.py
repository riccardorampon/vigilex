import psycopg

GLOBAL = "__global__"  # one row per day across all sessions: a new login cannot reset it

_BUMP = (
    "INSERT INTO session_quota (session_id, day, count) VALUES (%s, current_date, 1) "
    "ON CONFLICT (session_id, day) DO UPDATE SET count = session_quota.count + 1 "
    "RETURNING count"
)


def consume(conn: psycopg.Connection, session_id: str, limit: int, global_limit: int) -> bool:
    own = conn.execute(_BUMP, (session_id,)).fetchone()
    total = conn.execute(_BUMP, (GLOBAL,)).fetchone()
    conn.commit()
    assert own is not None and total is not None
    return own[0] <= limit and total[0] <= global_limit

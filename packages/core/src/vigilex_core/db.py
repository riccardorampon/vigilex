import psycopg
from pgvector.psycopg import register_vector


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn

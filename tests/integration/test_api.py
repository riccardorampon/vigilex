from helpers import article, login
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def test_healthz_is_public(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "db": "ok"}


def test_api_requires_session(client):
    for path in ("/api/config", "/api/articles/1", "/api/eval/runs", "/api/rag/answer?q=x"):
        r = client.get(path)
        assert r.status_code == 401 and r.json() == {"detail": "login required"}, path


def test_wrong_passcode_is_401_and_sets_no_cookie(client):
    r = client.post("/login", data={"passcode": "nope"}, follow_redirects=False)
    assert r.status_code == 401 and "vigilex_session" not in r.cookies
    assert "Passcode errato" in r.text


def test_login_sets_secure_httponly_cookie_and_logout_clears_it(client):
    r = login(client)
    cookie = r.headers["set-cookie"].lower()
    assert "vigilex_session=" in cookie and "httponly" in cookie and "secure" in cookie
    assert "samesite=lax" in cookie and r.headers["location"] == "/"
    assert client.get("/api/config").status_code == 200
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/api/config").status_code == 401


def test_config_reports_live_retriever(client):
    login(client)
    body = client.get("/api/config").json()
    assert body["name"] == "hybrid" and len(body["hash"]) == 16 and body["top_k"] == 10


def test_article_endpoint(client, conn, tmp_path, write_release):
    corpus = [article(7, "Art. 7 testo")]
    ingest(conn, load_release(write_release(tmp_path, corpus)), FakeEmbedder())
    login(client)
    assert client.get("/api/articles/7").json() == {
        "id": 7,
        "source_act": "act",
        "path": "article_7",
        "text": "Art. 7 testo",
    }
    assert client.get("/api/articles/8").status_code == 404

from helpers import login


def test_pages_redirect_to_login(client):
    for path in ("/", "/eval"):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login", path


def test_chat_page(client):
    login(client)
    page = client.get("/").text
    assert 'id="ask"' in page and "/static/app.js" in page and "hybrid" in page


def test_login_page_and_static_assets(client):
    assert 'name="passcode"' in client.get("/login").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/pico.min.css").headers["content-type"].startswith("text/css")

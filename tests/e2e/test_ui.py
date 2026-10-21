import os
import time

import psycopg
from playwright.sync_api import Page, expect

PASSCODE = os.environ.get("VIGILEX_PASSCODE", "dev")
DB = "postgresql://vigilex:vigilex@localhost:5432/vigilex"


def login(page: Page, base_url: str) -> None:
    page.goto(base_url + "/")
    page.fill("input[name=passcode]", PASSCODE)
    page.click("button[type=submit]")
    expect(page.locator("#ask")).to_be_visible()


def session_count(page: Page) -> int:
    cookie = next(c["value"] for c in page.context.cookies() if c["name"] == "vigilex_session")
    sid = cookie.rsplit(".", 1)[0]
    with psycopg.connect(DB) as conn:
        row = conn.execute(
            "SELECT count FROM session_quota WHERE session_id = %s AND day = current_date", (sid,)
        ).fetchone()
    return row[0] if row else 0


def test_ask_shows_cited_answer_and_article(page: Page, base_url: str):
    login(page, base_url)
    page.fill("#question", "Quali sono gli obblighi di adeguata verifica della clientela?")
    page.click("#ask button")
    expect(page.locator("#hits li")).to_have_count(10)
    expect(page.locator("#answer")).to_contain_text("[art:")
    page.locator("#answer a").first.click()
    expect(page.locator("#article-dialog")).to_be_visible()
    expect(page.locator("#article-text")).not_to_be_empty()


def test_one_question_costs_one_quota_unit(page: Page, base_url: str):
    login(page, base_url)
    page.fill("#question", "obblighi di segnalazione")
    page.click("#ask button")
    expect(page.locator("#answer")).not_to_be_empty()
    time.sleep(4)  # an EventSource left open would reconnect within ~3 s and count again
    assert session_count(page) == 1


def test_eval_page(page: Page, base_url: str):
    login(page, base_url)
    page.goto(base_url + "/eval")
    expect(page.locator("table")).to_contain_text("hybrid")

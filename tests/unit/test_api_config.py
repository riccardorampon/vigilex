import pytest
from vigilex_api.config import ApiConfig
from vigilex_core.settings import Settings

ENV = (
    "VIGILEX_PASSCODE",
    "VIGILEX_SECRET_KEY",
    "VIGILEX_RETRIEVER_MODE",
    "VIGILEX_RERANK",
    "VIGILEX_DAILY_QUOTA",
    "VIGILEX_GLOBAL_DAILY_QUOTA",
    "VIGILEX_INSECURE_DEV",
)
PASS = "p" * 16
KEY = "k" * 32
SETTINGS = Settings("u", "fake", "eu-central-1", "emb", "c", "r")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def test_fails_closed_without_secrets():
    with pytest.raises(RuntimeError, match="VIGILEX_PASSCODE.*VIGILEX_SECRET_KEY"):
        ApiConfig.from_env(SETTINGS)


def test_defaults(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", PASS)
    monkeypatch.setenv("VIGILEX_SECRET_KEY", KEY)
    c = ApiConfig.from_env(SETTINGS)
    assert (c.passcode, c.secret_key) == (PASS, KEY.encode())
    assert (c.daily_quota, c.global_daily_quota) == (30, 300)
    assert c.retriever.name() == "hybrid+rerank" and c.retriever.embed_model_id == "emb"


def test_overrides_and_bad_mode(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", PASS)
    monkeypatch.setenv("VIGILEX_SECRET_KEY", KEY)
    monkeypatch.setenv("VIGILEX_RETRIEVER_MODE", "vector")
    monkeypatch.setenv("VIGILEX_RERANK", "false")
    monkeypatch.setenv("VIGILEX_DAILY_QUOTA", "5")
    c = ApiConfig.from_env(SETTINGS)
    assert c.retriever.name() == "vector" and c.daily_quota == 5
    monkeypatch.setenv("VIGILEX_RETRIEVER_MODE", "magic")
    with pytest.raises(ValueError, match="magic"):
        ApiConfig.from_env(SETTINGS)


def test_short_secrets_are_refused(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", "dev")
    monkeypatch.setenv("VIGILEX_SECRET_KEY", KEY)
    with pytest.raises(RuntimeError, match="VIGILEX_PASSCODE.*16"):
        ApiConfig.from_env(SETTINGS)
    monkeypatch.setenv("VIGILEX_PASSCODE", PASS)
    monkeypatch.setenv("VIGILEX_SECRET_KEY", "short")
    with pytest.raises(RuntimeError, match="VIGILEX_SECRET_KEY.*32"):
        ApiConfig.from_env(SETTINGS)


def test_insecure_dev_allows_short_secrets(monkeypatch):
    monkeypatch.setenv("VIGILEX_PASSCODE", "dev")
    monkeypatch.setenv("VIGILEX_SECRET_KEY", "dev-only-secret")
    monkeypatch.setenv("VIGILEX_INSECURE_DEV", "1")
    assert ApiConfig.from_env(SETTINGS).passcode == "dev"

import pytest
from vigilex_core.settings import Settings


def test_defaults(monkeypatch):
    for k in (
        "VIGILEX_DATABASE_URL",
        "VIGILEX_LLM",
        "VIGILEX_AWS_REGION",
        "VIGILEX_EMBED_MODEL_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env()
    assert s.database_url == "postgresql://vigilex:vigilex@localhost:5432/vigilex"
    assert s.llm == "bedrock"
    assert s.aws_region == "eu-central-1"
    assert s.embed_model_id == "cohere.embed-multilingual-v3"


def test_invalid_llm_rejected(monkeypatch):
    monkeypatch.setenv("VIGILEX_LLM", "openai")
    with pytest.raises(ValueError, match="VIGILEX_LLM"):
        Settings.from_env()

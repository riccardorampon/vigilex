import os
from dataclasses import dataclass
from typing import Literal, cast

LLM_MODES = ("bedrock", "fake")


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm: Literal["bedrock", "fake"]
    aws_region: str
    embed_model_id: str
    chat_model_id: str
    rerank_model_id: str

    @classmethod
    def from_env(cls) -> "Settings":
        llm = os.environ.get("VIGILEX_LLM", "bedrock")
        if llm not in LLM_MODES:
            raise ValueError(f"VIGILEX_LLM must be one of {LLM_MODES}, got {llm!r}")
        return cls(
            database_url=os.environ.get(
                "VIGILEX_DATABASE_URL", "postgresql://vigilex:vigilex@localhost:5432/vigilex"
            ),
            llm=cast(Literal["bedrock", "fake"], llm),
            aws_region=os.environ.get("VIGILEX_AWS_REGION", "eu-central-1"),
            embed_model_id=os.environ.get("VIGILEX_EMBED_MODEL_ID", "cohere.embed-multilingual-v3"),
            chat_model_id=os.environ.get("VIGILEX_CHAT_MODEL_ID", "eu.anthropic.claude-sonnet-4-6"),
            rerank_model_id=os.environ.get("VIGILEX_RERANK_MODEL_ID", "cohere.rerank-v3-5:0"),
        )

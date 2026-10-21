import os
from dataclasses import dataclass, field

from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig

MODES = ("bm25", "vector", "hybrid")


@dataclass(frozen=True)
class ApiConfig:
    passcode: str
    secret_key: bytes
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    daily_quota: int = 30
    global_daily_quota: int = 300
    max_question_chars: int = 1000

    @classmethod
    def from_env(cls, settings: Settings) -> "ApiConfig":
        missing = [k for k in ("VIGILEX_PASSCODE", "VIGILEX_SECRET_KEY") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"refusing to start without {', '.join(missing)}")
        mode = os.environ.get("VIGILEX_RETRIEVER_MODE", "hybrid")
        if mode not in MODES:
            raise ValueError(f"VIGILEX_RETRIEVER_MODE must be one of {MODES}, got {mode!r}")
        rerank = os.environ.get("VIGILEX_RERANK", "true").lower() in ("1", "true", "yes")
        return cls(
            passcode=os.environ["VIGILEX_PASSCODE"],
            secret_key=os.environ["VIGILEX_SECRET_KEY"].encode(),
            retriever=RetrieverConfig(
                mode=mode,  # type: ignore[arg-type]
                rerank=rerank,
                embed_model_id=settings.embed_model_id,
            ),
            daily_quota=int(os.environ.get("VIGILEX_DAILY_QUOTA", "30")),
            global_daily_quota=int(os.environ.get("VIGILEX_GLOBAL_DAILY_QUOTA", "300")),
        )

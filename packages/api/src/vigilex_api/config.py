import os
from dataclasses import dataclass, field

from vigilex_core.settings import Settings
from vigilex_rag.retrieval import RetrieverConfig

MODES = ("bm25", "vector", "hybrid")
MIN_PASSCODE_CHARS = 16  # no login throttling: length is what makes guessing impractical
MIN_SECRET_KEY_BYTES = 32


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
        passcode, secret_key = (
            os.environ["VIGILEX_PASSCODE"],
            os.environ["VIGILEX_SECRET_KEY"].encode(),
        )
        if os.environ.get("VIGILEX_INSECURE_DEV") != "1":  # local compose only, bound to 127.0.0.1
            if len(passcode) < MIN_PASSCODE_CHARS:
                raise RuntimeError(f"VIGILEX_PASSCODE must be at least {MIN_PASSCODE_CHARS} chars")
            if len(secret_key) < MIN_SECRET_KEY_BYTES:
                raise RuntimeError(
                    f"VIGILEX_SECRET_KEY must be at least {MIN_SECRET_KEY_BYTES} bytes"
                )
        mode = os.environ.get("VIGILEX_RETRIEVER_MODE", "hybrid")
        if mode not in MODES:
            raise ValueError(f"VIGILEX_RETRIEVER_MODE must be one of {MODES}, got {mode!r}")
        rerank = os.environ.get("VIGILEX_RERANK", "true").lower() in ("1", "true", "yes")
        return cls(
            passcode=passcode,
            secret_key=secret_key,
            retriever=RetrieverConfig(
                mode=mode,  # type: ignore[arg-type]
                rerank=rerank,
                embed_model_id=settings.embed_model_id,
            ),
            daily_quota=int(os.environ.get("VIGILEX_DAILY_QUOTA", "30")),
            global_daily_quota=int(os.environ.get("VIGILEX_GLOBAL_DAILY_QUOTA", "300")),
        )

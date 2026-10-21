import logging
import re
import time
from typing import Any, Protocol

from vigilex_core.bedrock import runtime_client
from vigilex_core.settings import Settings

log = logging.getLogger(__name__)
_FIRST_CITATION = re.compile(r"\[art:\d+\]")


class Chat(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class BedrockChat:
    def __init__(self, client: Any, model_id: str):
        self.client = client
        self.model_id = model_id

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        start = time.monotonic()
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = resp["usage"]
        log.info(
            "chat model=%s input_tokens=%d output_tokens=%d stop=%s latency_ms=%d",
            self.model_id,
            usage["inputTokens"],
            usage["outputTokens"],
            resp["stopReason"],
            (time.monotonic() - start) * 1000,
        )
        return "".join(b.get("text", "") for b in resp["output"]["message"]["content"])


class FakeChat:
    """Deterministic stand-in: cites the first article tag it was given."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        m = _FIRST_CITATION.search(user)
        return f"Risposta di prova {m.group(0)}" if m else "Risposta di prova."


def make_chat(settings: Settings) -> Chat:
    if settings.llm == "fake":
        return FakeChat()
    return BedrockChat(runtime_client(settings), settings.chat_model_id)

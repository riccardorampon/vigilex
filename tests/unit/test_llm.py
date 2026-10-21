from vigilex_core.llm import BedrockChat, FakeChat


class StubConverse:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 12, "outputTokens": 3},
            "stopReason": "end_turn",
        }


def test_bedrock_chat_sends_system_user_and_fixed_inference_config():
    stub = StubConverse("Roma [art:5]")
    out = BedrockChat(stub, "eu.anthropic.claude-sonnet-4-6").complete("sys", "domanda", 300)
    assert out == "Roma [art:5]"
    (call,) = stub.calls
    assert call["modelId"] == "eu.anthropic.claude-sonnet-4-6"
    assert call["system"] == [{"text": "sys"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "domanda"}]}]
    assert call["inferenceConfig"] == {"maxTokens": 300, "temperature": 0}


def test_fake_chat_cites_first_article_in_prompt():
    assert (
        FakeChat().complete("s", "[art:7] testo\n[art:9] altro", 10) == "Risposta di prova [art:7]"
    )
    assert FakeChat().complete("s", "nessun articolo", 10) == "Risposta di prova."

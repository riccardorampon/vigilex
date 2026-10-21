from vigilex_core.llm import BedrockChat, Completion, FakeChat


class StubConverse:
    def __init__(self, text: str, stop: str = "end_turn"):
        self.text = text
        self.stop = stop
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 12, "outputTokens": 3},
            "stopReason": self.stop,
        }


def test_bedrock_chat_sends_system_user_and_fixed_inference_config():
    stub = StubConverse("Roma [art:5]")
    out = BedrockChat(stub, "eu.anthropic.claude-sonnet-4-6").complete("sys", "domanda", 300)
    assert out == Completion("Roma [art:5]", truncated=False)
    (call,) = stub.calls
    assert call["modelId"] == "eu.anthropic.claude-sonnet-4-6"
    assert call["system"] == [{"text": "sys"}]
    assert call["messages"] == [{"role": "user", "content": [{"text": "domanda"}]}]
    assert call["inferenceConfig"] == {"maxTokens": 300, "temperature": 0}


def test_bedrock_chat_flags_answers_cut_by_max_tokens():
    out = BedrockChat(StubConverse("Gli obblighi sono", stop="max_tokens"), "m").complete(
        "s", "u", 5
    )
    assert out == Completion("Gli obblighi sono", truncated=True)


def test_fake_chat_cites_first_article_in_prompt():
    out = FakeChat().complete("s", "[art:7] testo\n[art:9] altro", 10)
    assert out == Completion("Risposta di prova [art:7]", truncated=False)
    assert FakeChat().complete("s", "nessun articolo", 10).text == "Risposta di prova."

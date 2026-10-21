import json


def event(name: str, data: object) -> str:
    """One Server-Sent Events frame. json.dumps never emits raw newlines, so data stays one line."""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

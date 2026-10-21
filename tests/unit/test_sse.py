import json

from vigilex_api.sse import event


def test_event_is_one_frame_with_single_line_json():
    frame = event("answer", {"text": "riga 1\nriga 2", "n": 1})
    assert frame.startswith("event: answer\ndata: ") and frame.endswith("\n\n")
    data_line = frame.split("\n")[1]
    assert json.loads(data_line.removeprefix("data: ")) == {"text": "riga 1\nriga 2", "n": 1}
    assert frame.count("\n") == 3  # newlines inside the text are JSON-escaped


def test_event_keeps_unicode():
    assert "perché" in event("x", {"t": "perché"})

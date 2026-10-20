# Bedrock's Cohere embed rejects texts over 2048 characters; keep a safety margin.
MAX_CHARS = 1800
OVERLAP = 200


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Greedy word-packing into chunks of at most max_chars, with ~overlap chars repeated."""
    words: list[str] = []
    for w in text.split():
        words.extend(w[i : i + max_chars] for i in range(0, len(w), max_chars))

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, size = start + 1, len(words[start])
        while end < len(words) and size + 1 + len(words[end]) <= max_chars:
            size += 1 + len(words[end])
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        back, kept = end, 0
        while back - 1 > start and kept + 1 + len(words[back - 1]) <= overlap:
            back -= 1
            kept += 1 + len(words[back])
        start = back
    return chunks

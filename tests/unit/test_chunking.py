from vigilex_rag.chunking import chunk_text


def test_empty_text_has_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("  \n\t ") == []


def test_short_text_is_one_normalized_chunk():
    assert chunk_text("Art. 1\nObblighi   di verifica") == ["Art. 1 Obblighi di verifica"]


def test_long_text_chunks_are_bounded_overlapping_and_complete():
    words = [f"parola{i}" for i in range(2000)]
    chunks = chunk_text(" ".join(words), max_chars=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.split()[-1] in b.split()  # consecutive chunks overlap
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words)


def test_giant_token_is_split():
    chunks = chunk_text("inizio " + "x" * 5000 + " fine", max_chars=1800, overlap=200)
    assert all(len(c) <= 1800 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks).count("x") >= 5000

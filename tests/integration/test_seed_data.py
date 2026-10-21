from pathlib import Path

import pytest
from vigilex_core.embeddings import FakeEmbedder
from vigilex_rag.seed import load_seed

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "data" / "seed"


class NoCalls(FakeEmbedder):
    def embed(self, texts, input_type):
        if texts:
            raise AssertionError(f"seed must be fully cached, {len(texts)} misses")
        return []


def test_committed_seed_loads_without_embedding_calls(conn):
    if not (SEED / "embeddings.parquet").exists():
        pytest.fail("data/seed not built yet — run Task 7 step 6")
    from vigilex_core.embeddings import CachedEmbedder
    from vigilex_core.settings import Settings

    model = Settings.from_env().embed_model_id
    stats = load_seed(conn, Path(SEED), CachedEmbedder(conn, model, NoCalls(), store=False))
    assert stats.questions == 37
    assert 1500 <= stats.articles <= 2000

import argparse
import logging
import random
from pathlib import Path

from vigilex_core.db import connect
from vigilex_core.embeddings import EMBED_DIM, bedrock_embedder, make_embedder
from vigilex_core.llm import make_chat
from vigilex_core.migrate import migrate
from vigilex_core.rerank import make_reranker
from vigilex_core.settings import Settings

from vigilex_rag.answer import answer
from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release
from vigilex_rag.retrieval import RetrieverConfig
from vigilex_rag.seed import load_seed, select_seed, write_seed


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="vigilex")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="apply SQL migrations")
    m.add_argument("--dir", type=Path, default=Path("migrations"))
    i = sub.add_parser("ingest", help="ingest a JuriFindIT-shaped directory")
    i.add_argument("--source", type=Path, required=True)
    sub.add_parser("check-bedrock", help="embed one text to verify Bedrock access")
    b = sub.add_parser("build-seed", help="build data/seed from a full release (calls Bedrock)")
    b.add_argument("--release", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/seed"))
    s = sub.add_parser("load-seed", help="load the committed seed subset (no Bedrock calls)")
    s.add_argument("--dir", type=Path, default=Path("data/seed"))
    a = sub.add_parser("ask", help="answer a question with cited articles")
    a.add_argument("question")
    a.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    a.add_argument("--rerank", action="store_true")
    args = p.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "migrate":
        print("applied:", migrate(settings.database_url, args.dir) or "nothing")
    elif args.cmd == "ingest":
        with connect(settings.database_url) as conn:
            print(ingest(conn, load_release(args.source), make_embedder(settings, conn)))
    elif args.cmd == "check-bedrock":
        (v,) = bedrock_embedder(settings).embed(["prova"], "search_query")
        assert len(v) == EMBED_DIM, f"expected {EMBED_DIM} dims, got {len(v)}"
        print(f"ok: {settings.embed_model_id} in {settings.aws_region} returned {len(v)} dims")
    elif args.cmd == "build-seed":
        seed = select_seed(load_release(args.release), random.Random(0))
        write_seed(seed, args.out, bedrock_embedder(settings), settings.embed_model_id)
        print(f"seed: {len(seed.articles)} articles, {len(seed.questions)} questions -> {args.out}")
    elif args.cmd == "load-seed":
        with connect(settings.database_url) as conn:
            print(load_seed(conn, args.dir, make_embedder(settings, conn)))
    elif args.cmd == "ask":
        cfg = RetrieverConfig(
            mode=args.mode, rerank=args.rerank, embed_model_id=settings.embed_model_id
        )
        with connect(settings.database_url) as conn:
            result = answer(
                conn,
                args.question,
                cfg,
                make_embedder(settings, conn),
                make_chat(settings),
                make_reranker(settings),
            )
        print(result.text, "\n")
        for h in result.hits:
            mark = "*" if h.id in result.citations else " "
            print(f"{mark} [art:{h.id}] {h.score:.4f} {h.source_act} {h.path}")
        if result.invalid_citations:
            print("invalid citations (not retrieved):", result.invalid_citations)

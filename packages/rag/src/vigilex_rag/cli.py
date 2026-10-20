import argparse
import logging
from pathlib import Path

from vigilex_core.db import connect
from vigilex_core.embeddings import EMBED_DIM, bedrock_embedder, make_embedder
from vigilex_core.migrate import migrate
from vigilex_core.settings import Settings

from vigilex_rag.ingest import ingest
from vigilex_rag.release import load_release


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="vigilex")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("migrate", help="apply SQL migrations")
    m.add_argument("--dir", type=Path, default=Path("migrations"))
    i = sub.add_parser("ingest", help="ingest a JuriFindIT-shaped directory")
    i.add_argument("--source", type=Path, required=True)
    sub.add_parser("check-bedrock", help="embed one text to verify Bedrock access")
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

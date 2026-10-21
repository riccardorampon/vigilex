import logging
import re
from dataclasses import dataclass

import psycopg
from vigilex_core.embeddings import Embedder
from vigilex_core.llm import Chat
from vigilex_core.rerank import Reranker

from vigilex_rag.retrieval import ArticleHit, RetrieverConfig, retrieve

log = logging.getLogger(__name__)

CITATION = re.compile(r"\[art:(\d+)\]")
MAX_ANSWER_TOKENS = 1024
PROMPT_DOC_CHARS = 4000
NOT_FOUND = "Non ho trovato fonti pertinenti per rispondere a questa domanda."
SYSTEM_PROMPT = (
    "Sei un assistente per la compliance antiriciclaggio e privacy (diritto italiano ed europeo). "
    "Rispondi in italiano usando SOLO gli articoli forniti. "
    "Dopo ogni affermazione cita l'articolo da cui proviene con il suo tag, ad esempio [art:123]. "
    "Se gli articoli non bastano per rispondere, dillo esplicitamente. "
    "Non citare articoli che non sono nell'elenco. "
    "Le risposte sono informative e non costituiscono consulenza legale."
)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[int]
    invalid_citations: list[int]
    hits: list[ArticleHit]


def build_prompt(query: str, hits: list[ArticleHit]) -> str:
    docs = "\n\n".join(
        f"[art:{h.id}] ({h.source_act} — {h.path})\n{h.text[:PROMPT_DOC_CHARS]}" for h in hits
    )
    return f"Articoli:\n\n{docs}\n\nDomanda: {query}\n"


def check_citations(text: str, hits: list[ArticleHit]) -> tuple[list[int], list[int]]:
    allowed = {h.id for h in hits}
    cited = list(dict.fromkeys(int(m) for m in CITATION.findall(text)))
    return [c for c in cited if c in allowed], [c for c in cited if c not in allowed]


def answer(
    conn: psycopg.Connection,
    query: str,
    cfg: RetrieverConfig,
    embedder: Embedder,
    chat: Chat,
    reranker: Reranker | None = None,
) -> Answer:
    hits = retrieve(conn, query, cfg, embedder, reranker)
    if not hits:
        return Answer(NOT_FOUND, [], [], [])
    text = chat.complete(SYSTEM_PROMPT, build_prompt(query, hits), MAX_ANSWER_TOKENS)
    valid, invalid = check_citations(text, hits)
    if invalid:
        log.warning("answer cited articles that were not retrieved: %s", invalid)
    return Answer(text, valid, invalid, hits)

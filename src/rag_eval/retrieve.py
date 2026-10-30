import json
import re
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from .config import RERANK_MODEL
from .index import chunks_path, client, embedder

MODES = ("vector", "hybrid", "hybrid_rerank")


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


@lru_cache
def _bm25(chunking: str):
    chunks = [json.loads(line) for line in chunks_path(chunking).read_text().splitlines()]
    return chunks, BM25Okapi([_tok(c["text"]) for c in chunks])


@lru_cache
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL)


def vector(chunking: str, q: str, n: int) -> list[dict]:
    qv = next(iter(embedder().query_embed(q))).tolist()
    return [p.payload for p in client().query_points(chunking, query=qv, limit=n).points]


def bm25(chunking: str, q: str, n: int) -> list[dict]:
    chunks, index = _bm25(chunking)
    scores = index.get_scores(_tok(q))
    return [chunks[i] for i in sorted(range(len(chunks)), key=scores.__getitem__, reverse=True)[:n]]


def rrf(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion over ranked lists of chunks."""
    score, by_id = {}, {}
    for ranked in lists:
        for rank, c in enumerate(ranked):
            by_id[c["id"]] = c
            score[c["id"]] = score.get(c["id"], 0) + 1 / (k + rank + 1)
    return [by_id[i] for i in sorted(score, key=score.get, reverse=True)]


def search(q: str, chunking: str = "recursive", mode: str = "hybrid_rerank", k: int = 5) -> list[dict]:
    if mode == "vector":
        return vector(chunking, q, k)
    hits = rrf([vector(chunking, q, 50), bm25(chunking, q, 50)])[:20]
    if mode == "hybrid_rerank":
        scores = list(_reranker().rerank(q, [h["text"] for h in hits]))
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
    return hits[:k]

from fastapi import FastAPI
from pydantic import BaseModel

from .generate import answer
from .index import build
from .retrieve import MODES, search

app = FastAPI(title="rag-eval")


class Query(BaseModel):
    question: str
    chunking: str = "recursive"
    mode: str = "hybrid_rerank"
    k: int = 5


@app.post("/search")
def _search(q: Query):
    return search(q.question, q.chunking, q.mode, q.k)


@app.post("/ask")
def ask(q: Query):
    hits = search(q.question, q.chunking, q.mode, q.k)
    return {**answer(q.question, hits), "sources": [{"n": i, "doc": h["doc"], "text": h["text"]}
                                                    for i, h in enumerate(hits, 1)]}


@app.post("/ingest")
def ingest(chunking: str = "recursive"):
    return {"chunks": build(chunking)}


@app.get("/health")
def health():
    return {"ok": True, "modes": MODES}

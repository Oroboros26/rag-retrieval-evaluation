"""Chunk, embed and upsert. Re-runnable: chunk ids are content hashes and the collection is rebuilt."""
import json
import sys
import uuid
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from .chunk import CHUNKERS
from .config import DATA, EMBED_MODEL


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(path=str(DATA / "qdrant"))


@lru_cache
def embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


def chunks_path(chunking: str):
    return DATA / f"chunks_{chunking}.jsonl"


def build(chunking: str) -> int:
    chunks = []
    for f in sorted((DATA / "raw").glob("*.txt")):
        title, _, body = f.read_text().partition("\n\n")
        for text in CHUNKERS[chunking](body):
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunking}:{title}:{text}"))
            chunks.append({"id": cid, "doc": title, "text": text})
    chunks_path(chunking).write_text("\n".join(json.dumps(c) for c in chunks))

    vectors = list(embedder().embed([c["text"] for c in chunks]))
    c = client()
    c.recreate_collection(chunking, vectors_config=models.VectorParams(
        size=len(vectors[0]), distance=models.Distance.COSINE))
    c.upsert(chunking, points=[
        models.PointStruct(id=ch["id"], vector=v.tolist(), payload=ch)
        for ch, v in zip(chunks, vectors)])
    return len(chunks)


if __name__ == "__main__":
    for name in sys.argv[1:] or CHUNKERS:
        print(name, build(name), "chunks")

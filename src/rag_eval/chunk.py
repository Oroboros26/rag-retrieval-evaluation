import re


def fixed(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    pieces = (text[i:i + size] for i in range(0, len(text), size - overlap))
    return [p for p in pieces if p.strip()]


def recursive(text: str, size: int = 800) -> list[str]:
    """Pack paragraphs, falling back to sentences for oversized paragraphs."""
    chunks, cur = [], ""
    for para in re.split(r"\n{2,}", text):
        for piece in re.split(r"(?<=[.!?])\s+", para) if len(para) > size else [para]:
            if cur and len(cur) + len(piece) + 1 > size:
                chunks.append(cur)
                cur = ""
            cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


CHUNKERS = {"fixed": fixed, "recursive": recursive}

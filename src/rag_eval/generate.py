import re
from functools import lru_cache

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL

REFUSAL = "INSUFFICIENT_CONTEXT"
SYSTEM = (
    "Answer using ONLY the numbered context. Cite sources inline like [1][2]. "
    f"If the context does not contain the answer, reply exactly: {REFUSAL}"
)


@lru_cache
def llm() -> Groq:
    return Groq(api_key=GROQ_API_KEY, max_retries=6)


def parse(text: str) -> dict:
    return {
        "answer": text,
        "refused": REFUSAL in text,
        "citations": sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)}),
    }


def answer(question: str, chunks: list[dict]) -> dict:
    ctx = "\n\n".join(f"[{i}] ({c['doc']}) {c['text']}" for i, c in enumerate(chunks, 1))
    r = llm().chat.completions.create(
        model=GROQ_MODEL, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
    )
    out = parse(r.choices[0].message.content.strip())
    out["tokens_in"], out["tokens_out"] = r.usage.prompt_tokens, r.usage.completion_tokens
    return out

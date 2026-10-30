"""Run every chunking x retrieval config over the golden set; write eval/results/<ts>.json + table."""
import itertools
import json
import statistics
import sys
import time
from datetime import UTC, datetime

from rag_eval.chunk import CHUNKERS
from rag_eval.config import EVAL, GROQ_MODEL, PRICE_IN, PRICE_OUT
from rag_eval.generate import answer, llm
from rag_eval.retrieve import MODES, search

K = 5


def judge(question: str, ref: str, pred: str) -> bool:
    r = llm().chat.completions.create(
        model=GROQ_MODEL, temperature=0, max_tokens=3,
        messages=[{"role": "user", "content":
                   f"Question: {question}\nReference answer: {ref}\nCandidate answer: {pred}\n"
                   "Does the candidate agree with the reference? Reply YES or NO."}])
    return r.choices[0].message.content.strip().upper().startswith("YES")


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def run_config(chunking: str, mode: str, golden: list[dict]) -> dict:
    lat, cost, hit5, rr, correct, cited, refused_un, refused_ans = [], [], [], [], [], [], [], []
    for g in golden:
        t0 = time.perf_counter()
        hits = search(g["question"], chunking, mode, K)
        out = answer(g["question"], hits)
        lat.append(time.perf_counter() - t0)
        cost.append((out["tokens_in"] * PRICE_IN + out["tokens_out"] * PRICE_OUT) / 1e6)
        if g["answerable"]:
            docs = [h["doc"] for h in hits]
            hit5.append(g["doc"] in docs)
            rr.append(1 / (docs.index(g["doc"]) + 1) if g["doc"] in docs else 0)
            refused_ans.append(out["refused"])
            correct.append(not out["refused"] and judge(g["question"], g["answer"], out["answer"]))
            cited.append(bool(out["citations"]) and max(out["citations"]) <= len(hits))
        else:
            refused_un.append(out["refused"])
    m = statistics.mean
    return {
        "chunking": chunking, "mode": mode,
        "recall@5": m(hit5), "mrr": m(rr), "correct": m(correct), "citation_ok": m(cited),
        "false_refusal": m(refused_ans), "refusal_on_unanswerable": m(refused_un),
        "p50_s": pct(lat, 0.5), "p95_s": pct(lat, 0.95), "cost_per_1k_usd": m(cost) * 1000,
    }


def table(rows: list[dict]) -> str:
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None  # quick run: first n answerable + all unanswerable
    golden = [json.loads(line) for line in (EVAL / "golden.jsonl").read_text().splitlines()]
    if n:
        golden = [g for g in golden if g["answerable"]][:n] + [g for g in golden if not g["answerable"]]
    rows = [run_config(c, m, golden) for c, m in itertools.product(CHUNKERS, MODES)]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (EVAL / "results" / f"{ts}.json").write_text(json.dumps(
        {"timestamp": ts, "llm": GROQ_MODEL, "n_questions": len(golden), "results": rows}, indent=2))
    print(table(rows))


if __name__ == "__main__":
    main()

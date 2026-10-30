"""Draft the golden set: LLM-written Q/A from sampled chunks + hand-written unanswerable questions.
Review eval/golden.jsonl by hand before trusting the numbers."""
import json
import random

from rag_eval.config import EVAL, GROQ_MODEL
from rag_eval.generate import llm
from rag_eval.index import chunks_path

N_ANSWERABLE = 30
UNANSWERABLE = [
    "What was the retail price of the 2027 flagship alpine ski of a leading brand?",
    "Who won the men's overall alpine skiing World Cup in 2035?",
    "What is the phone number of the Chamonix tourist office?",
    "Which restaurant in Zermatt serves the best fondue?",
    "What is the weather forecast for Mont Blanc tomorrow?",
    "What is the capital of Australia?",
    "How do I make a Margherita pizza?",
    "What is the current share price of a major ski equipment company?",
]


def main() -> None:
    chunks = [json.loads(line) for line in chunks_path("recursive").read_text().splitlines()]
    random.Random(42).shuffle(chunks)
    rows = []
    for c in (c for c in chunks if len(c["text"]) > 400):
        if len(rows) == N_ANSWERABLE:
            break
        r = llm().chat.completions.create(
            model=GROQ_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       "Write one specific, self-contained factual question answerable ONLY from this "
                       "passage (no 'the passage' wording), and its short answer. "
                       'Reply as JSON {"question": "...", "answer": "..."}.\n\n' + c["text"]}])
        qa = json.loads(r.choices[0].message.content)
        rows.append({"question": qa["question"], "answer": qa["answer"], "doc": c["doc"], "answerable": True})
    rows += [{"question": q, "answer": None, "doc": None, "answerable": False} for q in UNANSWERABLE]
    (EVAL / "golden.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(len(rows), "questions")


if __name__ == "__main__":
    main()

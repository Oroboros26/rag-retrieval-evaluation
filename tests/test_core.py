from rag_eval.chunk import fixed, recursive
from rag_eval.generate import REFUSAL, parse
from rag_eval.retrieve import rrf


def test_rrf_prefers_docs_ranked_high_in_both_lists():
    a, b, c = ({"id": x} for x in "abc")
    assert next(d["id"] for d in rrf([[a, b], [b, c]])) == "b"


def test_chunkers_respect_size_and_cover_text():
    text = "Sentence one. " * 200
    assert all(len(c) <= 800 for c in fixed(text) + recursive(text))
    assert len(recursive(text)) > 1


def test_parse_citations_and_refusal():
    assert parse("It is X [2][1].")["citations"] == [1, 2]
    assert parse(REFUSAL)["refused"] and not parse("Fine [1]")["refused"]

PY = .venv/bin/python

.PHONY: setup data run test eval golden lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

data:
	$(PY) -m rag_eval.corpus && $(PY) -m rag_eval.index

run:
	.venv/bin/uvicorn rag_eval.api:app --reload

test:
	.venv/bin/pytest -q

golden:
	$(PY) eval/make_golden.py

eval:
	$(PY) eval/run.py

lint:
	.venv/bin/ruff check .

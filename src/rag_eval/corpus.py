"""Download the Wikipedia subset listed in eval/corpus_titles.txt into data/raw/."""
import re
import time

import requests

from .config import DATA, EVAL

API = "https://en.wikipedia.org/w/api.php"


def fetch(title: str) -> str:
    for wait in (2, 5, 15, 30, 60):
        r = _get(title)
        if r.status_code != 429:
            break
        time.sleep(wait)
    r.raise_for_status()
    return next(iter(r.json()["query"]["pages"].values())).get("extract", "")


def _get(title: str) -> requests.Response:
    return requests.get(
        API,
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "format": "json", "redirects": 1},
        headers={"User-Agent": "rag-eval/0.1 (portfolio project)"},
        timeout=30,
    )


def main() -> None:
    out = DATA / "raw"
    out.mkdir(parents=True, exist_ok=True)
    titles = [t.strip() for t in (EVAL / "corpus_titles.txt").read_text().splitlines() if t.strip()]
    ok = 0
    for t in titles:
        dest = out / (re.sub(r"\W+", "_", t) + ".txt")
        if dest.exists():
            ok += 1
            continue
        time.sleep(1)
        text = fetch(t)
        if len(text) < 500:
            print(f"skip (missing/short): {t}")
            continue
        dest.write_text(f"{t}\n\n{text}")
        ok += 1
    chars = sum(f.stat().st_size for f in out.glob("*.txt"))
    print(f"{ok}/{len(titles)} articles, {chars / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

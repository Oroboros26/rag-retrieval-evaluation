import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
EVAL = ROOT / "eval"

# GROK_API_KEY accepted as a common misspelling of GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# USD per 1M tokens; unverified defaults, check console.groq.com pricing
PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0.59"))
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0.79"))

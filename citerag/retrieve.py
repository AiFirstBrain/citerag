"""BM25 keyword retrieval. Zero cost, zero setup, surprisingly strong baseline.

Planned upgrade (and a good public post): add embeddings, fuse with BM25 (hybrid), and show the eval delta.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

# BM25Plus, not BM25Okapi: Okapi's IDF = log((N-n+.5)/(n+.5)) is exactly 0 for a term found in half the
# chunks (e.g. any term in a 2-chunk corpus), so every score collapsed to 0 and the system abstained on
# everything. Plus keeps IDF positive. Found by the Day 1 eval run.
from rank_bm25 import BM25Plus

CHUNKS = Path("data/chunks.jsonl")
_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on for and or is are was were be by with as at from that this it what which who how does do can".split())


def tokenize(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s.lower()) if t not in STOP]


@lru_cache(maxsize=1)
def _index(path: str = str(CHUNKS)):
    chunks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not chunks:
        raise RuntimeError("No chunks. Run: python -m citerag.ingest")
    toks = [tokenize(c["text"]) for c in chunks]
    return chunks, [set(t) for t in toks], BM25Plus(toks)


def search(query: str, k: int = 5) -> list[dict]:
    chunks, vocab, bm25 = _index()
    q = tokenize(query)
    scores = bm25.get_scores(q)
    # BM25Plus gives every chunk a small positive score, so keep only chunks sharing at least one query term.
    top = sorted((i for i in range(len(chunks)) if vocab[i] & set(q)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**chunks[i], "score": round(float(scores[i]), 3)} for i in top]

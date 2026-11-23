"""Retrieve -> grounded answer with [n] citations -> validate citations.

Run: python -m citerag.answer "your question"
"""
from __future__ import annotations

import re
import sys

from citerag.llm import complete
from citerag.retrieve import search

ABSTAIN = "I don't know based on the provided documents."

SYSTEM = f"""You answer questions using ONLY the numbered sources provided.
Rules:
- Every factual sentence ends with a citation like [1] or [2][3], pointing to the source(s) that support it.
- Never use outside knowledge. If the sources do not contain the answer, reply exactly: "{ABSTAIN}"
- Be concise: 1-4 sentences."""


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        loc = f"{h['source']}" + (f", p.{h['page']}" if h.get("page") else "")
        blocks.append(f"[{n}] ({loc})\n{h['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})
    valid = [c for c in cited if 1 <= c <= n_sources]
    invalid = [c for c in cited if c not in valid]
    return valid, invalid


def answer(question: str, k: int = 5) -> dict:
    hits = search(question, k=k)
    if not hits:  # nothing retrieved: abstain without spending a call
        return {"question": question, "answer": ABSTAIN, "abstained": True, "citations": [], "invalid_citations": [],
                "sources": [], "llm": None}
    r = complete(build_prompt(question, hits), system=SYSTEM, tag="answer")
    valid, invalid = parse_citations(r.text, len(hits))
    return {
        "question": question,
        "answer": r.text.strip(),
        "abstained": ABSTAIN.lower().rstrip(".") in r.text.lower(),
        "citations": valid,
        "invalid_citations": invalid,
        "sources": hits,
        "llm": {"provider": r.provider, "model": r.model, "in": r.input_tokens, "out": r.output_tokens,
                "latency_s": r.latency_s, "cost_usd": r.cost_usd},
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many days of annual leave do employees get?"
    out = answer(q)
    print(out["answer"], "\n")
    for c in out["citations"]:
        s = out["sources"][c - 1]
        print(f"[{c}] {s['source']}" + (f" p.{s['page']}" if s.get("page") else "") + f" (bm25={s['score']})")
    if out["invalid_citations"]:
        print("WARNING: cited sources that don't exist:", out["invalid_citations"])
    print("\n", out["llm"])

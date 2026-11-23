"""Eval harness. These numbers go in your README, your posts and your applications.

questions.jsonl, one per line:
  {"q": "...", "source": "file.md", "must_include": ["15"], "answerable": true}
  {"q": "...", "answerable": false}          # should abstain

Metrics:
  retrieval_hit@k   expected source file appears in top-k retrieved chunks
  citation_valid    answer cites >=1 source and never a non-existent one
  cited_correct     at least one cited chunk comes from the expected source
  keyword_recall    fraction of must_include strings present in the answer
  abstain_correct   on unanswerable questions, the system said it doesn't know
  latency, tokens, cost

Run: python eval/run_eval.py [--k 5]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citerag.answer import answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--questions", default="eval/questions.jsonl")
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    rows = []
    for item in qs:
        out = answer(item["q"], k=args.k)
        retrieved_sources = [s["source"] for s in out["sources"]]
        cited_sources = [out["sources"][c - 1]["source"] for c in out["citations"]]
        row = {"q": item["q"], "answer": out["answer"], "answerable": item.get("answerable", True)}
        if row["answerable"]:
            must = item.get("must_include", [])
            row.update(
                retrieval_hit=item.get("source") in retrieved_sources,
                citation_valid=bool(out["citations"]) and not out["invalid_citations"],
                cited_correct=item.get("source") in cited_sources,
                keyword_recall=(sum(m.lower() in out["answer"].lower() for m in must) / len(must)) if must else None,
                wrongly_abstained=out["abstained"],
            )
        else:
            row["abstain_correct"] = out["abstained"]
        if out["llm"]:
            row.update(latency_s=out["llm"]["latency_s"], tokens=out["llm"]["in"] + out["llm"]["out"], cost_usd=out["llm"]["cost_usd"])
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    kr = [r["keyword_recall"] for r in ans if r["keyword_recall"] is not None]
    lat = [r["latency_s"] for r in rows if "latency_s" in r]
    summary = {
        "provider": os.getenv("LLM_PROVIDER", "mock"), "model": os.getenv("MODEL", ""), "k": args.k,
        "n_answerable": len(ans), "n_unanswerable": len(unans),
        f"retrieval_hit@{args.k}_%": pct([r["retrieval_hit"] for r in ans]),
        "citation_valid_%": pct([r["citation_valid"] for r in ans]),
        "cited_correct_source_%": pct([r["cited_correct"] for r in ans]),
        "keyword_recall_%": round(100 * st.mean(kr), 1) if kr else None,
        "wrongly_abstained_%": pct([r["wrongly_abstained"] for r in ans]),
        "abstain_correct_%": pct([r["abstain_correct"] for r in unans]),
        "p50_latency_s": round(st.median(lat), 2) if lat else None,
        "total_cost_usd": round(sum(r.get("cost_usd", 0) for r in rows), 5),
    }

    print(json.dumps(summary, indent=2))
    fails = [r for r in rows if (r["answerable"] and not (r["retrieval_hit"] and r["citation_valid"] and r["cited_correct"]))
             or (not r["answerable"] and not r["abstain_correct"])]
    if fails:
        print(f"\n{len(fails)} failing questions (your 'what broke' material):")
        for r in fails:
            print(" -", r["q"], "->", r["answer"][:120])

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

# CiteRAG

A retrieval-augmented QA system that answers **only** from your documents, **cites every claim**, and **says "I don't know"** instead of hallucinating, with an eval harness that measures all three.

> **Live demo:** _<add link>_ · **60s walkthrough:** _<add Loom>_ · **Latest eval:** see table below

## Why this exists
Most RAG demos show a happy-path answer. Production RAG fails in three ways: it retrieves the wrong chunk, cites sources that don't support the claim, or confidently answers questions the documents can't answer. CiteRAG treats each of these as a measured metric.

## Eval results
| Run | Provider / model | Retrieval hit@5 | Valid citations | Correct-source citations | Keyword recall | Correct abstentions | p50 latency | Cost |
|---|---|---|---|---|---|---|---|---|
| Day 1 | extractive baseline (no LLM) | 100% | 100% | 100% | 100% | 100% | 0s | $0 |
| _next_ | _LLM_ | | | | | | | |

_Day 1 numbers are on a 6-question sample corpus. Swap in the real corpus and a 30+ question set before quoting numbers anywhere._

## What broke and how I fixed it
- **Day 1: system abstained on every question.** BM25Okapi's IDF is `log((N-n+0.5)/(n+0.5))`, which is exactly 0 when a term appears in half the chunks. On a small corpus every score collapsed to 0 and retrieval returned nothing. Switched to BM25Plus (IDF stays positive) and filtered by query-term overlap instead of `score > 0`. Eval went from 0% to 100% retrieval hit on the sample set.

## Architecture
```
data/raw/*.md|txt|pdf → ingest (section-aware chunks, page numbers kept)
→ BM25 retrieval (top-k) → grounded prompt with numbered sources
→ LLM answer with [n] citations → citation validation (flags non-existent sources)
→ every call logged: tokens, latency, cost (logs/calls.jsonl)
```
Provider-agnostic: Anthropic, Gemini, Groq or OpenAI via one env var. An offline extractive baseline (`LLM_PROVIDER=mock`) runs with no key and is the number the LLM has to beat.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add a key, set LLM_PROVIDER + MODEL
python hello_llm.py                                      # first LLM call
python -m citerag.ingest                                 # chunk data/raw → data/chunks.jsonl
python -m citerag.answer "your question"
python eval/run_eval.py                                  # metrics → eval/results/
streamlit run app.py                                     # demo UI
python -m pytest -q                                      # offline tests
```

## Roadmap
- [ ] Real corpus + 30–50 question eval set (including unanswerable ones)
- [ ] Hybrid retrieval (BM25 + embeddings) with eval delta
- [ ] LLM-judged faithfulness: does each cited chunk actually support its sentence?
- [ ] Deploy + basic monitoring

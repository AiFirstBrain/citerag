"""Day 1 checkpoint: first working LLM call. Run: python hello_llm.py"""
from citerag.llm import complete

r = complete(
    "In one sentence: what is retrieval-augmented generation, and what is its most common failure mode?",
    system="You are a concise senior AI engineer.",
    tag="hello",
)
print(r.text)
print(f"\n[{r.provider}/{r.model}] {r.input_tokens} in / {r.output_tokens} out tokens, {r.latency_s}s, ${r.cost_usd}")

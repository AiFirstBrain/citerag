"""Live demo. Run locally: streamlit run app.py   Deploy: Streamlit Community Cloud or Hugging Face Spaces (free)."""
import json
from pathlib import Path

import streamlit as st

from citerag.answer import answer
from citerag.ingest import OUT, ingest

if not OUT.exists():  # fresh deploy: chunks are gitignored, build them on first boot
    ingest()

st.set_page_config(page_title="CiteRAG", page_icon="📎")
st.title("📎 CiteRAG")
st.caption("Answers only from the documents, cites every claim, and says \"I don't know\" instead of guessing.")

latest = sorted(Path("eval/results").glob("*.json"))
if latest:
    s = json.loads(latest[-1].read_text())["summary"]
    cols = st.columns(3)
    hit_key = next((k for k in s if k.startswith("retrieval_hit")), None)
    cols[0].metric("Retrieval hit", f"{s.get(hit_key)}%" if hit_key else "–")
    cols[1].metric("Correct-source citations", f"{s.get('cited_correct_source_%')}%")
    cols[2].metric("Abstains when it should", f"{s.get('abstain_correct_%')}%")

q = st.text_input("Ask a question about the corpus")
if q:
    with st.spinner("Retrieving and answering…"):
        out = answer(q)
    st.markdown(out["answer"])
    if out["invalid_citations"]:
        st.warning(f"Model cited non-existent sources: {out['invalid_citations']}")
    for n, s in enumerate(out["sources"], 1):
        mark = "✅ cited" if n in out["citations"] else "not cited"
        loc = s["source"] + (f" p.{s['page']}" if s.get("page") else "")
        with st.expander(f"[{n}] {loc} · bm25 {s['score']} · {mark}"):
            st.write(s["text"])
    if out["llm"]:
        l = out["llm"]
        st.caption(f"{l['provider']}/{l['model']} · {l['in']}+{l['out']} tokens · {l['latency_s']}s · ${l['cost_usd']}")

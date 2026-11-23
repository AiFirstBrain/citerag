"""Offline smoke tests (mock LLM, no API key). Run: python -m pytest -q"""
import os

os.environ["LLM_PROVIDER"] = "mock"

from citerag.answer import ABSTAIN, answer, parse_citations  # noqa: E402
from citerag.ingest import chunk_text, ingest  # noqa: E402


def test_chunking_respects_size():
    chunks = chunk_text("para one.\n\n" * 300, size=200, overlap=20)
    assert chunks and all(len(c) <= 300 for c in chunks)


def test_parse_citations_flags_invalid():
    assert parse_citations("A [1]. B [2][7].", n_sources=3) == ([1, 2], [7])


def test_end_to_end_cites_and_abstains():
    ingest()
    good = answer("How many days of paid annual leave?")
    assert good["citations"] and "24" in good["answer"]
    bad = answer("xyzzy plugh frobnicate")
    assert bad["answer"] == ABSTAIN

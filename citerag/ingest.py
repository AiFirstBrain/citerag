"""Turn files in data/raw (.md, .txt, .pdf) into overlapping chunks in data/chunks.jsonl.

Each chunk keeps its source file and page so answers can cite exactly where a fact came from.
Run: python -m citerag.ingest
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/chunks.jsonl")
CHUNK_CHARS = 900
OVERLAP_CHARS = 150


def read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    return [(None, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Pack paragraphs into ~size-char chunks; carry a tail of the previous chunk forward for context."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and p.startswith("#"):  # markdown heading = hard section boundary, no overlap across sections
            chunks.append(cur)
            cur = p
        elif cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > size * 1.5:  # very long paragraph: hard split
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def ingest(raw: Path = RAW, out: Path = OUT) -> int:
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"})
    if not files:
        sys.exit(f"No .md/.txt/.pdf files in {raw}/ — add your corpus first.")
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for path in files:
            for page, text in read_pages(path):
                for i, c in enumerate(chunk_text(text)):
                    f.write(json.dumps({"id": f"{path.name}:{page or 0}:{i}", "source": path.name, "page": page, "text": c}) + "\n")
                    n += 1
    print(f"Ingested {len(files)} files -> {n} chunks -> {out}")
    return n


if __name__ == "__main__":
    ingest()

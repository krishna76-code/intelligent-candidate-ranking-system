"""
loader.py — stream and parse candidates.jsonl safely.

Handles both plain .jsonl and gzipped .jsonl.gz (the validator/README mention both).
Streams line-by-line so we never load all 100K dicts at once unnecessarily.

API to implement:
    def load_candidates(path) -> Iterator[dict]      # streams one candidate dict per line
    def build_text_blob(candidate) -> str            # summary + headline + career DESCRIPTIONS
                                                     # (NO skills list — anti-stuffer rule)

TODO: implement. Placeholder skeleton.
"""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator

# import gzip
# import json
# from pathlib import Path


def load_candidates(path):
    """Yield one candidate dict per line from .jsonl or .jsonl.gz."""
    path = Path(path)

    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                candidate = json.loads(line)
                yield candidate

            except json.JSONDecodeError:
                continue

def build_text_blob(candidate: dict) -> str:
    """
    Build text for embeddings.

    Uses:
    - headline
    - summary
    - career descriptions

    Intentionally excludes skills list to prevent
    keyword stuffers from gaming embeddings.
    """

    chunks = []

    headline = candidate.get("headline", "")
    summary = candidate.get("summary", "")

    if headline:
        chunks.append(headline)

    if summary:
        chunks.append(summary)

    for exp in candidate.get("career_history", []):

        title = exp.get("title", "")
        description = exp.get("description", "")

        if title:
            chunks.append(title)

        if description:
            chunks.append(description)

    return "\n".join(chunks)
"""
loader.py — stream and parse candidates.jsonl safely.

Handles both plain .jsonl and gzipped .jsonl.gz (the validator/README mention both).
Streams line-by-line so we never load all 100K dicts at once unnecessarily.

API:
    load_candidates(path) -> Iterator[dict]   # streams one candidate dict per line
    build_text_blob(candidate) -> str         # profile.summary + profile.headline +
                                              # career-history descriptions
                                              # (NO skills list — anti-stuffer rule)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def _s(value) -> str:
    """Coerce any value to a clean string (None -> '')."""
    if value is None:
        return ""
    return str(value).strip()


def load_candidates(path) -> Iterator[dict]:
    """Yield one candidate dict per line from .jsonl or .jsonl.gz."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip a malformed line rather than crash the whole run.
                continue


def build_text_blob(candidate: dict) -> str:
    """
    Build the text we embed.

    Includes:
        - profile.headline
        - profile.summary
        - each career_history entry's title + description

    Intentionally EXCLUDES the skills list, so keyword stuffers who paste
    "PyTorch, LLM, RAG" into skills cannot game the embedding. The embedding
    reflects what the candidate actually DID (their story + work), not what
    they clicked.
    """
    candidate = candidate if isinstance(candidate, dict) else {}
    profile = candidate.get("profile") or {}

    chunks = []

    # NOTE: headline & summary live under "profile", not at the top level.
    headline = _s(profile.get("headline"))
    summary = _s(profile.get("summary"))
    if headline:
        chunks.append(headline)
    if summary:
        chunks.append(summary)

    career = candidate.get("career_history") or []
    for exp in career:
        exp = exp if isinstance(exp, dict) else {}
        title = _s(exp.get("title"))
        description = _s(exp.get("description"))
        if title:
            chunks.append(title)
        if description:
            chunks.append(description)

    return "\n".join(chunks)

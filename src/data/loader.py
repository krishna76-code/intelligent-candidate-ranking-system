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

# import gzip
# import json
# from pathlib import Path


def load_candidates(path):
    """Yield one candidate dict per line from .jsonl or .jsonl.gz."""
    raise NotImplementedError


def build_text_blob(candidate: dict) -> str:
    """summary + headline + all career-history descriptions. NO skills list."""
    raise NotImplementedError

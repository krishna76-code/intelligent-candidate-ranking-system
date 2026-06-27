# evaluation/

Reproducible validation for the ranker. These are **dev tools** — they are *not*
part of the graded `rank.py` step (they read the full pool + embeddings and have
no time limit). They let anyone (teammates, reviewers) re-check the ranking the
same way we did.

Because the official ground truth is hidden, we validate three ways: by
inspection, against our own hand labels, and by measuring embedding quality.

## Prerequisites
Run from the **repo root**, with the embeddings present (`artifacts/*.npy`, shipped
via Git LFS) and the candidate pool at `./data/candidates.jsonl`.

```bash
# (if needed) regenerate embeddings first — offline, no time limit:
python -m src.precompute --candidates ./data/candidates.jsonl --cache-model
```

## The three checks

### 1. `eyeball_test.py` — is the top 100 sensible?
Prints the top picks (title / years / company) and reports how many are real
engineering/AI titles vs keyword-stuffers vs honeypots.
```bash
python -m evaluation.eyeball_test --candidates ./data/candidates.jsonl --show 20
```
**Pass = 0 stuffers, 0 honeypots in the top 100.** Exit code 0/1 (CI-friendly).

### 2. `labeled_eval.py` — does it agree with human judgment?
Runs the ranking and checks our ~25 hand-labeled candidates (`eval_labels.json`):
good ones should rank high, bad ones should stay out of the top 100.
```bash
python -m evaluation.labeled_eval --candidates ./data/candidates.jsonl
```
**Pass = no bad candidate leaks into the top 100.** (Labels are a *check*, never
training input — no leakage.)

### 3. `embedding_separation.py` — is the embedding model actually helping?
Measures mean JD-similarity for clearly-strong vs clearly-weak candidates. A bigger
gap = the model distinguishes good from bad.
```bash
python -m evaluation.embedding_separation --candidates ./data/candidates.jsonl
```
Reference: MiniLM ≈ 0.007 (noise); **bge-base ≈ 0.048 (~7× better)**.

## Files
| File | What |
|---|---|
| `_common.py` | Shared loader/scorer (reuses the exact `rank.py` pipeline) |
| `eval_labels.json` | ~25 hand-labeled real candidates (12 good, 13 bad) |
| `eyeball_test.py` | Top-100 inspection + stuffer/honeypot counts |
| `labeled_eval.py` | Ranking vs the hand-labeled set |
| `embedding_separation.py` | Embedding discrimination measurement |

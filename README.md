# Intelligent Candidate Discovery & Ranking

Rank the **top 100** best-fit candidates out of **100,000** for one role —
**Senior AI Engineer — Founding Team** at Redrob AI — and output a CSV.

> **Strategy:** filter first, rank second. Only ~726 of 100K candidates even have an
> AI/ML title. Hard-filter the obvious rejects with cheap rules, then score the
> survivors with a **transparent weighted feature model** where embedding similarity
> is ONE feature (~18%), not the backbone. See [docs/implementation_plan.md](docs/implementation_plan.md).

---

## Reproduce the submission (the graded ≤5-min step)

```bash
# 1. Create env & install (CPU-only torch)
python -m venv .venv
. .venv/Scripts/activate            # Windows;  source .venv/bin/activate on Linux/Mac
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. (offline, once) build embeddings — NO time limit. Skipped if artifacts/*.npy
#    are already present (they are committed via Git LFS). Use --cache-model the
#    first time to download the embedding model. GPU optional (speeds it up).
python -m src.precompute --candidates ./data/candidates.jsonl --cache-model

# 3. Produce the submission CSV — CPU, no network, <=5 min.
#    NOTE: run as a MODULE (-m), not "python src/rank.py", so package imports resolve.
python -m src.rank --candidates ./data/candidates.jsonl --out ./output/<participant_id>.csv

# 4. Validate before uploading
python scripts/validate_submission.py ./output/<participant_id>.csv
```

> **torch note:** `requirements.txt` pins `torch` but the CPU build must come from the
> PyTorch CPU index (step 1 line). On a fresh machine, install torch first from that
> index, then the rest, so you don't pull a CUDA build.

---

## Repository layout

```
src/            ranking logic
  rank.py         MAIN ENTRY — produces the CSV (the 5-min step)
  precompute.py   offline: builds embeddings (no time limit)
  scoring/        hard_filters, features, penalties, behavioral, reasoning
  data/           loader (candidates.jsonl) + embedder (model wrapper)
  utils/          constants (consulting list, company_founded, term sets)
config/         weights.yaml — feature weights & thresholds (no magic numbers)
artifacts/      precomputed embeddings + cached model (runtime needs no network)
data/           candidates.jsonl (gitignored), job_description.txt, sample
sandbox/        Streamlit demo (the required live link)
tests/          pytest — filters, features, output-CSV validation
scripts/        validate_submission.py (organizer's validator)
docs/           plan, rules, architecture diagram
output/         the final <participant_id>.csv
deck/           explainer slides (PDF)
```

---

## Compute constraints (the `rank.py` step)

| Constraint | Limit |
|---|---|
| Runtime | ≤ 5 minutes wall-clock |
| Memory | ≤ 16 GB RAM |
| Compute | CPU only — no GPU |
| Network | OFF — no LLM API calls |
| Disk | ≤ 5 GB |

Precomputation (embeddings) runs offline with no time limit; only the ranking step
that writes the CSV must fit the budget.

---

## Status

🛠️ **Skeleton** — module stubs with docstrings and the intended API. Implement
`src/` bottom-up: `loader` → `hard_filters` → `features` → `rank.py`.

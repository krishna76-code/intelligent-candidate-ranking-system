# Intelligent Candidate Discovery and Ranking

A ranking system that finds the **top 100** best-fit candidates out of **100,000**
for a single role — **Senior AI Engineer, Founding Team** at Redrob AI — and
produces a validated submission CSV.

**Live demo:** https://intelligent-candidate-discovery-ranking-nnbgwtgrgqgmdmbzzuymn2.streamlit.app/

---

## The problem

The dataset is adversarial by design. Only about 726 of the 100,000 candidates
even hold an AI/ML title, and it is seeded with traps: keyword stuffers (a
Marketing Manager who lists "PyTorch, RAG, LLM"), honeypots (impossible profiles
such as eight years at a company founded three years ago), and behavioral twins
(profiles stuffed with the right words whose own descriptions admit they are weak).
A naive "most AI keywords wins" ranker fails; the challenge is to read the gap
between what a profile says and what it means.

## The approach: filter first, rank second

1. **Hard filters** remove the obvious rejects with cheap, explainable rules:
   honeypots (impossible founding dates, fake "expert" skills, inflated
   experience), consulting-only careers, candidates who cannot relocate to India,
   and non-engineering titles claiming AI skills.

2. **A transparent weighted feature model** scores the survivors. Embedding
   similarity is only one feature (~14% weight), not the backbone — the decisive
   signals are the candidate's title-role match and their real career evidence
   (retrieval / ranking / recommendation / NLP work described in their history).

3. **A behavioral multiplier** adjusts for reachability (recruiter response rate,
   recency of activity, notice period). A perfect-on-paper candidate who never
   replies to recruiters is, for hiring purposes, not actually available.

This design is deliberate: on this synthetic, templated dataset embeddings barely
separate strong candidates from weak ones (measured strong-vs-weak similarity gap
~0.04), so the structured features carry the ranking while embeddings act as a
semantic tie-breaker. Full rationale in
[docs/implementation_plan.md](docs/implementation_plan.md).

---

## Results

On the full 100,000-candidate pool:

- Top 100 are **100% genuine AI/ML engineers** at real product companies.
- **Zero keyword-stuffers** and **zero honeypots** in the top 100 (the honeypot
  disqualification threshold is 10%).
- Every hand-labeled poor-fit candidate is kept out of the top 100.
- Ranking runs in about **30-46 seconds** on CPU, well under the 5-minute limit.

---

## Reproduce the submission

The ranking step is CPU-only, uses no network, and stays well under 5 minutes
because it loads precomputed embeddings rather than running a model.

```bash
# 1. Environment and dependencies (CPU-only torch)
python -m venv .venv
. .venv/Scripts/activate              # Windows; on Linux/Mac: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Fetch the committed embeddings (Git LFS). Skip if already present.
git lfs pull

# 3. Produce the submission CSV. Run as a module (-m) so package imports resolve.
python -m src.rank --candidates ./data/candidates.jsonl --out ./output/<participant_id>.csv

# 4. Validate before uploading
python scripts/validate_submission.py ./output/<participant_id>.csv
```

The candidate pool (`candidates.jsonl`, ~487 MB) is not committed; place it under
`data/` or pass its path with `--candidates`.

### Rebuilding embeddings (offline, optional)

Embeddings ship via Git LFS, so `rank.py` runs without this. To regenerate them
(no time limit, GPU used automatically if available; only the ranking step is
constrained):

```bash
python -m src.precompute --candidates ./data/candidates.jsonl --cache-model
```

---

## Scoring model

`final = (weighted feature sum) x (JD-penalty multiplier) x (behavioral multiplier)`,
normalized to `[0, 1]`.

| Feature | Weight | What it measures |
|---|---|---|
| title_role_match | 0.24 | Current/recent title is genuinely ML/AI/DS/eng |
| career_evidence | 0.22 | Descriptions show retrieval/ranking/recommendation/NLP work (depth-aware) |
| embedding_similarity | 0.14 | bge-base cosine similarity between the JD and the career text |
| experience_fit | 0.12 | 5-9 year band, with soft edges |
| product_vs_services | 0.10 | Product-company experience over pure consulting |
| skill_trust | 0.08 | duration x endorsements, discounting unbacked skills |
| assessment_bonus | 0.05 | Relevant Redrob assessment present (bonus only, never a penalty) |
| writes_code_recently | 0.05 | Penalizes architecture/tech-lead-only profiles with no recent code |

**JD penalties (soft multipliers):** title-chaser, LangChain-only, CV/speech-only,
research-only — straight from the job description's "things we do not want".

**Embeddings:** `BAAI/bge-base-en-v1.5` (768-dim), generated offline. The JD is
embedded with the BGE query prefix; candidate career text (summary + headline +
career descriptions, never the skills list) is embedded as documents.

**Reasoning:** each row gets a 1-2 sentence justification that references specific
facts, connects to a JD requirement, and honestly names concerns at lower ranks.

---

## Repository layout

```
src/
  rank.py           main entry - produces the submission CSV (CPU, no network)
  precompute.py     offline: builds embeddings (no time limit)
  scoring/          hard_filters, features, penalties, behavioral, reasoning
  data/             loader (candidates.jsonl) and embedder (model wrapper)
  utils/            constants (consulting firms, company_founded, term sets)
config/weights.yaml feature weights and thresholds
artifacts/          precomputed embeddings (.npy via Git LFS); model cache regenerated
data/               candidates.jsonl (gitignored), job_description.txt, sample
sandbox/            Streamlit demo (the hosted live link)
evaluation/         reproducible validation: eyeball, labeled, embedding-separation
tests/              pytest - filters, features, output-CSV validation
scripts/            validate_submission.py (organizer's validator)
docs/               implementation plan
output/             the final <participant_id>.csv (gitignored)
```

---

## Live demo (sandbox)

The Streamlit app runs the exact same pipeline on a small sample (uploaded or
bundled), embedding it live on CPU. It ranks the candidates, shows each one's
reasoning and score breakdown, and lists what was filtered out and why. Because it
handles at most 100 candidates, live embedding finishes in seconds.

```bash
streamlit run sandbox/app.py
```

Deploy instructions: [sandbox/README.md](sandbox/README.md).
Live URL: https://intelligent-candidate-discovery-ranking-nnbgwtgrgqgmdmbzzuymn2.streamlit.app/

---

## Evaluation

The official ground truth is hidden, so the ranking is validated three independent
ways (details in [evaluation/README.md](evaluation/README.md)):

```bash
python -m evaluation.eyeball_test         --candidates ./data/candidates.jsonl
python -m evaluation.labeled_eval         --candidates ./data/candidates.jsonl
python -m evaluation.embedding_separation --candidates ./data/candidates.jsonl
```

- **eyeball_test** — inspects the top 100 and counts engineers / stuffers / honeypots.
- **labeled_eval** — checks a small hand-labeled set (good candidates rank high,
  bad ones stay out); labels are only a check, never training input.
- **embedding_separation** — measures how well the embeddings distinguish strong
  from weak candidates.

---

## Tests

```bash
python -m pytest tests/ -q      # 36 tests: hard filters, features, output-CSV format
```

---

## Compute constraints (the ranking step)

| Constraint | Limit | This project |
|---|---|---|
| Runtime | 5 minutes | ~30-46 s on the full 100K |
| Memory | 16 GB | ~1-2 GB |
| Compute | CPU only, no GPU | rank.py loads .npy only, never a model |
| Network | off | no API calls at ranking time |
| Disk | 5 GB | ~0.3 GB of artifacts |

Precomputation (embeddings) runs offline with no time limit and may use a GPU;
only the ranking step that writes the CSV must fit the budget above. The graded
`rank.py` imports no model, GPU, or network code.

---

## Tech stack

Python 3.12, PyTorch (CPU at ranking time), sentence-transformers / bge-base for
embeddings, NumPy, PyYAML, Streamlit for the demo, pytest for tests, Git LFS for
the committed embeddings.

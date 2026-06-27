# Sandbox — Live Demo

The hackathon **requires** a working hosted demo where the ranker runs on a small
candidate sample (≤100). This is that app.

It runs the **exact same pipeline** as `rank.py` (hard filters → 8 weighted
features → JD penalties → behavioral multiplier → reasoning), but embeds the
uploaded sample **live on CPU** instead of using precomputed vectors — so it works
on candidates the judges upload themselves.

> **Constraints:** only a small sample (≤100) is embedded live, which takes a few
> seconds on CPU (model load is cached). The full 100K pool is embedded offline by
> `precompute.py`; the graded `rank.py` never runs a model. Nothing here violates
> the ≤5-min / CPU / small-sample sandbox rules.

## Run locally
```bash
# from the repo root, with the venv active:
streamlit run sandbox/app.py
```
Then open the printed URL, and either upload a candidate file or click
**"Use the bundled 50-candidate sample"**.

Input format: a JSON **list** of candidate objects (same schema as
`data/sample_candidates.json`), or a `.jsonl` file with one candidate per line.

## Deploy a public link (free) — Streamlit Cloud
1. Push this repo to GitHub (done).
2. Go to https://share.streamlit.io → **New app**.
3. Pick the repo/branch, set **Main file path** to `sandbox/app.py`.
4. Under **Advanced settings**, set the requirements file to
   `sandbox/requirements.txt`.
5. Deploy. First run downloads the `bge-base` model (one-time); after that it's
   cached. You get a public URL like `https://<your-app>.streamlit.app`.
6. Paste that URL into `submission_metadata.yaml` → `sandbox_link`.

> The repo's embeddings (`artifacts/*.npy`) come via **Git LFS**. Streamlit Cloud
> pulls LFS automatically; the sandbox doesn't strictly need them (it embeds live),
> but they don't hurt.

### Alternative: HuggingFace Spaces
Create a **Streamlit** Space, point it at this repo (or copy `sandbox/app.py` +
`sandbox/requirements.txt` and the `src/`, `config/`, `data/` folders). Same result.

## What you'll see
A ranked list of the uploaded candidates — rank, name, title, years, company,
score, and a 1–2 sentence reason — plus a "filtered out (and why)" section and a
per-candidate score breakdown.

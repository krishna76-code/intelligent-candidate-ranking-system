# Sandbox — Live Demo

The hackathon **requires** a working hosted demo where the ranker runs on a small
candidate sample (≤100). This folder holds that app.

## Run locally
```bash
streamlit run sandbox/app.py
```

## Deploy (pick one — free tier is fine)
- **Streamlit Cloud**: connect this GitHub repo, set the app file to `sandbox/app.py`.
- **HuggingFace Spaces**: create a Streamlit Space, point it at this repo / copy `app.py`.

## Constraints (must match `rank.py`)
- CPU only, no GPU
- No calls to hosted LLM APIs
- Completes within ~5 minutes on the small sample

Paste the resulting public URL into `submission_metadata.yaml` → `sandbox_link`.

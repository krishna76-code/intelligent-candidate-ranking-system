"""
app.py — Streamlit sandbox / live demo for the Redrob candidate ranker.

What it does:
  - Accepts a small candidate sample (<=100) via upload, or uses the bundled sample.
  - Embeds those candidates LIVE on CPU (a small sample is fast — see note below),
    then runs the EXACT same pipeline as rank.py:
        hard filters -> 8-feature score -> JD penalties -> behavioral multiplier
  - Shows the ranked results with scores, reasoning, and a per-candidate breakdown.

Constraint note: this demo only ever handles a SMALL sample (<=100), so live
embedding finishes in seconds. The full 100K pool is embedded OFFLINE by
precompute.py; the graded rank.py loads those precomputed vectors and never runs
a model. The model is forced to CPU here; the model load is cached so repeat runs
are instant.

Run locally:   streamlit run sandbox/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# --- Guard against the Streamlit file-watcher + PyTorch crash ---------------- #
# Streamlit's watcher walks module attributes on rerun and crashes when it hits
# torch.classes (a known torch+Streamlit bug) -> shows as "Connection error".
# Belt-and-suspenders: also disabled via .streamlit/config.toml (fileWatcherType="none").
try:
    import torch  # noqa
    torch.classes.__path__ = []  # stop the watcher from inspecting torch.classes
except Exception:
    pass

# Make the repo root importable so `import src...` works on Streamlit Cloud.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rank import Constants, sanitize_candidate
from src.data.loader import build_text_blob
from src.data.embedder import load_model, embed_texts, BGE_QUERY_PREFIX
from src.scoring.hard_filters import is_disqualified
from src.scoring.features import feature_score
from src.scoring.penalties import penalty_multiplier
from src.scoring.behavioral import behavioral_multiplier
from src.scoring.reasoning import build_reasoning

import yaml

MAX_CANDIDATES = 100  # spec cap for the sandbox
SAMPLE_PATH = REPO_ROOT / "data" / "sample_candidates.json"
JD_PATH = REPO_ROOT / "data" / "job_description.txt"
CONFIG_PATH = REPO_ROOT / "config" / "weights.yaml"


# --------------------------------------------------------------------------- #
# Cached resources (loaded once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_model():
    # Force CPU — the demo mirrors the graded CPU-only environment.
    return load_model(device="cpu")


@st.cache_data
def get_weights():
    cfg = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8")) or {}
    return cfg.get("feature_weights", cfg)


@st.cache_data
def get_jd_vector():
    """Embed the JD once (with the BGE query prefix)."""
    model = get_model()
    jd_text = JD_PATH.read_text(encoding="utf-8")
    vec = embed_texts(model, [BGE_QUERY_PREFIX + jd_text], batch_size=1)[0]
    return vec


@st.cache_data
def load_sample():
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Core ranking (same pipeline as rank.py, embeddings computed live)
# --------------------------------------------------------------------------- #
def rank_candidates(candidates: list[dict]):
    model = get_model()
    weights = get_weights()
    jd_vec = get_jd_vector()
    constants = Constants()

    candidates = [sanitize_candidate(c) for c in candidates][:MAX_CANDIDATES]

    # Live-embed the uploaded candidates (documents -> no query prefix).
    blobs = [build_text_blob(c) for c in candidates]
    embs = embed_texts(model, blobs, batch_size=32)

    results = []
    filtered = 0
    for c, emb in zip(candidates, embs):
        dq, reason = is_disqualified(c)
        if dq:
            filtered += 1
            results.append({"candidate": c, "score": 0.0, "filtered": True,
                            "filter_reason": reason, "breakdown": {},
                            "penalties": [], "behavioral": {}})
            continue
        sim = float(np.dot(emb, jd_vec))
        feat, breakdown = feature_score(c, sim, weights, constants)
        pen, pen_reasons = penalty_multiplier(c, constants)
        beh, beh_breakdown = behavioral_multiplier(c)
        results.append({
            "candidate": c, "score": feat * pen * beh, "filtered": False,
            "filter_reason": "", "breakdown": breakdown,
            "penalties": pen_reasons, "behavioral": beh_breakdown,
        })

    # Survivors ranked best-first (ties by candidate_id asc, like rank.py);
    # filtered candidates listed after.
    survivors = [r for r in results if not r["filtered"]]
    filtered_out = [r for r in results if r["filtered"]]
    max_raw = max((r["score"] for r in survivors), default=1.0) or 1.0
    for r in survivors:
        r["score"] = round(r["score"] / max_raw, 4)  # normalize to [0,1] like rank.py
    survivors.sort(key=lambda r: (-r["score"], str(r["candidate"].get("candidate_id", ""))))
    return survivors, filtered_out, filtered


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.markdown(
    """
    <style>
    /* Dark emerald palette: deep teal-slate surfaces + emerald/mint accents */
    .main-title {
        font-size: 2.2rem; font-weight: 800; margin-bottom: 0.15rem;
        color: #f0faf6; letter-spacing: -0.01em;
    }
    .accent-bar {
        height: 3px; width: 72px; border-radius: 3px; margin-bottom: 0.9rem;
        background: linear-gradient(90deg, #10b981 0%, #6ee7b7 100%);
    }
    .subtitle { color: #94b3ac; font-size: 1.02rem; margin-bottom: 1.2rem; line-height: 1.55; }
    .card {
        border: 1px solid #24404a; border-left: 3px solid #10b981;
        border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;
        background: #13232e;
    }
    .rank-badge {
        background: #10b981; color: #05130d; border-radius: 999px;
        padding: 2px 12px; font-weight: 800; font-size: 0.85rem;
    }
    .cand-name { color: #f0faf6; font-weight: 700; font-size: 1.02rem; }
    .score { color: #6ee7b7; font-weight: 700; font-size: 0.9rem; }
    .muted { color: #9fbdb5; font-size: 0.9rem; }
    .pill {
        display: inline-block; background: #0f3d31; color: #a7f3d0;
        border-radius: 8px; padding: 2px 10px; font-size: 0.78rem;
        margin: 4px 6px 2px 0; border: 1px solid #10b981;
    }
    /* metric numbers in emerald */
    div[data-testid="stMetricValue"] { color: #6ee7b7; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Redrob Candidate Ranker</div>', unsafe_allow_html=True)
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ranks candidates for <b>Senior AI Engineer &mdash; Founding Team</b> '
    'by genuine fit (real retrieval / ranking experience), not keyword stuffing. '
    'Runs the live pipeline: hard filters &rarr; weighted features &rarr; behavioral signals.</div>',
    unsafe_allow_html=True,
)

# "How it works" moved inline as a collapsed panel (was crowding the sidebar).
with st.expander("How it works"):
    st.markdown(
        "- **Filters** drop honeypots, consulting-only, can't-relocate, stuffers\n"
        "- **8 weighted features** score the survivors (title, career evidence, "
        "embeddings, experience, product-vs-services, skill trust, assessments, recency-of-code)\n"
        "- **Behavioral multiplier** weights reachability (response rate, recency)\n"
        "- Embeddings: `bge-base-en-v1.5`, computed live for your sample (CPU)\n"
        f"- Sample is capped at {MAX_CANDIDATES} candidates (sandbox spec)"
    )

st.subheader("1. Choose candidates")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload a candidate sample (.json list or .jsonl, max 100 candidates)",
        type=["json", "jsonl"],
    )
with col2:
    use_sample = st.button("Use the bundled 50-candidate sample", use_container_width=True)

candidates = None
if uploaded is not None:
    try:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            candidates = [json.loads(line, strict=False) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw, strict=False)
            candidates = data if isinstance(data, list) else [data]
        st.success(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.error(f"Could not parse the file: {e}")
elif use_sample:
    candidates = load_sample()
    st.success(f"Loaded {len(candidates)} candidates from the bundled sample.")

if candidates:
    if len(candidates) > MAX_CANDIDATES:
        st.warning(f"Showing the first {MAX_CANDIDATES} (sandbox cap).")
    st.subheader("2. Results")
    with st.spinner("Embedding and ranking candidates..."):
        survivors, filtered_out, n_filtered = rank_candidates(candidates)

    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates in", min(len(candidates), MAX_CANDIDATES))
    m2.metric("Passed filters", len(survivors))
    m3.metric("Filtered out", n_filtered)

    st.markdown("#### Ranked (best fit first)")
    for i, r in enumerate(survivors, start=1):
        c = r["candidate"]; p = c.get("profile", {})
        name = p.get("anonymized_name") or c.get("candidate_id", "Candidate")
        title = p.get("current_title", "")
        yrs = p.get("years_of_experience", "")
        company = p.get("current_company", "")
        reasoning = build_reasoning(c, i, r["breakdown"], r["penalties"], r["behavioral"])

        with st.container():
            st.markdown(
                f'<div class="card"><span class="rank-badge">#{i}</span> '
                f'&nbsp;<span class="cand-name">{name}</span> &nbsp;'
                f'<span class="score">score {r["score"]:.4f}</span><br>'
                f'<span class="pill">{title}</span>'
                f'<span class="pill">{yrs} yrs</span>'
                f'<span class="pill">{company}</span><br>'
                f'<span class="muted">{reasoning}</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Score breakdown"):
                bd = {k: round(v, 3) for k, v in r["breakdown"].items()}
                st.json({"features": bd,
                         "penalties_applied": r["penalties"] or "none",
                         "behavioral": {k: round(v, 3) for k, v in r["behavioral"].items()}})

    if filtered_out:
        with st.expander(f"{len(filtered_out)} candidates filtered out (and why)"):
            for r in filtered_out:
                c = r["candidate"]; p = c.get("profile", {})
                st.markdown(
                    f"- **{p.get('anonymized_name') or c.get('candidate_id')}** "
                    f"({p.get('current_title','')}) — _{r['filter_reason']}_"
                )
else:
    st.info("Upload a file or click **Use the bundled sample** to see the ranking.")

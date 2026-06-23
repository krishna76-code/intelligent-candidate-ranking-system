"""
app.py — Streamlit demo (the REQUIRED live sandbox link).

Deploy free on Streamlit Cloud or HuggingFace Spaces. Must:
    - accept a small candidate sample (<=100) via upload OR use the preloaded sample
    - run the ranking end-to-end and show the ranked table + reasoning
    - complete within the compute budget (CPU, no network to LLM APIs)

Run locally:
    streamlit run sandbox/app.py

TODO: wire this to src/rank.py once the pipeline is implemented.
"""

import streamlit as st

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("🎯 Intelligent Candidate Discovery & Ranking")
st.caption("Senior AI Engineer — Founding Team · ranks candidates by genuine fit, not keywords.")

st.info(
    "Skeleton placeholder. Once src/rank.py is implemented, this app will:\n\n"
    "1. Load a small candidate sample (upload or preloaded data/sample_candidates.json)\n"
    "2. Run the filter -> feature score -> behavioral pipeline\n"
    "3. Show the ranked table with score + reasoning + a per-candidate score breakdown",
    icon="🛠️",
)

uploaded = st.file_uploader("Upload a candidate sample (.json / .jsonl, <=100 candidates)", type=["json", "jsonl"])

if st.button("Rank candidates"):
    st.warning("Ranking pipeline not implemented yet — this is the skeleton placeholder.")
    # TODO:
    # from src.data.loader import load_candidates
    # candidates = [...]
    # results = run_ranking(candidates)
    # st.dataframe(results)

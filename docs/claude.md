# Instructions for Claude — Project Guidelines & Constraints

Strict rules, constraints, and verified data facts that **must be kept in mind at all times** for the Redrob Intelligent Candidate Discovery & Ranking project. Facts below are **verified against the real 100,000-candidate `candidates.jsonl`** and organizer docs. See `implementation_plan.md` for the full plan.

---

## 1. Core Objective
- **Target Role (the ONLY role):** Senior AI Engineer — Founding Team, Redrob AI.
- **Goal:** rank the **top 100** best-fit candidates from **100,000** (`candidates.jsonl`). Output one CSV.
- **Key focus:** real product-engineering experience building/shipping ML/retrieval/ranking systems — NOT buzzword-listers.
- **Strategy:** filter first, rank second. Only ~726 candidates even have an AI/ML title. Use a **transparent weighted feature scorer**; embeddings are ONE feature (~18%), not the spine.

---

## 2. Runtime Constraints (`rank.py` step only)
- Runtime ≤ 5 min wall-clock · Memory ≤ 16 GB · CPU only (no GPU) · Network OFF (no OpenAI/Anthropic/Gemini/etc.) · Disk ≤ 5 GB.
- Solution: dual-phase. Precompute embeddings offline (no time limit) → `candidate_embeddings.npy`; load at runtime for instant CPU similarity. Cache model weights in repo so runtime needs no internet.

---

## 3. Honeypot Signatures (auto-DQ risk — >10% in top 100 = disqualified)
Verified counts in real data:
1. **Krutrim** (founded Dec 2023): **38** candidates with start year < 2023 → reject.
2. **Sarvam AI** (founded 2023): **36** candidates with start year < 2023 → reject.
3. **Zero-duration experts:** **21** candidates with ≥3 skills `proficiency=="expert"` AND `duration_months==0` → reject.
   - Use threshold **≥3** zero-duration expert skills (NOT 5; spec example is "10 skills, 0 years").
4. Keep an extensible `company_founded` dict; flag any `start_year < founded_year` for known-young companies.

---

## 4. Anti-Buzzword Stuffer Rules
1. **Embed career DESCRIPTIONS only** — never the skills list. Blob = summary + headline + career-history descriptions.
2. **Title-to-role check:** non-engineering current/primary title (Marketing/HR/Sales/Graphic/Accountant/Civil/Mechanical/etc.) + AI skills → hard filter to 0. (~4,931 such profiles.)
3. **Skill trust:** `duration_months × endorsements`, capped; discount 0-endorsement, <6-month skills.
4. **Assessment cross-ref (BONUS ONLY):** only **24,244 / 100,000** have `skill_assessment_scores`. Use a relevant high score as a positive bonus; **NEVER penalize absence** (most strong candidates lack the relevant one).

---

## 5. JD-Specific Disqualifiers (the JD's "Things we do NOT want")
Apply as soft penalties (not zero):
- **Title-chaser:** ≥4 jobs, ≥3 tenures < 20 months (~5,019). JD wants 3+ year commitment.
- **LangChain-tutorial only:** only recent LangChain/OpenAI work, no pre-2022 real ML.
- **CV/speech/robotics only:** no NLP/retrieval/IR exposure.
- **Research-only:** academic titles, no "production/deployed/shipped/users/scale" in descriptions.
- **Architecture/tech-lead only:** no hands-on code in last 18 months ("this role writes code").

---

## 6. Relocation & Recency (corrected facts)
- **Relocation:** hard-filter only `country != "India"` AND `willing_to_relocate == False` (~17,756). India-based `willing_to_relocate` matters far less. JD welcomes Hyderabad/Pune/Mumbai/Delhi NCR; outside India is "case-by-case", not auto-reject.
- **Recency:** today is **2026**. Last-active 2025 is **NOT** automatically stale (31,472 in 2025, 68,528 in 2026). Use continuous decay by months-since-active, not a 2025 cutoff.

---

## 7. Behavioral Signals (multiplier, ~0.5–1.1)
Multiply feature score by reachability: `recruiter_response_rate` (low = big down-weight), `last_active_date` decay, `open_to_work_flag`, short `notice_period_days`. A 5%-response candidate is effectively unhireable.

---

## 8. CSV Format (any violation = auto-reject; run `validate_submission.py`)
- Filename: `<registered_participant_id>.csv`, UTF-8.
- Columns exact order: `candidate_id,rank,score,reasoning`.
- Exactly **100 data rows** + 1 header.
- `candidate_id` matches `^CAND_[0-9]{7}$` and exists in `candidates.jsonl`.
- Ranks: unique integers 1–100. Scores **non-increasing** by rank. Ties broken by `candidate_id` **ascending**.

---

## 9. Reasoning Column (Stage-4 scored; 10 rows sampled)
1–2 sentences. Must: reference **specific facts** (company + what they built, years, signals); connect to a **JD requirement**; **acknowledge concerns** at lower ranks; vary across candidates; **no hallucination** (no skills/companies not in profile); tone matches rank. The shipped `sample_submission.csv` is a deliberate LOW-quality anti-example — do not imitate its templated style.

---

## 10. Deliverables (all required)
1. `<participant_id>.csv` (validated). 2. GitHub repo + `requirements.txt` + README single-command reproduce; **commit incrementally** (git history is reviewed). 3. `submission_metadata.yaml` at root (honest AI-tools declaration — declaring Claude is NOT penalized). 4. **Live sandbox/demo link** (HuggingFace Space / Streamlit Cloud, CPU-only, no network, ranks ≤100 sample) — REQUIRED. 5. Explainer PDF.

---

## 11. Scoring (what we optimize for)
`Final = 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10`. **80% of the grade is the top 50** — getting the ordering of the best ~150 right matters most. Max 3 submissions; last valid counts; no live leaderboard.

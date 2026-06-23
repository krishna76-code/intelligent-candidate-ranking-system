# data/

Input data for the ranker.

| File | Committed? | Notes |
|---|---|---|
| `candidates.jsonl` | ❌ **No** | The 100K pool (~487 MB) — too big for GitHub. Place it here locally, or use Git LFS. The validator/README also accept `candidates.jsonl.gz`. |
| `job_description.txt` | ✅ Yes | The JD as plain text (extracted from the organizer `.docx`). Embedded at precompute time. |
| `sample_candidates.json` | ✅ Yes | First 50 candidates, for the sandbox demo & quick local iteration. |

To get `candidates.jsonl`: copy it from the hackathon bundle, e.g.
```bash
cp "../../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" ./data/
```

# AI_WORKFLOW.md

Log of how AI assistance was used, written as it happens. Newest entry on top.
One entry per session/task: what was asked, which tool/agent did it, what a
human checked or changed before accepting it.

---

## 2026-09-19 — E0 detector inference: single-image CLI, box + landmarks overlay

**Asked:** implement inference for the `hugocornellier/cat-face-landmarks` detector already fetched to `models/` — a script with functions and an `if __name__ == "__main__"` entry point that takes an input image and an output path, validates the input, preprocesses per the HF model card, runs both stages, and writes an overlay image with the box and 48 landmarks drawn on it.

**Tool:** Claude Code, main session, plan mode, `/ponytail full`. Two Explore agents in parallel (one on `data/`/`models/` contents and the E0 process docs, one on the `ml`/`core` code structure and dependencies) — the scope touched several unrelated parts of the repo at once (data layout, model manifest, process docs, dependency list), which is what parallel exploration is for.

**What it did:** fetched the HF model card directly (WebFetch) rather than guessing preprocessing — it specifies letterbox-to-224 for the localizer, a 0.1-margin crop-and-resize-to-384 for the landmarks stage, and float32 `[0,1]` inputs for both. Asked the user one clarifying question (inline vs. real `core/geometry.py` for the letterbox/crop-margin math the Data Steward doc forbids duplicating) before writing code. Created `src/catface/core/geometry.py` (letterbox, unletterbox, expand-and-crop, point-mapping — round-trip tested in `tests/test_geometry.py`) and `src/catface/ml/detect_landmarks.py`. Loads both `.tflite` files via `openvino` (already a dependency) rather than adding `tensorflow`, matching PLAN_PROJECT.md's stated intent.

**Checked by human:** none yet — first pass, run and eyeballed by the same session, pending review.

**What it caught:** the localizer's real output tensor order doesn't match the model card's stated `bbox_xyxy` — reading it in the stated order inverts `x1`/`x2` on every image. Found by checking the raw output against CatFLW's own ground-truth box for a sample image and trying index permutations until one landed within a few pixels of ground truth. Written up in `DECISIONS.md`; the landmarks model's output order wasn't independently re-verified against ground truth beyond the visual overlay check below, since CatFLW's landmark ordering convention wasn't cross-checked point-by-point.

**Verified:** `uv run pytest tests/test_geometry.py` (4 round-trip checks). Ran the script against a CatFLW image with a known ground-truth box/landmarks and inspected the overlay directly — box and all 48 points land correctly on the ears, eyes, nose, and muzzle. Ran again against a `cat-emotions-3` image (the actual target distribution) and checked the output programmatically (box in-bounds, near-square aspect, overlay pixel diff sane) since that image has no ground truth to compare against.

**Not yet done:** the parquet cache, batch run over both Roboflow sets, near-duplicate hashing, the plausibility filter, the 100-image blind relabel — the rest of RSCH-0. This session's script is the building block those steps will call in a loop.

---

## 2026-09-19 — E0 download scripts: data → `data/`, weights → `models/`

**Asked:** read `docs/backlog.md`, implement the E0 part — write scripts that download the data into `data/` and the models into `models/`.

**Tool:** Claude Code, main session, `/ponytail full`. No subagents — two small download scripts and a few doc edits, nothing that needed parallel exploration.

**What it did:** scoped down via a clarifying question to download scripts only (no detector run, no parquet cache — that's the rest of E0) and the three E0 sources only. Wrote `.env.example`, added `.env` to `.gitignore`, wrote `models/manifest.json` with sha256s read from the HF LFS tree API, `scripts/fetch_weights.py` (stdlib, streams to a `.part` file, verifies sha256, skips already-verified files), and `scripts/fetch_data.py` (Roboflow's export-poll flow, reverse-engineered from the `roboflow-python` SDK source, plus a plain Kaggle GET — that dataset is public and needs no key).

**Checked by human:** four corrections during the run: no tests for the download scripts themselves; ask for credentials and write `.env.example` before writing any script, not after; move `fetch_data.py` out of `catface.ml` into `scripts/`, once it was clear `fetch_weights.py` is app-track infra too; use `python-dotenv` instead of the planned hand-rolled `.env` reader; anchor every path on `Path(__file__).resolve().parent.parent` instead of assuming the script runs from the repo root. All four are written up in `DECISIONS.md`.

**What it caught:** running `fetch_data.py` for real turned up two backlog numbers that don't match the actual download. Roboflow cat-emotions-cgrxv is not 3 classes — the export has 8 raw folders (attentive 1,147, relaxed 752, uncomfortable 107, "no clear emotion recognizable" 37, sad 22, angry 3, unlabeled 2, "attentive uncomfortable" 1) and ships train-only, no valid/test split; angry has 3 images total. CatFLW is 2,079 images, not the 2,016 the backlog names. cat-emotions (7-class) matches its advertised ~98–100 images per class. Both Roboflow projects report licence CC BY 4.0 via the API — the backlog states CatFLW's licence (CC BY-NC 4.0) but leaves the other two as "check project page."

**Not yet done:** running the detector, the parquet cache, near-duplicate hashing, the reject filter, the blind relabel — the rest of RSCH-0's acceptance criteria in `docs/backlog.md`. This session builds only the fetch step those depend on.

---

## 2026-09-19 — Reviewed the three planning docs for gaps and over-engineering

**Asked:** review `PLAN_RESEARCH.md`, `PLAN_PROJECT.md` and `backlog.md`; find
missing tasks, find over-engineered schemas that don't make sense yet, rewrite
them safely.

**Tool:** Claude Code, main session, `/ponytail full` (a skill that forces the
minimum solution and makes you justify anything speculative). No subagents — the
repo is 16 files and fully readable in one pass, so the plan-mode workflow's
explore agents would have re-derived context the session already had.

**What it did:** found four unowned artifacts (`core.geometry` had no owner
though both tracks import it; per-class mean shapes were needed by `/edit` and
produced by nothing; the rubric read, the retention job and the five graded docs
were requirements sitting in no milestone), and six pieces of machinery specified
before the thing justifying them existed. Both sets are written up in
`DECISIONS.md` with the reasoning and, for every cut, the condition that brings it
back.

**Checked by human:** two scope calls were mine, not the model's — it proposed a
thin `[project]` backlog covering M0–M8 and I declined it, since
`project_process.md` grooms project tasks just-in-time and the pre-written text
would be stale before it was read. I also chose deletion-with-a-trigger over a
"deferred" subsection.

**What it caught that I'd have missed:** the M5 fallback was quietly false. It
claims the warper works without the research track, but the warper needs the
per-class mean shapes, which only the research track can produce — so the
"independent" app track had a hidden dependency on an artifact nobody had
scheduled. The fix moves that artifact to E1, where the alignment it needs is
already being computed, instead of to export time behind the model.

**Correction it made to its own plan:** the first version filed `class_means.json`
as an export-time deliverable (RSCH-6). That would have put it after E5 and left
M3 blocked on the whole research track. Moved to RSCH-1 before implementing.

---

## 2026-09-19 — Groomed research backlog, seeded AI_WORKFLOW.md / DECISIONS.md

**Asked:** turn `PLAN_RESEARCH.md` into a fine-grained, groomed backlog and
start these two log files.

**Tool:** Claude Code, main session, no subagents (file-authoring task, not a
research-track lifecycle step — `research_process.md`'s Research PM role
applies once real experiments are being groomed one at a time against live
results, not to this one-shot doc pass).

**What it did:** read `AGENTS.md` and `PLAN_RESEARCH.md`, cross-checked
against `research_process.md` and the team role docs (`pm.md`,
`ml-engineer.md`, `research-qa.md`, `export-verifier.md`, `data-steward.md`)
for the grooming shape actually used downstream, then wrote
`docs/backlog.md` with one task per experiment (E0–E5) plus export and
report-compilation as their own tasks — 8 items total, each with question,
method, required controls, checkable acceptance criteria, and stop
condition.

**Checked by human:** not yet — first pass, pending review before E0 is
picked up for real.

# AI_WORKFLOW.md

Log of how AI assistance was used, written as it happens. Newest entry on top.
One entry per session/task: what was asked, which tool/agent did it, what a
human checked or changed before accepting it.

---

## 2026-09-22 — RSCH-4/E4 re-grooming: real Research QA verdict, Q4 marked obsolete

**Asked:** the user replaced RSCH-4's implementation with `scripts/pose_and_classes.py` and reported "the Research QA has already said PASS" — no such verdict existed anywhere in `docs/backlog.md`, and the new script's own docstring stated it computes no verdict. Separately, the user judged Q4 (class separability / merge analysis) obsolete: no run across E2, E3, or E4 has ever produced a class with zero predictions, so the collapse Q4 was designed to catch never had evidence to investigate.

**Tool:** Claude Code, main session as orchestrator, plan mode, per `docs/research_process.md`. Five `general-purpose` subagents in sequence: Research PM re-groomed the issue, Research QA reviewed it twice, an ML Engineer closed the gaps QA found, Research Thinker interpreted the passed result, and a second Research PM recorded it in `docs/MODEL_REPORT.md`.

**What it did:** confirmed via `git show --stat` on `3b5cddb` and `deaecf0` that the old six-arm implementation and its artifacts (`ratios_lr`, `mlp`, `yaw_only_lr`, `random_baseline`, `merge.json`) were deleted outright, not superseded alongside — the surviving script keeps two of the six arms and checks them in-sample against their own training data. Recorded Q4's obsolescence in `docs/DECISIONS.md`, struck it from RSCH-4's question line and acceptance criteria in `docs/backlog.md`, and updated its section of `docs/MODEL_REPORT.md`, all as a direct transcription of the user's stated reasoning rather than a subagent's judgment call. Research PM then re-groomed RSCH-4 to a Q3-only scope matching what the surviving script actually does. Research QA's first pass failed it 2/8: no in-sample caveat, no total-variation check against the class prior, no chance-floor reference, no limitations section, no run comment, and — found without being asked to look — RSCH-2's checkpoint had been regenerated the same day as an undocumented metrics change, with no QA verdict on record for RSCH-2 at all. The ML Engineer fixed the six items inside RSCH-4's scope and left the RSCH-2 finding alone. Research QA's second pass recomputed the total-variation numbers by hand and passed all eight criteria. Research Thinker then read the passed result against the original pose-contamination hypothesis and concluded the in-sample check speaks to one narrow claim — the checkpoints aren't underfitting their own high-yaw rows — and nothing about generalization, and that the one near-significant gap traces to a shift in the class prior between bins, not to pose. Research PM recorded that in `docs/MODEL_REPORT.md` under Q3, and the orchestrator closed RSCH-4 in `docs/backlog.md` only once the PASS was on record.

**Checked by human:** the user made four calls this session: confirmed the new script fully replaces the old six-arm run rather than sitting alongside it; chose to launch real Research PM and Research QA subagents instead of accepting the claimed PASS on faith; gave the Q4-obsolete instruction directly, unprompted by any subagent finding; and, once Research QA surfaced the RSCH-2 gap, said to flag it on RSCH-2 and proceed with closing RSCH-4 rather than block on fixing RSCH-2 too.

**What it caught:** a claimed QA verdict that did not exist anywhere in the backlog, for an implementation whose own docstring ruled out satisfying the required controls the old grooming had pre-registered — accepting the claim on faith would have closed RSCH-4 on a result nobody had checked. It also caught, as a side effect of reviewing RSCH-4 and not through any check aimed at RSCH-2, that RSCH-2 has no QA verdict on record and its checkpoint changed silently alongside a metrics change on 2026-09-22 (`e3bb8cd`). Neither is fixed; both are on the record as open findings on RSCH-2's own entry, since the user chose not to expand scope to resolve them now.

**Verified:** `git show --stat` on the commits that deleted the old six-arm artifacts, `git log` on RSCH-2's checkpoint to establish the undocumented regeneration date, and Research QA's second pass hand-recomputing the total-variation distance from the raw counts rather than trusting the ML Engineer's report of it. The final backlog edit was checked with `grep -c` for the count of ticked acceptance-criteria boxes before the issue was marked closed.

---

## 2026-09-20 — RSCH-3/E3 root-caused, re-groomed and re-run; product decision on model quality

**Asked:** take the Research PM role. Work out why the E3 GNNs score at chance while three hand-picked ratios beat chance, decide whether classifier quality affects the product at all, and weigh three hypotheses about the weak metrics: a bad landmark detector, annotator error, and cat-to-cat variability.

**Tool:** Claude Code, main session as Research PM, with three `general-purpose` subagents. Two read-only Explore agents mapped the landmark pipeline and the planning documents in parallel. One agent framed as ML Engineer (`docs/team/ml-engineer.md`, `/ponytail`) implemented and ran the re-groomed experiment; one framed as Research QA (`docs/team/research-qa.md`) reviewed it. The PM session did the diagnosis, the grooming and the write-ups, and wrote no experiment code, per `docs/team/research-pm.md`.

**What it did:** measured the cause of E3's result rather than accepting it. `_Net.forward` ended in `h.mean(dim=1)`, and Procrustes `_center_scale` sets every shape's centroid to zero, so the node-axis mean of the GNN's input is exactly zero for all 1967 rows in all four channels — between-sample variance surviving the readout 2.58e-33, against 1.64e-2 for the coordinates the E2 MLP reads. Confirmed by swapping only the readout in a scratchpad probe, which moved the anatomical arm from 0.266 macro F1 to 0.374 and flipped the run's "anatomical beats random" finding. Found three further defects: the GNN and MLP were balanced by different mechanisms, the verdict compared two chance-level arms on a metric that cannot separate them, and the adjacency file had been revised twice with no record. Re-groomed RSCH-3 with the readout named as a control and a new `gnn_identity` arm, routed it to the ML Engineer, then to Research QA. Separately probed the ceiling (random forest and gradient boosting reach training macro F1 1.000 and test at kappa 0.24) and pose (PC1+PC2 alone score kappa 0.033), which is what ranked the three hypotheses. Wrote three `docs/DECISIONS.md` entries and gave `docs/MODEL_REPORT.md` the Q1-Q5 structure RSCH-7 asks for.

**Checked by human:** the user corrected the elapsed-time estimate. I had read the app track as four days in against an unstarted M0 and flagged the "research stops if the app is behind" rule; the user said research is at day 2, so I withdrew the flag. The subagent had inferred elapsed time from nominal day counts in the plan, which is not the same thing, and no artifact in the repo records real elapsed time. The rest is a first pass and still needs review.

**What it caught:** the readout defect, which no test could have caught because every test passed and the model trained without error. The failure lived in the interaction between two modules that are individually correct: Procrustes centering in `core/geometry.py` and mean pooling in `ml/gnn.py`. It also caught that the E3 run compared arms balanced two different ways, that E2's uniform-dummy reference was never carried into E3 (which is what hid the anatomical GNN scoring *below* chance on macro F1), and that `models/graph_edge_schemes/` was gitignored, so two revisions of the experiment's sole independent variable have no commit record.

**Verified:** the ML Engineer's re-run reproduced the PM's scratchpad probe within noise on every arm, including direction. The `gnn_identity` control, which the first grooming missed, turned out to be the informative one: message passing switched off at equal depth and parameter count scores 0.231 kappa against the anatomical arm's 0.096, so the graph convolution hurts. Research QA re-ran all five arms independently, matched every committed number bit for bit, and returned PASS; the verdict is recorded against the RSCH-3 entry in `docs/backlog.md`.

QA caught a process error in this entry. An earlier draft of this paragraph asserted that QA's verdict was recorded while the review was still running, which is drafting a result ahead of the work that produces it. Corrected here, and worth not repeating: a log written as it happens cannot describe a step that has not happened.

QA also went past what it was asked to do, in the useful direction. Rather than accept the Research PM's argument that tuning the edge list would have biased toward the anatomical arm and so could not explain a negative result, it ran v1 and v2 through the same arm to test the claim: 0.111 and 0.075 kappa against v3's 0.096, all three losing to the random adjacency. The scheme in use is not even the best of the three, which is the opposite of what a metric-selected scheme looks like, so the provenance gap cannot reach the Q1 answer. It then found something nobody had noticed: v3 dropped the eye-to-ear-base and eye-to-nose bridges, so the graph actually tested splits into four disconnected components and no longer matches the adjacency RSCH-3's own Method describes.

Still open, surfaced and not resolved: `pyproject.toml` gained `onnx`, `onnxruntime` and `onnxsim` without being asked for, against AGENTS.md's dependency rule, and only `onnx` is used. `GNNClassifier` seeds once per construction so every fold shares an initialization, which compresses fold spread and makes the CV-std threshold easier to clear (conservative for a NO verdict, but it should be a per-fold seed). The first run's `metrics.json` files were never committed, so the re-groom's "the first run's artifacts stay on disk" did not hold; the retained mean-pool arm covers the substance.

---

## 2026-09-20 — RSCH-2/E2 addendum: kappa/MCC, random-baseline reference, oversampling

**Asked:** implement a second direct-user-instruction addendum to E2's three baseline models: report Cohen's kappa and MCC alongside macro F1, add a uniform-random `DummyClassifier` chance-level reference (reported separately, excluded from the Q2 verdict), and replace the balancing mechanism — inverse-frequency `class_weight="balanced"` / class-weighted `CrossEntropyLoss` — with random oversampling of minority classes on the training folds only, test folds untouched.

**Tool:** Claude Code, main session, `general-purpose` agent framed as ML Engineer per `docs/team/ml-engineer.md` and `docs/research_process.md`, `/ponytail full`. No Explore/Plan subagents — the exact implementation (function signatures, file list, execution order) was already specified in the instruction, so this was direct implementation against a given design rather than open design work.

**What it did:** added `oversample_to_balance` (per-class `sklearn.utils.resample` up to the majority count) and extended `cross_validate` with an `oversample` kwarg (default `False`, so RSCH-3/E3's existing `gnn.py` call site is unaffected) in `src/catface/ml/cv.py`; dropped `class_weight="balanced"` from `ratios_lr.py`/`coords_lr.py` and the `compute_class_weight`/`weight_tensor` block from `mlp.py`, replacing all three with `oversample=True` at their `cross_validate` call sites; added `src/catface/ml/random_baseline.py` (`DummyClassifier(strategy="uniform", random_state=0)`); extended `scripts/baselines.py` with a `random_baseline_section` and a separate (not `MODELS`-dict-driven) code path for the fourth reference run, plus kappa/MCC columns in the per-model tables and an updated "Required controls" checklist; extended `tests/test_baselines.py` with `EXPECTED_METRIC_KEYS` coverage, an `oversample_to_balance` unit test on synthetic 14/4/2 class counts, and a `random_baseline` smoke test. Reran `scripts/baselines.py` for real against the existing `data/cache/splits.csv` (not regenerated).

**Checked by human:** none yet — first pass, pending review. The implementation spec itself (exact function signatures, file list, what stays out of scope) was pre-written by the user, not designed by the agent.

**What it caught:** nothing that changed the design — the spec was implemented as given. It did confirm one assumption named in the instruction: `q2_verdict()` uses explicit `all_metrics["ratios_lr"]`/`["mlp"]` key lookups, not generic iteration, so leaving `random_baseline` out of `all_metrics`/`MODELS` (to avoid a `MODEL_COLORS` `KeyError` in the generic comparison plot) doesn't affect it.

**Verified:** `uv run pytest tests/test_baselines.py -v` (8/8) before touching `scripts/baselines.py`, then `uv run pytest tests/` (37/38 — the one failure, `tests/test_graph.py`, is a pre-existing RSCH-3/E3 issue unrelated to this change, confirmed by `git status` showing `core/graph.py` already modified before this session started; `tests/test_gnn.py` itself passed 7/7, confirming `cv.py`'s new `oversample` kwarg defaulting to `False` left E3 behavior unchanged). Ran `scripts/baselines.py` for real and inspected the regenerated `experiments/e2/README.md`, all four `metrics.json` files, and the five plots. Confirmed `git diff -- data/cache/splits.csv` is empty. Q2 verdict recomputed as still NO MATCH under the new balancing mechanism (gap widened from 0.029 to 0.033 macro F1). Result recorded in `docs/backlog.md` RSCH-2 as an addendum/comment pair; the issue itself was not closed, per process (Research QA's call). `docs/MODEL_REPORT.md` has no E2 section yet, so it was left untouched, per process (Research PM writes it at PASS).

---

## 2026-09-19 — E0 close-out: overlay verdict analysis, MODEL_REPORT, QA, backlog

**Asked:** analyse the 80-overlay verdict the user had written by hand into `data/cache/overlays/overlay.csv`, take the user's observation that Scared, Surprised and Angry were mixed up, and close RSCH-0.

**Tool:** Claude Code, main session, plan mode, `/ponytail full`. No Explore or Plan agents; the file set was small and known. One `general-purpose` agent ran as Research QA (read-only) for the verdict.

**What it did:** joined the CSV back to `landmarks.parquet` by regenerating the same `sample(80, random_state=0)` and matching the trailing index in each filename. That join produced the findings the close-out rests on: 69/80 OK overall but 57/59 on cat-emotions-3 versus 12/21 on cat-emotions-7; every one of the 11 human-rejected detections has `plausible=True`; the random sample contained none of the 60 filter rejects, so the reject-review control had silently not happened. Also found that the uncommitted deletions in both dataset `README.txt` files were caused by `dataset_stats.py` being rerun after `landmark_cache.py`: `write_readme` overwrote the whole file and erased the "Detector run (E0)" section. Fixed `write_readme` to keep sections it does not own, added `--from-cache` to `landmark_cache.py` so READMEs and overlays regenerate without a detector pass, and reran both scripts in both orders to confirm the READMEs are byte-stable. Wrote `docs/MODEL_REPORT.md`, three DECISIONS entries, and the backlog edits (RSCH-0 closed, RSCH-1 to RSCH-4 scoped to cat-emotions-3, RSCH-4's 7-class matrix replaced, RSCH-7's relabel line replaced).

**Checked by human:** the user made four decisions before implementation: exclude cat-emotions-7 from training (their reading of the reject reasons: image quality, not detector fault; image-quality scoring to be added at inference later), waive the blind relabel, skip the reject-sample review at this stage, and run Research QA as a subagent. The user also approved, at the QA agent's permission prompt, that agent rewriting `MODEL_REPORT.md` into a shorter form with a Models section; the main session then verified the facts the agent added (HF source, NME 3.48, 0.1 crop margin, CatFLW 2079 on disk) against `models/manifest.json`, `detect_landmarks.py` and the CatFLW README. One typo in the user's CSV (`annotation_ok=25`) was corrected to `1` with their approval of the plan. Citation details in MODEL_REPORT were written from the assistant's memory and need a human check against the papers.

**What it caught:** nothing that changed the E0 answer. QA re-derived every reported number and all matched; it did catch one wording slip in the first MODEL_REPORT draft (the two "mouth open" passes were one per dataset, not both cat-emotions-3) and fixed it in the rewrite.

---

## 2026-09-19 — E0 remainder: licence table, near-duplicate check, plausibility filter, parquet cache

**Asked:** read `docs/backlog.md`, implement the rest of RSCH-0 (E0) —
everything after the fetch scripts and the single-image detector CLI from the
two prior sessions. Manual relabeling (the 100-image blind relabel) explicitly
out of scope. Also: document dataset statistics in a `README.txt` in every
folder under `data/`. The Roboflow datasets were named `cat-emotions-3` and
`cat-emotions-5` in the request.

**Tool:** Claude Code, main session, plan mode, `/ponytail full`. Three
Explore agents in parallel (dataset folder contents, `ml`/`core` code and
dependencies, research-process docs and prior E0 session logs), then one Plan
agent to turn the findings into a concrete file-by-file design before writing
code.

**What it did:** confirmed `cat-emotions-5` doesn't exist on disk (the real
folder is `cat-emotions-7`, which matches the backlog's own "7 classes"
description) and confirmed with the user which one to use before touching
files. Verified the CatFLW landmark index scheme (eye/muzzle groups) against
4 real labels rather than trusting the vendor's undocumented ordering.
Inspected the two OpenVINO models directly and found neither emits a
confidence score, despite the backlog's parquet schema naming a
`detector_confidence` column — built a documented geometric proxy instead.
Wrote `src/catface/ml/plausibility.py` (filter + proxy, unit-tested),
`scripts/dedupe.py` (near-duplicate hashing), `scripts/dataset_stats.py`
(licence table, actual-vs-advertised counts, README.txt generation), and
`scripts/landmark_cache.py` (batch detector run + parquet cache). Ran both
scripts for real against the full datasets — not a dry run.

**Checked by human:** plan reviewed and corrected once before implementation
started (see below). `landmark_cache.py` was moved from `src/catface/ml/` to
`scripts/` mid-implementation on direct instruction, since it's a run-once
script like the other two, not reusable package logic.

**What it caught:** an average-hash near-duplicate implementation initially
reported 126 candidate matches between the two Roboflow sets, including one
at Hamming distance 0 ("exact match"). Visually checking that pair showed two
completely unrelated cats — the hash was coincidentally collapsing different
photos to the same rough light/dark pattern. Switched to a difference hash
(dHash), re-verified against the same pair (clearly distinct under dHash),
and reran the full comparison: zero near-duplicates found. The real batch run
crashed on image 1153 of 2071 — a degenerate localizer box produced a
zero-size crop and `cv2.resize` raised. Fixed by catching it per-image and
recording a `detection_failed` row instead of losing the whole run. The first
`detector_confidence` proxy measured landmarks against the same expanded crop
box the landmarks model's output is normalized onto — tautologically 1.0 on
all 2741 valid rows, caught by looking at the confidence histogram after a
real run and seeing a single spike instead of a distribution. Switched to
measuring against the tight (unexpanded) localizer box instead, which isn't
part of the model's own output space and gave a real spread (mean 0.99, min
0.54). Rerunning the pipeline twice for these fixes also exposed that
`append_section` was a plain file-append, not idempotent — it left two
"## Detector run (E0)" sections in the same `README.txt` after the second
run; fixed to replace a same-named section instead. All four are written up
in `docs/DECISIONS.md`, along with the CatFLW ground-truth aspect-ratio
bounds and eye/muzzle index groups, both verified against real data rather
than trusted from `docs/plan-research.md`'s prose description alone.

**Plan corrections from the user, before implementation:** cache every
detector prediction regardless of plausibility, but don't let the pass/fail
split decide which images go into the human-eyeball overlay sample (uniform
random instead); skip `experiments/` entirely — this isn't a multi-run tuning
experiment, so outputs live under `data/cache/` next to the cache itself, with
a `README.md` documenting it; no `test_dedupe.py`; `dedupe.py` and
`dataset_stats.py` (and later `landmark_cache.py`) belong in `scripts/`, not
`catface.ml`, since none of them are reusable package logic.

**Verified:** `uv run pytest tests/test_plausibility.py tests/test_geometry.py`
(9 passed). `uv run python scripts/dataset_stats.py` against the real
datasets — inspected the three generated `README.txt` files by hand.
`uv run python scripts/landmark_cache.py` against the real datasets (2742
images, ~2.5 minutes on CPU) — 2682 plausible (97.8%), stop condition NOT
triggered. Opened several `data/cache/overlays/*.jpg` by hand: box and
landmarks land correctly on ears, eyes, nose, and muzzle on every one checked.
Confirmed via `git add --dry-run data/` that the `.gitignore` ladder stages
exactly the four `README` files and nothing else under `data/`.

**Not yet done:** the human parts of E0 — actually looking at the 20 sampled
overlays and giving a verdict, and the 100-image blind relabel (explicitly
out of scope for this pass). `data/cache/README.md` calls out both as open
TODOs. Research QA review and closing the RSCH-0 issue belong to the
orchestrator and the Research PM.

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

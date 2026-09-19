# Backlog

Tagged by track. `[research]` items below are groomed from `PLAN_RESEARCH.md`
per `research_process.md`'s lifecycle (one issue per experiment, in order).
`[project]` items are deliberately absent. `project_process.md` has PM groom
a project task immediately before it's implemented, so that track runs off
`PLAN_PROJECT.md`'s milestone list until M0 grooming starts — pre-grooming M8
today produces text that's stale before anyone reads it. This file is
research-only on purpose, not by omission.

Grooming shape follows `research_process.md` step 2 (question, method, required
controls, stop condition) since no generic task template exists in this repo.

---

## [research] RSCH-0: E0 — landmark cache

**Owner:** Data Steward · **Blocked by:** none · **Time box:** 0.5 day · **Status:** closed 2026-09-19, QA verdict below

**Question:** Does the landmark detector (trained on CatFLW) work on the
Roboflow image distribution at all?

**Answer:** yes on cat-emotions-3 (57/59 spot-checked detections OK, 2.0% filter drop rate). cat-emotions-7 is excluded from training for image-quality reasons; see `docs/DECISIONS.md` 2026-09-19 and `docs/MODEL_REPORT.md` E0.

**Method:**
- Record the licence for every source: Roboflow cat-emotions-cgrxv (2,071
  imgs, 3 classes), Roboflow cat-emotions (671 imgs, 7 classes), CatFLW
  (2,016 imgs, 48 landmarks, CC BY-NC 4.0)
- Hash images across both Roboflow sets, report near-duplicates or shared
  sources
- Report actual post-download class balance against the advertised numbers
- Run the detector once over both Roboflow sets, cache to parquet:
  `image_id, class, 48x2 landmarks, face_box, detector_confidence`
- Filter implausible detections (points inside box, eyes above muzzle, no
  degenerate/collapsed configs, no absurd aspect ratios); report drop count
- Eyeball random overlays (80 in practice); confirm the detector works on
  this image distribution
- Plot class distribution and detector confidence
- ~~Relabel 100 images blind, compare to dataset labels, report agreement~~
  waived, `docs/DECISIONS.md` 2026-09-19
- ~~Eyeball a sample of rejects; confirm the filter isn't throwing away
  hard-but-valid cases~~ dropped from scope, `docs/DECISIONS.md` 2026-09-19

**Required controls:** near-duplicate hash check. (Reject-sample review was a
required control at grooming; re-scoped out on 2026-09-19, see DECISIONS.)

**Acceptance criteria:**
- [x] Licence table filled in for all three sources (`data/*/README.txt`, `docs/MODEL_REPORT.md`)
- [x] Near-duplicate result reported (0 pairs at dHash Hamming ≤ 5, `data/cache/near_duplicates.csv`)
- [x] Actual vs advertised class balance reported (cat-emotions-3: 8 folders, not 3)
- [x] Parquet cache exists at `data/cache/landmarks.parquet`
- [x] Reject count reported (42 of 2071 on cat-emotions-3; 60 of 2742 overall)
- [x] Overlay spot check done and reported (80 images, `data/cache/overlays/overlay.csv`, summary in MODEL_REPORT)
- [x] Stop condition answered explicitly: NOT triggered
- ~~100-image blind relabel agreement number reported~~ waived
- ~~Manual reject-sample verdict~~ dropped from scope

**Stop condition:** if the detector fails on most images, halt the research
track entirely and escalate to Research PM before E1 starts.

**Research QA verdict (2026-09-19), subagent, read-only:**

> ## QA: PASS
> - [x] Re-scopes on record: three 2026-09-19 entries in `docs/DECISIONS.md` (cat-emotions-7 excluded, relabel waived, reject review dropped); backlog strikes both and points at them
> - [x] Required control, near-duplicate hash check: `near_duplicates.csv` header-only, 0 pairs at dHash Hamming <= 5, closest pair 6; both dataset READMEs agree
> - [x] Licence table: CC BY 4.0 / CC BY 4.0 / CC BY-NC 4.0 with URL and confirmation source in each `data/*/README.txt` and MODEL_REPORT
> - [x] Actual vs advertised: cat-emotions-3 8 folders not 3 (2071 matches parquet); cat-emotions-7 7 classes; CatFLW 2079 on disk vs 2016 advertised, stated in both places
> - [x] Parquet cache at `data/cache/landmarks.parquet`, 2742 rows, all 12 declared columns
> - [x] Reject count re-derived from parquet: cat-emotions-3 2071 / 2029 / 42 (eyes_below_muzzle 21, degenerate 20, detection_failed 1, bad_aspect_ratio 1); cat-emotions-7 671 / 653 / 18; matches reported exactly
> - [x] Overlay spot check reproduced: `valid.sample(80, random_state=0)` matches all 80 filenames; 59/57 and 21/12, 69/80 total; all 11 rejected rows `plausible == True`; proxy means 0.958 vs 0.993. Matches
> - [x] Stop condition stated verbatim as NOT triggered in `data/cache/README.md` (with the 50% definition) and MODEL_REPORT
>
> Result: every reported number re-derives from the parquet and CSV; the one remaining required control ran and produced a documented negative. Detector works on cat-emotions-3 (57/59 by human judgment, 2.0% filter drop).

---

## [research] RSCH-1: E1 — shape space

**Owner:** ML Engineer · **Blocked by:** RSCH-0, `core.geometry` (app track
M2 — Procrustes lives there per AGENTS.md, and E1 imports it rather than
writing a local copy) · **Time box:** 0.5 day

**Question:** Is there visible class signal before training anything, and
does any early PC track a confound (ear position, head yaw) instead?

**Method:**
- Procrustes-align the cached landmarks (via `core.geometry`), cat-emotions-3
  rows only (`docs/DECISIONS.md` 2026-09-19)
- PCA, plot the first six components
- Check whether any component visibly tracks ear position or head yaw
- Check whether classes visibly separate in the first few PCs
- Write `models/class_means.json`: per-class mean of the aligned landmarks,
  one groupby over the alignment this experiment already computes. The
  app's warper (M3) and its M5 manual-picker fallback block on this file
  and not on the trained model, so it ships here, not at export.

**Acceptance criteria:**
- [ ] Procrustes + PCA run on the E0 cache (not raw landmarks)
- [ ] First-six-PC plot committed to the run directory
- [ ] Explicit call-out of any PC that tracks a pose confound
- [ ] Explicit statement on whether classes separate visibly
- [ ] `models/class_means.json` committed, one 48×2 mean per class, and the
      app track told it exists

**Stop condition:** none named in the plan — this is a look, not a gate.

---

## [research] RSCH-2: E2 — baselines

**Owner:** ML Engineer · **Blocked by:** RSCH-0 · **Time box:** 1 day

**Question:** Q2 — does a learned model beat two geometric ratios (eye
aperture, ear angle) fed to logistic regression?

**Method:** cat-emotions-3 rows only (`docs/DECISIONS.md` 2026-09-19); which
of its 8 label folders become classes is decided here at grooming. Same
stratified splits for all three, 5-fold, balanced class weights:
1. Logistic regression on eye aperture + ear angle
2. Logistic regression on all Procrustes-aligned coordinates
3. MLP, two hidden layers, same input as (2)

**Required controls:** identical splits across all three models, 5-fold,
macro F1 (not accuracy alone), confusion matrix per model

**Acceptance criteria:**
- [ ] All three models trained on the same stratified 5-fold splits
- [ ] Macro F1 + confusion matrix reported for each
- [ ] Explicit Q2 verdict: does (1) match (3) within CV noise or not
- [ ] `experiments/<run>/metrics.json`, config, git sha committed for each
      of the three runs

**Stop condition:** none — a match between (1) and (3) is a valid (negative)
answer to Q2, not a failure.

---

## [research] RSCH-3: E3 — the GNN

**Owner:** ML Engineer · **Blocked by:** RSCH-2 · **Time box:** 1–2 days

**Question:** Q1 — does a dense graph conv over the anatomical adjacency
beat the E2 MLP?

**Method:**
- Hand-written dense graph conv, `H' = act(A_hat @ H @ W)`, three layers,
  mean pool, linear head (~40 lines). **No PyTorch Geometric** — its sparse
  scatter ops break ONNX export; the dense form is identical maths on a
  fixed 48-node graph
- Adjacency hand-written from anatomy in `core/graph.py`: eyelid contours
  as rings, ear base→tip, whisker pads→nose, nose→mouth, eyes→ear bases.
  Normalise `D^-1/2 (A+I) D^-1/2`
- Node features: aligned `(x,y)` plus offset from the mean shape; same
  cat-emotions-3-only rows and splits as E2
- **Before training it:** export the untrained model to ONNX once. The
  no-PyG decision above is an assumption about export behaviour, and this
  is the cheap place to test it — an export problem found here changes the
  architecture, found at RSCH-6 it costs a retrain against the deadline
- **Required ablation:** same architecture with a random adjacency of the
  same edge count

**Acceptance criteria:**
- [ ] Untrained-model ONNX export attempted and its result recorded before
      any training run
- [ ] Anatomical-adjacency GNN trained, macro F1 + confusion matrix, 5-fold
- [ ] Random-adjacency ablation trained under identical conditions
- [ ] Q1 verdict computed as: GNN beats E2 MLP by >1 CV std **and**
      anatomical adjacency beats random adjacency — both conditions
      checked, not just the headline F1
- [ ] Negative outcome (random does as well) reported as such, not omitted
- [ ] No `torch_geometric` import anywhere in the run

**Stop condition:** none — a random-adjacency tie is the Q1 answer.

---

## [research] RSCH-4: E4 — class structure and pose

**Owner:** ML Engineer · **Blocked by:** RSCH-2, RSCH-3 · **Time box:** 1 day

**Question:** Q3 (pose contamination) and Q4 (class separability).

**Method:**
- Confusion matrix over cat-emotions-3's classes as E2 defined them (the
  7-class set is excluded, `docs/DECISIONS.md` 2026-09-19; usable folders are
  attentive 1147, relaxed 752, uncomfortable 107, the other five are too
  small to be classes). Merge classes that are inseparable and state why.
  The E0 spot check found Scared/Surprised/Angry interchangeable on
  cat-emotions-7; if that set ever comes back, that is the expected merge.
- Estimate head yaw from landmark asymmetry, bin it, report accuracy per
  bin

**Acceptance criteria:**
- [ ] Confusion matrix over the E2 classes produced, merge decisions stated with reason
- [ ] Yaw estimated and binned; accuracy-per-bin table produced
- [ ] A one-sentence pose verdict in the shape "stable under N degrees,
      degrades past that" — this becomes the app's confidence penalty input
- [ ] Q3 and Q4 both answered explicitly, including if the answer is
      negative or inconclusive

**Stop condition:** none.

---

## [research] RSCH-5: E5 — my cat

**Owner:** ML Engineer · **Blocked by:** RSCH-4 · **Time box:** 0.5 day

**Question:** Q5 — does the final model transfer to photos outside both
training datasets?

**Method:** Run the final (winning) model from E2–E4 over the researcher's
own cat photos. Qualitative only.

**Acceptance criteria:**
- [ ] Model run over own-cat photo set
- [ ] Three failure images selected with a written explanation each
- [ ] Q5 answered as a qualitative verdict, not a metric (none exists here)

**Stop condition:** none.

---

## [research] RSCH-6: export to ONNX/OpenVINO

**Owner:** Export Verifier · **Blocked by:** RSCH-5 (E5 closed) · **Time box:** unboxed, gates delivery

**Question:** Does the exported artifact match the model Research QA
evaluated?

**Note:** this is the *second* of the two cross-track artifacts, not the
only one. `models/class_means.json` shipped back at RSCH-1 and the app's
warper has been running on it since M3.

**Method:**
- `torch.onnx.export`, opset 17, static shapes, input `[1,48,2]`, via a
  script (not notebook cells) — the same script CI runs on every model
  change
- Check ONNX output matches PyTorch within 1e-5
- `ov.convert_model`, check OpenVINO output matches ONNX within 1e-5
- Benchmark p50/p95 latency and peak RSS on the deploy CPU
- Write `models/manifest.json`: name, file, sha256, source, licence, input
  shape, metrics

**Acceptance criteria:**
- [ ] ONNX-vs-PyTorch parity recorded (numeric diff, not "it matched")
- [ ] OpenVINO-vs-ONNX parity recorded, same tolerance
- [ ] p50/p95 + peak RSS recorded on the deploy CPU, not a dev machine
- [ ] Manifest entry committed with sha256 matching the committed file
- [ ] Export script is the committed artifact CI runs, no hand-run export

**Stop condition:** parity failure at either step blocks the handoff; this
is an export bug, not a Research QA or model-quality matter. Report exact
diff values, don't round tolerance up.

---

## [research] RSCH-7: compile MODEL_REPORT.md

**Owner:** Research PM · **Blocked by:** nothing — this one runs alongside
RSCH-0..RSCH-6 and gets a final pass once RSCH-6 closes · **Time box:**
none named

**Method:** Research PM records each experiment's result — including
negative ones — under the matching question as it closes (per
`research_process.md` step 6), not all at once at the end. Listing this as
blocked by every other item would have contradicted that method and turned
the report into the end-of-project scramble the process exists to prevent.

**Acceptance criteria:**
- [x] Dataset table with licences, the E0 near-duplicate result, and the
      cat-emotions-7 exclusion with its reason
- [x] Label quality: blind relabel waived (`docs/DECISIONS.md` 2026-09-19),
      the informal spot-check note recorded instead
- [ ] Results for all four models (2× logistic regression, MLP, GNN) plus
      the random-adjacency ablation
- [ ] Q1–Q5 answered explicitly, negative answers included
- [ ] Confusion matrices included
- [ ] Three E5 failure photos with explanations
- [ ] Limitations section: perceived-emotion labels, internet photos, no
      clinical validity, nothing claimed about pain/welfare
- [ ] Citations to CatFLW and the Finka landmark scheme

**Stop condition:** none.

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
writing a local copy) · **Time box:** 0.5 day · **Status:** closed 2026-09-19, QA verdict below

**Question:** Is there visible class signal before training anything, and
does any early PC track a confound (ear position, head yaw) instead?

**Answer:** no on both counts, correctly so. PC1 tracks ear position (r = 0.84) and PC2 tracks head yaw (r = 0.90) before any class signal shows up, and the three usable classes (attentive, relaxed, uncomfortable) do not visibly separate in the first six PCs (best ratio 0.044, PC3). See `experiments/e1/README.md` and `docs/MODEL_REPORT.md` E1.

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
- [x] Procrustes + PCA run on the E0 cache (not raw landmarks)
- [x] First-six-PC plot committed to the run directory
- [x] Explicit call-out of any PC that tracks a pose confound
- [x] Explicit statement on whether classes separate visibly
- [x] `models/class_means.json` committed, one 48×2 mean per class, and the
      app track told it exists

**Stop condition:** none named in the plan — this is a look, not a gate.

**Research QA verdict (2026-09-19), subagent, read-only:**

> ## QA: PASS
> - [x] Procrustes + PCA run on the E0 cache, not raw landmarks — `scripts/shape_space.py` reads `data/cache/landmarks.parquet`, filters `dataset == "cat-emotions-3" & plausible == True` (2071 -> 2029 rows), runs `generalized_procrustes` from `core/geometry.py`, then `sklearn.decomposition.PCA(n_components=6)` on the aligned coordinates. Re-ran the script myself: reproduces row counts (2071/2029), explained variance, correlations, and bit-identical `models/class_means.json` and byte-identical plot PNGs.
> - [x] Procrustes math correct — `procrustes_align` centers+scales `shape`/`reference` independently, fits rotation via SVD (Kabsch), and flips the sign of U's last column whenever `det(R) < 0` to forbid reflections; `generalized_procrustes` iterates this to a fixed point. `tests/test_geometry.py` (8 tests, all pass) includes a non-tautological reflection test that first shows an unconstrained Kabsch fit *would* land exactly on the reference via an improper rotation (det<0, residual ~1e-16), then confirms `procrustes_align` refuses that solution and returns a proper rotation instead.
> - [x] First-six-PC plot(s) committed to `experiments/e1/plots/` — `pc_explained_variance.png` (bars for PC1-6 + cumulative line), `pc_scatter_grid.png` (PC1v2, PC3v4, PC5v6 colored by class), `pc_confound_scatter.png` (strongest confound hit). Visually confirmed both: scatter grid shows heavy class overlap, confound plot shows a clean linear PC2/head_yaw trend consistent with the reported r=0.90.
> - [x] Explicit call-out of PC tracking a pose confound — `experiments/e1/README.md` states in prose (not just a plot reference): "**ear_position**: PC1 tracks it, r = 0.84" and "**head_yaw**: PC2 tracks it, r = 0.90," against a stated |r|>0.3 threshold. Re-ran the correlations myself from the same aligned array; matched to 3 significant figures (0.836, 0.896).
> - [x] Explicit statement on class separation — README states in prose: "Classes do not visibly separate in the first six PCs — the best ratio (PC3, 0.044) is small." This is a between/within variance ratio computed across the three usable classes, re-derivable from the same aligned array; the scatter-grid plot backs it up visually (no separable clusters by color).
> - [x] `models/class_means.json` committed (not gitignored) and correct — valid JSON, exactly 3 top-level keys (`attentive`, `relaxed`, `uncomfortable`), each a 48x2 array. Independently re-ran Procrustes over the same parquet filter and re-computed the `relaxed` class mean myself: max abs diff from the committed file is 0.0 (bit-exact, as expected since GPA+groupby is deterministic). `.gitignore`'s blanket `experiments/` line is gone; `git check-ignore` confirms `experiments/e1/README.md`, its plots, and `models/class_means.json` are all trackable.
> - [x] App track told class_means.json exists — `docs/PLAN_PROJECT.md` documents it in four places (dataset table, M3 "committed not fetched" note, M5 fallback dependency, fallback table), predating this run. One stale spot found, not blocking: `AGENTS.md`'s "Cross-track dependency" section still calls the ONNX export "the only handoff between tracks," which `docs/DECISIONS.md`'s own "models/class_means.json is an E1 deliverable" entry already flagged as needing a one-line fix that was never applied — pre-existing grooming-era gap, not something E1 introduced, doesn't block this PASS.
> - [x] Required-control check — RSCH-1 names no control beyond its acceptance criteria ("this is a look, not a gate"); no implicit control skipped.
> - [x] Full test suite — `uv run pytest tests/ -v`: 13/13 pass (8 geometry + 5 plausibility).
>
> Result: negative on both fronts, and correctly so — PC1 tracks ear position (r=0.84) and PC2 tracks head yaw (r=0.90) before any class signal shows up, and the three usable classes (attentive/relaxed/uncomfortable) do not visibly separate in the first six PCs (best between/within ratio 0.044). This is RSCH-1's expected "look, not a gate" outcome, correctly reported rather than hidden, and every number reproduces exactly from the E0 cache and the committed script.

---

## [research] RSCH-2: E2 — baselines

**Owner:** ML Engineer · **Blocked by:** RSCH-0 · **Time box:** 1 day

**Question:** Q2 — does a learned model beat two geometric ratios (eye
aperture, ear angle) fed to logistic regression?

**Method:** cat-emotions-3 rows only, three usable classes (see Grooming notes below). Same stratified splits for all three, 5-fold, balanced class weights:
1. Logistic regression on eye aperture + ear angle
2. Logistic regression on all Procrustes-aligned coordinates
3. MLP, two hidden layers, same input as (2)

**Grooming notes (Research PM, 2026-09-19):**

- **Classes:** the three usable classes are `attentive`, `relaxed`, `uncomfortable`. This isn't a fresh call — it locks in the precedent already set by `models/class_means.json` (E1) and RSCH-4's method above, both grounded in `docs/DECISIONS.md`'s 2026-09-19 "`models/class_means.json` covers only the three usable cat-emotions-3 classes" entry: the other five label folders (`no clear emotion recognizable`, `sad`, `angry`, `Unlabeled`, `attentive uncomfortable`) are single/low-double-digit counts or aren't real expression labels at all. E2 trains on the same three classes E1 already computed means for and RSCH-4 already assumes; there is no open question left here.
- **Code layout:** the two required derived-geometry functions, `eye_aspect_ratio` and `ear_angle`, go in `src/catface/core/geometry.py` — they're preprocessing formulas the app track's UI will also use as readouts, not private to the research track, so they belong with the rest of `core` rather than in `ml`. Everything else here is research-only and goes under `src/catface/ml`: `splits.py` (dataset filter + `StratifiedKFold` + cache read/write, kept separate from training code specifically so RSCH-3 can import the split logic without importing training code too — its own backlog entry requires reusing "the same rows and splits"), `features.py` (the two ratio features for model 1, and Procrustes-aligned flattened coordinates for models 2 and 3, shared by all three model files rather than duplicated), `cv.py` (the fold-by-fold macro F1 + confusion-matrix scoring, identical logic for all three models), and one file per model — `ratios_lr.py` (model 1), `coords_lr.py` (model 2), `mlp.py` (model 3) — each just defining its model plus a `run()` that composes `features.py`, `splits.py`, and `cv.py`, mapping 1:1 onto the three `experiments/e2/<name>/` run directories the acceptance criteria already require. Two run-driving scripts sit in `scripts/`, matching the existing `landmark_cache.py`/`shape_space.py` pattern of "script drives, module does the work": `scripts/make_splits.py` writes `data/cache/splits.csv`, and `scripts/baselines.py` calls the three models' `run()` and writes to `experiments/e2/...`.
- **Split persistence:** the CV split — `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` — is computed once, not recomputed inline per model script, and written to `data/cache/splits.csv` with columns `image_path,label,split` (`split` is the fold index, 0-4). Seed 0 is picked to match the seed already in use elsewhere in this repo (`scripts/shape_space.py`'s overlay sample). This file is the artifact RSCH-3 reuses for "the same rows and splits" per its own backlog entry — it has to exist as a committed file, not just as reproducible code, so RSCH-3 doesn't silently redraw folds when it runs later.
- **MLP framework:** model (3) is a small PyTorch MLP with two hidden layers, not sklearn's `MLPClassifier` — sklearn's implementation has no `class_weight`/`sample_weight` support, and the required control here is balanced class weights identical across all three models. The PyTorch version applies them via `nn.CrossEntropyLoss(weight=<balanced class weights>)`. `torch` is already a `train`-group dependency in `pyproject.toml`, so this needs no new dependency, only a framework choice within what's already installed.
- **Muzzle-spread readout:** ship a third derived-geometry function, `muzzle_spread_ratio(shape, muzzle_indices, left_eye_center, right_eye_center) -> float`, in `core/geometry.py` alongside the two required ones. It's the same category as eye aperture and ear rotation — a formula, no learning, no labels — and `docs/PLAN_RESEARCH.md` (around lines 46-48) already names whisker-pad spread as computable from the lower-face points, while excluding a *tension* feature specifically because "tension has no label source anywhere." This ships the spread, not the tension: defined purely geometrically as the single farthest-apart pair among the 22 `MUZZLE` points (`src/catface/ml/plausibility.py`), found by max pairwise distance rather than a named whisker-pad index pair, divided by inter-ocular distance. It's defined this way because neither CatFLW's GitHub repo nor its arXiv paper (2305.04232) publish an index-to-anatomy map for `MUZZLE`'s 22 points — the group is defined only by exclusion (nose, whiskers, and mouth lumped together) — so no code can name "this index is the left whisker pad," and max-pairwise-distance is a defensible stand-in since whisker pads are anatomically the widest bilateral points on the lower face. Two hard constraints follow: it is **not** part of Q2's required two-ratio baseline for model (1) — model (1) stays exactly eye aperture + ear angle, per the question as posed — and it is **not** called "tension" anywhere in code or docs, because there is still no label for what "tension" would mean; call it what it measures, a spread.
- **Citation:** `docs/MODEL_REPORT.md`'s Citations section already names the CatFLW paper and the Finka landmark scheme. Add one more at E2: the feline pain-assessment paper (Scientific Reports, https://www.nature.com/articles/s41598-023-49031-2), which uses 35 geometric descriptors — angles, distance ratios, area ratios grouped by action unit — over cat facial landmarks, and reports that ear-position and orbital-tightening descriptors gave the smallest prediction error of the set. This belongs in `experiments/e2/README.md` and in `docs/MODEL_REPORT.md`'s Citations section as external validation that eye aperture and ear angle are a reasonable pair to baseline against, not as a requirement to add features beyond Q2's stated two.

**Addendum (direct user instruction, 2026-09-19, overrides the muzzle-spread grooming note above):** the user asked, after grooming, to wire `muzzle_spread_ratio` into model (1)'s training input rather than keeping it as a descriptive-only readout. `ratio_features()` in `src/catface/ml/features.py` now returns 3 features (eye aspect ratio, ear angle, muzzle spread) instead of 2; `ratios_lr.py`'s `n_features` is 3; `tests/test_baselines.py`'s shape assertion and `scripts/baselines.py`'s README generation (the "Extra readout" section is removed, since the readout is now a real model input) were updated to match. This makes model (1) diverge from Q2's original "two geometric ratios" framing as literally posed — `docs/PLAN_RESEARCH.md`'s Q2 text is intentionally left unchanged; this note is the record of the divergence. `experiments/e2/` was regenerated; see the updated result comment below.

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

**Comment (ML Engineer, 2026-09-19):** implemented and run, `experiments/e2/`. 5-fold macro F1: ratios_lr (model 1, eye aperture + ear angle) 0.391 ± 0.051, coords_lr (model 2, 96 Procrustes-aligned coordinates) 0.419 ± 0.026, mlp (model 3, same input as model 2) 0.411 ± 0.018. Q2 verdict: NO MATCH — model (1) and model (3) differ by 0.019 macro F1, which is more than model (3)'s own 0.018 CV std, so the learned model does find signal past the two hand-built ratios. Full per-fold numbers, confusion matrices, and the muzzle-spread extra readout are in `experiments/e2/README.md`; per-model `metrics.json`/config/git sha are in `experiments/e2/<name>/` and `experiments/e2/config.json`. Not closing this issue — that's Research QA's call.

**Comment (ML Engineer, 2026-09-19, addendum re-run):** re-ran after wiring `muzzle_spread_ratio` into model (1) per the addendum above. Updated 5-fold macro F1: ratios_lr (model 1, now eye aperture + ear angle + muzzle spread, `n_features=3`) 0.382 ± 0.018, coords_lr (model 2, unchanged) 0.419 ± 0.026, mlp (model 3, unchanged) 0.411 ± 0.018. Q2 verdict is still NO MATCH — model (1) and model (3) now differ by 0.029 macro F1, versus model (3)'s own 0.018 CV std — the gap widened slightly rather than closing, so adding the third ratio did not change Q2's answer. `uv run pytest tests/` is green, 24/24. `experiments/e2/` (README, config, `ratios_lr/metrics.json`, `plots/macro_f1_comparison.png`, `plots/ratios_lr_confusion_matrix.png`) regenerated; `data/cache/splits.csv` untouched, splits are unaffected by this change. Not closing this issue — that's Research QA's call.

**Addendum (direct user instruction, 2026-09-20, extends the required controls above):** the user asked for two further changes to the three baseline models. First, report two new metrics alongside macro F1 — Cohen's kappa and MCC — plus a fourth reference run, a uniform-random `DummyClassifier` "chance level" classifier, reported separately from the three real models (not part of the Q2 verdict, not in the macro-F1 comparison plot). Second, replace the balancing mechanism: the existing `class_weight="balanced"` (both LogisticRegressions) and class-weighted `nn.CrossEntropyLoss` (MLP) reweight the loss but never touch the actual training data, which is heavily imbalanced (attentive 1147 / relaxed 752 / uncomfortable 107). Replaced with random oversampling of minority classes on the training folds only (`catface.ml.cv.oversample_to_balance`, via `sklearn.utils.resample`), applied inside `cross_validate` right before each fold's `.fit()`; test folds keep the original imbalanced distribution untouched, which the re-run confirms. No new dependency: `cohen_kappa_score`, `matthews_corrcoef`, `resample`, and `DummyClassifier` are all already available via the existing `scikit-learn` dependency.

**Comment (ML Engineer, 2026-09-20):** implemented and re-ran, `experiments/e2/`. New 5-fold numbers (mean ± std): ratios_lr macro F1 0.385 ± 0.030, kappa 0.154 ± 0.046, MCC 0.161 ± 0.046; coords_lr macro F1 0.426 ± 0.032, kappa 0.174 ± 0.048, MCC 0.182 ± 0.050; mlp macro F1 0.418 ± 0.023, kappa 0.155 ± 0.046, MCC 0.161 ± 0.047. Random-classifier reference (excluded from the Q2 verdict): macro F1 0.302 ± 0.000, kappa 0.017 ± 0.001, MCC 0.018 ± 0.001 — all three real models clear chance level by a wide margin on every metric. `data/cache/splits.csv` confirmed untouched (`git diff -- data/cache/splits.csv` empty). `uv run pytest tests/` — 37/38 passed; the one failure (`tests/test_graph.py::test_build_cat_edges_default_path_covers_all_48_nodes`) is pre-existing and out of scope: `src/catface/core/graph.py` was already modified, uncommitted, before this session started (RSCH-3/E3 territory, edge-count assertion on `models/graph_edges_manual.txt`), unrelated to this addendum's files. `tests/test_gnn.py` (explicitly named as the check that `cv.py`'s new `oversample` kwarg, defaulting to `False`, doesn't change RSCH-3's behavior) passed 7/7. Q2 verdict is unchanged by the balancing switch: still NO MATCH — model (1) and model (3) now differ by 0.033 macro F1 (was 0.029 under class weights), versus model (3)'s own 0.023 CV std, so oversampling widened the gap slightly rather than closing it, same conclusion as before. Not closing this issue — that's Research QA's call.

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

**Addendum (direct user instruction, 2026-09-20, overrides the Method line's mechanism above):** the Method line above ("Adjacency hand-written from anatomy in `core/graph.py`") is still true in spirit but the mechanism changed. `core/graph.py` no longer derives edges geometrically from a reference shape — it parses a hand-authored edge list at `models/graph_edges_manual.txt`, itself rewritten by a human reviewing rendered overlays of the earlier geometric version (`scripts/graph_review_manual.py`). Same "a human decided this because CatFLW publishes no index map" rationale, same output shape (76 unique edges over all 48 nodes), different mechanism. See `docs/DECISIONS.md` 2026-09-20 for the full rationale; this note is the record of the divergence, not a rewrite of the Method text above.

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

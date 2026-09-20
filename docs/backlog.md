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

**Owner:** ML Engineer · **Blocked by:** RSCH-2 · **Time box:** 1–2 days · **Status:** re-groomed and re-run 2026-09-20, Research QA PASS below, result recorded in `docs/MODEL_REPORT.md` under Q1. Ready for the orchestrator to close.

**Question:** Q1 — does a dense graph conv over the anatomical adjacency
beat the E2 MLP?

**Answer:** no, and the graph does worse than no graph. Decided on kappa, both conditions failing: the anatomical arm scores 0.096 against the E2 MLP's 0.155 and the random adjacency's 0.124, with its own CV std at 0.021. The `gnn_identity` control (message passing switched off at equal depth and parameter count) scores 0.231 and beats every other arm including the MLP, so the convolution costs accuracy rather than the adjacency lacking signal. The first run's contrary finding came from a mean-pool readout that discarded the input; see `docs/DECISIONS.md` 2026-09-20 and `docs/MODEL_REPORT.md` Q1 for the two riders (the v3 edge list is disconnected, and absolute numbers are an upper bound pending a grouped-split check).

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

**Re-grooming (Research PM, 2026-09-20). Supersedes the Method and Acceptance criteria above. The first run's artifacts stay on disk as the record of what was measured.**

The first E3 run completed and reported Q1 = NO. I am sending it back on methodology, per `research_process.md` step 5: the run did not measure what Q1 asks. Four defects, in descending order of how much they matter.

*Defect 1, the readout zeroes the signal.* `_Net.forward` ends in `h.mean(dim=1)`, a mean over the 48-node axis. Node features come from `generalized_procrustes`, and `_center_scale` subtracts each shape's centroid, so the node-axis mean of the GNN's input is exactly zero for all 1967 rows in all four channels. Measured between-sample variance surviving that readout: 2.58e-33, against 1.64e-2 for the flattened coordinates the MLP reads. The pooling step discards the input before the classifier head sees it, and only ReLU asymmetry leaks anything through. The GNN and the MLP differ in two ways at once, adjacency and readout, and the readout term dominates, so the run cannot attribute its result to the graph. Q1 needs the readout held constant and the adjacency varied alone.

*Defect 2, the conclusion changes once the readout is fixed.* Holding convs, adjacency, splits and epochs fixed and swapping only the readout: anatomical mean-pool 0.266 macro F1 / 0.018 kappa, anatomical flatten 0.374 / 0.096, random-adjacency flatten 0.398 / 0.124. The random adjacency beats the anatomical one. The reported "Condition B HOLDS" is an artifact of the broken readout, and the corrected result is the negative `docs/PLAN_RESEARCH.md` asks for by name: if random does as well, the structure carries nothing.

*Defect 3, the two arms were balanced differently.* `gnn.run` calls `cv.cross_validate` without `oversample=True` and applies `compute_class_weight("balanced")` inside `fit`; `mlp.run` passes `oversample=True` and no loss weights. The GNN kept the mechanism the 2026-09-20 instruction replaced for E2, so "identical conditions" in the run's control list is not true against the MLP. This one does not drive the result (mean-pool with oversampling scores 0.269, against 0.266 with class weights), but it has to go before the comparison means anything.

*Defect 4, the verdict used a metric that cannot see the difference.* Condition B compared two arms sitting at kappa 0.018 and 0.013, both at chance, and declared a winner on macro F1. Between two chance-level classifiers, macro F1 reflects the prediction marginals, not discrimination. E3 also never picked up the `DummyClassifier` reference E2 gained on 2026-09-20, which would have shown the anatomical GNN's 0.266 macro F1 sitting *below* a uniform dummy's 0.302.

**Revised method.** Same hand-written dense graph conv, same no-PyG constraint, same frozen splits in `data/cache/splits.csv`. Five arms, every one of them flatten readout and `oversample=True`:

| arm | adjacency | what it is for |
|---|---|---|
| `gnn_anatomical` | v3, 98 edges | the hypothesis |
| `gnn_random_ablation` | random, 98 edges, seed 1 | the ablation RSCH-3 already required |
| `gnn_identity` | `A_hat = I` | new control, see below |
| `gnn_anatomical_meanpool` | v3, 98 edges | keeps the original defect on the record |
| `random_baseline` | none | chance reference, as E2 |

`gnn_identity` is the control the first grooming missed. It runs the same depth and parameter count with message passing switched off. Without it, an anatomical arm that beat the MLP would not tell you whether the graph did the work or the extra depth did, and the ablation against a random adjacency cannot separate those either, because a random graph still passes messages. Name it now rather than discover it at QA.

**Adjacency provenance.** `core/graph.py` loads `models/graph_edge_schemes/graph_edges_manual_v3.txt`, 100 edge lines, 98 unique. `docs/DECISIONS.md` 2026-09-20 documents the v1 file, 78 lines and 76 unique, at `models/graph_edges_manual.txt`, a path that no longer exists. Nobody recorded v2 or v3. The edge list is the sole independent variable of this experiment and it was revised twice without an entry, so the ML Engineer states in the run comment whether any scheme was chosen by looking at E3 metrics. If one was, say so and I will re-scope again: selecting the independent variable against the outcome invalidates the comparison regardless of what the numbers say.

**Revised acceptance criteria:**
- [ ] Untrained-model ONNX export attempted and recorded before any training run, for the flatten head (the head shape changed, so the earlier export result does not carry over)
- [ ] All five arms trained on the frozen splits, macro F1 + kappa + MCC + confusion matrix, 5-fold
- [ ] Readout identical across the four trained GNN arms; balancing identical across all arms (`oversample=True`, no in-`fit` class weights)
- [ ] `gnn_identity` reported with the same detail as the other arms, not as a footnote
- [ ] Q1 verdict computed on kappa, with macro F1 reported beside it, as both conditions: beats the E2 MLP by more than the anatomical arm's own CV std, **and** beats the random adjacency by the same margin
- [ ] Adjacency file path and sha256 pinned in `experiments/e3/config.json`
- [ ] Run comment states whether any edge scheme was selected against E3 metrics
- [ ] Negative outcome reported as such
- [ ] No `torch_geometric` import anywhere in the run

**Stop condition:** unchanged, and none fires. A random-adjacency tie is the Q1 answer.

**Expected shape of the result,** recorded before the run so the ML Engineer cannot be accused of writing the conclusion afterwards: anatomical near 0.374 macro F1 / 0.096 kappa, random near 0.398 / 0.124, both under the MLP's 0.418 / 0.155. That reads Q1 = NO, on the ground that a hand-authored anatomical graph does no better than a random graph of equal density. If the run lands somewhere else, the run is right and this paragraph is wrong.

**Comment (ML Engineer, 2026-09-20, re-groomed re-run):** implemented the revised method and re-ran, `experiments/e3/`. All five arms on the frozen `data/cache/splits.csv`, 5-fold, flatten readout and `oversample=True` everywhere except the arm that exists to vary one of those. 5-fold means ± std, macro F1 / kappa / MCC: `gnn_anatomical` 0.374 ± 0.015 / 0.096 ± 0.021 / 0.102 ± 0.022; `gnn_random_ablation` 0.398 ± 0.024 / 0.124 ± 0.037 / 0.129 ± 0.039; `gnn_identity` 0.466 ± 0.012 / 0.231 ± 0.022 / 0.235 ± 0.021; `gnn_anatomical_meanpool` 0.269 ± 0.021 / 0.019 ± 0.018 / 0.023 ± 0.022; `random_baseline` 0.302 ± 0.000 / 0.017 ± 0.001 / 0.018 ± 0.001. Per-fold tables, confusion matrices and plots are in `experiments/e3/README.md` and `experiments/e3/<arm>/metrics.json`.

**Q1 verdict: NO,** a negative result, and both conditions fail rather than one. On kappa, with macro F1 beside it: condition A, the anatomical arm scores 0.096 against the E2 MLP's 0.155, diff -0.059 against its own std of 0.021, DOES NOT HOLD (macro F1 0.374 vs. 0.418). Condition B, the anatomical arm scores 0.096 against the random adjacency's 0.124, diff -0.028 against the same std, DOES NOT HOLD (macro F1 0.398 for random). This lands within noise of the expected shape recorded above, including the direction: a random graph of equal density does better than the hand-authored one.

**The identity control is the sharpest reading here and it was worth adding.** `A_hat = I`, message passing off, same three layers and same parameter count: 0.231 kappa / 0.466 macro F1, above both graph arms and above the E2 MLP (0.155 / 0.418). Turning the graph off raises the score, so no part of the anatomical arm's result can be credited to the adjacency. The convolution averages each node with its neighbours and smooths away landmark-position signal that the same layers keep when they act per-node. Without this arm the run would have shown two graph arms below the MLP and left open whether depth or adjacency was at fault; it was the adjacency. `gnn_anatomical_meanpool` reproduces the first run's defect on the record at 0.019 kappa / 0.269 macro F1, below the uniform dummy's 0.302 macro F1, which is the reading defect 4 predicted.

**Edge-scheme selection, asked for by name:** I cannot determine it from git history, and here is exactly why. `models/*` is gitignored with per-file exceptions, so `models/graph_edge_schemes/` was never tracked. One commit in the whole history touches any edge scheme, `cd0fbdd` "Add graph edges v1" (2026-09-20 12:58), for `models/graph_edges_manual.txt`; `git log --follow -- models/graph_edge_schemes/` returns nothing, and v2 and v3 have no commit, no message and no author record. What the on-disk record does show, and I report it as circumstantial rather than as an answer: mtimes put v1 at 11:47, the first E3 run's output directories at 13:06 and 13:07, v2 at 13:34 and v3 at 13:59, so both revisions were authored after E3 metrics for v1 existed on disk. Whether whoever wrote them read those metrics first, nothing in the repository records. For my own part: I ran against `core.graph.DEFAULT_EDGES_PATH` (v3, 98 unique edges over all 48 nodes) as I found it, changed no edge, and compared no scheme against another. The adjacency file path and its sha256 `86b29330044500b04f1a9bf38d814537aea8624da0e2fb0b18f6a75bf0a97fb4` are now pinned in `experiments/e3/config.json`, and I added a `.gitignore` exception so the three scheme files are tracked from here on and that sha256 can be checked against something.

Housekeeping in the same pass: `GNNClassifier.fit` no longer applies `compute_class_weight("balanced")` and `gnn.run` passes `oversample=True`, so balancing matches E2's MLP; `readout` is a constructor argument on `_Net`/`GNNClassifier` defaulting to `flatten`; the ONNX export check ran before any training against the flatten head (opset 17, input `[1,48,4]`, head now `Linear(1536, 3)`) and succeeded; `A_hat` stays a registered buffer and the conv stays hand-written dense matmuls, no `torch_geometric` anywhere, AST-walk tested. Stale `models/graph_edges_manual.txt` references fixed in `core/graph.py`, `scripts/gnn.py`, `scripts/graph_review_manual.py` and the E3 README. `tests/test_graph.py`'s edge-count assertion, which a previous pass deleted rather than updated, is back and asserts 98. `uv run pytest` is green, 43 passed. `git diff -- data/cache/splits.csv` is empty. Not closing this issue.

**QA: PASS (Research QA / Methodology Reviewer, 2026-09-20, re-groomed re-run).** The re-run measures what Q1 asks. I reproduced all five arms from a scratch script against `catface.ml.gnn` without writing into `experiments/e3/`: every mean, every standard deviation, every per-fold value and every confusion matrix matches the committed `metrics.json` exactly. The README matches `metrics.json`. I changed no code, no data and no run artifact.

Control verdicts, one line each:

- [x] Readout identical across the arms that carry the verdict, node identity preserved — PASS. `_Net.forward` branches on `self.readout`; `gnn_anatomical`, `gnn_random_ablation` and `gnn_identity` all run `flatten` (`config.json` `arms`, and `metrics.json` carries `"readout": "flatten"` in each). `h.flatten(1)` concatenates node-major, so node 7's hidden vector lands in its own slice of the 1536-wide head input and node identity survives the readout; `tests/test_gnn.py::test_mean_readout_discards_which_node_is_which_flatten_does_not` pins that as a permutation property. `gnn_anatomical_meanpool` is the one arm on `mean`, which is the arm's whole purpose per the re-grooming's table, so the criterion's literal "all four" reads against the re-grooming's own design; the three arms the verdict rests on are identical.
- [x] Balancing identical across all arms — PASS. `gnn.run` passes `oversample=True` to `cv.cross_validate` (`src/catface/ml/gnn.py:115`), same as `mlp.run`, `ratios_lr`, `coords_lr` and `random_baseline`. `grep -rn "compute_class_weight\|class_weight\|weight=" src scripts tests` returns no live hit: the only survivors are two prose strings in `scripts/gnn.py` describing the removal. Defect 3 is closed in code, not in the README.
- [x] `gnn_identity` run and reported at full detail — PASS. `A_hat = np.eye(48)` built in `scripts/gnn.py`, own run directory, own per-fold table, own confusion matrix, own plot, and its own paragraph in the verdict. 0.231 ± 0.022 kappa, reproduced exactly.
- [x] Random-adjacency ablation under identical conditions — PASS. Same `GraphConv`/`_Net` with a different `A_hat` buffer, 98 edges against the anatomical 98, drawn once at seed 1 and not per fold, same `flatten`, same 200 epochs, same `oversample=True`, same frozen splits. The only difference is the adjacency matrix.
- [x] Q1 verdict on kappa, both conditions, margin = the anatomical arm's own CV std — PASS. `q1_verdict` reads `mean_kappa`/`std_kappa` and requires `diff > std` on both conditions; macro F1 appears beside each, never as the decider. Condition A -0.059 against std 0.021, condition B -0.028 against the same std, both fail, verdict NO. The headline number never carries the call.
- [x] Uniform `DummyClassifier` chance reference — PASS. `catface.ml.random_baseline` reused from E2 unchanged, same folds through the same `cross_validate`, 0.302 macro F1 / 0.017 kappa. It does the work defect 4 asked of it: `gnn_anatomical_meanpool` at 0.269 macro F1 sits below it and is visibly not classifying.
- [x] Untrained-model ONNX export against the flatten head, before any training — PASS. `scripts/gnn.py:378` runs `onnx_export_check` on a freshly constructed `_Net` inside a `TemporaryDirectory` before the arms loop starts; default `readout="flatten"`, so it exercised the `Linear(1536, 3)` head, not the old `Linear(32, 3)`. `config.json` records the result with opset, input shape, exporter and the two deprecation warnings.
- [x] No `torch_geometric` anywhere — PASS. Repo-wide grep finds it only in the two AST-walk tests and in prose. `pyproject.toml` has no PyG dependency.
- [x] Adjacency path and sha256 pinned and correct — PASS. `sha256sum models/graph_edge_schemes/graph_edges_manual_v3.txt` returns `86b29330044500b04f1a9bf38d814537aea8624da0e2fb0b18f6a75bf0a97fb4`, byte-identical to `config.json`. The file is now tracked, so the pin has something to check against.
- [x] Splits untouched and identical to E2's — PASS. `git diff -- data/cache/splits.csv` is empty and `git status --porcelain` on it is clean; its last commit is `23e659b`, E2's own implementation commit, so E2 and E3 read the same 1967 rows and the same five folds. sha256 `488c674e6cde9a239b481a5f52091b2372cf1196d2106b6fee86851ecbecee47`. The MLP comparison is against like.
- [x] Edge-scheme provenance — PASS with the reservation recorded below, and I tested it rather than reasoning about it.

**On the edge scheme, the one item that could have sunk this.** The ML Engineer's account is accurate: v2 (13:34) and v3 (13:59) both post-date the first run's output at 13:06-13:07, `git log` covers only v1, and nothing in the repo says whether whoever wrote v2 and v3 had read the v1 metrics. The direction-of-bias argument holds as far as it goes — tuning an edge list against E3 metrics would push the anatomical arm up, and the anatomical arm loses to both the random and the identity arms, so the bias runs opposite to the conclusion. I did not want to rest a report section on an argument, so I ran the other two schemes through the same arm, same splits, same everything: v1 (76 edges) 0.383 macro F1 / 0.111 kappa, v2 (106 edges) 0.361 / 0.075, against v3's 0.374 / 0.096. Every scheme in the repository loses to the random adjacency (0.124), to the identity control (0.231) and to the E2 MLP (0.155). v3 is not even the best of the three, which is what a scheme selected for E3 performance would have been. Q1 = NO under all three, so the selection question cannot reach the answer, and the negative result is safe to accept.

Two things the PM should record while the scheme is in view. v3 deleted the `bridge_left_eye_to_ear_base`, `bridge_right_eye_to_ear_base` and eye-to-nose groups that v2 added, and those bridges were the only edges tying the eyes to the rest of the face: the adjacency actually run splits into four connected components (left eye, right eye, both ears via `top_head`, and the nose/mouth/muzzle/whisker mass), where v1 and v2 were connected graphs. RSCH-3's Method text names "eyes→ear bases" as an anatomical edge family, so the arm labelled anatomical no longer matches the method's own description. It does not change the verdict, since connected v1 loses too, but "a hand-authored anatomical graph carries nothing" should be written knowing the graph tested was disconnected.

**Split check, per my role doc.** The E0 near-duplicate check ran across the two Roboflow sets and found 0 pairs at dHash Hamming ≤ 5, and `splits.py` filters to `dataset == "cat-emotions-3"` alone, so no cross-set pair can straddle a fold here. Within-dataset duplicates were never hashed, as `docs/MODEL_REPORT.md` already states, and the folds stratify on class alone, so photos of one cat can sit on both sides of a split. All five arms and E2's MLP read the same `splits.csv`, so any such leakage inflates every arm together and the Q1 comparison survives it; what it threatens is the absolute level of every number, which belongs in the report as an upper bound. One caveat on the identity control specifically: if the leakage is real, the arm best able to memorize an individual cat benefits most, and that is the per-node arm rather than the smoothed graph arms. The gap is large (0.231 against 0.096) and the direction of the Q1 answer does not depend on it, but "the graph convolution costs signal" is the claim that would most repay a grouped-split re-run.

**Not blocking, for the PM's attention.** (1) `pyproject.toml` gained `onnx`, `onnxruntime` and `onnxsim` in `422cc39` with no mention in the run comment; only `onnx` is used, by `pytest.importorskip` in `tests/test_gnn.py`, and nothing imports the other two. `AGENTS.md` says dependencies are not added without asking. (2) The first E3 run's `metrics.json` files were never committed — `git log --diff-filter=A` puts their first appearance at `2e681a7`, the re-run — so the re-grooming's "the first run's artifacts stay on disk as the record" did not hold; `gnn_anatomical_meanpool` reproduces the defect measurement, which covers the substance. (3) `GNNClassifier.__init__` calls `torch.manual_seed(0)` on every construction, so all five folds start from one initialisation, where `mlp.run` seeds once per run; this is uniform across the GNN arms and so cannot tilt the comparison, and it compresses fold-to-fold spread, which makes the std threshold easier to clear rather than harder. (4) the uncommitted working copy of `docs/AI_WORKFLOW.md` already carries a line stating that Research QA's verdict is recorded against this entry, written while this review was still running; it happens to name the same two open items I found, but a QA verdict should not be drafted ahead of the review. I did not run `uv run pytest` myself; the ML Engineer reports 43 passed and the session that invoked me confirmed it.

Result: negative, obtained by correct method. The anatomical graph conv does not beat the E2 MLP and does not beat a random graph of equal density; switching message passing off beats both. That is the Q1 answer, and the Research PM can write it into `docs/MODEL_REPORT.md` as it stands, with the disconnected-v3 note and the within-dataset leakage caveat carried alongside it. Not closing this issue.

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

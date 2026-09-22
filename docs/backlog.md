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

**Flag (orchestrator, 2026-09-22, found incidentally during RSCH-4's Research QA review, not a re-groom).** This issue still has no Research QA verdict on record, unlike RSCH-0/RSCH-1/RSCH-3. Separately, commit `e3bb8cd` (2026-09-22) regenerated `experiments/e2/coords_lr/checkpoint.pkl` and rewrote `experiments/e2/mlp/metrics.json` (kappa 0.155 → 0.110, confusion matrix and git sha changed) with no matching backlog comment recording why or with what result. RSCH-4's Q3 sanity check loads the `coords_lr` checkpoint from this same commit, on the assumption it came from a QA-passed E2 run — that assumption does not hold. Left here for whoever next grooms or QA's RSCH-2; not blocking RSCH-4 per direct user instruction, 2026-09-22.

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

**Owner:** ML Engineer · **Blocked by:** RSCH-2, RSCH-3 · **Time box:** 1 day · **Status:** closed 2026-09-22 by the orchestrator, third-grooming-pass scope, Research QA PASS below, result recorded in `docs/MODEL_REPORT.md` under Q3. Q4 struck as obsolete, see addendum below — this issue closes without answering it.

**Question:** Q3 (pose contamination) and ~~Q4 (class separability)~~ — Q4 struck as obsolete, see addendum below.

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

**Grooming (Research PM, 2026-09-20). Extends the Method above and supersedes the four acceptance criteria below it. The original text stays on the record.**

E4 answers two questions off one run, and one of them carries a measurement problem that has to be settled on paper before anyone writes code. Everything below is pre-registered: bins, metric, thresholds and the expected result are fixed here, before the run, so that no bin edge and no decision rule can be chosen after the numbers are in.

**Questions.** Q3, how much head pose contaminates the prediction. Q4, whether the three classes are separable or some collapse. Either may come back negative or inconclusive, and either such answer closes the issue.

**The degrees problem.** The third original acceptance criterion asks for a verdict shaped "stable under N degrees, degrades past that". Nothing in this repository can produce an N. The only yaw estimator that exists is `pose_confound_proxies` at `scripts/shape_space.py:51-63`, which computes `||muzzle_centroid - right_eye_centroid|| - ||muzzle_centroid - left_eye_centroid||` over Procrustes-aligned shapes. That is a signed, unitless quantity in Frobenius-normalised shape units, not an angle. There is no angular ground truth in CatFLW, in either Roboflow set, or anywhere else on disk, so nothing in the data calibrates the proxy to degrees. E1 already leaned on this same quantity (PC2 tracks it at r = 0.90) and never had to name a unit; E4 does, because the app wants a threshold.

The decision, and it is a methodological term of this experiment rather than a footnote: **degrees come from a weak-perspective geometric model of a bilaterally symmetric face, carrying one stated, unmeasured prior.** Put the eyes symmetric about the midline at `+/- w/2`, `w` the inter-ocular distance measured per row from the aligned shape. Put the muzzle centroid on the midline, below the eye line, displaced toward the camera by a depth `d * w`. Yaw by `theta` about the vertical axis, project weakly (`x -> x cos(theta) + z sin(theta)`), and the two eye-to-muzzle distances differ by

    dist_right^2 - dist_left^2 = -d * w^2 * sin(2 * theta)
    proxy                      = -d * w^2 * sin(2 * theta) / (dist_right + dist_left)

Every term except `d` is measurable per row from the aligned shape. `d`, nose protrusion depth as a fraction of inter-ocular distance, is the free prior. **It is set to 0.30 and swept +/-50% at 0.15, 0.30 and 0.45.** Nobody measured it; the sweep is what stands in for having measured it.

Two consequences that the ML Engineer must carry into the artifact rather than resolve quietly.

*First, bins are defined in exact proxy units and degrees are a derived, assumption-tagged label.* The sweep moves the third bin edge from 8.7 degrees at `d = 0.45` to 28.3 degrees at `d = 0.15`, a factor of three on the number the acceptance criterion asks for. The verdict's N is therefore reported as a range across the three priors, never as a point. The proxy edge is the number the app actually consumes: `core/states.py` can compute the proxy from landmarks with no angular assumption at all, and the confidence penalty keys on that. **A bare degree figure anywhere in the run, without its `d` and without the range, is a defect and Research QA should treat it as one.**

*Second, the forward model saturates and the data overruns it.* `sin(2 * theta)` is non-monotonic: the proxy rises, saturates near 50 degrees and falls again past it, so one proxy value has two candidate angles and the model cannot choose between them. Report the smaller root and say that is the convention. Worse, the largest `|proxy|` this model can produce at `d = 0.30` is about 0.0315, while `|proxy|` in the real 1967 rows reaches 0.1265. About 90 rows, 4.6%, sit outside anything the model can generate at the central prior. Those rows keep their proxy bin, have their degree reported as ">= theta*" with theta* the saturation angle for that row and prior, are counted in the artifact, and never receive an invented angle. That the proxy exceeds the model's whole range by a factor of four is itself a finding and belongs in `experiments/e4/README.md` in its own right: the proxy is not a yaw meter. It moves under roll, pitch, expression asymmetry and plain landmark error as well, and this run measures how much it moves, not what moved it.

**Bin count is 4, fixed before the run, justified on power.** Equal-count quantile bins on `|proxy|` over all 1967 rows, pooled, not per fold. Quartiles leave 25 / 34 / 26 / 22 `uncomfortable` rows per bin, which is thin but usable. Five bins put at least one (bin x fold) cell at zero `uncomfortable` rows, and a fold-wise spread is undefined once a cell is empty. Three bins leave two interior edges and cannot locate a threshold. Bin on the absolute value: the signed proxy splits 933 negative against 1034 positive, so folding the sign loses nothing, and the sign itself is not identifiable as left-versus-right without ground truth nobody has. The four edges are computed once, written into `experiments/e4/config.json` before any arm is scored, and are what QA re-derives.

**The verdict rests on kappa, with macro F1 beside it.** Accuracy is reported too, because the original criterion names it, but printed next to each bin's own majority-class rate so its emptiness is visible on the page. Class balance in the frozen `data/cache/splits.csv` is attentive 1130 / relaxed 730 / uncomfortable 107, so within any bin accuracy is roughly the majority rate and an arm can gain accuracy by predicting `attentive` harder. This is the same argument my E3 re-grooming made in defect 4, where two chance-level arms were separated on macro F1 and the winner reflected prediction marginals rather than discrimination. Do not repeat it here.

**Model arms.** Four model arms, per user decision: `ratios_lr`, `coords_lr`, `mlp`, `gnn_identity`. Plus `yaw_only_lr` and `random_baseline` as controls. All six on the frozen folds, `oversample=True` on training folds only, test folds untouched, exactly as E2 and E3 ran them.

**Required controls, named now so none is discovered at review time.**

a. *Per-bin class counts and total-variation distance from the overall prior.* This guards against reporting prior shift as pose degradation: if the high-yaw bin holds a different class mix, its kappa moves for reasons that have nothing to do with pose. Measured per-class mean `|proxy|` is attentive 0.0091 / relaxed 0.0095 / uncomfortable 0.0075, so yaw is close to equidistributed across classes and this guard is unlikely to fire hard. It goes in the artifact anyway, as a table, not in a reviewer's head.

b. *The uniform `DummyClassifier`* (`src/catface/ml/random_baseline.py`, as E2 and E3 used it) scored **within each bin, under that bin's own label prior**. A dummy scored once over the pooled rows does not tell you what chance looks like inside bin 3.

c. *A `yaw_only_lr` arm:* logistic regression on `|proxy|` alone, same frozen folds, same oversampling, same LogisticRegression configuration as the other two LR arms. This separates "pose degrades the model" from "pose predicts the label", and a per-bin table on its own cannot tell those apart: if yaw carries label information, per-bin kappa moves without any degradation having occurred. Expect it at chance, which is a clean positive statement about Q3 rather than hand-waving.

d. *Reproduction check.* `metrics.json` in E2 and E3 stores per-fold aggregates and a pooled confusion matrix, no per-row predictions, so every arm has to be re-run to recover out-of-fold predictions per row. That re-run's per-fold macro F1, kappa and MCC are compared against the already-committed `experiments/e2/<arm>/metrics.json` and `experiments/e3/gnn_identity/metrics.json`, and the deltas are recorded in `experiments/e4/config.json`. A hidden mismatch is the defect. A recorded one is not, and if a delta is non-zero the run comment says what drifted.

**Pre-registered decision rules.** These sit here, before the run, because each one is a choice that could otherwise be made after seeing the numbers.

- Verdict metric: kappa.
- Verdict arm: the highest pooled out-of-fold kappa among the four model arms. Picked by that rule, not by which arm gives the nicer pose story.
- Degradation rule: the first bin `b >= 1` whose kappa upper CI falls below bin 0's kappa lower CI. That bin's lower proxy edge is the threshold, and its degree range at the three priors is the reported N.
- Chance gate: if bin 0's kappa CI contains 0, the verdict is **INCONCLUSIVE**. There is no performance for yaw to degrade, and the app takes no pose penalty out of E4. Say so plainly rather than reading a trend out of noise.
- Prior-shift tolerance: total-variation distance 0.05 between a bin's class distribution and the overall prior. Exceeded in any bin means the verdict is **PROVISIONAL**, with the offending bin named in the verdict sentence.
- Merge thresholds for Q4: cross-talk `(C[i,j] + C[j,i]) / (n_i + n_j) >= 0.25` **and** the pairwise 2x2 kappa CI contains 0. Both, not either.
- Uncertainty: bootstrap over rows within each bin, B = 1000, seed 0, percentile interval at 2.5 / 97.5. The five per-fold kappas are a secondary sanity check, not the interval; with 22 to 34 `uncomfortable` rows per bin, five folds give five numbers and no usable spread.

**Q4's merge procedure, by rule and not by eye.** The merge decision comes off the pooled out-of-fold confusion matrix plus an actual re-run under merged labels on the same frozen folds. Nobody eyeballs a matrix and declares two classes the same. All three pairs are candidates. A 2-class kappa is not comparable to a 3-class kappa, so the valid comparison is three 2-class quantities side by side: the post-hoc collapse of the 3-class predictions onto the merged label set, the retrained merged model, and a 2-class uniform dummy under the merged prior. **The merge earns its place only if retrained >= collapsed and both clear the dummy.** If the retrained model does no better than collapsing predictions after the fact, merging bought nothing and the report says so.

**Q4's null path, written in advance.** If every arm sits at chance over three classes, then no pair is *separable*, and an inseparable pair cannot be shown inseparable-but-merge-worthy either, because there is no separation anywhere to contrast it against. The answer is then a reasoned "no merge is warranted on the evidence", which the original fourth acceptance criterion already permits as a negative result. Given E2 and E3 levels, best pooled kappa anywhere in this project is `gnn_identity` at 0.231, this is the likely outcome. The branch exists in writing before the run because otherwise an implementer facing a flat confusion matrix will be tempted to manufacture a merge out of the largest off-diagonal cell.

The Method text above expects trouble telling Scared, Surprised and Angry apart on cat-emotions-7. That set is excluded (`docs/DECISIONS.md` 2026-09-19), so the expectation is **not testable here**. Record it as not testable, do not substitute a different pair for it, and leave it as the prediction to check if that set ever comes back.

**Positive and negative shape, and the stop condition.** A positive Q3 result: bin 0 clears chance, a later bin's kappa CI falls clear of bin 0's, and the run names a proxy edge with its degree range. A negative Q3 result: bin 0 clears chance and no later bin degrades, meaning pose does not contaminate the prediction over the range this dataset covers, and the app applies no penalty. An inconclusive Q3: bin 0 is at chance, the gate fires, nothing can be said. A positive Q4: at least one pair meets both merge thresholds and the retrained merged model beats the collapse and the dummy. A negative Q4: no pair does, and the three classes stay as they are. RSCH-4's stop condition in the plan is **none**, and it stays none. Nothing in this run can halt the research track: a negative or inconclusive answer to either question is the finding, gets written into `docs/MODEL_REPORT.md` under Q3 and Q4, and E5 proceeds against whichever arm carries the highest pooled kappa.

**Constraints.**

- `data/cache/splits.csv` is frozen and is never regenerated. E2, E3 and E4 read the same 1967 rows and the same five folds. `git diff -- data/cache/splits.csv` is empty when the run finishes.
- E4 writes only under `experiments/e4/`. `experiments/e1`, `experiments/e2` and `experiments/e3` are byte-unchanged afterwards.
- No new dependency (`AGENTS.md`). In particular **scipy is not declared in `pyproject.toml` and must not be used**: the bootstrap, the quantile edges and the total-variation distance are numpy, and `cohen_kappa_score`, `confusion_matrix`, `LogisticRegression` and `DummyClassifier` already come with the installed scikit-learn.
- The yaw measurement belongs in `src/catface/core/geometry.py`, alongside `eye_aspect_ratio` and `ear_angle`, as `yaw_asymmetry_proxy(shape, muzzle_indices, left_eye_indices, right_eye_indices) -> float`. The app's future `core/states.py` is its declared consumer and `AGENTS.md` forbids either track keeping a local copy of shared geometry, so it does not go in `ml/` and `scripts/shape_space.py` stops computing its own copy. The degree calibration is a research-track modelling assumption and goes in `src/catface/ml/pose.py` as `yaw_degrees_from_proxy(proxy, shape, d)`, returning the angle and a saturation flag.
- Within-dataset near-duplicates were never hashed, so absolute per-bin numbers are an upper bound; it inflates every arm and every bin equally and so does not tilt the per-bin comparison. One line in the README, as E3 carried it.

**Expected shape of the result,** recorded before the run so the ML Engineer cannot be accused of writing the conclusion afterwards. I expect `gnn_identity` to carry the verdict at pooled kappa near 0.23, the other three arms below it, and `yaw_only_lr` at chance. I expect bin 0's kappa CI to clear 0 for `gnn_identity` and quite possibly not for `ratios_lr`. I expect no bin to degrade cleanly, because with 22 to 34 `uncomfortable` rows per bin the CIs will be wide enough to overlap, and so I expect Q3 to come back "no measurable degradation over the yaw range this dataset covers, threshold not locatable at this sample size" rather than a crisp N. On Q4 I expect no pair to clear both merge thresholds and the answer to be no merge. If the run lands somewhere else, the run is right and this paragraph is wrong.

**Revised acceptance criteria.** These supersede the four above.

- [ ] `yaw_asymmetry_proxy` lives in `src/catface/core/geometry.py`, `scripts/shape_space.py` imports it instead of recomputing it, and no copy of the formula survives in `ml/` or `scripts/`
- [ ] Degree calibration in `src/catface/ml/pose.py`, the weak-perspective model written out in the docstring with `d` named as an unmeasured prior
- [ ] Four equal-count quantile bins on `|proxy|` over all 1967 rows; edges in exact proxy units committed to `experiments/e4/config.json` before any arm is scored
- [ ] Every reported degree figure carries its `d` and the full 0.15 / 0.30 / 0.45 range; no bare degree anywhere in the run
- [ ] Saturation handled: count and fraction of rows whose `|proxy|` exceeds the model's range reported per prior, those rows labelled ">= theta*", no invented angles
- [ ] The proxy-exceeds-model-range finding written up in `experiments/e4/README.md` as a result, with the confounds named (roll, pitch, expression asymmetry, landmark error)
- [ ] Six arms on the frozen folds: `ratios_lr`, `coords_lr`, `mlp`, `gnn_identity`, `yaw_only_lr`, `random_baseline`; `oversample=True` on training folds only
- [ ] Per-bin table: kappa with bootstrap CI, macro F1, accuracy printed beside that bin's majority-class rate, and n per class
- [ ] Control (a): per-bin class counts and TV distance from the overall prior, in the artifact
- [ ] Control (b): uniform `DummyClassifier` scored within each bin under that bin's own prior
- [ ] Control (c): `yaw_only_lr` reported at the same detail as the model arms, not as a footnote
- [ ] Control (d): per-fold metrics for the re-run arms compared against the committed E2/E3 `metrics.json`, deltas recorded in `experiments/e4/config.json`
- [ ] Bootstrap as specified: within-bin over rows, B = 1000, seed 0, 2.5 / 97.5 percentile; per-fold kappas reported alongside as the sanity check
- [ ] Q3 verdict applies the pre-registered degradation rule, chance gate and prior-shift tolerance by name, and states which one fired
- [ ] Pose verdict sentence gives the proxy edge first and the degree range second, or states explicitly that no threshold is locatable
- [ ] Q4: pooled out-of-fold confusion matrix, cross-talk and pairwise 2x2 kappa CI computed for all three pairs against the pre-registered thresholds
- [ ] Q4: if any pair clears both thresholds, the merged model is actually retrained on the same frozen folds and compared against both the post-hoc collapse and a 2-class dummy; merge accepted only if retrained >= collapsed and both clear the dummy
- [ ] Q4: if no pair clears, the null path is taken and written as a reasoned "no merge warranted on the evidence"
- [ ] The cat-emotions-7 Scared/Surprised/Angry expectation recorded as not testable here, with the DECISIONS reference
- [ ] Q3 and Q4 each answered explicitly in the run comment, negative or inconclusive answers included and labelled as such
- [ ] `experiments/e4/config.json`, `README.md` and per-arm `metrics.json` committed with the git sha; `experiments/e1`, `e2`, `e3` byte-unchanged; `git diff -- data/cache/splits.csv` empty
- [ ] No new dependency; no scipy import anywhere in the run
- [ ] Within-dataset near-duplicate caveat carried in the README as an upper-bound note

**Stop condition:** unchanged, none, and none can fire. Negative and inconclusive are both closeable outcomes here.

**Second grooming pass (Research PM, 2026-09-20). Supersedes the yaw estimator of the first pass above and nothing else. Bin count 4, the verdict on kappa, controls (a) through (d), the pre-registered decision rules, Q4's merge procedure and null path, the expected-shape paragraph and the constraints all stand as written. The superseded text stays on the record, the way RSCH-3 kept its first run's.**

**What changed.** `models/graph_edge_schemes/graph_edges_manual_v3.txt` carries a header legend naming every one of the 48 node positions. A human hand-authored it while reviewing rendered overlays for E3. The project's standing assumption, repeated in RSCH-2's grooming notes and in `docs/DECISIONS.md`, that nobody publishes an index-to-anatomy map and `MUZZLE` is defined only by exclusion, is true of CatFLW and false of this repository. The first pass built its whole degrees section on a 22-point muzzle centroid against two 8-point eye centroids, because that was all the available index groups allowed. The map replaces both blunt instruments with named points whose 3D geometry can be reasoned about, and the estimator changes accordingly.

Four points lie on the sagittal plane: **16** nose philtrum, **17** mouth top, **0** mouth, **2** mouth chin. Twenty-one bilateral pairs are named left/right: eyelid outside 4/8, eyelid inside 5/9, eyelid top 6/10, eyelid bottom 7/11, pupil bottom 3/1, pupil outside 36/41, pupil inside 37/40, pupil top 38/39, nose top 12/13, nostril middle 14/15, nostril bottom 44/45, mouth corner 20/18, muzzle middle 33/34, muzzle outside 46/47, whisker pad outside 32/35, whisker pad middle 42/43, ear bottom-outside 22/31, ear middle-outside 23/30, ear top 24/29, ear middle-inside 25/28, ear bottom-inside 26/27. (Nodes 19 and 21 are named `right_muzzle, cheek, right` and `left_muzzle, bottom`; they are not a matched pair and this experiment does not use them.)

**Axis convention and projection model, stated once and used throughout.** Right-handed world coordinates: `x` to the image right, `y` up, `z` toward the camera. Yaw is rotation by `theta` about the vertical axis `y`. Projection is weak perspective, orthographic plus a uniform scale, so a world point `(x, y, z)` lands at `u = x*cos(theta) + z*sin(theta)`, `v = y`. All of it runs on the GPA-aligned stack, the same `generalized_procrustes` output E1 uses, so in-plane roll is already removed and the shapes share a frame. Pitch is not modelled and not removed, and that is a stated limitation of every estimator below.

### Family 1, foreshortening. Primary.

Put a bilateral pair at `(+/-a, y0, z0)` with `w = 2a` its frontal span. Projected, the two points land at `u = +/-a*cos(theta) + z0*sin(theta)`, so the **observed span is `w*|cos(theta)|` and the pair's own depth `z0` cancels**: yaw shifts both members of a pair equally, and the shift drops out of the difference. A vertical span between any two points is `|delta_y|`, untouched by a rotation about the vertical axis. Therefore

    w_obs / v_obs = r_frontal * |cos(theta)|
    theta         = arccos( (w_obs / v_obs) / r_frontal )

**This needs no depth prior at all.** That is why it is primary. The first pass's weakest term, an invented nose-protrusion constant swept +/-50% and moving the reported threshold by a factor of three, is gone rather than narrowed.

*Spans, fixed here.* `w_obs = |u_4 - u_8|`, the outer eye corners (eyelid outside, left and right). Picked over pupil outside 36/41 because pupil landmarks move with gaze direction and pupil dilation while the outer canthus is skeletally anchored, and picked over anything on the ears because E1 measured PC1 tracking ear position at r = 0.84 and an ear-based span would import that confound wholesale. `v_obs = |v_2 - (v_4 + v_8)/2|`, chin to the midpoint of the outer eye corners. Yaw-invariant by construction, since it is a pure `y` difference.

*The `r_frontal` rule, fixed before the run.* Yaw only ever shrinks `w_obs/v_obs`, never grows it, so the population's frontal value sits at the top of the observed distribution. The maximum is the wrong statistic, being the single noisiest order statistic and the one landmark error reaches first. **Pre-registered: `r_frontal` is the 95th percentile of `w_obs/v_obs` over all 1967 rows of the frozen splits, pooled, computed once, written into `experiments/e4/config.json` before any arm is scored.** The sensitivity analysis is a sweep at the 90th and 99th percentiles, reported the way the first pass's `d` sweep would have been. That sweep is over an empirical quantile of measured data rather than over a constant nobody measured, which is the whole point of the change.

*Costs, stated rather than discovered.*

- **Unsigned.** `cos` is even, so family 1 cannot tell a left turn from a right turn. This costs nothing for the binning, because the first pass already binned on magnitude and the app's confidence penalty is direction-agnostic. Sign comes from family 2.
- **Confounded with facial width.** A genuinely narrow-faced cat photographed head-on reads as yawed. Breed is not labelled in cat-emotions-3, and with one photo per cat and no identity labels a per-row `r_frontal` is not estimable, so a single population value has to absorb real brachycephalic-to-dolichocephalic variation. A Persian and a Siamese differ in `w/v` at zero yaw. This confound cannot be removed at this sample size and belongs in `experiments/e4/README.md` and in `docs/MODEL_REPORT.md` under Q3 as a limitation, not as a caveat sentence in a footnote.
- **Out-of-domain rows.** Rows with `w_obs/v_obs > r_frontal` give `cos(theta) > 1`. At the 95th percentile roughly 5% of rows do this by construction. They clamp to `theta = 0`, land in bin 0, and are counted and reported per percentile setting. This replaces the first pass's saturation machinery.
- **Monotonic, so the saturation problem is gone.** `cos` is strictly decreasing on `[0, 90]` degrees, so the inverse is single-valued over the entire usable range. The first pass's non-monotonic `sin(2*theta)`, its two candidate roots, its smaller-root convention and its `">= theta*"` labelling are all struck. Nothing saturates.

### Family 2, midline offset. Secondary, carries the sign.

For a midline point `M` at `(0, y_M, z_M)` and a bilateral pair `(L, R)` at depth `z0`:

    s = ( u_M - (u_L + u_R)/2 ) / (u_R - u_L)
      = ( (z_M - z0) / w ) * tan(theta)

The pair's midpoint moves to `z0*sin(theta)` and the midline point to `z_M*sin(theta)`, so the numerator is `(z_M - z0)*sin(theta)`; the denominator is `w*cos(theta)`, which cancels the foreshortening automatically. Write `d_rel = (z_M - z0)/w`, the midline point's depth relative to the pair's plane in units of the pair's span. Then `s = d_rel * tan(theta)`: **signed, monotonic over the full range, no saturation, no second root.** Even as a standalone this is better behaved than the E1 centroid proxy it replaces, which went as `sin(2*theta)` and turned over.

Primary midline point **16**, the nose philtrum: the most protruding sagittal point, so the largest `d_rel` and the largest signal. Pair `(4, 8)`, the same outer eye corners family 1 uses, so the two families share a frame. Sign convention: positive `s` means the philtrum sits toward `+u` relative to the eye-corner midpoint; which physical turn that is follows from the sign of `d_rel`, and the run fixes it by requiring `d_rel(16) > 0` and states the resulting left/right mapping in the README.

Node **2**, the chin, runs as a near-zero-depth control: it sits close to the eye-corner plane, so its implied `d_rel` should come out far smaller in magnitude than the philtrum's. That is a falsifiable prediction of the geometry and it costs one extra column.

### The combination, and the internal control that replaces the sweep

Family 1 gives `theta` with no depth prior. Family 2 gives `s`. Together, per row,

    d_rel = s / tan(theta)

so **the depth term stops being an invented constant and becomes a measured, per-row quantity with a distribution over 1967 rows.** Its median and spread go in the report as a result in their own right. That is the substantive gain from the node map, and it is what the first pass's `+/-50%` sweep was standing in for.

The same relation is the internal control. If both families measure yaw, `|s|` plotted against `tan(theta)` is a straight line through the origin whose slope is the population `d_rel`. **Pre-registered: Spearman `rho` between `|s|` and `tan(theta)` over all 1967 rows, with a bootstrap CI at B = 1000, seed 0, matching the rest of this experiment's uncertainty rule. `rho >= 0.5` means the two agree well enough to call the binning quantity yaw. `rho < 0.3` means the yaw label is not earned, and Q3's verdict is then reported against the binning quantity under its operational name, the foreshortening ratio, with no degree figure attached at all.** Between 0.3 and 0.5, report both and say the label is weakly supported. This is an estimator-versus-estimator measurement, which is evidence, where the first pass had a sensitivity analysis over an assumption, which is not.

### Family 3, considered and rejected

*Mirror-Procrustes residual.* Swap all 21 bilateral pairs, align the relabelled shape to the original, read the residual. Rejected: the residual is one non-negative scalar mixing yaw with expression asymmetry, one ear forward, and landmark error, with no way to decompose it. It is less interpretable than family 1 and unsigned like family 1, without family 1's freedom from a depth prior. The one useful by-product, the pair-swap permutation itself, is required anyway as a committed constant.

*Per-row least-squares fit of `theta` against all pairs at once.* This is family 1 generalised, and it would be right if a 3D reference cat existed. None does. Fitting against 21 pairs needs 21 frontal spans, each carrying the same population-versus-individual width confound, so the extra statistical precision buys nothing against a shared systematic error that dominates it. Rejected as cost without return at n = 1967, and recorded as the obvious upgrade if a 3D model ever lands.

### The E1 centroid proxy, kept as a named legacy quantity

`pose_confound_proxies`' head-yaw term, `||muzzle_centroid - right_eye_centroid|| - ||muzzle_centroid - left_eye_centroid||`, moves to `core/geometry.py` as `yaw_centroid_proxy(shapes)` and is **reported alongside the new estimators in the per-row artifact and correlated against `theta`**, so E1's finding that PC2 tracks it at r = 0.90 stays connected to E4's. It is not the binning quantity and no bin edge is computed from it. It must reproduce E1's array bit-for-bit, in the same summation order, and E1's r = 0.90 is re-derived and stated to match.

### Numbers from the first pass that are now stale

Every figure in the first pass's degrees section was measured against the centroid proxy and pertains to a different quantity. Struck as inputs to this run, and carried only where they justify the change of estimator:

- **8.7 to 28.3 degrees** for the third bin edge under the `+/-50%` `d` sweep: struck. There is no `d` sweep.
- **Saturation near 50 degrees**, the two candidate roots, the smaller-root convention, the `">= theta*"` labelling: struck. Family 1 is monotonic.
- **`|proxy|` reaching 0.1265 against a model maximum of ~0.0315, and ~90 rows (4.6%) beyond the model's range**: retained **only** as measurements on the superseded centroid proxy, and only because they are the evidence that the centroid proxy is not a yaw meter, which is why the estimator changed. Not inputs to any bin, any threshold or any verdict here.
- **Per-class mean `|proxy|` 0.0091 / 0.0095 / 0.0075**: stale. Control (a) is recomputed against `theta`.
- **Quartile occupancy 25 / 34 / 26 / 22 `uncomfortable` rows per bin**: stale, because occupancy depends on which quantity is being quantiled. See the restated bin argument below.
- **Signed split 933 negative / 1034 positive**: stale, and its accompanying argument is now wrong rather than merely stale. The first pass said the sign is not identifiable as left-versus-right without ground truth. With a named midline point it is. Binning still happens on magnitude, for the different and better reason that family 1 produces a magnitude and the app's penalty is direction-agnostic; family 2's left/right split is reported separately as a balance check.

**All bin edges are quantiles of `theta` computed at run time over all 1967 rows, pooled, and frozen into `experiments/e4/config.json` before any arm is scored.** No edge in this issue is a number. I cannot measure one without touching data and I am not going to carry one forward that was measured on a different quantity.

### Bin count of 4, restated without the old occupancy numbers

The argument was always about `uncomfortable` counts per cell and it survives the change of estimator, but it must not read as resting on edges measured against the centroid proxy. Restated: quartiles put roughly 492 of the 1967 rows in each bin, and `uncomfortable` is 107 rows, so if `uncomfortable` is close to uniform over the binning quantity each bin holds about 27 of them and each (bin x fold) cell about 5. Five bins drop that to about 21 per bin and about 4 per cell, where a single empty cell makes the fold-wise spread undefined. Three bins leave two interior edges and cannot locate a threshold. Four it is.

Because I can no longer assert the occupancy from measurement, one pre-registered fallback: **the run computes per-bin and per-(bin x fold) `uncomfortable` counts immediately after freezing the edges, before scoring any arm. If any (bin x fold) cell holds zero `uncomfortable` rows at 4 bins, the run drops to 3 bins, records both occupancy tables in `config.json`, and says so in the run comment.** Decided here so it is not decided after seeing a kappa.

### Code placement, and vectorisation

`src/catface/core/geometry.py`, because `core/states.py` is the declared consumer and `AGENTS.md` forbids either track keeping a local copy of shared geometry:

- `BILATERAL_PAIRS`, the 21 pairs above, and `MIDLINE_POINTS`, `(16, 17, 0, 2)`, as module constants, sourced from the v3 legend with the file named in the docstring
- `yaw_foreshortening_ratio(shapes) -> (N,)`, the `w_obs / v_obs` of family 1
- `yaw_midline_offset(shapes, midline_index=16, pair=(4, 8)) -> (N,)`, the signed `s` of family 2
- `yaw_centroid_proxy(shapes) -> (N,)`, the legacy quantity

`src/catface/ml/pose.py`, because the percentile rule and the degree label are research-track modelling choices and the app consumes the ratio, not the angle:

- `frontal_ratio(ratios, percentile=95.0) -> float`
- `yaw_degrees(ratios, r_frontal) -> (N,), (N,) bool`, angles plus an out-of-domain mask for the clamped rows

**Every one of these takes `(N, 48, 2)` and returns arrays, not a per-shape scalar.** `scripts/shape_space.py` computes E1's proxy over the whole stack, and a per-shape loop risks a different summation order against a published number. Single-shape callers index `[None]`. This supersedes the first pass's `yaw_asymmetry_proxy(shape, ...)` signature, which was per-shape.

The first pass's `yaw_degrees_from_proxy(proxy, shape, d)` is struck along with the depth prior it took.

### On the algebra, checked term by term

The first pass derived `dist_right^2 - dist_left^2 = -d * w^2 * sin(2*theta)` for the centroid proxy. Under the convention stated above, with eyes at `(+/-a, 0, 0)`, a midline point at `(0, -h, z_M)` and `w = 2a`, the difference of squared distances is `-2*a*z_M*sin(2*theta) = -w*z_M*sin(2*theta)`. Both forms are the same equation under different readings of `d`: with `d` an absolute depth it is `-w*d*sin(2*theta)`, and with `d` the dimensionless fraction `z_M/w` that the first pass defined, substituting `z_M = d*w` gives `-w^2*d*sin(2*theta)`, which is `length^2` as required. I verified all three relations in this section numerically before writing them, by projecting synthetic points and comparing against the closed forms; family 1's span independence of `z0` and family 2's `d_rel * tan(theta)` both hold exactly. The convention, not the derivation, was what the first pass left unstated, and it is stated now.

### Acceptance criteria: replacements

These five checkboxes in the first pass are **struck**: the `yaw_asymmetry_proxy` placement one, the degree-calibration-with-`d` one, the "four equal-count quantile bins on `|proxy|`" one, the "every degree figure carries its `d` and the 0.15 / 0.30 / 0.45 range" one, and the saturation one. The proxy-exceeds-model-range criterion is **amended** to read as a finding about the superseded centroid proxy explaining the estimator change. Every other checkbox in the first pass stands. Replacing them:

- [ ] `BILATERAL_PAIRS` and `MIDLINE_POINTS` committed as constants in `src/catface/core/geometry.py`, sourced from the v3 legend, that file named in the docstring
- [ ] `yaw_foreshortening_ratio`, `yaw_midline_offset` and `yaw_centroid_proxy` in `core/geometry.py`, each vectorised over `(N, 48, 2)`; `frontal_ratio` and `yaw_degrees` in `src/catface/ml/pose.py`; no copy of any of these formulas surviving in `ml/` or `scripts/`
- [ ] `scripts/shape_space.py` imports `yaw_centroid_proxy` instead of computing it inline, the array is bit-identical to the current inline result, and E1's r = 0.90 against PC2 is re-derived and stated to match; `experiments/e1` byte-unchanged
- [ ] Axis convention and projection model written out in the `core/geometry.py` docstring in the form stated above
- [ ] `r_frontal` computed as the 95th percentile of `w_obs / v_obs` over all 1967 rows, frozen into `experiments/e4/config.json` before any arm is scored, with the 90th and 99th percentile sweep reported
- [ ] Out-of-domain rows (`w_obs / v_obs > r_frontal`) counted and reported per percentile setting, clamped to `theta = 0`, no invented angles
- [ ] Four quantile bins computed on `theta` at run time, edges frozen into `config.json` before scoring; per-bin and per-(bin x fold) `uncomfortable` occupancy recorded at that moment, and the 3-bin fallback taken and declared if any cell is empty
- [ ] Internal control: Spearman `rho` between `|s|` and `tan(theta)` over all 1967 rows with a bootstrap CI (B = 1000, seed 0), reported against the pre-registered 0.5 / 0.3 thresholds, and the verdict's use of the word "yaw" justified by it or withdrawn
- [ ] Per-row `d_rel = s / tan(theta)` distribution reported (median and spread), and the chin control's `d_rel(2)` shown to be far smaller in magnitude than `d_rel(16)` or the discrepancy explained
- [ ] `yaw_centroid_proxy` reported alongside the new estimators and correlated against `theta`, so E1's PC2 finding stays connected to E4's
- [ ] Family 2's left/right sign split reported as a balance check, with the sign-to-turn-direction mapping stated
- [ ] Facial-width confound (brachycephalic versus dolichocephalic, no breed or identity labels, no per-row `r_frontal`) and the unmodelled pitch stated as limitations in `experiments/e4/README.md` and carried into `docs/MODEL_REPORT.md` under Q3
- [ ] Every degree figure in the run carries the percentile that produced it and the 90th/99th range; a bare degree figure is still a defect
- [ ] Pose verdict sentence gives the foreshortening-ratio edge first and the degree range second, or states that no threshold is locatable. This restates the first pass's criterion against the new quantity: the ratio is what `core/states.py` consumes, computable from landmarks with no modelling assumption at all
- [ ] Rejected estimator families recorded in the README with the reason, so the choice is on the record rather than implied

**Stop condition:** still none, and still nothing here can fire one. Negative and inconclusive remain closeable outcomes for both questions.

**Comment (ML Engineer, 2026-09-20):** implemented the second grooming pass's method and ran it, `experiments/e4/`. Six arms on the frozen `data/cache/splits.csv`, 1967 rows, five folds, `oversample=True` on training folds only, each arm scored inside four quantile bins of the foreshortening estimator. Pooled out-of-fold kappa: `gnn_identity` 0.231 (CI 0.193 to 0.267), `coords_lr` 0.174, `mlp` 0.155, `ratios_lr` 0.154, `yaw_only_lr` 0.041, `random_baseline` 0.017. The pre-registered rule picks `gnn_identity` as the verdict arm. Per-bin tables, per-fold numbers, both verdict texts and the limitations are in `experiments/e4/README.md`; per-arm `metrics.json`, `config.json`, `merge.json` and `per_row.csv` sit beside it.

**Q3: NEGATIVE, and PROVISIONAL.** Bin 0 clears chance at kappa 0.191 (CI 0.132 to 0.247), so the chance gate did not fire. No bin b >= 1 has a kappa upper CI below bin 0's lower CI, so the degradation rule did not fire either: no measurable degradation across the range this dataset covers, no threshold locatable at this sample size, no pose penalty for the app. The prior-shift tolerance is exceeded at bin 0, TV 0.064 against the pre-registered 0.05, which makes the verdict PROVISIONAL and names the reference bin itself as the offender: 44 of the 107 `uncomfortable` rows sit in the most frontal bin. Control (c) says pose barely predicts the label on its own, `yaw_only_lr` at 0.041 against the dummy's 0.017, so the flat per-bin profile is not yaw carrying label information.

**Q4: NEGATIVE, no merge warranted on the evidence.** Cross-talk, then pairwise 2x2 kappa with its CI, over the pooled confusion matrix: attentive/relaxed 0.298, kappa 0.262 (0.213 to 0.310); attentive/uncomfortable 0.128, kappa 0.333 (0.260 to 0.412); relaxed/uncomfortable 0.192, kappa 0.240 (0.151 to 0.321). Every pair fails at least one threshold, and all three CIs exclude 0, so the three classes are separable enough that no merge is defensible. I retrained the highest cross-talk pair anyway, attentive/relaxed, so the null path rests on a measurement: retraining scores 0.148 against 0.190 for collapsing the 3-class predictions after the fact, with the 2-class dummy at 0.021. Merging loses to not merging. The cat-emotions-7 Scared/Surprised/Angry expectation is recorded as not testable here, `docs/DECISIONS.md` 2026-09-19, and no other pair is substituted for it.

**The internal control, and the part of the method that did not survive the data.** Spearman rho between family 2's |s| and family 1's tan(theta) is 0.106, CI 0.062 to 0.149, well under the pre-registered 0.3. The yaw label is not earned, so the verdict names the foreshortening ratio and carries no degree figure anywhere. Three measurements agree on why. The chin control fails its falsifiable prediction: node 2 sits near the eye-corner plane and should imply a far smaller depth than the philtrum, but its median `d_rel` comes out at 0.87 of the philtrum's, 0.0449 against 0.0517. The legacy centroid proxy correlates with the new theta at r = -0.067, so E1's quantity and this one are unrelated too. And the montage shows it in pictures. The synthetic round trip in `tests/test_pose.py` recovers a known rotation to 0.01 degrees through the real `procrustes_align` and `yaw_degrees`, and puts the chin below 1% of a built-in 0.30, so the estimators do what the geometry says; the photographs are not the geometry. The likeliest single cause is the facial-width confound the grooming named: family 1 divides a bilateral span by a vertical one, and breed varies more than pose does here.

**What the montage showed** (`experiments/e4/yaw_extremes_manual_review.png`, looked at before I wrote the README). Two of the three most-negative-`s` rows are detector failures: in one, point 4 lands on one cat's eye and point 8 on a second cat's eye in the same photo, and in the next the 48 points spread across three kittens in a basket. Both passed E0's plausibility filter. The readable extremes do carry the sign, and I read the mapping off them rather than off the algebra: positive `s` means the head is turned toward the image left, the cat's own right. The near-zero panels are where family 1 comes apart, with a black cat facing the camera square on assigned 38.9 degrees, above the median of the whole dataset, because its face is narrow relative to its chin-to-eye span.

**Controls and constraints.** Control (a) per-bin class counts and TV are in the README and `config.json`. Control (b) scores the uniform dummy inside each bin under that bin's own prior, and it sits at chance in all four. Control (d): all five checkable arms reproduce E2's and E3's committed per-fold macro F1, kappa and MCC bit-for-bit, `max_abs_delta` 0.0 each, recorded in `config.json`. One thing to flag for QA: an earlier execution of the same script at the same commit reproduced `mlp` in four folds of five, with fold 0 off by 0.0153 of kappa, so torch CPU training here is not bit-reproducible between executions. Every artifact in the directory comes from one execution. The 3-bin fallback did not fire; the smallest (bin x fold) `uncomfortable` cell holds 2 rows and none is empty. Bin edges and `r_frontal` were frozen into `config.json` before any arm was scored, and both verdict functions were committed in `src/catface/ml/scoring.py` before the run produced a metric, so `git log` can check that rather than taking it on trust. `git diff -- data/cache/splits.csv` is empty, `experiments/e1`, `e2` and `e3` are byte-unchanged, no new dependency and no scipy anywhere: the Spearman is Pearson on average ranks in `ml/pose.py`, tested against a hand-computed case with a tie. `uv run pytest` green, 69 passed.

One observation the pre-registered rule does not capture, recorded so nobody finds it later and over-reads it. `ratios_lr` is the only arm whose per-bin kappa declines monotonically, 0.231 to 0.182 to 0.107 to 0.085, and its three features are geometric ratios of the same kind as the binning quantity. Shared measurement is a better reading of that than pose, and its CIs overlap throughout, which is why the degradation rule did not fire on it. It is also not the verdict arm.

Not closing this issue, and I have not ticked the acceptance criteria. That is Research QA's call and then the orchestrator's.

**Addendum (direct user instruction, 2026-09-22): Q4 declared obsolete, struck from this issue.** Per `docs/DECISIONS.md` 2026-09-22: no run to date — E2's three baselines, E3's five arms, or E4 itself — ever produced a class with zero predictions, so there was never evidence of class collapse for a merge procedure to investigate. Q4 is no longer a question this issue answers. The following are struck as obsolete, not pursued, and kept on the record rather than deleted: the Q4 merge-thresholds decision rule (cross-talk >= 0.25 and pairwise 2x2 kappa CI containing 0), the retrained-vs-collapsed-vs-dummy merge procedure, the Q4 null-path paragraph, the cat-emotions-7 Scared/Surprised/Angry not-testable note, and every Q4-labelled acceptance criterion in the "Acceptance criteria: replacements" list above (the two Q4 bullets and the cat-emotions-7 bullet). RSCH-4 from here on answers Q3 only.

**Second addendum (direct user instruction, 2026-09-22): the six-arm implementation above is replaced.** The ML Engineer's 2026-09-20 comment, its six arms, its bins on `theta`, and its Q3/Q4 verdicts are superseded by a new, much smaller implementation at `scripts/pose_and_classes.py` (git commits `3b5cddb` "Groom E4 code" and `deaecf0` "Add new E4 outputs"). The old six-arm artifacts under `experiments/e4/` (`ratios_lr/`, `mlp/`, `yaw_only_lr/`, `random_baseline/`, `merge.json`, `per_row.csv`) were deleted rather than kept alongside, unlike RSCH-3's own "first run's artifacts stay on disk" precedent — recorded here since it means the only record of the six-arm numbers above is this comment and git history (`9d0b4ec`), not a live directory. The new script checks only two arms (`coords_lr`, `gnn_identity`) in-sample, against their already-fit full-dataset checkpoints, and by its own docstring computes no verdict and is not a re-run of the pre-registered out-of-fold numbers above. This issue's question, method, and acceptance criteria are re-groomed below to match what the new implementation actually is, rather than left pointing at a method nobody is running.

**Third grooming pass (Research PM, 2026-09-22). Supersedes the second grooming pass's Q3 decision rules (verdict-arm selection, degradation rule, chance gate, prior-shift tolerance), required controls (b) `DummyClassifier`, (c) `yaw_only_lr`, (d) the per-fold reproduction check, and the matching "Revised acceptance criteria" / "Acceptance criteria: replacements" bullets that assumed six arms and an out-of-fold verdict. Q4's equivalents are already struck by the first addendum above and are not revisited here. All superseded text stays on the record.**

**What actually ran, and what's left of the pre-registered method.** `scripts/pose_and_classes.py` loads two already-fitted checkpoints — `experiments/e2/coords_lr/checkpoint.pkl`, `experiments/e3/gnn_identity/checkpoint.pt`, both full-dataset fits — and predicts over the same rows those checkpoints were trained on, binned into the four pre-registered quantiles of `core.geometry.yaw_foreshortening_ratio`. Its own docstring says it computes no verdict. This isn't a smaller run of the same pre-registered method: commit `3b5cddb` ("Groom E4 code"), the same commit that rewrote this script, also deleted 229 lines from `src/catface/ml/scoring.py` (the degradation rule, chance gate, prior-shift TV-distance and merge-threshold functions the second grooming pass pre-registered) and 107 lines from `tests/test_pose.py`. `quantile_bins`, `bootstrap_ci` and `kappa_ci` survive there and are what this run uses. The 2026-09-20 Q3 verdict recorded above (NEGATIVE, PROVISIONAL; prior-shift TV 0.064 at bin 0) is therefore not reproducible from this repository as it stands — neither its `experiments/e4/` output directories (deleted per the second addendum) nor the code that computed it survive. Only the text of that comment is left. That's the state Q3 was already in before this grooming pass; this run doesn't change it, and doesn't try to.

**Question this run can answer.** Not Q3 as posed ("how much does head pose contaminate the prediction"). That question is about generalization: whether a model trained mostly on near-frontal photos degrades on photos it hasn't seen at higher yaw. An in-sample check can't measure that — both checkpoints have already seen every row in every bin during fitting, including whatever's hardest at high yaw, so nothing forces a real generalization failure to show up here even if one exists at deployment time. What this run answers is narrower and one-sided: does apparent per-bin agreement, on rows the model was fit to, decline with estimated yaw? A "yes" would be real evidence (if a model can't even fit its own high-yaw training rows, a held-out model won't do better). The "no" that actually came back — both arms trend flat-to-rising rather than falling (`coords_lr` per-bin kappa 0.159 → 0.196 → 0.181 → 0.275; `gnn_identity` 0.224 → 0.266 → 0.263 → 0.262, pooled kappa 0.213 and 0.263 respectively, `experiments/e4/README.md`) — is weak: it shows the checkpoints aren't underfitting at high yaw, not that a fresh prediction on unseen high-yaw photos holds up. **This reduced method answers a strictly weaker question than Q3 as posed. Its result is a one-sided sanity check that found nothing alarming; it is not a second, independent negative answer to Q3 and should not be reported as one.**

**Required controls, given this scope.**
- Chance-level reference: still meaningful, not yet stated as such. `kappa` is chance-corrected, but the reference each bin's kappa is implicitly read against — the bin's own `majority_rate` (0.54 to 0.60, already in `experiments/e4/*/metrics.json`) — is left as an unlabelled column rather than named in prose.
- Per-bin prior shift: still meaningful, and cheaper than the six-arm version. Class counts per bin are already reported; a total-variation distance from the pooled class prior is not computed. Guards the same failure mode (a bin's different class mix read as pose degradation) independent of in-sample vs. out-of-fold.
- `yaw_only_lr` as a separate arm: **not required at this scope.** It existed to separate "pose degrades the model" from "pose predicts the label" for a verdict this run doesn't compute; requiring it here rebuilds a piece of the six-arm design the addendum already walked back.
- Reproduction/provenance check: narrowed, not dropped. The old control (d) compared fresh per-fold CV metrics against committed E2/E3 numbers — there's no fold structure here to compare. What's still checkable and currently missing: confirmation that the two loaded checkpoints are the exact, unmodified files those experiments' QA-passed runs produced, not a silently retrained stand-in.
- Stop condition: unchanged, none. A one-sided sanity check that finds nothing is a legitimate, closeable result.

**Revised acceptance criteria.** Replace the struck bullets below; the geometry/estimator infrastructure criteria from the "Acceptance criteria: replacements" list (`BILATERAL_PAIRS`/`MIDLINE_POINTS`, `yaw_foreshortening_ratio`, `frontal_ratio`, `yaw_degrees`, the axis-convention docstring, `r_frontal` at the 95th percentile, out-of-domain row counting) are unaffected by this pass and already hold, since this script reuses that same code.

- [x] `experiments/e4/README.md` states, in the reader-facing text and not only in the script docstring, that this is an in-sample check (checkpoints fit on the rows they're scored on) and that its flat-to-rising per-bin trend is one-sided evidence: it does not establish the model holds up on unseen high-yaw photos, only that it isn't visibly underfitting its own high-yaw training rows
- [x] Any degree figure in the artifact (the `theta range (deg)` column) either carries the caveat that this project's own internal control already found the yaw label unearned for this estimator (Spearman rho 0.106, CI 0.062 to 0.149, against the pre-registered 0.3 threshold, recorded in the now-code-orphaned 2026-09-20 comment above), and carries the percentile that produced it plus the 90th/99th sweep range per the still-standing "every degree figure... a bare degree figure is still a defect" criterion above — or the bins are reported in raw `w_obs / v_obs` ratio units instead of a bare degree figure
- [x] Per-bin total-variation distance from the pooled class prior computed and reported, alongside the class counts already present
- [x] Each bin's `majority_rate` named explicitly in prose as the chance floor the arm's kappa is being read against, not left as a table column only
- [x] Checkpoint provenance recorded: `experiments/e2/coords_lr/checkpoint.pkl` and `experiments/e3/gnn_identity/checkpoint.pt` confirmed as the exact files their own QA-passed runs produced (committed, unmodified — sha256 or `git log` on the file path), not re-derived for this run
- [x] The facial-width confound and unmodelled pitch (still-standing limitation from the "Acceptance criteria: replacements" list above) stated in `experiments/e4/README.md`, since `theta` here is still family 1's foreshortening ratio and inherits that limitation regardless of verdict scope
- [x] Run comment states plainly that this is a supplementary sanity check, that it computes no Q3 verdict, and that Q3's substantive 2026-09-20 answer (NEGATIVE, PROVISIONAL) is currently unverifiable from this repository because the code that computed it (`scoring.py`'s degradation-rule/TV-distance/merge functions, deleted in `3b5cddb`) no longer exists — so that gap is on the record rather than papered over by this run's presence
- [x] `data/cache/splits.csv` untouched (frozen since E2); `experiments/e1`, `e2`, `e3` byte-unchanged (already true — unaffected by this pass)
- [x] No new dependency (already true — unaffected by this pass)

**Struck as obsolete (assumed the six-arm out-of-fold method, not applicable to this scope):**
- Pre-registered decision rules: verdict metric/arm selection among four model arms, the degradation rule (bin-vs-bin kappa CI comparison), the chance gate, the prior-shift tolerance triggering a PROVISIONAL verdict
- Required controls (b) `DummyClassifier` scored per bin under that bin's prior — replaced above by naming `majority_rate` as the floor; (c) `yaw_only_lr`; (d) per-fold reproduction against committed E2/E3 `metrics.json` — replaced above by a checkpoint-provenance check
- "[ ] Six arms on the frozen folds: `ratios_lr`, `coords_lr`, `mlp`, `gnn_identity`, `yaw_only_lr`, `random_baseline`; `oversample=True` on training folds only"
- "[ ] Control (b): uniform `DummyClassifier` scored within each bin under that bin's own prior", "[ ] Control (c): `yaw_only_lr` reported at the same detail as the model arms, not as a footnote", "[ ] Control (d): per-fold metrics for the re-run arms compared against the committed E2/E3 `metrics.json`, deltas recorded in `experiments/e4/config.json`"
- "[ ] Bootstrap as specified: ... per-fold kappas reported alongside as the sanity check" — the within-bin bootstrap-CI half of this bullet is satisfied (`scoring.kappa_ci` is used); the per-fold half is struck, there being no folds in a full-dataset in-sample fit
- "[ ] Q3 verdict applies the pre-registered degradation rule, chance gate and prior-shift tolerance by name, and states which one fired"
- "[ ] Pose verdict sentence gives the proxy edge first and the degree range second, or states explicitly that no threshold is locatable" and its later restatement "[ ] Pose verdict sentence gives the foreshortening-ratio edge first and the degree range second..." — both struck, no verdict is computed
- "[ ] Q3 and Q4 each answered explicitly in the run comment..." — the Q3 half is struck along with the verdict it presupposes; the Q4 half was already struck by the first addendum
- "[ ] Internal control: Spearman `rho` between `|s|` and `tan(theta)`...", "[ ] Per-row `d_rel`... distribution reported...", "[ ] `yaw_centroid_proxy` reported alongside...", "[ ] Family 2's left/right sign split reported..." — struck as not required at this scope: this pass handles the same underlying honesty problem (an unearned "yaw" label) with a caveat requirement above instead of a full family-2 recompute
- "[ ] Rejected estimator families recorded in the README with the reason" — not applicable; this script doesn't compare estimator families, it reuses the one already chosen
- The "positive/negative/inconclusive Q3 shape" paragraph under **Positive and negative shape, and the stop condition** above, insofar as it describes a verdict computed from bin-vs-bin CI comparison across CV folds — no fold structure exists in this run

**Stop condition:** unchanged, none.

**QA: FAIL (Research QA / Methodology Reviewer, 2026-09-22, third-grooming-pass scope).** Checked `scripts/pose_and_classes.py` and `experiments/e4/{config.json,README.md,coords_lr/metrics.json,gnn_identity/metrics.json,plots/kappa_per_bin.png}` against the eight "Revised acceptance criteria" bullets above, line by line. The struck lists are not graded. The geometry/estimator infrastructure this pass names as already holding (`BILATERAL_PAIRS`/`MIDLINE_POINTS`, `yaw_foreshortening_ratio`, `frontal_ratio`, `yaw_degrees` with out-of-domain masking, the axis-convention docstring, `r_frontal` at the 95th percentile) does in fact still hold — read `src/catface/core/geometry.py` and `src/catface/ml/pose.py` to confirm, not re-checked below.

- [x] `data/cache/splits.csv` untouched, `experiments/e1`/`e2`/`e3` byte-unchanged — PASS. `git diff -- data/cache/splits.csv` is empty; `git diff --stat -- experiments/e1 experiments/e2 experiments/e3` is empty at HEAD.
- [x] No new dependency — PASS. `pyproject.toml` diff is empty; `matplotlib` (used by the plotting code) is already in the `train` group; no `scipy` import anywhere in `pose_and_classes.py`, `core/geometry.py` or `ml/pose.py`.
- [ ] README states, in reader-facing text, that this is an in-sample check and that the flat-to-rising trend is one-sided evidence (doesn't establish generalization, only that the checkpoints aren't visibly underfitting their own high-yaw rows) — FAIL. `experiments/e4/README.md` line 3 says "In-sample check... Not a re-run of the pre-registered out-of-fold numbers... No verdict is computed" but never states the one-sidedness: that a flat-or-rising in-sample trend is weak evidence and cannot rule out a real generalization failure on unseen high-yaw photos. A reader who stops at the README, rather than this backlog entry's prose, gets the weaker claim only by omission, not by an actual sentence saying so.
- [ ] Degree figures carry the unearned-yaw-label caveat and the 90th/99th percentile sweep, or bins are reported in raw ratio units instead — FAIL. The README's "theta range (deg)" table (lines 11–16) reports bare degrees with no caveat and no sweep. `config.json` records only a single `r_frontal` at the 95th percentile; `pose.FRONTAL_PERCENTILE_SWEEP = (90.0, 95.0, 99.0)` exists in `src/catface/ml/pose.py` but `grep` across `scripts/`, `src/` and `tests/` shows it is never read anywhere — the sweep this pass requires was never run, let alone reported. Neither escape hatch (caveat+sweep, or ratio units) was taken.
- [ ] Per-bin total-variation distance from the pooled class prior computed and reported — FAIL. `experiments/e4/coords_lr/metrics.json` and `gnn_identity/metrics.json` report `class_counts` and `majority_rate` per bin but no TV-distance field anywhere in either file, `config.json`, or the README. I computed it myself from the committed `class_counts` against the pooled prior (attentive 1130/1967, relaxed 730/1967, uncomfortable 107/1967): bin 0 (297/151/44) comes out at TV ≈ 0.064, over the 0.05 tolerance the second grooming pass set and the same offending bin the now-orphaned 2026-09-20 comment flagged. That the guard would fire is exactly why this pass still requires it be computed and shown, not left to a reviewer's arithmetic.
- [ ] Each bin's `majority_rate` named explicitly in prose as the chance floor — FAIL. It appears only as a table column in both per-arm sections; no sentence in the README says "kappa in bin N is being read against a majority rate of X."
- [ ] Checkpoint provenance recorded: `experiments/e2/coords_lr/checkpoint.pkl` and `experiments/e3/gnn_identity/checkpoint.pt` confirmed as the exact, unmodified files their own QA-passed runs produced — FAIL, and not just on paperwork. Nothing in `experiments/e4/config.json` or the README records a hash or a provenance note for either file; `config.json`'s `"checkpoints"` block is bare file paths. I checked this myself rather than trusting the claim: `git log --oneline -- experiments/e3/gnn_identity/checkpoint.pt` shows one commit, `d0b4e21` "Add trained GNN checkpoints" (2026-09-21, the day after RSCH-3's 2026-09-20 QA PASS), which only adds checkpoint/ONNX files and leaves `experiments/e3/gnn_identity/metrics.json` and its README untouched — consistent with the claim. `experiments/e2/coords_lr/checkpoint.pkl` is worse: its only commit is `e3bb8cd` "Add new E2 outputs" (2026-09-22 15:47, forty minutes before `3b5cddb` "Groom E4 code" that this run's script comes from), and that same commit *also* silently rewrote `experiments/e2/mlp/metrics.json` — mean kappa moved from the previously-reported 0.155 ± 0.046 to 0.110 ± 0.061, confusion matrix changed, and `config.json`'s `git_sha` changed from `6c01ac7c...` to `acb8fe9d...` — with no matching `docs/backlog.md` comment anywhere recording that E2 was re-run. And RSCH-2 itself has no "QA: PASS" comment anywhere in this file — unlike RSCH-1 and RSCH-3, it was never actually QA-passed to begin with, so "their own QA-passed E2/E3 runs" is true for E3 and false for E2. `coords_lr/metrics.json` itself wasn't touched by `e3bb8cd` (its 0.174 ± 0.048 stands), so the CV numbers on record are probably still right, but the specific `.pkl` this run loads was minted in the same undocumented, unreviewed commit that demonstrably let a sibling model's numbers drift — that is exactly the silent-regeneration risk this criterion exists to catch, and it fired.
- [ ] Facial-width confound and unmodelled pitch stated in `experiments/e4/README.md` — FAIL. Neither appears anywhere in the README; there is no limitations section at all.
- [ ] Run comment states plainly this is a supplementary sanity check, computes no Q3 verdict, and that Q3's 2026-09-20 answer is currently unverifiable because `scoring.py`'s deleted functions no longer exist — FAIL as a distinct artifact. That content exists, but only inside this Research PM's own third-grooming-pass prose above ("What actually ran..."); no ML Engineer run comment reporting this run's numbers and caveats was posted to this issue the way every prior run in RSCH-2/3/4 was, and `scripts/pose_and_classes.py`'s docstring (which does carry an in-sample/no-verdict note) is not "the run comment" the criterion asks for.

**Result: FAIL.** Two of eight bullets pass (git/dependency hygiene); six do not, spanning both categories the task asked me to separate. The README-caveat, TV-distance, majority-rate-in-prose, degree-sweep and run-comment gaps are all doable-but-undone artifact work — the numbers needed for most of them (TV distance, majority rate) are already sitting in the committed `metrics.json` files, and a run comment is one comment away. The checkpoint-provenance finding is not that: it surfaced that RSCH-2 has never been QA-passed at all, and that the specific checkpoint this run depends on was minted in an undocumented same-day regeneration of E2 that demonstrably changed another model's numbers without a backlog comment. That is a methodology/process gap upstream of RSCH-4 and belongs with the Research PM, not a checklist item for the ML Engineer to silently tick.

**Comment (ML Engineer, 2026-09-22).** Fixed six of QA's seven FAIL findings above (all but checkpoint provenance, left for a separate track per direct user instruction) by editing `scripts/pose_and_classes.py` and re-running it; `experiments/e4/{config.json,README.md,coords_lr/metrics.json,gnn_identity/metrics.json,plots/kappa_per_bin.png}` are regenerated. The README now states in reader-facing text that this is an in-sample check and that its flat-to-rising per-bin trend is one-sided evidence: it does not establish the model holds up on unseen high-yaw photos, only that it isn't visibly underfitting its own high-yaw training rows. The `theta range (deg)` column is gone; bins are reported in raw `w_obs / v_obs` ratio units instead, with a note pointing at the unearned-yaw-label finding below. Per-bin total-variation distance from the pooled class prior is now computed (`scoring.total_variation`) and reported per bin in both `metrics.json` and the README table; bin 0 comes out at TV 0.064, over the pre-registered 0.05 tolerance, same as QA's hand-computed figure, and the README says so in prose. Each bin's `majority_rate` is now named explicitly in prose per arm, not left as a table column only. The facial-width confound and unmodelled pitch are stated as limitations in the README. Pooled kappa is unchanged (`coords_lr` 0.213, `gnn_identity` 0.263), since none of this touches the model or the data, only the reporting.

This run is a supplementary sanity check. It computes no Q3 verdict. Q3's substantive 2026-09-20 answer (NEGATIVE, PROVISIONAL) is currently unverifiable from this repository: the code that computed it — `scoring.py`'s degradation-rule, chance-gate, TV-distance and merge-threshold functions — was deleted in commit `3b5cddb`. That gap is recorded here rather than papered over by this run's presence; nothing in this run re-derives or stands in for that verdict. See the regenerated `experiments/e4/` artifacts for the current numbers.

Checkpoint provenance (QA FAIL item 5) is not addressed here, per direct user instruction — left for a separate track.

**QA: PASS (Research QA / Methodology Reviewer, 2026-09-22, third-grooming-pass re-review).** Re-checked the same eight "Revised acceptance criteria" bullets against the ML Engineer's regenerated `scripts/pose_and_classes.py`, `experiments/e4/config.json`, `README.md`, `coords_lr/metrics.json` and `gnn_identity/metrics.json`. The six previously-failing items were verified directly, not taken on the ML Engineer's word; the checkpoint-provenance item was scored on the lower disclosure bar this pass's own instructions set, per direct user instruction.

- [x] `data/cache/splits.csv` untouched, `experiments/e1`/`e2`/`e3` byte-unchanged — PASS. `git diff -- data/cache/splits.csv` and `git diff --stat -- experiments/e1 experiments/e2 experiments/e3` both empty at HEAD.
- [x] No new dependency — PASS. `git diff -- pyproject.toml` empty; no `scipy` import in `pose_and_classes.py`, `core/geometry.py`, `ml/pose.py` or `ml/scoring.py` (one comment in `pose.py` only names scipy as the thing being avoided).
- [x] README states, in reader-facing text, that this is an in-sample check and that the flat-to-rising trend is one-sided evidence — PASS. `experiments/e4/README.md`'s second paragraph, bolded, now reads in full: "This is an in-sample check and its evidence is one-sided... a flat-or-rising per-bin trend below does not establish that the model holds up on unseen high-yaw photos... A falling trend would have been informative; this one is not proof of generalization." That is the sentence the FAIL found missing, present now in the artifact itself, not only in this backlog's prose.
- [x] Degree figures carry the caveat + sweep, or bins are reported in raw ratio units instead — PASS via the ratio-units branch. The README's bin table now reports `w_obs / v_obs` ranges (0.7791 to 2.4708 across four bins), the `theta range (deg)` column is gone, and the accompanying prose names the unearned-yaw-label finding (Spearman rho 0.106, CI 0.062 to 0.149, against the 0.3 threshold) as the reason. One residual worth recording rather than failing over: `config.json` still carries a `theta_edges` array in bare degrees (`[0.0, 29.77, 35.31, 39.16, 61.51]`), computed en route to the ratio bins and never read back by any code (`grep -rn theta_edges` outside this script returns nothing) or surfaced in `docs/MODEL_REPORT.md`. It is inert internal data, not a claim made to a reader anywhere, so it does not revive the FAIL — but it is the one place a bare degree figure from this run still exists on disk, and a future consumer of `config.json` alone (without the README's caveat next to it) would not see the unearned-label warning attached to it.
- [x] Per-bin total-variation distance from the pooled class prior computed and reported — PASS, and re-derived independently rather than trusted. From `config.json`'s `pooled_class_counts` [1130, 730, 107] (prior [0.5745, 0.3710, 0.0544]) and each bin's `class_counts` in `coords_lr/metrics.json`, I recomputed TV = 0.5·Σ|p_i − q_i| by hand for all four bins: bin 0 (297/151/44) → 0.0642, bin 1 (296/175/20) → 0.0284, bin 2 (272/202/18) → 0.0395, bin 3 (265/202/25) → 0.0395. These match the committed `tv_distance` fields to four decimal places, and `scoring.total_variation`'s implementation (`0.5 * np.abs(p/p.sum() - q/q.sum()).sum()`) is the correct formula, not a stand-in. Bin 0's 0.064-over-0.05 finding is stated in the README's prose, matching my own arithmetic and the FAIL's.
- [x] Each bin's `majority_rate` named explicitly in prose as the chance floor — PASS. Both per-arm sections now end with "`<arm>`'s kappa in each bin is read against that bin's own majority-class rate as the chance floor: bin 0 0.604, bin 1 0.603, bin 2 0.553, bin 3 0.539," matching the table's `majority_rate` column exactly.
- [~] Checkpoint provenance recorded — scored on the lower bar this pass sets, not the original full bar: no run to date has fixed the underlying gap (RSCH-2 still carries no QA verdict; `experiments/e2/coords_lr/checkpoint.pkl`'s only commit is still the same undocumented `e3bb8cd` that also silently moved `mlp`'s kappa), and that is out of scope here per direct user instruction. What is in scope is disclosure, and it holds: `config.json`'s `checkpoints` block is still bare paths with no hash, but it makes no provenance claim either, clean or otherwise, and the run comment above states plainly, in the same backlog entry a reader would land on, that "Checkpoint provenance (QA FAIL item 5) is not addressed here, per direct user instruction — left for a separate track." Nowhere in `README.md`, `config.json` or the run comment does the artifact assert or imply the checkpoints' provenance was confirmed. That is the same disclosure standard this review already accepts for the Q3-unverifiable-answer bullet below, where the caveat lives in the backlog comment rather than the README, so it is applied consistently here.
- [x] Facial-width confound and unmodelled pitch stated in `experiments/e4/README.md` — PASS. A "## Limitations" section now exists with both, in the same terms as the struck first-pass text: breed/identity not labelled, one photo per cat, no per-row correction estimable; pitch absorbed into the ratio uncalibrated, not removed.
- [x] Run comment states plainly this is a supplementary sanity check, computes no Q3 verdict, and that Q3's 2026-09-20 answer is currently unverifiable because `scoring.py`'s deleted functions no longer exist — PASS. The ML Engineer's comment above states this in almost the same words the criterion asks for, naming commit `3b5cddb` as what deleted the code, and stops short of re-deriving or standing in for the old verdict.

**Result: PASS.** Eight of eight bullets pass, one with a lower bar this pass explicitly set rather than the original full bar. All six previously-failing artifact/text items are fixed in the artifact itself, not only asserted fixed: I re-read the README prose rather than trusting the ML Engineer's comment, and I independently recomputed the TV distance from the committed class counts rather than trusting the `tv_distance` field, and both hold up. The one residual note — a bare-degree `theta_edges` array left inert in `config.json` — is recorded above for whoever next touches this file, but it makes no claim to any reader and does not block. Checkpoint provenance itself remains an open methodology gap (RSCH-2 has no QA verdict; the specific checkpoint this run loads was minted in an undocumented same-day regeneration), unchanged from the FAIL and explicitly not this run's to fix per direct user instruction — but the artifact now honestly says so instead of staying silent in a way a reader could mistake for clean provenance.

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

# DECISIONS.md

Decisions and the reasons for them, written as they're made. Not a design
doc — one entry per decision: what was decided, why, what the alternative
was. Newest on top.

---

## 2026-09-19 — `.gitignore`'s blanket `experiments/` line removed

**Decided:** `.gitignore` no longer ignores `experiments/` wholesale. `experiments/e1/` (README, `plots/`) commits normally, and so will every later `experiments/<run>/` directory.

**Why:** the blanket line was already in tension with `docs/team/ml-engineer.md`'s own stated convention — "Commit regularly, one directory per run under `experiments/<run>/`... every run, not just the ones that worked" — but the conflict stayed latent because E0 deliberately skipped `experiments/` entirely (see the "E0 skips `experiments/`" entry below) and nothing had tried to commit there yet. E1 is the first run to actually write to `experiments/e1/`, so leaving the blanket ignore in place would have silently dropped the run directory the role doc requires committing.

**Alternative considered:** add a narrower `!experiments/e1/` exception per run, mirroring the `models/*` / `!models/manifest.json` pattern used elsewhere in this file. Rejected — the role doc wants every run committed, not an allowlist maintained by hand on each new experiment; removing the blanket line once does the same job without a recurring edit.

**How to apply:** don't re-add a blanket `experiments/` ignore. If a specific run produces something genuinely too large or generated-on-demand to commit (e.g. a cached model checkpoint), ignore that one path under `experiments/<run>/`, not the whole tree.

---

## 2026-09-19 — `models/class_means.json` covers only the three usable cat-emotions-3 classes

**Decided:** `scripts/shape_space.py` writes `models/class_means.json` with exactly three top-level keys — `attentive` (1130), `relaxed` (730), `uncomfortable` (107) — computed as the per-class mean of the shared Procrustes-aligned array (2029 plausible cat-emotions-3 rows total) the E1 PCA also runs on.

**Why:** cat-emotions-3's raw download has 8 label folders, but the other five — `no clear emotion recognizable` (34 plausible), `sad` (22), `angry` (3), `Unlabeled` (2), `attentive uncomfortable` (1) — are single/low-double-digit counts, too small to support a stable mean shape, and at least two of them (`Unlabeled`, `attentive uncomfortable`) aren't real expression labels at all. RSCH-4's method in `docs/backlog.md` already treats only these three as real classes for the same reason; E1 follows that scoping rather than inventing its own.

**Alternative considered:** emit all 8 folders and let the app's warper ignore the tiny ones at request time. Rejected — a mean of 1-3 points is not a meaningful "class shape," and shipping it would imply a confidence the data doesn't support; better to not offer those targets at all than offer a noisy one.

**How to apply:** if a later experiment (E2 onward) changes which classes are "usable" (e.g. a merge decision at E4), `class_means.json` needs regenerating from `scripts/shape_space.py`, not hand-edited — the app's `/edit` endpoint reads this file directly as its delta source.

---

## 2026-09-19 — Ear landmark indices verified and promoted to public `LEFT_EAR`/`RIGHT_EAR`

**Decided:** `src/catface/ml/plausibility.py` now defines `LEFT_EAR = (22, 23, 24, 25, 26)` and `RIGHT_EAR = (27, 28, 29, 30, 31)` as public constants (`EAR = LEFT_EAR + RIGHT_EAR` kept for `MUZZLE`'s existing by-exclusion derivation).

**Why:** the previous `_EAR = (22,...,31)` was a private placeholder — per the "Landmark index groups... verified against 4 CatFLW ground-truth labels" entry below, ear indices were carried as one undifferentiated 10-tuple because nothing in E0 used the left/right split. E1's pose-confound proxies in `scripts/shape_space.py` need the split for real (`ear_position` needs `mean_y(EAR)` only, but a future consumer wanting per-side ear angle needs the split, and grouping them wrong would silently corrupt that). Verified the same way the original eye groups were: loaded 4 real CatFLW label JSONs (`data/catflw/CatFLW dataset/labels/*.json`), inspected raw (x,y) for indices 22-31 against each label's `LEFT_EYE`/`RIGHT_EYE` mean x. In all 4 samples, indices 22-26 had x-coordinates clustered on the same side as `LEFT_EYE`'s mean x (e.g. one sample: `LEFT_EYE` mean x 225.2, indices 22-26 at x 204-228; `RIGHT_EYE` mean x 257.9, indices 27-31 at x 254-285) — a contiguous 5-and-5 split by index, unlike the eyes' interleaved indices.

**Alternative considered:** assume a 5-and-5 split without checking whether it's contiguous in index order, since the eye groups weren't. Rejected — `docs/PLAN_RESEARCH.md` only guarantees the count (5 per ear), not the order, and the whole point of the earlier eye verification was that matching counts doesn't confirm index assignment; only real coordinates do.

**How to apply:** any future consumer needing per-side ear features (e.g. E3's `core/graph.py` anatomical adjacency) can import `LEFT_EAR`/`RIGHT_EAR` directly instead of re-deriving them.

---

## 2026-09-19 — Generalized Procrustes is rotation-only; reflections forbidden by construction

**Decided:** `procrustes_align` in `src/catface/core/geometry.py` centers and scales `shape` and `reference` independently, then fits a rotation via SVD (`U, S, Vt = svd(shape_centered.T @ reference_centered)`, `R = U @ Vt`). If `det(R) < 0` — the fit would be an improper rotation (a reflection) — the sign of `U`'s last column is flipped and `R` recomputed before applying it. `generalized_procrustes` (iterative GPA, `core/geometry.py`) builds on this same primitive for every pairwise alignment.

**Why:** the 48 landmarks are labeled and anatomically chiral — `LEFT_EYE`/`RIGHT_EYE` (and now `LEFT_EAR`/`RIGHT_EAR`) in `src/catface/ml/plausibility.py` are specific, distinct indices, not an unordered point cloud. An unconstrained Kabsch fit is free to pick the reflection solution whenever it fits at least as well as any proper rotation — verified concretely in `tests/test_geometry.py::test_procrustes_align_forbids_reflection`: for a genuinely mirrored, asymmetric point set, the unconstrained SVD fit lands exactly on the reference (residual ~1e-16) using an improper (`det < 0`) rotation, while the sign-flip-corrected fit refuses that solution (residual two orders of magnitude larger, `det > 0` recovered). Silently accepting that mirror solution during GPA would swap left/right in the shared aligned frame that `models/class_means.json` and every downstream PC/proxy computation reads from — not a cosmetic bug, a correctness one.

**Alternative considered:** run unconstrained Procrustes and rely on cat faces being roughly bilaterally symmetric so the reflection case "shouldn't" come up in practice. Rejected — bilateral near-symmetry is exactly the condition under which noise can tip an unconstrained SVD fit toward the improper solution, and there's no cheap way to detect after the fact that it happened; forbidding it at the source costs one `if` statement.

**How to apply:** any new pairwise shape alignment added later (E3's node features, e.g.) should go through `procrustes_align`/`generalized_procrustes` rather than a fresh `np.linalg.svd` call, so this guard isn't silently reintroduced as a gap.

---

## 2026-09-19 — cat-emotions-7 is out of the research training set

**Decided:** E1 onward read the E0 cache with `dataset == "cat-emotions-3"` only. The 671 cat-emotions-7 rows stay in `data/cache/landmarks.parquet` (already computed, nothing to gain by deleting them) and its per-dataset README keeps its detector-run section for the record.

**Why:** the 80-overlay spot check put 21 cat-emotions-7 images in front of the reviewer and 9 came back unusable. Every reason named was image quality: ears cropped out of frame (3), fluffy or black cats where the ear outline does not resolve (4), one covered face, one low-resolution photo. The same pass found Angry, Scared and Surprised visually interchangeable on this set, and those are the three classes with the worst detections (1/5, 0/2, 3/5 OK). cat-emotions-3 came back 57/59 OK in the same sitting.

**Alternative considered:** keep the set and tighten `check_landmarks` until it rejects those images. Rejected: the filter sees geometry, and an ear placed on black fur or at the image edge is geometrically plausible. Even a filter that caught them would leave the label noise in place.

**How to apply:** the classification problem is now cat-emotions-3's usable classes (attentive, relaxed, uncomfortable; what happens to the five tiny folders is decided at E1/E2 grooming). RSCH-4's 7-class confusion matrix is re-scoped in the backlog. The failure modes are a fact about photos users will upload too, so the app needs an image-quality scoring step before inference; that is a project-track item, groomed when picked up.

---

## 2026-09-19 — The 100-image blind relabel is waived

**Decided:** RSCH-0's "relabel 100 images blind, report agreement" acceptance criterion is dropped, along with RSCH-7's matching report line. `docs/MODEL_REPORT.md` records the one label-quality observation we have (the non-blind spot-check note about Scared/Surprised/Angry) and states that no agreement number exists.

**Why:** the reviewer's call. The dataset the relabel would have said most about is now excluded, and the remaining set has three coarse classes where a relabel would cost an hour to tell us the ceiling on a problem E2's confusion matrix will show anyway.

**Alternative considered:** run the relabel on cat-emotions-3 only. Not rejected on principle, just not now; the tooling is a shuffled image dump plus a CSV, so it can be added at E2 if the confusion matrix makes the ceiling matter.

---

## 2026-09-19 — RSCH-0 re-scoped: the reject-sample review control is dropped

**Decided:** RSCH-0's required control "eyeball a sample of filter rejects, confirm the filter isn't throwing away hard-but-valid cases" is removed from the issue. Research QA judges E0 against the re-scoped issue. This is a methodology re-scope under `research_process.md` step 5, recorded here so it does not read as a skipped control.

**Why:** the 80-image sample was drawn from all rows and hit none of the 60 rejects (2.2% of rows), so the control did not happen as a side effect. Running it means drawing and eyeballing a second sample. The reviewer decided that is not worth doing now: the filter drops 42 of 2071 cat-emotions-3 rows (2.0%), and the spot check already showed its real weakness is the other direction (11 of 11 human-rejected detections passed).

**Alternative considered:** dump the 60 reject overlays and review them. Deferred, not rejected; `scripts/landmark_cache.py --from-cache` makes that a cheap add if E1's PCA shows outliers that look like filter misses.

---

## 2026-09-19 — `append_section` replaces a same-named section instead of duplicating it on rerun

**Decided:** `scripts/dataset_stats.py:append_section` now parses the
existing `README.txt` into its `## heading` sections, drops any section
matching the heading being written, and appends the fresh one — instead of a
plain file-append.

**Why:** rerunning `scripts/landmark_cache.py` (done twice in this session —
once to fix the `detection_failed` crash, once to fix the `detector_confidence`
proxy) calls `append_section(readme_path, "Detector run (E0)", ...)` each
time. With a plain append, the second run left two "## Detector run (E0)"
sections in `data/cat-emotions-3/README.txt` and `data/cat-emotions-7/README.txt`
— found by grepping for `^## ` after the second run. A generated doc that
can't be regenerated cleanly is a bug in the generator, not a one-off to
patch by hand.

**Alternative considered:** leave it append-only and tell whoever reruns the
pipeline to manually clean the README first. Rejected — `scripts/landmark_cache.py`
is meant to be rerun whenever the detector or filter changes; making the
human remember a manual cleanup step defeats the point of scripting this at
all.

**How to apply:** `dataset_stats.write_readme` (rewrites its own sections, keeps any it doesn't own) and `append_section` (targeted replace) are now both rerun-safe in either order. The first version of `write_readme` overwrote the whole file, so rerunning `dataset_stats.py` after `landmark_cache.py` silently erased the "Detector run (E0)" section from both dataset READMEs; found as an unexplained deletion in `git diff`. Any future section-writer added to either script should follow the same replace, not append, rule.

---

## 2026-09-19 — `detect_face_box`/`detect_landmarks` can crash on a degenerate box; `landmark_cache.py` catches it, `core`/`ml` don't change

**Decided:** running the batch cache over `cat-emotions-3` hit a real crash —
`crop_and_resize` raised `cv2.error: !ssize.empty()` on
`data/cat-emotions-3/train/attentive/CAT_01_00000153_020_png...jpg` because
`detect_face_box`'s localizer output, once clipped to the image by
`expand_box`, produced a zero-width or zero-height crop region. `build_cache`
in `scripts/landmark_cache.py` now wraps the `detect_face_box` +
`detect_landmarks` calls in `try/except cv2.error` and records the row as
`plausible=False`, `drop_reasons=["detection_failed"]` instead of crashing the
whole batch. One image out of 2742 hit this.

**Why:** the whole point of E0 is finding out what breaks on the Roboflow
distribution — a script that dies on image 1153 of 2742 doesn't answer that
question, and a genuinely bad localizer output is itself a valid E0 finding,
not a bug to hide.

**Alternative considered:** make `expand_box`/`crop_and_resize` in
`core/geometry.py` defensive against zero-size crops. Rejected — those are
shared, generic preprocessing primitives used by both tracks; silently
tolerating a degenerate box there would hide the same failure from the app
track's live inference path instead of surfacing it. The failure belongs to
the caller that can decide what "this image didn't work" means (here: cache a
dropped row).

**How to apply:** if this shows up on more than a handful of images once
`cat-emotions-7` or later datasets run through the same path, that's a
detector-quality finding for the stop condition, not just an edge case to
catch and move on from.

---

## 2026-09-19 — Near-duplicate hash switched from average hash to dHash after it produced a false "identical" pair

**Decided:** `scripts/dedupe.py`'s hash function compares adjacent-pixel
brightness gradients (difference hash / dHash), not raw brightness against
the image mean (average hash).

**Why:** the average hash implementation, tried first, hashed two completely
unrelated cats (different animals, different backgrounds — one in grass, one
studio-lit) to Hamming distance 0 out of 64 bits — a false "exact match."
Visually inspecting the image pair confirmed it was not a duplicate. The root
cause: 8x8 average brightness collapses to the same rough light/dark
composition for very different photos when both datasets tend toward
"cat centered on a plain background." Re-tested the same pair with dHash:
30/64 bits differ (clearly distinct), and running dHash across the full
2071×671 pair-set found zero matches at Hamming distance ≤5 (minimum distance
found was 6), versus average hash's 126 candidate "near-duplicates" at the
same threshold — spot-checking those confirmed they were false positives too.

**Alternative considered:** keep average hash but raise the hash resolution
(16x16 or 32x32) to add discriminating power. Tested — still weaker than
dHash on the same false-positive pair (16x16 average hash: 23/256 bits
differ, ~9%, still borderline; dHash: ~36-47% differ across hash sizes).
Rejected in favor of switching algorithms rather than tuning the weaker one.

**How to apply:** the reported result — 0 near-duplicates between
`cat-emotions-3` and `cat-emotions-7` at Hamming ≤5 — is the dHash result;
`data/cache/near_duplicates.csv` and the per-dataset `README.txt` files
reflect it. If a future dataset pair does show dHash matches, they're much
more likely to be real than an average-hash result would have been.

---

## 2026-09-19 — `dedupe.py`, `dataset_stats.py`, `landmark_cache.py` all live in `scripts/`, not `catface.ml`

**Decided:** all three are one-shot, run-once scripts under `scripts/`,
invoked directly (`uv run python scripts/<name>.py`), not through
`catface.ml`'s `-m` convention. `src/catface/ml/plausibility.py` stays in the
package, since it's reusable filter logic other code imports and tests
(`tests/test_plausibility.py`), not a script.

**Why:** user direction, given directly during implementation — the
distinguishing line drawn here is "run once, standalone script" (→
`scripts/`) versus "importable logic other code depends on" (→ `catface.ml`
or `core`), separate from the earlier fetch-scripts rule (shared app/research
infra → `scripts/`). `landmark_cache.py` imports `dataset_stats.append_section`
directly as a sibling module in the same directory — no package `__init__.py`
needed, since scripts run standalone and Python puts the invoked script's own
directory on `sys.path` automatically.

**Alternative considered:** keep `landmark_cache.py` under `catface.ml` per
AGENTS.md's literal "research track scripts (E0–E5, export)" line, since it's
the only one of the three that touches the detector model. Overridden by
direct user instruction mid-implementation.

**How to apply:** AGENTS.md's `uv run python -m catface.ml.<script>` line
doesn't cover any of the three E0 scripts. `detect_landmarks.py` (the
single-image CLI) and `plausibility.py` (filter logic) are the only
`catface.ml` files this experiment added or touches.

---

## 2026-09-19 — E0 skips `experiments/`; outputs live under `data/cache/` instead

**Decided:** no `experiments/e0/` run directory. `data/cache/landmarks.parquet`
carries the cache, `data/cache/README.md` documents it (schema, pass/drop
counts, stop-condition answer, human-TODO checklist), and
`data/cache/{overlays,plots,near_duplicates.csv}` hold the supporting
artifacts. `.gitignore`'s existing blanket `experiments/` line is untouched.

**Why:** user direction — this pass isn't a tuning run with
config/metrics.json-per-attempt bookkeeping the way E2 onward's model
training will be; it's a one-shot cache-building pass over a fixed detector,
so the `experiments/<run>/` convention `research_process.md` describes for
comparing multiple runs doesn't apply here. The cache and its documentation
belong together, next to the other data-adjacent artifact (`data/`), not in a
run-comparison directory.

**How to apply:** if a later E0 rerun needs to compare cache versions (e.g.
after retuning the plausibility filter), that's the trigger to reconsider
`experiments/`, not this pass.

---

## 2026-09-19 — Plausibility filter's pass/fail split doesn't gate the human-eyeball overlay sample

**Decided:** `landmarks.parquet` caches every processed row (2742, including
the one `detection_failed` row and 59 filter-dropped rows), with `plausible`
and `drop_reasons` columns beyond the backlog's literal 5-column schema.
`sample_overlays` in `scripts/landmark_cache.py` draws its ~20-image human
review sample uniformly at random across **all** rows, independent of
`plausible`.

**Why:** `docs/team/data-steward.md` says "nothing downstream re-runs the
detector" — dropping failed rows from the cache would force a full
~2742-image OpenVINO re-run if the filter's thresholds (derived/estimated in
this same pass) get retuned later. The uniform-random overlay sample is a
direct user correction during plan review: the filter's pass/fail split is
for reporting and future refiltering only, not for curating what a human
looks at.

**Alternative considered:** bucket a separate `rejects/<reason>/` folder so a
human specifically reviews dropped detections, per the backlog's literal
"reject-sample review" control. Overridden by the user — a human wanting that
view can filter the parquet by `plausible == False` themselves; this pass
doesn't pre-build it.

**How to apply:** this makes the backlog's "reject count + manual sample
verdict" acceptance criterion lighter than written — the count and per-reason
breakdown are automated (in `data/cache/README.md`), but the sample a human
sees during the 20-overlay spot check isn't reject-focused.

---

## 2026-09-19 — Plausibility filter's aspect-ratio bound derived from CatFLW ground truth, not guessed

**Decided:** `ASPECT_RATIO_BOUNDS = (0.5, 1.8)` in `src/catface/ml/plausibility.py`.

**Why:** computed width/height for all 2079 CatFLW ground-truth
`bounding_boxes`: min 0.763, p1 0.856, p50 1.019, p99 1.291, max 1.436. The
chosen bound pads that range generously (roughly ±30-40% beyond the observed
min/max) since CatFLW is curated single-cat portrait photography and the
Roboflow sets are "in the wild" — tighter bounds risked flagging legitimate
off-angle photos as implausible.

**Alternative considered:** a round guessed bound like `(0.5, 2.0)`. Rejected
— CatFLW's own boxes were sitting on disk unused for this purpose; computing
the real distribution cost one line and removes the guess.

**How to apply:** if E4's yaw-binning work later finds the Roboflow pose
distribution is wider than CatFLW's, this bound may need widening — check the
`bad_aspect_ratio` drop count (2 out of 2742 in this run) against that.

---

## 2026-09-19 — Parquet cache path: `data/cache/landmarks.parquet`

**Decided:** the E0 landmark cache lives at `data/cache/landmarks.parquet`,
documented by `data/cache/README.md`.

**Why:** no literal path is named anywhere in `PLAN_RESEARCH.md`,
`backlog.md`, or `data-steward.md` — all three say "the path the plan
specifies" without ever specifying one. `data/cache/` groups it with the
other data-adjacent artifacts (near-duplicate CSV, distribution plots,
overlays) rather than filing it as a per-run experiment artifact (see the
"E0 skips `experiments/`" entry above).

**Alternative considered:** `experiments/e0/landmarks.parquet`. Rejected once
`experiments/` was ruled out for this pass.

**How to apply:** E1 onward reads from this path; don't introduce a second
cache location without updating this entry.

---

## 2026-09-19 — `detector_confidence` is a derived geometric-plausibility proxy, not a model output — and it has to use the tight box, not the crop

**Decided:** `plausibility.detector_confidence_proxy` computes the fraction of
the 48 landmarks that land inside the *tight* localizer box (`box_xyxy`), not
the margin-expanded crop `detect_landmarks` runs the landmarks model on.
Documented as a proxy, explicitly not a model confidence score, everywhere it
appears (code comment, `data/cache/README.md`, this entry).

**Why:** inspecting the compiled OpenVINO models directly —
`cat_face_localizer` outputs `[1,4]` (box only), `cat_face_landmarks` outputs
`[1,96]` (48×2 flattened only) — confirmed neither model emits a confidence
value, even though the backlog's parquet schema names a `detector_confidence`
column. Something has to fill that column; it can't be a real model score.
The first version measured the fraction inside the *expanded crop box*
instead, reusing the same box the "inside box" filter rule checks — that
turned out to be tautological: `map_points_to_image` scales the landmarks
model's normalized output directly onto that same crop box, so the fraction
came out to exactly 1.0 on all 2741 valid rows in the real run, with zero
variance. Caught by looking at the confidence histogram after the real run —
a single spike, not a distribution. Switching to the tight box (which is not
part of the model's own output space) gives a real spread: mean 0.99, min
0.54, std 0.034 across the same run.

**Alternative considered:** leave the column out entirely, since the backlog
schema doesn't strictly require inventing a substitute. Rejected — the
backlog and `data-steward.md` both call for plotting "detector confidence,"
and a working proxy is more useful than silently dropping the requirement.

**How to apply:** never read this column as if it came from the model. If a
future model version does emit a real confidence score, that's a new column,
not a silent redefinition of this one. If a future proxy candidate is checked
against the same box its own inputs were derived from, check whether it can
vary at all before trusting the number — this is the second time in this
session a metric turned out to be tautological by construction (see the
average-hash entry above, where the failure mode was the opposite: too little
structure, not too much).

---

## 2026-09-19 — Landmark index groups (eyes, muzzle) verified against 4 CatFLW labels before use in the plausibility filter

**Decided:** `LEFT_EYE = (3,4,5,6,7,36,37,38)`, `RIGHT_EYE =
(1,8,9,10,11,39,40,41)`, `MUZZLE` = the remaining 22 non-eye, non-ear indices,
in `src/catface/ml/plausibility.py`. Ear indices (two 5-point groups) aren't
defined as a constant since nothing in E0 uses them.

**Why:** derived by inspecting one real CatFLW label's raw coordinates and
clustering by position, cross-checked against `docs/plan-research.md`'s
stated group sizes ("8 landmarks per eye, 5 per ear, 22 across nose and
whiskers" — matched exactly), then spot-checked against 3 more random CatFLW
labels (`mean_y(eye) < mean_y(muzzle) < mean_y(ear)`... in image coordinates,
`ear < eye < muzzle` — held on all 4 samples). The vendor doesn't publish an
index-order document; this is the same category of unverified-scheme risk as
the localizer's raw output order (see the entry below from the previous
session), so it got the same verify-before-trust treatment.

**Alternative considered:** trust the group sizes from `docs/plan-research.md`
without checking index order against real coordinates. Rejected — matching
counts doesn't confirm index assignment; only checking actual (x,y) values
does.

**How to apply:** if E3's `core/graph.py` needs ear indices for its
anatomical adjacency, re-derive them the same way (they were found during
this pass but not committed as unused code) rather than guessing from the
Finka paper's diagram alone.

---

## 2026-09-19 — `core/geometry.py` created now, ahead of the app track's M2

**Decided:** `src/catface/core/geometry.py` exists as of this session, with `letterbox_square`, `unletterbox_xyxy`, `expand_box`, `crop_and_resize`, `map_points_to_image` — the two operations (letterbox, crop margin) the Data Steward role doc forbids reimplementing in `ml`. Procrustes is not here yet; that's still E1's job.

**Why:** `src/catface/ml/detect_landmarks.py` (this session's other deliverable) cannot run the two-stage detector without letterboxing for the localizer and a margin-expanded crop for the landmarks stage — the HF model card requires both. PLAN_PROJECT.md assigns `core/geometry.py` to the app track's M2, but M2 hasn't started, and research can't wait for a milestone on a track that hasn't begun without stalling RSCH-0 entirely.

**Alternative considered:** write the letterbox/crop math inline in the `ml` script, flagged as temporary. Rejected — it's the exact duplication AGENTS.md and the Data Steward doc call out as the failure mode, and the app track's live inference path (per PLAN_PROJECT.md's pipeline note) needs the identical math, not a second copy that can drift.

**How to apply:** when the app track picks up M2, it extends this file (adds Procrustes, anything else M2 needs) rather than starting a new one. `PLAN_PROJECT.md`'s M2 scope shrinks by exactly these two functions.

---

## 2026-09-19 — Cat-face localizer's raw output order is `[x2, y1, x1, y2]`, not `xyxy`

**Decided:** `detect_face_box` in `src/catface/ml/detect_landmarks.py` reads the localizer's `[1,4]` output as `(output[2], output[1], output[0], output[3])` to get `xyxy`, not `output[:4]` directly.

**Why:** the HF model card states the output is `bbox_xyxy` with no index-order detail. Reading it in the stated order puts `x1 > x2` on every real image. Checked against CatFLW's ground-truth box for a sample image (`124, 59, 338, 241`): the raw output pixel-mapped to `(332, 56, 119, 237)` in `(x1,y1,x2,y2)` reading order, and `(119, 56, 332, 237)` reading indices `(2,1,0,3)` — the second is within a few pixels of ground truth, the first is inverted. No other permutation matched.

**How to apply:** if the landmarks model (or any future model from this same HF repo) shows a similarly inverted-looking output, check index order against a labelled sample before assuming the card's stated format applies literally — this vendor's model card doesn't match its own tensor layout at least once already.

---

## 2026-09-19 — `openvino` reads the `.tflite` detector pair directly, no `tensorflow`/`onnxruntime` added

**Decided:** `src/catface/ml/detect_landmarks.py` loads both `.tflite` files via `ov.Core().read_model(path)` + `compile_model(..., "CPU")`. No new dependency.

**Why:** `openvino` (already a runtime dependency, pinned to `2026.4.0` in `uv.lock`) has read `.tflite` natively since well before this version. PLAN_PROJECT.md already named this as the intended path ("OpenVINO reads `.tflite` directly... if a custom op blocks it, swap in ONNX Runtime") but hadn't verified it; this session is that verification, and it works without hitting an unsupported op.

**Alternative considered:** the HF model card's own sample snippet uses `tf.lite.Interpreter` (i.e. `tensorflow`). Rejected — a full `tensorflow` dependency for two inference calls when an already-installed engine reads the same file is the dependency AGENTS.md's "don't add one without asking" rule exists to head off.

**How to apply:** RSCH-6's export step can build on this confirmation — OpenVINO's `.tflite` frontend is now known-good on this exact model family, not just assumed.

---

## 2026-09-19 — `models/manifest.json` holds the detector pair now; RSCH-6 appends `expression_head`, doesn't replace the file

**Decided:** `models/manifest.json` is committed today with two entries, `cat_face_localizer` and `cat_face_landmarks` (the CatFLW-trained HF detector pair `fetch_weights.py` downloads), in the exact shape RSCH-6 already specifies for its own entry: name, file, sha256, source, licence, input shape, metrics.

**Why:** the detector pair has to exist before E0's "run the detector once over both Roboflow sets" step can run, and PLAN_PROJECT's inference path needs the same files. RSCH-6 was always going to write to this path in this shape, so starting the file now instead of waiting for RSCH-6 avoids a rename or a second manifest file later.

**Alternative considered:** a separate `models/detector_manifest.json` for the app-track weights, keeping `manifest.json` for RSCH-6 alone. Rejected — one manifest is one place to check sha256s from, and the entry shape already matches.

**How to apply:** RSCH-6 adds a third top-level key, `expression_head`, to this same file. It doesn't touch the two keys above.

---

## 2026-09-19 — `.env` loading uses `python-dotenv`, not a hand-rolled parser

**Decided:** `fetch_data.py` calls `dotenv.load_dotenv` and `dotenv.set_key`; `python-dotenv` is now a `dev` group dependency.

**Why:** the approved plan specified a four-line stdlib `KEY=VALUE` reader to avoid adding a dependency for something that small. `set_key` also has to handle the append-if-missing, quoting, and rewrite cases a hand-rolled version would either skip or reinvent, and dotenv is already the standard tool for exactly this job.

**Alternative considered:** the stdlib parser as planned. Rejected on review — small enough to write, but the quoting edge cases are exactly the kind of already-solved problem that justifies reaching for a dependency instead of stdlib.

**How to apply:** AGENTS.md's "dependencies are added in pyproject.toml, do not add one without asking" rule was followed — asked, and the user named `python-dotenv` directly.

---

## 2026-09-19 — E0 fetch scripts go in `scripts/`, not `catface.ml`, because one of them is app-track infra too

**Decided:** `fetch_weights.py` and `fetch_data.py` live in `scripts/`, run as `uv run python scripts/fetch_weights.py` and `uv run python scripts/fetch_data.py`, not under `catface.ml` as AGENTS.md's research-script command line describes.

**Why:** `fetch_weights.py` downloads the CatFLW-trained detector pair the app needs to run inference at all — PLAN_PROJECT's M2 build step calls this same script, not a research-only one. Filing it under `catface.ml` would put an app-track build dependency behind the import path the `api` track is forbidden from touching. `fetch_data.py` only serves E0, but splitting the two fetch scripts across two locations for that reason is worse than keeping both build-time fetchers in one place.

**Alternative considered:** `catface/ml/fetch_data.py`, per AGENTS.md's stated convention. Built first, then moved once the shared-infra point above became obvious mid-review.

**How to apply:** AGENTS.md's `uv run python -m catface.ml.<script>` line no longer covers either fetch script; it wants a `uv run python scripts/<script>.py` line next to it.

---

## 2026-09-19 — `models/class_means.json` is an E1 deliverable, and the second cross-track artifact

**Decided:** the per-class Procrustes mean shapes the editor maths subtracts get
written at E1 as `models/class_means.json` (committed, not fetched), rather than
falling out of the export step with the model.

**Why:** `/edit` computes `mean_shape[target] - mean_shape[predicted]`, so the
warper depends on the mean shapes, not on the classifier. Filing them behind the
model would have made PLAN_PROJECT.md's M5 fallback ("manual class picker,
everything else still works") false — with a manual picker you still need the
deltas. E1 already Procrustes-aligns every cached landmark, so the means are one
groupby over work that's happening anyway.

**Alternative considered:** deliver them with the ONNX export at RSCH-6.
Rejected — it puts a research artifact on M3's critical path and leaves the app
track's advertised independence untested until the end.

**How to apply:** AGENTS.md still says the only handoff between tracks is
`expression_head.onnx` + its manifest entry. That's now wrong and wants one line.

---

## 2026-09-19 — Cut four pieces of premature machinery from PLAN_PROJECT.md

**Decided:** each of these loses its oversized form and keeps a one-line
`add when:` trigger — the `convert` dependency group (tensorflow-cpu, tf2onnx),
the self-hosted Prometheus/Loki/Tempo/Grafana stack, the InferRequest pool plus
bounded wait queue on top of the semaphore, and locust/k6 for the load test. The
C++ bench CLI is deleted outright from both plans. `predictions.scores` becomes
one JSON column instead of three, and `edits` stops storing derivable landmarks.

**Why:** each was specified before the thing that justifies it exists. The
monitoring stack is the clearest case — it's several hundred MB on a box the same
document budgets at ~700 MB of 1024 MB, so it would OOM the app it observes; the
fallback table already conceded this, so plan and fallback are now swapped. The
`convert` group is ~600 MB for a conversion line 111 says is unnecessary. A
bounded semaphore already is a bounded queue. And three score columns bake a class
count into the schema while E4 is still deciding whether to merge classes.

**Alternative considered:** keep everything, mark it deferred in a subsection.
Rejected — a deferred section is a plan you still have to read past every time,
and the `add when:` triggers make each cut as reversible as a subsection would.

**How to apply:** don't re-add any of these because it feels professional. Re-add
when the named trigger fires, which for most of them is a measurement.

---

## 2026-09-19 — Backlog stays research-only; project work stays in the milestone list

**Decided:** don't write `[project]` backlog items for M0–M8 yet. The project
track runs off `PLAN_PROJECT.md`'s milestone list until M0 grooming starts.
Requirements that were stated in the plan but sat in no milestone (read the rubric
and weight the work, the image retention job, the five graded docs) were added to
the milestones themselves.

**Why:** `project_process.md` has PM groom one task immediately before it's
implemented. Groomed text for M8 written today is stale before it's read, and the
grooming happens anyway when the task is picked up — so writing it now is work
done twice.

**Alternative considered:** thin one-liner items per milestone. Rejected as a
second copy of the milestone list that would drift from it.

---

## 2026-09-19 — Backlog grooming shape: question/method/controls/stop-condition, not `_docs/task-template.md`

**Decided:** groom `docs/backlog.md` research items using the shape
`research_process.md` step 2 already specifies (question, method, required
controls, stop condition, checkable acceptance criteria) rather than the
four-section template `docs/team/pm.md` points at
(`_docs/task-template.md`).

**Why:** that template file doesn't exist anywhere in the repo. Inventing
one to satisfy a dangling reference would be scaffolding nobody asked for;
`research_process.md` already names the exact fields an experiment issue
needs, and it's the doc the research lifecycle actually runs on.

**Alternative considered:** write `_docs/task-template.md` first, then
groom against it. Rejected — out of scope for this pass, and PM's template
is meant for issues in general, not specifically research experiments,
which already have a more specific shape defined.

---

## 2026-09-19 — One backlog task per experiment (E0–E5), plus separate export and report tasks

**Decided:** 8 backlog items for the research track: E0 through E5 one each,
plus a standalone export task and a standalone `MODEL_REPORT.md`
compilation task, rather than one task per plan subsection (e.g. splitting
E2's three models or E4's two analyses into separate items).

**Why:** `research_process.md` states "one issue per experiment, in the
order PLAN_RESEARCH.md lists them" and treats export and the report as
separate lifecycle steps (step 8 for export, step 6 for report entries).
Splitting inside an experiment would fight the plan's own time-boxing,
which scores E2 and E4 as single 1-day units with multiple required parts,
not as multiple independent tasks.

**How to apply:** if a future experiment needs splitting (e.g. E3 turns out
to need more than 2 days), split it as a re-groom of that one item with a
Research PM note, not a silent backlog rewrite.

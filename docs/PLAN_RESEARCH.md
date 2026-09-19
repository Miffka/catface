# PLAN_RESEARCH.md

Research notes for the model side. Open questions, not a spec. Shares the repo,
`pyproject.toml`, and `catface.core` with PLAN_PROJECT.md.

Delivers two artifacts:

- `models/expression_head.onnx` — classifies cat facial expression from 48
  landmarks. If it turns out badly, the app still ships the landmark viewer, the
  derived geometry readouts, and the warper, none of which depend on it.
- `models/class_means.json` — per-class Procrustes mean shapes, out of E1. The
  app's `/edit` endpoint *does* depend on this one, so it ships early and
  separately from the model. See "Editor maths".

## Setup

`src/catface/ml/`: data loaders, feature code, `mlp.py`, `gcn.py`,
train/evaluate/export. May import `core`. Never imported by `api`.

`uv sync --group train --group dev`.

Preprocessing comes from `core.geometry`. Never reimplement letterboxing, the crop
margin, or Procrustes here. That's the classic route to a model that works in the
notebook and fails in prod.

`core.geometry` is delivered by the app track's M2, and research starts at M4, so
it exists before E1 needs it. If the order ever slips, E1 blocks on it rather than
growing a local copy — the duplication is the failure mode, not the delay.

One directory per run under `experiments/`: config, metrics.json, git sha, notes.

## Two kinds of output, don't confuse them

**Derived geometry — formulas, no learning, no labels.** The 48-point scheme is
8 landmarks per eye, 5 per ear, 22 across nose and whiskers.

- *Eye aperture*: vertical eyelid distance over corner-to-corner distance. Eight
  points per eye is an eyelid contour, so this is just the eye aspect ratio.
- *Ear rotation*: angle of the ear base-to-tip vector against the inter-ocular axis.
  The dog work on this landmark scheme found ear-base points separated conditions
  most strongly, so there is real signal here.

These go straight into the UI as interpretable readouts, and they are the baseline
the learned model has to beat.

There is no muzzle tension here. You can compute whisker-pad spread and mouth-corner
positions from the lower-face points, but "tension" has no label source anywhere, so
it stays out.

**Learned classification — needs labels, this is the actual research.** Expression
class from the full landmark configuration.

## Data

| Source | Use | Licence |
|---|---|---|
| Roboflow cat-emotions-cgrxv, 2,071 imgs, 3 classes (angry / attentive / no clear emotion) | **primary** training set | check project page |
| Roboflow cat-emotions, 671 imgs, 7 classes | secondary experiment | CC BY 4.0 |
| CatFLW, 2,016 imgs, 48 landmarks, no class labels | shape prior, mean shape, detector sanity check | CC BY-NC 4.0 |
| My own cat photos | qualitative test set, demo material | mine |

3 classes at ~690 each beats 7 classes at ~96 each. Start there.

Honest framing for the README: these are human-assigned *perceived* emotion labels
on internet photos, not validated affective states. State it as a limitation and
move on. It doesn't invalidate the work.

### Getting landmarks onto the labelled images

Run the detector over both Roboflow sets once, cache to a parquet file
(`image_id, class, 48x2 landmarks, face_box, detector_confidence`). Everything
downstream reads that cache, so the detector runs once, not per experiment.

Filter implausible detections before training: points inside the box, eyes above the
muzzle, no degenerate or collapsed configurations, no absurd aspect ratios. Report
how many were dropped and eyeball a sample of the rejects to check the filter isn't
throwing away hard-but-valid cases.

Two things to check before combining datasets:
- **Near-duplicates across the two Roboflow sets.** Hash the images. They may share
  sources, and overlap across a train/test split would inflate every number.
- **Actual class balance after download**, not the advertised one.

Cheap sanity check worth doing: relabel 100 images yourself blind, compare to the
dataset labels, report the agreement. That number is roughly your accuracy ceiling
and it costs an hour.

## Questions

- **Q1** Does a graph conv over the anatomical adjacency beat an MLP on flattened
  coordinates? Prior: maybe not at this size. The published cat pipeline flattens
  landmarks into an autoencoder plus XGBoost, so message passing here is untried
  rather than settled.
- **Q2** Does the learned model beat two geometric ratios (eye aperture, ear angle)
  fed to logistic regression? If not, the honest headline is that cat expression
  classification from landmarks is mostly ear angle.
- **Q3** How much does head pose contaminate the prediction? Pose and expression are
  entangled, and internet photos vary wildly in angle.
- **Q4** Are all classes separable, or do some collapse?
- **Q5** Does anything transfer to my own cat's photos?

## Experiments

### E0 landmark cache (0.5 day)
Run the detector over both sets, filter, cache. Plot class distribution and
detector confidence. Look at 20 random overlays to confirm the detector works on
this image distribution — it was trained on CatFLW, and Roboflow photos may differ.

Stop condition: if the detector fails on most of these images, the whole plan
changes, and better to find out on day one.

### E1 shape space (0.5 day)
Procrustes over the cached landmarks, PCA, plot the first six components. Check
whether any component visibly tracks ear position or head yaw, and whether the
classes separate at all in the first few PCs.

This is the cheap look at whether the signal exists before training anything.

Also write `models/class_means.json` here: the per-class mean of the aligned
landmarks, which is one groupby over the alignment E1 already computes. The app's
warper blocks on this file and not on the model, so producing it on day two of
research instead of at export time is what makes PLAN_PROJECT.md's M5 fallback
("manual class picker, everything else still works") actually true.

### E2 baselines (1 day)
Same splits for all of these, stratified by class:
1. Logistic regression on two derived ratios (eye aperture, ear angle)
2. Logistic regression on all Procrustes-aligned coordinates
3. MLP, two hidden layers, same input

Report macro F1 and a confusion matrix, 5-fold. Balanced class weights.

If (1) matches (3), that's the Q2 answer and it's a legitimate finding.

### E3 the GNN (1–2 days)
Dense graph conv by hand: `H' = act(A_hat @ H @ W)`, three layers, mean pool, linear
head. About 40 lines.

**No PyTorch Geometric.** Its sparse scatter ops are exactly what breaks ONNX export
(`scatter_reduce` with `include_self=False` gets rejected, and models that do export
crash on `ScatterElements`). On a fixed 48-node graph the dense form is identical
maths and exports to plain matmuls.

**Export the untrained model before training it.** Thirty minutes, and it tests the
assumption the whole architecture choice rests on. Finding an export problem here
means changing the architecture; finding it at export time, after E5, means
retraining with a deadline overhead.

Adjacency hand-written from anatomy in `core/graph.py`: eyelid contours as rings,
ear base to ear tip, whisker pads to nose, nose to mouth, eyes to ear bases.
Normalise `D^-1/2 (A+I) D^-1/2`. Node features: aligned `(x,y)` plus offset from the
mean shape.

Q1 counts as yes only if the GNN beats the MLP by more than one CV standard
deviation **and** the anatomical adjacency beats a random adjacency with the same
edge count. If random does as well, the structure carries nothing. Report it.

### E4 class structure and pose (1 day)
- Confusion matrix on the 7-class set. Merge classes that are inseparable and say
  why. Expect trouble telling disgusted from surprised.
- Estimate head yaw from landmark asymmetry, bin it, and report accuracy per bin.
  Output should be a sentence like "stable under 25 degrees of yaw, degrades past
  that", which becomes a confidence penalty in the app.

### E5 my cat (0.5 day)
Run the final model over my own photos. Qualitative only: does it behave sensibly,
where does it fail. Three failure images with explanations go in the report.

## Editor maths

Falls out of the class labels directly, no extra work:

```
align all training landmarks (Procrustes)
mean_shape[c] = mean of aligned landmarks for class c
delta = mean_shape[target] - mean_shape[predicted]
edited = user_landmarks + intensity * delta
```

The first two lines are E1's job and their output is `models/class_means.json`
(class name → 48×2 aligned mean). The last two run in the app, in `core`. The app
never recomputes a mean shape at request time.

Un-align back to image coordinates, triangulate, warp. This keeps the cat's identity
and moves only the expression, because the delta is a class difference rather than a
move toward a generic average cat.

Ears stay read-only in v1: a 2D warp can't do the out-of-plane rotation that real ear
flattening involves, and it will smear fur and leave holes.

## Export

`torch.onnx.export`, opset 17, static shapes, input `[1,48,2]`. Check ONNX matches
PyTorch within 1e-5, then `ov.convert_model` and check again. Benchmark p50/p95 and
peak RSS on the deploy CPU. Write the manifest entry with sha256 and metrics.

A script, not notebook cells. Runs in CI on every model change. E3 has already run
the export path once on an untrained model, so this should hold no surprises.

## Report

`docs/MODEL_REPORT.md`: dataset table with licences and the near-duplicate check,
label quality from the 100-image agreement test, results for all four models plus the
random-adjacency ablation, answers to Q1–Q5 including the negative ones, confusion
matrices, three failure photos, limitations (perceived-emotion labels, internet
photos, no clinical validity, nothing about pain or welfare), citations to CatFLW
and the Finka landmark scheme.

Negative results with error bars beat a headline number without them.

## Time boxes

E0 0.5d · E1 0.5d · E2 1d · E3 2d · E4 1d · E5 0.5d. Five and a half days, running
alongside app milestones from M4 on.

If the app is behind, research stops. The app is what gets graded.

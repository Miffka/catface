# MODEL_REPORT.md

## Datasets

| Source | Images | Classes | Licence | Used for |
|---|---|---|---|---|
| Roboflow cat-emotions-cgrxv (`data/cat-emotions-3`) | 2071, train only | advertised 3, actual 8 label folders | CC BY 4.0 | classifier training |
| Roboflow cat-emotions (`data/cat-emotions-7`) | 671 (train 502, valid 169) | 7, as advertised | CC BY 4.0 | excluded from training, see spot check |
| CatFLW (`data/catflw`) | advertised 2016, actual 2079, 48 landmarks each | none | CC BY-NC 4.0 | landmark detector (upstream training set), shape prior, detector sanity check |

Licence URLs, confirmation sources and per-class counts are in each `data/*/README.txt`.

cat-emotions-3 folders: attentive 1147, relaxed 752, uncomfortable 107, no clear emotion recognizable 37, sad 22, angry 3, Unlabeled 2, attentive uncomfortable 1. Three classes are usable; the five small folders are an E1/E2 grooming question.

Near-duplicates between the two Roboflow sets: 0 pairs at dHash Hamming distance 5 or less (closest pair: 6). `data/cache/near_duplicates.csv`.

Labels are the uploader's perceived emotion. No blind relabel was run, so no inter-rater agreement number exists (`docs/DECISIONS.md`, 2026-09-19).

## Models

Two-stage cat face landmark detector from [hugocornellier/cat-face-landmarks](https://huggingface.co/hugocornellier/cat-face-landmarks) (CC BY-NC 4.0, trained on CatFLW), loaded from `.tflite` via OpenVINO. Files and sha256s in `models/manifest.json`.

| Model | File | Input | Reported metric |
|---|---|---|---|
| Face localizer | [cat_face_localizer.tflite](https://huggingface.co/hugocornellier/cat-face-landmarks/resolve/main/cat_face_localizer.tflite) | 1x224x224x3, letterboxed | none published |
| 48-point landmarks | [cat_face_landmarks_full.tflite](https://huggingface.co/hugocornellier/cat-face-landmarks/resolve/main/cat_face_landmarks_full.tflite) | 1x384x384x3, 0.1-margin crop | NME/IOD 3.48 on the CatFLW validation split |

Neither model emits a confidence score. `detector_confidence` in the cache is a geometric proxy: the fraction of the 48 landmarks inside the tight localizer box (`src/catface/ml/plausibility.py`).

## Detector metrics on the Roboflow sets

Cache: `data/cache/landmarks.parquet`, one row per image, every row kept whether or not it passed the plausibility filter (points inside box, eyes above muzzle, no degenerate shape, sane aspect ratio). Schema in `data/cache/README.md`.

| Dataset | Processed | Plausible | Dropped | Drop reasons (an image can fail more than one) |
|---|---|---|---|---|
| cat-emotions-3 | 2071 | 2029 (98.0%) | 42 | eyes_below_muzzle 21, degenerate 20, detection_failed 1, bad_aspect_ratio 1 |
| cat-emotions-7 | 671 | 653 (97.3%) | 18 | eyes_below_muzzle 9, degenerate 9, bad_aspect_ratio 1 |

Stop condition for E0 (detector fails on most images, defined as plausible rate under 50%): NOT triggered.

## 80-overlay spot check

80 images sampled uniformly from all rows with landmarks, `df[df["landmarks"].notna()].sample(80, random_state=0).reset_index(drop=True)`; overlays in `data/cache/overlays/`, one human verdict per image in `overlays/overlay.csv` (join on the trailing index in the filename).

| Dataset | Reviewed | OK | Not OK, with reasons |
|---|---|---|---|
| cat-emotions-3 | 59 | 57 | 2: head turned, mouth open |
| cat-emotions-7 | 21 | 12 | 9: ears out of frame or undetectable 6, face covered 1, mouth open with ears down 1, low quality 1 |
| total | 80 | 69 (86%) | 11 |

cat-emotions-7 per class: Angry 1/5 OK, Scared 0/2, Surprised 3/5, Happy 3/4, Normal 4/4, Disgusted 1/1. The reviewer could not tell Angry, Scared and Surprised apart by eye on this set. Two more images (one per dataset) passed with the note "mouth open"; that pose is where the detector is least stable.

All 11 rejected detections have `plausible=True`. The confidence proxy separates them weakly: mean 0.96 on rejected vs 0.99 on accepted, ranges overlap (0.90 to 1.0 vs 0.94 to 1.0). None of the 80 images was a filter reject (60 in 2742 rows), so the filter's own rejects were not reviewed (dropped from scope, `docs/DECISIONS.md`, 2026-09-19).

Outcome: cat-emotions-7 is excluded from training. Every reject reason on it is image quality (fluffy or black cats, ears cropped, covered face, low resolution), and its worst-detected classes are the ones the reviewer found interchangeable. Its rows stay in the cache; downstream experiments filter `dataset == "cat-emotions-3"`.

## E1: shape space (Procrustes + PCA)

Question (RSCH-1): is there visible class signal before training anything, and does any early PC track a confound (ear position, head yaw) instead? Method and full numbers: `experiments/e1/README.md` and its plots in `experiments/e1/plots/`. Procrustes alignment via `core/geometry.py`, PCA via `scripts/shape_space.py`, on the E0 cache filtered to `dataset == "cat-emotions-3" & plausible == True` (2071 -> 2029 rows).

**Explained variance, first six PCs:**

| PC | explained variance | cumulative |
|---|---|---|
| PC1 | 34.5% | 34.5% |
| PC2 | 26.5% | 61.0% |
| PC3 | 10.2% | 71.2% |
| PC4 | 5.3% | 76.6% |
| PC5 | 3.5% | 80.1% |
| PC6 | 2.3% | 82.4% |

**Pose confounds:** PC1 tracks ear position (r = 0.84). PC2 tracks head yaw (r = 0.90). Both exceed the stated |r| > 0.3 threshold, and both land ahead of any PC carrying class signal.

**Class separation:** the three usable classes (attentive, relaxed, uncomfortable) do not visibly separate in the first six PCs. Best between-class/within-class variance ratio is 0.044, at PC3.

Answer to RSCH-1: no on both counts, and correctly so — this experiment names no stop condition ("this is a look, not a gate"), and a negative result here is the expected outcome, not a failure. Reported as a finding, per the process this report follows.

`models/class_means.json` now exists: a 48x2 Procrustes-aligned mean per class for the three usable classes (attentive 1130, relaxed 730, uncomfortable 107 plausible rows). The app's `/edit` endpoint consumes it as the M3 warper's data dependency and, per `docs/DECISIONS.md` 2026-09-20, as the data behind `/edit`'s primary manual class picker.

## Findings by question

All classifier results below train on cat-emotions-3, plausible rows, the three usable classes, 1967 rows (attentive 1130, relaxed 730, uncomfortable 107), on one frozen set of stratified 5-fold splits cached at `data/cache/splits.csv`. Minority classes are oversampled on training folds only; test folds keep the real class distribution. A uniform `DummyClassifier` on the same folds scores macro F1 0.302 and kappa 0.017, which is the chance level every number below should be read against.

### Q1 — does a graph conv over the anatomical adjacency beat an MLP on flattened coordinates?

Answered: no, and the graph does worse than no graph at all. Research QA PASS on the re-run (`docs/backlog.md` RSCH-3), which reproduced every committed number independently. Full detail in `experiments/e3/README.md`.

| arm | adjacency | readout | macro F1 | kappa | MCC |
|---|---|---|---|---|---|
| `gnn_anatomical` | v3, 98 edges | flatten | 0.374 +/- 0.015 | 0.096 +/- 0.021 | 0.102 |
| `gnn_random_ablation` | random, 98 edges | flatten | 0.398 +/- 0.024 | 0.124 +/- 0.037 | 0.129 |
| **`gnn_identity`** | **none (`A_hat = I`)** | flatten | **0.466 +/- 0.012** | **0.231 +/- 0.022** | **0.235** |
| `gnn_anatomical_meanpool` | v3, 98 edges | mean | 0.269 +/- 0.021 | 0.019 +/- 0.018 | 0.023 |
| E2 `mlp` (reference) | none | flatten | 0.418 +/- 0.023 | 0.155 +/- 0.046 | 0.161 |
| uniform dummy | none | none | 0.302 +/- 0.000 | 0.017 +/- 0.001 | 0.018 |

Both verdict conditions fail, decided on kappa. The anatomical arm trails the E2 MLP by 0.059 against its own CV standard deviation of 0.021, and trails the random adjacency by 0.028. A hand-authored anatomical graph does not beat a random graph of the same edge count, which is the outcome `PLAN_RESEARCH.md` names in advance: if random does as well, the structure carries nothing.

**The identity control carries the finding.** `gnn_identity` runs the same three layers at the same depth and parameter count with message passing switched off, and it scores higher than both graph arms and higher than the E2 MLP. Turning the graph off improves the model. So the result is not that this particular adjacency is uninformative; it is that the convolution itself costs accuracy. Each layer replaces a landmark's representation with an average over its neighbours, and the signal in these features is precisely the small relative displacement between nearby landmarks.

That suggests, as interpretation rather than measurement, why the anatomical graph does worse than a random one. An anatomically sensible edge list connects the landmark pairs whose relative positions encode expression: eyelid contour points to each other, ear base to ear tip. Averaging along exactly those edges destroys exactly those contrasts. A random graph mostly connects unrelated points across the face, so it damages less of what matters. Being anatomically correct makes the smoothing more targeted, not more useful. Nobody has tested this directly, and it would take an experiment of its own.

Read the Q1 answer with three riders.

*The first run answered a different question.* RSCH-3's original implementation pooled over the node axis with `h.mean(dim=1)`, and Procrustes centering sets every shape's node-axis mean to exactly zero, so that readout discarded the entire input (between-sample variance 2.58e-33). The `gnn_anatomical_meanpool` row above retains that configuration, and at kappa 0.019 it sits at chance with a macro F1 below the uniform dummy's. The first run's reported "anatomical beats random" finding was an artifact of it. See `docs/DECISIONS.md` 2026-09-20.

*The graph actually tested was disconnected.* Research QA found that the v3 edge list dropped the eye-to-ear-base and eye-to-nose bridges, splitting the adjacency into four components: left eye, right eye, ears, and nose/mouth/muzzle. RSCH-3's Method names "eyes→ear bases" explicitly, so the arm labelled anatomical no longer matches the adjacency the experiment was groomed around, and three convolution layers could not move information between those four regions. QA tested whether this changed the answer by running the two earlier schemes, both of which are connected: v1 (76 edges) scores 0.111 kappa and v2 (106 edges) 0.075, against v3's 0.096. All three lose to the random adjacency, to the identity arm and to the MLP. The verdict holds across every edge scheme ever authored here, and the scheme in use is not even the best of the three, which also disposes of the concern that the edge list might have been tuned against E3's own metrics.

*The absolute numbers are an upper bound.* Folds stratify on class alone and within-dataset near-duplicates were never hashed, so photos of one cat can straddle a split. Every arm shares one `splits.csv`, so this does not threaten the comparison between arms. It does threaten the headline: QA notes that leakage would most benefit whichever arm can best memorize individual cats, which is the per-node identity arm, so "the graph convolution costs signal" is the specific claim that would most repay a re-run under a grouped split.

One implementation note for anyone rebuilding this: `GNNClassifier` seeds once per construction, so all five folds share an initialization. That compresses fold-to-fold spread and makes the CV-standard-deviation threshold easier to clear. It is conservative for a NO verdict and uniform across arms, but it should become a per-fold seed before anyone reports a positive result this way.

### Q2 — does a learned model beat geometric ratios fed to logistic regression?

Answered: the learned model wins, by a margin larger than its own fold-to-fold noise. Full detail in `experiments/e2/README.md`.

| model | features | macro F1 | kappa | MCC |
|---|---|---|---|---|
| `ratios_lr` | 3 geometric ratios | 0.385 +/- 0.030 | 0.154 | 0.161 |
| `coords_lr` | 96 aligned coordinates | 0.426 +/- 0.032 | 0.174 | 0.182 |
| `mlp` | 96 aligned coordinates | 0.418 +/- 0.023 | 0.155 | 0.161 |
| uniform dummy | none | 0.302 +/- 0.000 | 0.017 | 0.018 |

The MLP clears the ratio baseline by 0.033 macro F1 against its own 0.023 CV standard deviation, so Q2's answer is NO MATCH: coordinates carry signal that eye aperture, ear angle and muzzle spread do not. Read it with the caveat that Q2 as posed in `PLAN_RESEARCH.md` names two ratios, and a direct instruction on 2026-09-19 added muzzle spread as a third (`docs/backlog.md` RSCH-2). The baseline the MLP beat is therefore slightly stronger than the one Q2 described.

The honest headline Q2 was written to test, that cat expression classification from landmarks is mostly ear angle, does not hold. Three hand-picked ratios get about two thirds of the way to the best coordinate model, which is more than nothing and well short of the whole story.

### Q3 — how much does head pose contaminate the prediction?

Partly answered by E1, and the answer is more interesting than "pose leaks the label". E1 found PC1 tracking ear position (r = 0.84) and PC2 tracking head yaw (r = 0.90), both ahead of any PC carrying class signal. A Research PM diagnostic (ungroomed, see the note at the end of this section) then trained logistic regression on PC1 and PC2 alone: macro F1 0.323, kappa 0.033, which is chance. So pose is not a shortcut the classifier exploits. It is the dominant direction of shape variation, and it crowds out expression rather than standing in for it. E1's between-class over within-class variance ratio of 0.044 says the same thing from the other side.

RSCH-4 still owes the yaw-degradation sentence that `core/states.py` turns into a confidence penalty.

### Q4 — are all classes separable, or do some collapse?

Partly answered, pending RSCH-4. Every E2 confusion matrix shows the same pattern: attentive and relaxed bleed into each other heavily, and uncomfortable behaves differently depending on the model. `coords_lr` recovers 62 of 107 uncomfortable rows while `ratios_lr` recovers 28. With 107 examples against 1130, no number about uncomfortable is stable, and oversampling replicates the same 107 images rather than adding information.

### Q5 — does anything transfer to my own cat's photos?

Not started (RSCH-5).

## What the model quality means for the product

It means very little, and that is a deliberate property of the design rather than a consolation. `models/class_means.json` shipped at E1 and carries the warp. `core/states.py`, not the model, writes every user-facing string. The classifier's only job is choosing a default target class in `/edit`.

Nothing fitted on these landmarks has cleared kappa 0.25. At that level a preselected default is wrong more often than right, and a wrong default costs the user more than no default at all. `docs/DECISIONS.md` 2026-09-20 records the resulting decision: the manual class picker is `/edit`'s primary path, the model becomes an optional suggestion beside it, and no app milestone gates on classifier accuracy.

## Where the ceiling is, and what to try next

A Research PM diagnostic (ungroomed, not QA-reviewed, run in a scratchpad and not committed) fitted stronger models on the same folds to find out whether 0.42 is a model limit or a data limit:

| model | test macro F1 | test kappa | training-fold macro F1 |
|---|---|---|---|
| logistic regression, coords + ratios | 0.444 +/- 0.029 | 0.194 | 0.536 |
| random forest | 0.456 +/- 0.018 | 0.237 | 1.000 |
| hist gradient boosting | 0.459 +/- 0.026 | 0.248 | 1.000 |

Two tree ensembles separate every training example perfectly and generalize at kappa 0.24. Linear, ensemble and neural models all land between 0.42 and 0.46 macro F1. Capacity is not the binding constraint, and neither are the landmark coordinates, since they carry enough information to memorize the labels outright. Treat these numbers as a pointer, not a result; they need grooming and a QA pass before anyone cites them.

E3's `gnn_identity` arm, which did get a QA pass, lands in the same place from a different direction: 0.466 macro F1 and 0.231 kappa from a per-node network on the same coordinates. Four model families built on three different feature treatments all stop between kappa 0.19 and 0.25. That is what a data ceiling looks like.

Three follow-ups were raised. Ranked by information per hour:

**1. Label noise (strongest).** Labels are one uploader's perceived emotion with no protocol and no agreement number. The dataset advertises three classes and ships eight folders, including `no clear emotion recognizable` (37 images) and one folder named `attentive uncomfortable` (1 image), which is an annotator who could not decide, preserved in the directory tree. Perfect training fit with kappa 0.24 on held-out folds is the label-noise signature. The 100-image blind relabel waived on 2026-09-19 carries its own reopening condition, "can be added at E2 if the confusion matrix makes the ceiling matter", and that condition has now fired. Roughly half a day, and it would tell us whether 0.46 is near the human ceiling or far below it.

**2. Per-cat variability (second, and partly a threat to results already reported).** The near-duplicate check ran across the two Roboflow sets, never within cat-emotions-3, and folds are stratified on class alone. Photos of the same cat can therefore straddle the train/test boundary, and perfect training fit is equally consistent with the trees memorizing individual cats. If that leakage exists, every number in E2 and E3 is optimistic rather than pessimistic. Roughly half a day to cluster within-dataset near-duplicates and re-run one model under a grouped split.

**3. Landmark detector error (weakest as posed).** Replacing the detector is not worth doing. The repo already runs `hugocornellier/cat-face-landmarks`, the full variant, at a reported NME/IOD of 3.48; cat-emotions-3 spot-checks at 57 of 59 overlays acceptable with a 2.0% filter drop rate, and the failures are occlusion, fluff and pose rather than systematic error. The memorization result above settles it: the coordinates are not the bottleneck. Training a detector on CatFLW would cost days to chase a model that is already good enough for the thing it feeds.

One cheap piece of it is worth doing as verification. Nobody has measured our own pipeline's landmark error. CatFLW sits on disk with 2079 ground-truth 48-point label files, our inference path carries three undocumented assumptions (the `[x2, y1, x1, y2]` tensor order, the 0.1 crop margin, and RGB versus BGR channel order), and `docs/AI_WORKFLOW.md` already notes that point ordering was never cross-checked against ground truth. 3.48 is upstream's number on upstream's split, not ours. Half a day either retires this question with a measurement or finds a pipeline bug that has been quietly degrading everything downstream.

## Limitations

- The plausibility filter checks geometry only. An ear placed on black fur or at the image edge is geometrically plausible, so it passed 11 of 11 human-rejected detections. Expect the same misses on user uploads until an image-quality scoring step exists.
- Detector failure modes seen in the spot check: fluffy or dark cats where ear outlines do not resolve, ears cropped out of frame, covered faces, low-resolution photos, open mouths, turned heads. Photos like these will still produce landmarks and a class.
- `detector_confidence` is a derived proxy, not a model score, and does not reliably flag bad detections.
- The detector was validated by one reviewer on 80 images, 21 of them from the excluded set; 59 images is the whole evidence base for cat-emotions-3.
- Labels are perceived emotion assigned by uploaders, with no agreement measurement. Three usable classes remain.
- Images are internet photos with uncontrolled lighting, breed and framing. Nothing here has clinical validity and nothing is claimed about pain or welfare.
- Folds are stratified on class alone. The near-duplicate check covered the two Roboflow sets against each other, not cat-emotions-3 against itself, so photos of one cat may appear on both sides of a split. Every classifier number in this report should be read as an upper bound until someone measures that.
- No classifier here has cleared kappa 0.25. Read the macro F1 figures against the 0.302 chance level rather than against 1.0; a model at 0.46 macro F1 is closer to the dummy than the number looks.

## Citations

- Martvel, G., Shimshoni, I., Zamansky, A. Automated Detection of Cat Facial Landmarks. 2023. https://github.com/martvelge/CatFLW
- Finka, L. R. et al. Geometric morphometrics for the study of facial expressions in non-human animals, using the domestic cat as an exemplar. Scientific Reports 9, 9883 (2019). The 48-landmark scheme.
- Cornellier, H. cat-face-landmarks model weights. https://huggingface.co/hugocornellier/cat-face-landmarks
- Feline pain-assessment descriptors. Scientific Reports (2023). https://www.nature.com/articles/s41598-023-49031-2 — 35 geometric descriptors (angles, distance ratios, area ratios by action unit) over cat facial landmarks, of which ear-position and orbital-tightening descriptors gave the smallest prediction error. This is the external support for picking eye aperture and ear angle as E2's ratio baseline, required at RSCH-2 grooming.

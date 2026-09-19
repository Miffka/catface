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

`models/class_means.json` now exists: a 48x2 Procrustes-aligned mean per class for the three usable classes (attentive 1130, relaxed 730, uncomfortable 107 plausible rows). The app's `/edit` endpoint consumes it as the M3 warper's data dependency and the M5 manual-picker fallback.

## Limitations

- The plausibility filter checks geometry only. An ear placed on black fur or at the image edge is geometrically plausible, so it passed 11 of 11 human-rejected detections. Expect the same misses on user uploads until an image-quality scoring step exists.
- Detector failure modes seen in the spot check: fluffy or dark cats where ear outlines do not resolve, ears cropped out of frame, covered faces, low-resolution photos, open mouths, turned heads. Photos like these will still produce landmarks and a class.
- `detector_confidence` is a derived proxy, not a model score, and does not reliably flag bad detections.
- The detector was validated by one reviewer on 80 images, 21 of them from the excluded set; 59 images is the whole evidence base for cat-emotions-3.
- Labels are perceived emotion assigned by uploaders, with no agreement measurement. Three usable classes remain.
- Images are internet photos with uncontrolled lighting, breed and framing. Nothing here has clinical validity and nothing is claimed about pain or welfare.

## Citations

- Martvel, G., Shimshoni, I., Zamansky, A. Automated Detection of Cat Facial Landmarks. 2023. https://github.com/martvelge/CatFLW
- Finka, L. R. et al. Geometric morphometrics for the study of facial expressions in non-human animals, using the domestic cat as an exemplar. Scientific Reports 9, 9883 (2019). The 48-landmark scheme.
- Cornellier, H. cat-face-landmarks model weights. https://huggingface.co/hugocornellier/cat-face-landmarks

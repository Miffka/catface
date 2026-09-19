# data/cache — E0 landmark cache

## What this is
`landmarks.parquet` holds every detector prediction over cat-emotions-3 (train) and
cat-emotions-7 (train+valid) — every processed row is kept, plausible or not, so a
later retune of the plausibility filter can refilter this cache instead of re-running
the detector. Produced by `uv run python scripts/landmark_cache.py`.

## Schema
- `image_id`, `dataset`, `class`, `split` — identifiers
- `box_x1, box_y1, box_x2, box_y2` — detected face box, image pixel coordinates
- `landmarks` — flat list of 96 floats, `[x0, y0, x1, y1, ...]` for the 48 points
- `detector_confidence` — **not a native model output**: neither OpenVINO model in
  `models/manifest.json` emits a confidence score. This is a derived geometric
  proxy (fraction of the 48 landmarks that land inside the tight localizer box);
  see `src/catface/ml/plausibility.py:detector_confidence_proxy`.
- `plausible` — result of `src/catface/ml/plausibility.py:check_landmarks`
- `drop_reasons` — list of failed rule names when `plausible` is false

## Detector run
Processed 2742 images. Plausible: 2682 (97.8%). Dropped: 60.
Drop reasons (an image can fail more than one rule):
  eyes_below_muzzle: 30
  degenerate: 29
  bad_aspect_ratio: 2
  detection_failed: 1

## Stop condition
NOT triggered — the detector fails on most images is defined here as a plausible-rate under 50%; actual rate is 97.8%.
TRIGGERED means halt the research track and escalate to the Research PM before E1.

## Plots
`plots/class_distribution_cat-emotions-3.png`, `plots/class_distribution_cat-emotions-7.png`
(written by `scripts/dataset_stats.py`), `plots/detector_confidence_hist.png`.

## Near-duplicates
See `near_duplicates.csv` and the per-dataset README.txt (written by
`scripts/dataset_stats.py`).

## 80-overlay spot check
80 images sampled uniformly at random (not filtered by `plausible`) into
`overlays/`. The human verdict per image is `overlays/overlay.csv`
(`img_fn, annotation_ok, reason`); the summary is in `docs/MODEL_REPORT.md`.

## 100-image blind relabel
Waived — see `docs/DECISIONS.md` (2026-09-19).
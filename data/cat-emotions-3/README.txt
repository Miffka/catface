# cat-emotions-3 — dataset notes

## Licence
CC BY 4.0 — https://universe.roboflow.com/cat-emotion-classification/cat-emotions-cgrxv
confirmed in data/cat-emotions-3/README.roboflow.txt

## Advertised vs actual
Advertised: 2071 images, 3 classes
Actual: 2071 images, 8 classes/label folders
Advertised as 3 classes; the actual download has 8 raw label folders, train split only, no valid/test split:
  Unlabeled: 2
  angry: 3
  attentive: 1147
  attentive uncomfortable: 1
  no clear emotion recognizable: 37
  relaxed: 752
  sad: 22
  uncomfortable: 107

## Near-duplicates against cat-emotions-7
0 near-duplicate pairs found between cat-emotions-3 and cat-emotions-7 (difference hash, Hamming distance <= 5).
None found at this threshold.

## Detector run (E0)
Processed 2071 images: 2029 plausible, 42 dropped.
detector_confidence is a derived geometric-plausibility proxy, not a native model output — see data/cache/README.md.
  eyes_below_muzzle: 21
  degenerate: 20
  detection_failed: 1
  bad_aspect_ratio: 1


# cat-emotions-7 — dataset notes

## Licence
CC BY 4.0 — https://universe.roboflow.com/cats-xofvm/cat-emotions
confirmed in data/cat-emotions-7/README.roboflow.txt

## Advertised vs actual
Advertised: 671 images, 7 classes
Actual: 671 images, 7 classes/label folders
train + valid splits, per-class counts:
  Angry: train 70, valid 29
  Disgusted: train 61, valid 18
  Happy: train 61, valid 37
  Normal: train 74, valid 24
  Sad: train 63, valid 35
  Scared: train 79, valid 20
  Surprised: train 94, valid 6

## Near-duplicates against cat-emotions-3
0 near-duplicate pairs found between cat-emotions-3 and cat-emotions-7 (difference hash, Hamming distance <= 5).
None found at this threshold.

## Detector run (E0)
Processed 671 images: 653 plausible, 18 dropped.
detector_confidence is a derived geometric-plausibility proxy, not a native model output — see data/cache/README.md.
  eyes_below_muzzle: 9
  degenerate: 9
  bad_aspect_ratio: 1


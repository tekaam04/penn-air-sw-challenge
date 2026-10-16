# Measured results

Regenerate with `python tools/evaluate.py`.

## Part 1 — static image accuracy

| shape | IoU vs colour GT | centre error (px) | vertices |
| --- | --- | --- | --- |
| quad | 0.9903 | 0.28 | 4 |
| triangle | 0.9895 | 0.30 | 3 |
| pentagon | 0.9891 | 0.45 | 5 |
| circle | 0.9843 | 0.41 | — |
| quad | 0.9856 | 0.36 | 4 |
| **mean** | **0.9878** | **0.36** | 5 found / 5 expected |

## Speed vs working resolution

60 frames of 1080p video, single thread.

| working width | ms / frame | fps | shapes / frame |
| --- | --- | --- | --- |
| 640 px | 12.8 | 78 | 4.18 |
| 960 px (default) | 19.3 | 52 | 4.18 |
| 1280 px | 30.2 | 33 | 4.25 |
| 1920 px | 62.5 | 16 | 4.57 |

## Parts 2/3 — stability over the full videos

**dynamic.mp4** — 1837 frames, 19.9 ms/frame (50 fps)

* shapes per frame: mean 4.55, median 5, max 6
* frames with the full set visible (5): 63.7%  (the rest have shapes genuinely off-frame)
* frames with 0 detections: 0
* distinct track IDs: 36 (shapes leave and re-enter the frame)
* circle usable as depth reference: 1191/1837 frames (64.8%)
* fitted circle radius: 101.06 ± 0.13 px
* recovered depth: 254.0 ± 0.32 in (21.17 ft), spread 0.13% of range

**dynamic_hard.mp4** — 1841 frames, 22.1 ms/frame (45 fps)

* shapes per frame: mean 4.47, median 5, max 6
* frames with the full set visible (5): 59.2%  (the rest have shapes genuinely off-frame)
* frames with 0 detections: 0
* distinct track IDs: 35 (shapes leave and re-enter the frame)
* circle usable as depth reference: 1096/1841 frames (59.5%)
* fitted circle radius: 100.74 ± 0.14 px
* recovered depth: 254.8 ± 0.35 in (21.24 ft), spread 0.14% of range


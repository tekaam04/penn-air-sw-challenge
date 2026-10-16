# Report — approach, problems, and what changed

The deliverables ask for a brief explanation per part, the challenges faced, and
the adjustments made. This is that, in the order the work actually happened.
Every number here is reproducible with `python tools/evaluate.py`.

---

## Part 1 — static image

### First look

Before writing anything I looked at all three inputs, and the third one decided
the design. The static image and `dynamic.mp4` are saturated shapes on green
grass — trivially separable by colour. `dynamic_hard.mp4` is dark speckled
gravel, and the shapes are filled with *gradients*: a triangle that runs red to
yellow, a pentagon navy to gold, a trapezoid white to grey.

So a colour-based detector would have to be thrown away at Part 3. Since the
brief says the same algorithm should survive a background change, I designed for
the hard case first and let the easy cases fall out.

### The cue

The invariant across all three inputs is not colour, it is *texture*:

> Backgrounds are textured. Shapes are smooth.

Grass churns pixel to pixel. Gravel churns more. A solid shape — flat or
smoothly shaded — does not churn at all. That test never mentions a colour, so
it transfers to a background the code has never seen.

### Pipeline

1. **Texture energy** — CIE-Lab, subtract the local mean, local RMS of the
   residual. Two box filters: O(1) per pixel regardless of window size.
2. **Robust threshold** — shapes are low-side outliers: `median − k·MAD` in log
   space, recomputed per frame, smoothed across frames with an EMA.
3. **Morphological clean-up** — opening for speckle, closing, hole fill.
4. **Watershed refinement** — the texture test loses ~half a window's width at
   every edge (a window straddling a boundary is never smooth), so each blob's
   eroded core is flooded back out to the true boundary.
5. **Model fitting** — fit n-gons (3–8) and an ellipse, keep whichever explains
   the outline best.
6. **Measurement** — moments for the centre, plus area, circularity, solidity.

### Results

| shape | IoU vs colour GT | centre error |
| --- | --- | --- |
| rectangle | 0.9903 | 0.28 px |
| triangle | 0.9895 | 0.30 px |
| pentagon | 0.9891 | 0.45 px |
| circle | 0.9843 | 0.41 px |
| trapezoid | 0.9856 | 0.36 px |
| **mean** | **0.9878** | **0.36 px** |

Ground truth is a colour flood grown from one hand-placed seed inside each
shape — a completely different cue from the one the detector uses, so it is an
independent check and not a restatement of the algorithm.

---

## Challenges, and what each one changed

These are the four things that did not work the first time. Each was found by
measuring, not by guessing.

### 1. Local variance discards gradient fills

**Symptom.** On the hard video only the red half of the red-to-yellow triangle
was detected. The gradient half was being thrown away with the gravel.

**Cause.** The first implementation used a local standard deviation. A gradient
fill genuinely *has* variance inside a window — colour changes across it — so a
ramp region scored 8.1 while the gravel background scored ~15. Separation of
less than 2×, and the threshold fell in between.

**Fix.** Measure deviation from the *local mean* rather than spread about it. A
box filter reproduces a linear ramp exactly (average a plane over a symmetric
window and the plane comes back), so `I − boxfilter(I)` is identically zero
across a gradient fill and large on texture.

| region | local std | detrended RMS |
| --- | --- | --- |
| gradient fill inside a shape | 8.1 | **0.8** |
| flat fill inside a shape | 1.5 | 0.6 |
| gravel background | 19.5 | 16.2 |

2× separation became 20×, and it still costs two box filters.

### 2. Otsu thresholds the background against itself

**Symptom.** Even with the better cue, gradient regions kept falling outside the
mask. The computed threshold was 4.9 where the gradient regions sat at 8.1.

**Cause.** Otsu splits a histogram into two comparable classes. Here the
background is ~97% of the pixels, so the two classes it found were "darker
gravel" and "brighter gravel" — a perfectly good split of the wrong thing.

**Fix.** Turn the imbalance into an advantage. Because the background dominates,
the frame's median and MAD *are* the background's statistics, so the shapes can
be picked out as low-side outliers at `median − 3·MAD` (in log space). Still one
number derived per frame, no constant tied to any background. Otsu is kept only
as a fallback for the degenerate case where the "outliers" exceed 60% of the
frame, which is precisely when its balanced-class assumption holds instead.

### 3. The overlap splitter was the problem, not the solution

**Symptom.** Shapes with gradients were cut in half; overlapping shapes
sometimes merged anyway.

**Cause.** I had written an explicit splitter: find internal step edges and cut
the mask along them. A first derivative cannot tell a gradient fill from a seam
between two shapes — both move a long way in colour — so I moved it to the
Laplacian (a linear ramp has zero curvature, a step does not). That was better
but still fragile: with flat interiors the MAD of the curvature goes to zero and
the threshold collapses.

**Fix.** Delete the whole stage. Once the texture cue was correct, the seam
between two overlapping shapes *is* a step edge, so it fails the smoothness test
and the mask already arrives with a hairline gap along it. Verified on the
frames where shapes overlap: the raw mask yields one connected component per
shape, every time. The splitter's only reliable effect had been to cut gradient
fills apart.

This is the change I am happiest with — the accuracy came from removing code.

### 4. Corners were being rounded off, and counted wrong

**Symptom.** Outlines sat a pixel or two inside the shapes, corners were
visibly cut, and labels were unstable: the same triangle came back as a
pentagon or a hexagon depending on the frame.

**Cause.** Two separate things. Morphology rounds corners, and `approxPolyDP`
puts its vertices *on* that rounded contour, so every corner is systematically
pulled inward. And the vertex count from a single epsilon is close to a coin
flip once corners are soft.

**Fix.** Fit models rather than reading off approximations:

* each edge gets a least-squares line from the middle 60% of its points (the
  ends are the rounded part), and consecutive lines are intersected — corners
  land where the geometry says they should;
* the initial edge split is refined twice by reassigning every contour point to
  its nearest fitted line, because the starting corners can be 15 points off;
* every n from 3 to 8 is fitted, plus an ellipse, and the winner is chosen by
  mean residual — the simplest polygon within a sub-pixel tolerance, or the
  ellipse when it beats every polygon outright.

Circularity of the detected circle went from 0.897 to 0.990 (1.0 is perfect),
the triangle from 0.754 to 0.605 (0.605 is exact for an equilateral triangle),
and labels became stable.

---

## Part 2 — video, treated as a stream

The brief is explicit that the video is a stream, not a file to be studied, so
`FrameStream` reads one frame at a time and nothing downstream may seek. The
only state carried between frames is a threshold EMA and the tracker.

**Performance.** ~20 ms per 1080p frame single-threaded (≈50 fps), against a
33 ms budget at 30 fps. What made that possible:

* **Work at 960 px wide, not 1920.** Halving the width quarters the pixels:
  63 ms → 19 ms. Contours are scaled back up afterwards. The texture cue is
  scale-dependent, so this is not purely a cost saving — 960 px keeps grass
  "rough" relative to a 9 px window. Nothing measurably improves above it
  (IoU changes in the third decimal).
* **Box filters over explicit windows.** The texture map is O(1) per pixel
  whatever the window size.
* **No per-pixel Python.** Every stage is an OpenCV call or a NumPy expression.
* **Model fitting scores on 120 sampled boundary points**, not all ~400.

| working width | ms / frame | fps |
| --- | --- | --- |
| 640 px | 12.8 | 78 |
| **960 px (default)** | **19.3** | **52** |
| 1280 px | 30.2 | 33 |
| 1920 px | 62.5 | 16 |

**Stability over 1 837 frames:** mean 4.55 shapes/frame, median 5, and *zero*
frames with no detections. Frames reporting fewer than five were checked by
hand — the missing shapes are genuinely outside the frame as the camera pans.

---

## Part 3 — background agnostic

No modifications were needed, which was the point of designing for it first.
The same command, the same parameters:

```bash
python main.py video data/dynamic_hard.mp4 -o outputs/hard_annotated.mp4 --3d
```

What makes it transfer:

1. **No colour anywhere in the decision.** The only photometric quantity is
   "how much does this neighbourhood churn".
2. **Every threshold is derived per frame** from that frame's own statistics.
   There is no "grass is green at 60 ± 20" constant to invalidate.
3. **Gradient fills are handled by construction**, because removing the local
   mean removes any linear shading exactly.

**Result:** 1 841 frames, mean 4.47 shapes/frame, zero empty frames, 22 ms per
frame. The cross-check that matters most: both videos show the same physical
scene, and the independently recovered depths agree — 254.0 in from grass,
254.8 in from gravel, 0.3% apart.

The test suite pushes this further with four synthetic backgrounds — green,
dark gravel, bright concrete, and a purple one nobody would tune for — all
detected with unchanged parameters.

---

## Part 4 — 3D

`Z = f·R/r` from the 10 in circle, then back-projection for X and Y, with the
flat-surface assumption extending the circle's depth to every shape on the
plane. The intrinsics are rescaled when the frame resolution differs from the
1920×1080 the matrix belongs to (the static image is half size, so its focal
length is halved — otherwise its depth would come out 2× too far).

**Two robustness problems showed up in the measurements.**

A hexagon with circularity 0.903 was accepted as the reference circle in one
frame and moved the entire plane; and in another, a small partly-visible blob
was fitted as a circle of radius 34 px instead of 101. Together they inflated
the depth spread on the hard video to 5.8%. The reference is now required to be
fully visible, genuinely round (circularity ≥ 0.95), *and* ellipse-fitted, and a
candidate implying a >25% jump in scale between frames is rejected as physically
impossible — with a counter that re-acquires if the rejection persists, since a
real altitude change would look the same.

| | grass | hard |
| --- | --- | --- |
| frames where the circle is a valid ruler | 64.8% | 59.5% |
| fitted radius | 101.06 ± 0.13 px | 100.74 ± 0.14 px |
| recovered depth | 254.0 ± 0.32 in | 254.8 ± 0.35 in |

0.13–0.14% spread on a rigid scene, and the two videos agree to 0.3%. When the circle
is not usable the last good scale is held and every affected reading is flagged
`held` rather than silently reported.

**On the supplied `K`:** `cx = cy = 0` puts the principal point at the top-left
corner, which biases X and Y by half the sensor (~19 in at this range). The
default assumes the image centre; `--literal-principal-point` uses the matrix
verbatim. Depth is unaffected by the choice.

---

## Part 5 — ROS 2

Standard two-package layout: `pennair_msgs` (`ament_cmake`, because message
generation needs it) and `pennair_vision` (`ament_python`). The detector node is
a thin wrapper over the same `pennair` package the CLI uses, so there is exactly
one implementation of the algorithm to keep correct. Outlines are published both
as polygons in the message and as RViz markers. Full topic list and commands in
the [README](../README.md#part-5--ros-2).

---

## Part 6 — extras

**Tracking through overlap.** Per-frame detection is enough to draw outlines but
not to *follow* a shape: when two overlap, one's centroid jumps and it may
vanish entirely. Each track carries a constant-velocity estimate; detections are
matched by a gated cost that adds mean colour and area to position, because
position alone swaps IDs exactly when two shapes are on top of each other. An
unmatched track coasts on its prediction for up to 15 frames (half a second)
before being dropped, and is drawn dashed and labelled `occluded` so a consumer
can tell a prediction from an observation.

One bug worth recording: tracks that left the frame at the edge were being
coasted as "occluded", inventing shapes off-screen. A track that stops being
detected while at the frame boundary is now dropped immediately.

**Also:** geometric model fitting (above), shape classification for free from
the fit, a 17-test synthetic suite that needs no media, and `tools/evaluate.py`
to regenerate every number in these documents.

---

## What I would do next

* **Amodal completion** — fit the polygon model to the *visible* part of an
  occluded shape and extrapolate the hidden corners, so a half-covered rectangle
  reports its true centre rather than the centroid of what shows. This is the
  single biggest accuracy gap left, and the model-fitting machinery is already
  there.
* **Multi-scale texture** — run the cue at two window sizes and combine, to
  handle a background whose grain is close to the current window.
* **Per-shape depth** — with more than one known size, drop the single-plane
  assumption and solve for each shape's depth independently.
* **SIMD/GPU** — the pipeline is box filters and morphology, which map directly
  onto a GPU or a Jetson's VPI if a real airframe needed the headroom.

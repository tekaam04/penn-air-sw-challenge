"""
Background-agnostic detection of solid shapes.

The core idea
-------------
Every background in this challenge (grass, gravel/asphalt) is *textured*: at the
scale of a few pixels its color bounces around a lot.  Every foreground shape is
*solid*: flat color, or at worst a smooth gradient.  So instead of asking "what
color is the background?" (which breaks the moment the background changes) we
ask "where is the image locally smooth?".

That makes the whole pipeline color-blind, which is exactly what Part 3 wants:

  1. texture energy of the CIE-Lab image: the local RMS of the image minus its
     own local mean.  Two box filters, so O(1) per pixel whatever the window
     size, and identically zero across a gradient fill (see _local_sigma).
  2. robust threshold -- the shapes are low-side outliers of that map, at
     median - k*MAD.  Derived from the frame itself, so no constant ever has to
     be retuned for a new background.
  3. clean up: opening for speckle, closing, hole fill.  Shapes that overlap do
     not need splitting here: the seam between them is a step edge, so it fails
     the smoothness test in step 1 and the mask arrives already cut.
  4. watershed, seeded with each blob's eroded core, to snap the outline back
     onto the true boundary (the texture mask is short by ~half the window
     radius, because a window straddling an edge is never smooth)
  5. geometric sanity filtering (area, solidity, fill ratio) and moment centers
  6. model fitting: the shapes really are polygons and circles, so every n-gon
     from 3 to 8 and an ellipse are fitted and the best-explaining one wins.
     Corners come from intersecting least-squares edge lines, which undoes the
     morphological rounding and makes the circle radius -- which Part 4 turns
     into depth -- much steadier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One detected shape, in *original* (full-resolution) image coordinates."""

    contour: np.ndarray  # (N, 1, 2) int32, OpenCV contour convention
    center: tuple[float, float]  # centroid (x, y) in pixels
    area: float  # contour area in px^2
    perimeter: float
    circularity: float  # 4*pi*A / P^2 ; 1.0 == perfect circle
    solidity: float  # area / convex-hull area
    vertices: int  # corner count from polygon approximation
    label: str  # "circle" / "triangle" / ... / "polygon"
    equiv_radius: float  # radius of the circle with the same area, px
    mean_color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # BGR
    touches_border: bool = False
    refined: bool = False  # True if a polygon/ellipse model was fitted
    fit_radius: float | None = None  # ellipse-fit radius for circles, px
    extras: dict = field(default_factory=dict)  # 3D pose etc., filled in later


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ShapeDetector:
    """Detects solid shapes on an arbitrary textured background.

    Parameters are given in *relative* units wherever possible so that the same
    detector works on the 960x540 static image and on 1080p video without
    retuning.

    Parameters
    ----------
    work_width:
        Frames are resized to this width before processing and results are
        scaled back up.  This is the main speed/accuracy knob: the texture cue
        is scale-dependent, and ~960px keeps grass "rough" while staying fast.
    sigma_window:
        Side length (px, at work resolution) of the texture-energy window.  Must
        be larger than the background's texture grain and smaller than the
        smallest shape.
    min_area_frac, max_area_frac:
        Accepted contour area as a fraction of the frame area.
    outlier_k:
        How many robust standard deviations below the frame's median texture a
        pixel has to be to count as "shape".  Higher = stricter.
    min_solidity:
        Reject non-convex blobs (grass clumps, shadows).  Every shape in the
        challenge is convex; 0.85 leaves room for occluded shapes.
    temporal_smoothing:
        EMA factor for the auto threshold across frames (0 disables).  Keeps the
        mask from flickering when a shape enters or leaves the frame and shifts
        the texture histogram.
    """

    def __init__(
        self,
        work_width: int = 960,
        sigma_window: int = 9,
        min_area_frac: float = 0.0012,
        max_area_frac: float = 0.35,
        min_solidity: float = 0.85,
        outlier_k: float = 3.0,
        min_fill_ratio: float = 0.30,
        refine_watershed: bool = True,
        temporal_smoothing: float = 0.7,
    ) -> None:
        self.work_width = work_width
        self.sigma_window = sigma_window | 1  # force odd
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_solidity = min_solidity
        self.outlier_k = outlier_k
        self.min_fill_ratio = min_fill_ratio
        self.max_fit_residual = 1.5  # px at working resolution
        self.refine_watershed = refine_watershed
        self.temporal_smoothing = temporal_smoothing

        self._thresh_ema: float | None = None
        self.debug: dict[str, np.ndarray] = {}

    # -- public API ---------------------------------------------------------

    def detect(self, frame: np.ndarray, collect_debug: bool = False) -> list[Detection]:
        """Run the full pipeline on one BGR frame. Stateless except for the EMA."""
        if frame is None or frame.size == 0:
            return []

        small, scale = self._resize(frame)
        lab = cv2.cvtColor(cv2.GaussianBlur(small, (3, 3), 0), cv2.COLOR_BGR2Lab)

        sigma = self._local_sigma(lab)
        mask = self._smooth_mask(sigma)
        mask = self._clean(mask)
        markers, n_labels = self._seed_markers(mask)

        if self.refine_watershed and n_labels > 0:
            mask_refined = self._watershed_refine(small, markers, mask, n_labels)
        else:
            mask_refined = mask

        detections = self._contours_to_detections(mask_refined, small, scale, frame.shape)

        if collect_debug:
            self.debug = {
                "small": small,
                "sigma": cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                "mask": mask,
                "mask_refined": mask_refined,
            }
        return detections

    # -- pipeline stages ----------------------------------------------------

    def _resize(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale to the working width. Returns (small, upscale_factor)."""
        h, w = frame.shape[:2]
        if w <= self.work_width:
            return frame, 1.0
        scale = self.work_width / float(w)
        small = cv2.resize(
            frame, (self.work_width, int(round(h * scale))), interpolation=cv2.INTER_AREA
        )
        return small, 1.0 / scale

    def _local_sigma(self, lab: np.ndarray) -> np.ndarray:
        """Local texture energy: RMS of the image minus its own local mean.

        The obvious cue here is a local standard deviation, but it fails on the
        extra-credit case: a shape filled with a red-to-yellow gradient has a
        lot of variance inside any window, so it scores like grass and gets
        thrown away.

        Subtracting the *local mean* first fixes that, because a box filter
        reproduces a linear ramp exactly -- average a plane over a symmetric
        window and you get the plane back.  So the residual is identically zero
        across a gradient fill and stays large on texture, which is the
        difference we actually care about.  Measured on the hard video, a
        gradient region drops from 8.1 (std) to 0.8 (this), against a background
        of ~15: a 20x separation instead of 2x.

        Two box filters, so still O(1) per pixel regardless of window size.
        """
        k = (self.sigma_window, self.sigma_window)
        f = lab.astype(np.float32)
        residual = f - cv2.boxFilter(f, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
        energy = cv2.boxFilter(
            residual * residual, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT
        )
        rms = np.sqrt(np.maximum(energy, 0.0))
        # L counts once, a/b (chroma) count once together: mono texture shows up
        # in luminance, colored speckle shows up in chroma.
        return rms[..., 0] + 0.5 * (rms[..., 1] + rms[..., 2])

    def _smooth_mask(self, sigma: np.ndarray) -> np.ndarray:
        """Threshold the texture map into smooth (shape) vs rough (background).

        Otsu is the reflex here and it is wrong: the background is ~97% of the
        pixels, so Otsu's two classes end up being "darker grass" and "brighter
        grass" and the threshold lands far below the shapes.  Measured on the
        hard video it returned 4.9 where the gradient shapes sat at 8.1.

        Since the background dominates, robust statistics of the whole frame
        *are* the background's statistics, so the shapes can be picked out as
        low-side outliers: median - k*MAD in log space.  That is one derived
        number per frame, with no constant tied to any particular background.
        """
        log_sigma = np.log1p(sigma)
        med = float(np.median(log_sigma))
        mad = float(np.median(np.abs(log_sigma - med))) + 1e-6
        t_sigma = float(np.expm1(med - self.outlier_k * 1.4826 * mad))
        t_sigma = max(t_sigma, 0.5)  # a floor, for a background of pure noise-free color

        if self.temporal_smoothing > 0 and self._thresh_ema is not None:
            a = self.temporal_smoothing
            t_sigma = a * self._thresh_ema + (1.0 - a) * t_sigma
        self._thresh_ema = t_sigma

        mask = (sigma < t_sigma).astype(np.uint8) * 255

        # Guard against the degenerate case the robust estimate cannot see: if
        # "outliers" cover most of the frame, the frame is mostly smooth and the
        # median no longer describes the background.  Fall back to Otsu, which
        # is well behaved exactly when the classes are balanced.
        if cv2.countNonZero(mask) > 0.6 * mask.size:
            u8 = cv2.normalize(log_sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, inv = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = inv
        return mask

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """Remove speckle and fill interior holes."""
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # One opening pass only: the watershed stage recovers the true edge, so
        # aggressive erosion here just rounds off corners for nothing.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=1)
        return self._fill_holes(mask)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Flood fill from the border, invert, OR back in — classic hole fill."""
        h, w = mask.shape
        ff = mask.copy()
        pad = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, pad, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        return cv2.bitwise_or(mask, holes)

    def _seed_markers(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        """Label one watershed seed per blob.

        Overlapping shapes need no explicit splitting: the seam between two
        shapes *is* a step edge, so it fails the smoothness test in step 1 and
        the mask already arrives with a hairline gap along it.  (Measured on
        both videos: every frame where shapes overlap still yields one connected
        component per shape.)  An earlier version cut blobs apart along
        Laplacian seams; deleting it made the detector both simpler and more
        accurate, since the only thing it reliably split was gradient fills.

        Two shapes of the *same* colour that overlap would merge — but then no
        purely photometric method can separate them either.

        Returns an int32 marker image in OpenCV's watershed convention:
        1 = sure background, 2..N+1 = shape cores, 0 = unknown.
        """
        ek5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        min_px = max(24.0, self.min_area_frac * mask.size * 0.25)

        n_blobs, blobs, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        markers = np.zeros(mask.shape, np.int32)
        next_label = 2

        for b in range(1, n_blobs):
            if stats[b, cv2.CC_STAT_AREA] < min_px:
                continue
            x, y, w, h = (
                stats[b, cv2.CC_STAT_LEFT],
                stats[b, cv2.CC_STAT_TOP],
                stats[b, cv2.CC_STAT_WIDTH],
                stats[b, cv2.CC_STAT_HEIGHT],
            )
            roi = slice(y, y + h), slice(x, x + w)
            blob = (blobs[roi] == b).astype(np.uint8) * 255
            # Erode to a core so the watershed has room to find the real edge;
            # keep the whole blob if it is too thin to survive erosion.
            core = cv2.erode(blob, ek5)
            if cv2.countNonZero(core) < min_px * 0.5:
                core = blob
            markers[roi][core > 0] = next_label
            next_label += 1

        # The unknown band has to be wide enough to contain the true edge: the
        # texture mask is short by roughly half the texture window, and by more
        # than that at a sharp corner.
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.sigma_window, self.sigma_window))
        sure_bg = cv2.dilate(mask, grow, iterations=2)
        markers[sure_bg == 0] = 1
        return markers, next_label - 2

    def _watershed_refine(
        self, small: np.ndarray, markers: np.ndarray, fallback: np.ndarray, n_labels: int
    ) -> np.ndarray:
        """Grow the cores back to the true edges, keeping shapes separated.

        Watershed labels every unknown pixel, so overlapping shapes come out as
        separate regions with a 1px ridge between them; painting each region
        into its own mask keeps them from merging back together.
        """
        # Median blur kills the background's speckle (which would otherwise trap
        # the flood) while leaving the shape's own step edge sharp.
        guide = cv2.medianBlur(small, 5)
        ws = markers.copy()
        cv2.watershed(guide, ws)

        out = np.zeros(small.shape[:2], np.uint8)
        for lbl in range(2, n_labels + 2):
            region = (ws == lbl).astype(np.uint8)
            if region.any():
                # Close the 1px watershed ridge so the contour is continuous.
                region = cv2.morphologyEx(
                    region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                )
                out[region > 0] = 255
        if not out.any():
            return fallback
        # Hand the label image on as well: _contours_to_detections traces each
        # basin separately, so two shapes that touch cannot merge back into one
        # contour just because their masks are adjacent.
        self._ws_labels = ws
        self._ws_n = n_labels
        return out

    # -- measurement --------------------------------------------------------

    def _contours_to_detections(
        self, mask: np.ndarray, small: np.ndarray, scale: float, orig_shape: tuple
    ) -> list[Detection]:
        """Turn each region into a measured Detection in original coordinates."""
        regions: list[np.ndarray] = []
        ws = getattr(self, "_ws_labels", None)
        if self.refine_watershed and ws is not None and getattr(self, "_ws_n", 0) > 0:
            # One binary image per watershed basin -> shapes that touch stay apart
            for lbl in range(2, self._ws_n + 2):
                region = ((ws == lbl) & (mask > 0)).astype(np.uint8) * 255
                if region.any():
                    regions.append(region)
            self._ws_labels = None
        else:
            regions.append(mask)

        frame_area = float(orig_shape[0] * orig_shape[1])
        h_s, w_s = mask.shape
        out: list[Detection] = []

        for region in regions:
            # CHAIN_APPROX_NONE: the edge-fitting stage needs every boundary
            # pixel, not the compressed corner-only representation.
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                det = self._measure(c, small, scale, frame_area, (w_s, h_s))
                if det is not None:
                    out.append(det)

        out.sort(key=lambda d: d.area, reverse=True)
        return out

    def _measure(
        self,
        contour: np.ndarray,
        small: np.ndarray,
        scale: float,
        frame_area: float,
        small_wh: tuple[int, int],
    ) -> Detection | None:
        """Geometry + sanity checks for one contour (in working resolution)."""
        area_small = cv2.contourArea(contour)
        if area_small < 16:
            return None

        area = area_small * scale * scale
        if not (self.min_area_frac * frame_area <= area <= self.max_area_frac * frame_area):
            return None

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_small / hull_area if hull_area > 0 else 0.0

        w_s, h_s = small_wh
        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x <= 1 or y <= 1 or x + w >= w_s - 1 or y + h >= h_s - 1

        # An occluded / clipped shape is legitimately less solid.
        if solidity < (self.min_solidity - 0.15 if touches_border else self.min_solidity):
            return None
        if w * h > 0 and area_small / float(w * h) < self.min_fill_ratio:
            return None

        perim_small = cv2.arcLength(contour, True)
        if perim_small <= 0:
            return None
        circularity = 4.0 * math.pi * area_small / (perim_small * perim_small)

        # --- model selection: which shape explains this outline best? -------
        # Rather than reading a corner count off one approxPolyDP epsilon (which
        # happily turns a triangle into a hexagon when the corners come back
        # slightly rounded), fit every candidate model -- triangle through
        # octagon, plus an ellipse -- and keep the one with the smallest mean
        # residual, preferring the simpler model when residuals are comparable.
        fit = None if touches_border else self._best_model(contour, area_small)

        refined = False
        fit_radius = None
        if fit is not None:
            fitted, label_fit, fit_radius = fit
            contour = fitted
            area_small = cv2.contourArea(contour)
            area = area_small * scale * scale
            perim_small = cv2.arcLength(contour, True)
            circularity = 4.0 * math.pi * area_small / (perim_small * perim_small)
            refined = True
            vertices = 0 if label_fit == "circle" else len(contour)
            label = label_fit
        else:
            approx = cv2.approxPolyDP(contour, 0.025 * perim_small, True)
            vertices = len(approx)
            label = self._classify(vertices, circularity)

        m = cv2.moments(contour)
        if m["m00"] == 0:
            return None
        cx = m["m10"] / m["m00"] * scale
        cy = m["m01"] / m["m00"] * scale

        mask_c = np.zeros(small.shape[:2], np.uint8)
        cv2.drawContours(mask_c, [contour], -1, 255, cv2.FILLED)
        mean = cv2.mean(small, mask=cv2.erode(mask_c, np.ones((5, 5), np.uint8)))[:3]

        # A fitted model is already exact; only raw blobs need smoothing before
        # being scaled up (a working-resolution contour is visibly jagged at 2x).
        if not refined:
            contour = cv2.approxPolyDP(contour, 0.004 * perim_small, True)
        contour_full = np.round(contour.astype(np.float32) * scale).astype(np.int32)

        return Detection(
            contour=contour_full,
            center=(cx, cy),
            area=area,
            perimeter=perim_small * scale,
            circularity=circularity,
            solidity=solidity,
            vertices=vertices,
            label=label,
            equiv_radius=math.sqrt(area / math.pi),
            mean_color=(mean[0], mean[1], mean[2]),
            touches_border=touches_border,
            refined=refined,
            fit_radius=(fit_radius * scale) if fit_radius else None,
        )

    # -- model fitting ------------------------------------------------------

    def _best_model(
        self, contour: np.ndarray, area_small: float
    ) -> tuple[np.ndarray, str, float | None] | None:
        """Pick the geometric model that explains this outline best.

        Candidates are the n-gons (n = 3..8) and an ellipse, each scored by the
        mean absolute distance from the observed contour points to the fitted
        model.  Two different rules, because the two families are asymmetric:

        * Between polygons, take the *simplest* one whose residual is within a
          sub-pixel tolerance of the best.  On a clean square every n from 4 up
          fits to a fraction of a pixel, and splitting hairs there just promotes
          squares to pentagons.
        * Between the ellipse and the polygons, take whichever residual is
          lower, full stop.  A many-sided polygon can always creep up on a
          circle, so ranking by simplicity would let an octagon win; on anything
          that is not round the ellipse loses this comparison by a wide margin.

        Returns (contour, label, radius), radius set only for the ellipse, or
        None when nothing fits well enough to be worth trusting.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 24:
            return None
        # Score on a subsample: 120 boundary points pin a model down fine and
        # keep this loop off the per-frame budget.
        probe = pts[:: max(1, len(pts) // 120)]

        def plausible(model: np.ndarray) -> bool:
            """A fit that changed the area this much mis-split an edge."""
            return 0.85 * area_small <= cv2.contourArea(model) <= 1.35 * area_small

        polygons: list[tuple[int, float, np.ndarray]] = []
        for n in range(3, 9):
            poly = self._fit_ngon(contour, n)
            if poly is None or not plausible(poly):
                continue
            res = self._residual(probe, poly)
            if res is not None:
                polygons.append((n, res, poly))

        ellipse_poly, ellipse_r = self._fit_circle(contour)
        ellipse_res = None
        if ellipse_poly is not None and plausible(ellipse_poly):
            ellipse_res = self._residual(probe, ellipse_poly)

        best_poly_res = min((p[1] for p in polygons), default=None)

        # Round wins outright when it explains the outline better.
        if ellipse_res is not None and (best_poly_res is None or ellipse_res < best_poly_res):
            if ellipse_res > self.max_fit_residual:
                return None
            return ellipse_poly, "circle", ellipse_r

        if not polygons:
            return None
        if best_poly_res > self.max_fit_residual:
            return None

        tol = best_poly_res * 1.5 + 0.6
        viable = sorted((p for p in polygons if p[1] <= tol), key=lambda p: (p[0], p[1]))
        _, _, poly = viable[0]
        poly = self._drop_straight_vertices(poly)
        return poly, self._classify(len(poly), 0.0), None

    @staticmethod
    def _drop_straight_vertices(poly: np.ndarray, min_turn_deg: float = 12.0) -> np.ndarray:
        """Remove vertices that do not actually turn a corner.

        When the edge split puts two seeds on one straight side, the fit returns
        a valid polygon with a redundant ~180-degree vertex — geometrically fine,
        but it makes a pentagon report as a hexagon.
        """
        pts = poly.reshape(-1, 2).astype(np.float64)
        changed = True
        while changed and len(pts) > 3:
            changed = False
            for i in range(len(pts)):
                a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
                v1, v2 = b - a, c - b
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                if math.degrees(math.acos(cosang)) < min_turn_deg:
                    pts = np.delete(pts, i, axis=0)
                    changed = True
                    break
        return np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _residual(probe: np.ndarray, model: np.ndarray) -> float | None:
        """Mean absolute distance from sampled contour points to a model."""
        model_f = model.astype(np.float32)
        try:
            d = [
                abs(cv2.pointPolygonTest(model_f, (float(p[0]), float(p[1])), True)) for p in probe
            ]
        except cv2.error:
            return None
        return float(np.mean(d)) if d else None

    @staticmethod
    def _fit_circle(contour: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        """Least-squares ellipse fit, redrawn as a dense polygon."""
        if len(contour) < 5:
            return None, None
        try:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
        except cv2.error:
            return None, None
        if d1 <= 0 or d2 <= 0 or max(d1, d2) / min(d1, d2) > 1.35:
            return None, None  # too eccentric to be our circle seen flat-on
        pts = cv2.ellipse2Poly(
            (int(round(cx)), int(round(cy))),
            (int(round(d1 / 2)), int(round(d2 / 2))),
            int(round(angle)),
            0,
            360,
            5,
        )
        if len(pts) < 8:
            return None, None
        return pts.reshape(-1, 1, 2).astype(np.int32), 0.25 * (d1 + d2)

    @staticmethod
    def _fit_ngon(contour: np.ndarray, n: int) -> np.ndarray | None:
        """Fit an n-sided polygon: least-squares line per edge, then intersect.

        `approxPolyDP` puts its vertices *on* the contour, so morphological
        rounding pulls every corner inward.  Fitting each straight edge from the
        middle 60% of its points (the ends are the rounded part) and intersecting
        consecutive edges puts the corners back where the geometry says they are.

        The initial edge split comes from approxPolyDP and is often a little
        off — a vertex can land 15 points past the real corner, which drags that
        edge's line with it — so the split is refined twice by reassigning every
        contour point to its closest fitted line and refitting.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 12 * n:
            return None

        seeds = ShapeDetector._corner_seeds(contour, n)
        if seeds is None:
            return None

        lines = ShapeDetector._fit_edges(pts, seeds, n)
        for _ in range(2):
            if lines is None:
                return None
            better = ShapeDetector._resplit(pts, lines, n)
            if better is None or better == seeds:
                break
            seeds = better
            lines = ShapeDetector._fit_edges(pts, seeds, n)
        if lines is None:
            return None

        corners = []
        for i in range(n):
            p = ShapeDetector._intersect(lines[i - 1], lines[i])
            if p is None:
                return None
            corners.append(p)
        corners_arr = np.array(corners, np.float32)

        # Sanity: a corner may move out by a few pixels, never across the shape.
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        observed = pts[np.array(seeds)]
        if span <= 0 or np.linalg.norm(corners_arr - observed, axis=1).max() > 0.25 * span:
            return None
        return corners_arr.reshape(-1, 1, 2).astype(np.int32)

    @staticmethod
    def _fit_edges(pts: np.ndarray, seeds: list[int], n: int) -> list[np.ndarray] | None:
        """Least-squares line for each edge, ignoring the rounded corner ends."""
        lines = []
        for i in range(n):
            a, b = seeds[i], seeds[(i + 1) % n]
            seg = pts[a:b] if b > a else np.vstack([pts[a:], pts[:b]])
            if len(seg) < 10:
                return None
            trim = max(1, int(0.2 * len(seg)))
            seg = seg[trim:-trim]
            if len(seg) < 5:
                return None
            lines.append(cv2.fitLine(seg, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        return lines

    @staticmethod
    def _resplit(pts: np.ndarray, lines: list[np.ndarray], n: int) -> list[int] | None:
        """Reassign contour points to the nearest fitted edge, return new seeds.

        Each point's distance to a line is the cross product with the line's
        unit direction.  Points group into n contiguous runs around the contour;
        the start of each run is the refined corner index.
        """
        vx = np.array([ln[0] for ln in lines], np.float32)
        vy = np.array([ln[1] for ln in lines], np.float32)
        x0 = np.array([ln[2] for ln in lines], np.float32)
        y0 = np.array([ln[3] for ln in lines], np.float32)
        dx = pts[:, 0:1] - x0[None, :]
        dy = pts[:, 1:2] - y0[None, :]
        dist = np.abs(dx * vy[None, :] - dy * vx[None, :])  # (N, n)
        owner = np.argmin(dist, axis=1)

        # Runs around the cyclic contour.
        changes = np.flatnonzero(owner != np.roll(owner, 1))
        if len(changes) != n:
            return None  # ambiguous split; keep what we had
        seeds = sorted(int(c) for c in changes)
        if len(set(seeds)) != n:
            return None
        # Every edge still needs enough support to fit a line.
        gaps = [(seeds[(i + 1) % n] - seeds[i]) % len(pts) for i in range(n)]
        if min(gaps) < 10:
            return None
        return seeds

    @staticmethod
    def _corner_seeds(contour: np.ndarray, n: int) -> list[int] | None:
        """Contour indices of n corner candidates.

        Bisecting approxPolyDP's epsilon to land on exactly n vertices sounds
        natural but the vertex count is a step function of epsilon and often
        skips the value wanted — which silently removed the *correct* model from
        the running.  Instead: approximate once, finely, then decimate to n by
        repeatedly dropping the vertex that matters least (Visvalingam-Whyatt:
        the one whose triangle with its neighbours has the smallest area).  That
        always produces a candidate, and the residual comparison in _best_model
        decides whether it deserves to win.
        """
        perim = cv2.arcLength(contour, True)
        if perim <= 0:
            return None
        pts = contour.reshape(-1, 2).astype(np.float32)

        poly = cv2.approxPolyDP(contour, 0.004 * perim, True).reshape(-1, 2).astype(np.float64)
        if len(poly) < n:
            return None

        while len(poly) > n:
            prev = np.roll(poly, 1, axis=0)
            nxt = np.roll(poly, -1, axis=0)
            # Twice the area of the triangle (prev, v, next) for every vertex.
            weight = np.abs(
                (nxt[:, 0] - prev[:, 0]) * (poly[:, 1] - prev[:, 1])
                - (poly[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
            )
            poly = np.delete(poly, int(np.argmin(weight)), axis=0)

        seeds = sorted(
            int(np.argmin(np.sum((pts - v.astype(np.float32)) ** 2, axis=1))) for v in poly
        )
        if len(set(seeds)) != n:
            return None
        # Every edge needs enough points to fit a line through.
        gaps = [(seeds[(i + 1) % n] - seeds[i]) % len(pts) for i in range(n)]
        if min(gaps) < 10:
            return None
        return seeds

    @staticmethod
    def _intersect(l1: tuple, l2: tuple) -> tuple[float, float] | None:
        """Intersection of two lines given in point-direction form."""
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        den = vx1 * vy2 - vy1 * vx2
        if abs(den) < 1e-6:  # parallel edges never met
            return None
        t = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / den
        return (x1 + t * vx1, y1 + t * vy1)

    @staticmethod
    def _classify(vertices: int, circularity: float) -> str:
        """Name the shape from its polygon approximation."""
        if circularity > 0.87 and vertices > 6:
            return "circle"
        return {
            3: "triangle",
            4: "quad",
            5: "pentagon",
            6: "hexagon",
        }.get(vertices, "polygon")

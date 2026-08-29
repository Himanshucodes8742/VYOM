"""
refiner.py – Sub-pixel refinement for lunar image registration (Phase 4).

Takes the geometrically verified inlier correspondences from Phase 3
(:class:`~src.geometry.VerificationResult`) and sharpens each match to
sub-pixel accuracy using phase-correlation on local image patches.

The workflow for each point pair is:

1. **Patch isolation** – extract a small window (e.g. 31×31 or 63×63)
   centred on the matched pixel in both the source and reference images.
2. **Phase correlation** – ``cv2.phaseCorrelate`` on the two patches
   returns the translational shift ``(dx, dy)`` at sub-pixel precision.
3. **Quality gate** – discard shifts whose phase-correlation response
   is below a configurable confidence threshold.
4. **Coordinate update** – apply the accepted shifts to produce
   ``float64`` refined coordinates.

Usage::

    from src.geometry import GeometricVerifier
    from src.refiner   import SubPixelRefiner

    verifier = GeometricVerifier()
    vr       = verifier.verify(src_pts, ref_pts, image_shape=shape)

    refiner  = SubPixelRefiner(patch_size=63)
    result   = refiner.refine(source_image, reference_image,
                              vr.src_pts, vr.ref_pts)

    print(result.src_pts_refined)   # (M, 2) float64
    print(result.mean_shift_px)     # e.g. 0.37
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PointRefinement:
    """Per-point refinement diagnostics.

    Attributes
    ----------
    index : int
        Index of this point in the original input arrays.
    dx : float
        Sub-pixel shift in x applied to the source point.
    dy : float
        Sub-pixel shift in y applied to the source point.
    phase_response : float
        Peak-response value from ``cv2.phaseCorrelate``; higher values
        indicate a sharper, more reliable correlation peak.
    accepted : bool
        Whether the shift was accepted (``True``) or rejected and the
        original coordinate retained (``False``).
    reject_reason : str
        Empty string if accepted; otherwise a short code explaining the
        rejection (``"low_response"``, ``"border"``, ``"large_shift"``).
    """

    index: int
    dx: float
    dy: float
    phase_response: float
    accepted: bool
    reject_reason: str = ""


@dataclass(frozen=True)
class RefinementResult:
    """Output of :pymeth:`SubPixelRefiner.refine`.

    Attributes
    ----------
    src_pts_refined : ndarray (M, 2) float64
        Source points after sub-pixel adjustment.
    ref_pts_refined : ndarray (M, 2) float64
        Reference points (passed through unchanged unless
        ``refine_reference=True``).
    num_refined : int
        Number of points where a sub-pixel shift was successfully applied.
    num_rejected : int
        Number of points where the shift was rejected (original coords kept).
    mean_shift_px : float
        Mean Euclidean magnitude of the accepted sub-pixel shifts.
    max_shift_px : float
        Maximum shift magnitude across all accepted points.
    per_point : list[PointRefinement]
        Detailed per-point diagnostics.
    """

    src_pts_refined: npt.NDArray[np.float64]
    ref_pts_refined: npt.NDArray[np.float64]
    num_refined: int
    num_rejected: int
    mean_shift_px: float
    max_shift_px: float
    per_point: List[PointRefinement] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _to_f64_gray(image: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Convert any image to single-channel float64 for ``phaseCorrelate``.

    ``cv2.phaseCorrelate`` requires ``CV_32F`` or ``CV_64F`` input.
    We use float64 for maximum precision in the sub-pixel domain.
    """
    # Multi-band → first band (rasterio convention: bands, H, W)
    if image.ndim == 3:
        if image.shape[0] <= image.shape[2]:
            image = image[0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.float64:
        return image

    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float64)

    # Integer types → normalise to [0, 1] float64
    lo = float(image.min())
    hi = float(image.max())
    if hi - lo < 1e-12:
        return np.zeros(image.shape, dtype=np.float64)
    return ((image.astype(np.float64) - lo) / (hi - lo))


def _apply_hann_window(patch: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply *patch* by a 2-D Hann window to reduce spectral leakage.

    Phase correlation accuracy degrades when the patch has strong edge
    discontinuities.  A Hann (raised-cosine) taper forces the borders
    toward zero smoothly.
    """
    h, w = patch.shape
    win_h = np.hanning(h).astype(np.float64)
    win_w = np.hanning(w).astype(np.float64)
    window = np.outer(win_h, win_w)
    return patch * window


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════


class SubPixelRefiner:
    """Phase-correlation-based sub-pixel refinement for match points.

    Parameters
    ----------
    patch_size : int
        Side length of the square patch extracted around each keypoint
        (must be odd so the keypoint sits at the exact centre).
        Typical values: 31 for well-localised SIFT matches,
        63 for LoFTR or texture-poor terrain.
    min_phase_response : float
        Minimum ``phaseCorrelate`` response to accept a shift.
        Points below this threshold keep their original integer
        coordinates.  0.15–0.25 is a reasonable range for lunar data.
    max_shift_px : float
        Maximum allowable shift magnitude (pixels).  Shifts larger
        than this are physically implausible for a refinement step
        and indicate a failed correlation; the point is rejected.
    use_hann_window : bool
        Apply a Hann window to each patch before phase correlation
        (recommended).
    refine_reference : bool
        If ``True``, also refine the *reference* points by applying
        the negative of the computed shift.  Default ``False`` (shifts
        are applied only to source points).

    Examples
    --------
    >>> refiner = SubPixelRefiner(patch_size=63, min_phase_response=0.2)
    >>> result  = refiner.refine(ohrc_img, lroc_img, src_pts, ref_pts)
    >>> print(result.mean_shift_px)
    0.34
    """

    def __init__(
        self,
        patch_size: int = 31,
        min_phase_response: float = 0.20,
        max_shift_px: float = 5.0,
        use_hann_window: bool = True,
        refine_reference: bool = False,
    ) -> None:
        if patch_size < 3:
            raise ValueError(f"patch_size must be ≥ 3, got {patch_size}.")
        if patch_size % 2 == 0:
            patch_size += 1
            logger.warning(
                "patch_size must be odd; bumped to %d.", patch_size
            )

        self._patch_size = patch_size
        self._half = patch_size // 2
        self._min_response = min_phase_response
        self._max_shift = max_shift_px
        self._hann = use_hann_window
        self._refine_ref = refine_reference

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def refine(
        self,
        source_image: npt.NDArray[Any],
        reference_image: npt.NDArray[Any],
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
    ) -> RefinementResult:
        """Refine all point pairs to sub-pixel accuracy.

        Parameters
        ----------
        source_image : ndarray
            Full source (moving) image – any dtype or band count.
        reference_image : ndarray
            Full reference (fixed) image.
        src_pts : ndarray (N, 2) float32
            Source matched points (from Phase 3 output).
        ref_pts : ndarray (N, 2) float32
            Corresponding reference matched points.

        Returns
        -------
        RefinementResult
        """
        src_pts = np.asarray(src_pts, dtype=np.float64)
        ref_pts = np.asarray(ref_pts, dtype=np.float64)
        self._validate_inputs(src_pts, ref_pts)

        src_gray = _to_f64_gray(source_image)
        ref_gray = _to_f64_gray(reference_image)

        src_h, src_w = src_gray.shape
        ref_h, ref_w = ref_gray.shape

        refined_src = src_pts.copy()
        refined_ref = ref_pts.copy()
        diagnostics: List[PointRefinement] = []
        accepted_shifts: List[float] = []

        for i in range(src_pts.shape[0]):
            sx, sy = src_pts[i]
            rx, ry = ref_pts[i]

            # ---- Patch isolation (Task 1) --------------------------------
            src_patch = self._extract_patch(src_gray, sx, sy, src_h, src_w)
            ref_patch = self._extract_patch(ref_gray, rx, ry, ref_h, ref_w)

            # Border rejection
            if src_patch is None or ref_patch is None:
                diagnostics.append(
                    PointRefinement(
                        index=i,
                        dx=0.0,
                        dy=0.0,
                        phase_response=0.0,
                        accepted=False,
                        reject_reason="border",
                    )
                )
                continue

            # ---- Sub-pixel shift (Task 2) --------------------------------
            dx, dy, response = self._phase_correlate(src_patch, ref_patch)

            # ---- Quality gates -------------------------------------------
            shift_mag = np.hypot(dx, dy)

            if response < self._min_response:
                diagnostics.append(
                    PointRefinement(
                        index=i,
                        dx=dx,
                        dy=dy,
                        phase_response=response,
                        accepted=False,
                        reject_reason="low_response",
                    )
                )
                continue

            if shift_mag > self._max_shift:
                diagnostics.append(
                    PointRefinement(
                        index=i,
                        dx=dx,
                        dy=dy,
                        phase_response=response,
                        accepted=False,
                        reject_reason="large_shift",
                    )
                )
                continue

            # ---- Coordinate update (Task 3) ------------------------------
            refined_src[i, 0] += dx
            refined_src[i, 1] += dy

            if self._refine_ref:
                refined_ref[i, 0] -= dx
                refined_ref[i, 1] -= dy

            accepted_shifts.append(float(shift_mag))
            diagnostics.append(
                PointRefinement(
                    index=i,
                    dx=dx,
                    dy=dy,
                    phase_response=response,
                    accepted=True,
                )
            )

        num_refined = len(accepted_shifts)
        num_rejected = src_pts.shape[0] - num_refined
        mean_shift = float(np.mean(accepted_shifts)) if accepted_shifts else 0.0
        max_shift_val = float(np.max(accepted_shifts)) if accepted_shifts else 0.0

        logger.info(
            "Sub-pixel refinement: %d / %d points refined "
            "(mean shift %.4f px, max %.4f px, %d rejected).",
            num_refined,
            src_pts.shape[0],
            mean_shift,
            max_shift_val,
            num_rejected,
        )

        return RefinementResult(
            src_pts_refined=refined_src.astype(np.float64),
            ref_pts_refined=refined_ref.astype(np.float64),
            num_refined=num_refined,
            num_rejected=num_rejected,
            mean_shift_px=mean_shift,
            max_shift_px=max_shift_val,
            per_point=diagnostics,
        )

    # ==================================================================
    # Task 1 – Patch isolation
    # ==================================================================

    def _extract_patch(
        self,
        image: npt.NDArray[np.float64],
        cx: float,
        cy: float,
        img_h: int,
        img_w: int,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Extract a square patch centred on ``(cx, cy)``.

        Returns ``None`` if the patch would extend beyond the image
        boundaries (border rejection).

        Parameters
        ----------
        image : ndarray (H, W) float64
        cx, cy : float
            Centre coordinates (x = column, y = row).
        img_h, img_w : int
            Image dimensions for bounds checking.

        Returns
        -------
        ndarray (patch_size, patch_size) float64 | None
        """
        # Round centre to nearest integer pixel
        ix = int(round(cx))
        iy = int(round(cy))

        y0 = iy - self._half
        y1 = iy + self._half + 1
        x0 = ix - self._half
        x1 = ix + self._half + 1

        if y0 < 0 or x0 < 0 or y1 > img_h or x1 > img_w:
            return None

        return image[y0:y1, x0:x1].copy()

    def extract_patch_safe(
        self,
        image: npt.NDArray[Any],
        cx: float,
        cy: float,
        patch_size: Optional[int] = None,
    ) -> Tuple[Optional[npt.NDArray[np.float64]], int, int]:
        """Public-facing patch extractor with adaptive border handling.

        When the full patch doesn't fit, the patch is shrunk to the
        largest odd window that does fit (down to a minimum of 5×5).
        The actual patch size used is returned alongside the patch.

        Parameters
        ----------
        image : ndarray
            Source image (any dtype, any band count).
        cx, cy : float
            Centre coordinates.
        patch_size : int, optional
            Override the instance patch size for this call.

        Returns
        -------
        patch : ndarray (S, S) float64 | None
            Extracted patch, or ``None`` if even the minimum window
            cannot fit.
        actual_size : int
            Side length of the returned patch.
        center_offset : int
            Half-width of the returned patch (``actual_size // 2``).
        """
        gray = _to_f64_gray(image)
        img_h, img_w = gray.shape
        ps = patch_size if patch_size is not None else self._patch_size

        ix = int(round(cx))
        iy = int(round(cy))

        # Shrink until the patch fits (minimum 5×5)
        while ps >= 5:
            half = ps // 2
            y0, y1 = iy - half, iy + half + 1
            x0, x1 = ix - half, ix + half + 1
            if y0 >= 0 and x0 >= 0 and y1 <= img_h and x1 <= img_w:
                return gray[y0:y1, x0:x1].copy(), ps, half
            ps -= 2  # stay odd

        return None, 0, 0

    # ==================================================================
    # Task 2 – Sub-pixel shift via phase correlation
    # ==================================================================

    def _phase_correlate(
        self,
        src_patch: npt.NDArray[np.float64],
        ref_patch: npt.NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        """Compute sub-pixel translational shift between two patches.

        Uses ``cv2.phaseCorrelate`` which applies the Fourier-Mellin
        theorem: the cross-power spectrum of two images that differ
        only by a translation has a peak at the translational offset,
        and the peak location is interpolated to sub-pixel resolution
        via parabolic fitting.

        Parameters
        ----------
        src_patch, ref_patch : ndarray (P, P) float64
            Equally-sized image patches.

        Returns
        -------
        dx : float
            Sub-pixel shift in x (positive = source is right of reference).
        dy : float
            Sub-pixel shift in y (positive = source is below reference).
        response : float
            Peak response value (0–1); higher = more confident.
        """
        if src_patch.shape != ref_patch.shape:
            raise ValueError(
                f"Patch shape mismatch: {src_patch.shape} vs {ref_patch.shape}."
            )

        # Optional Hann windowing to suppress spectral leakage
        if self._hann:
            src_patch = _apply_hann_window(src_patch)
            ref_patch = _apply_hann_window(ref_patch)

        # cv2.phaseCorrelate expects float32 or float64, single channel
        (dx, dy), response = cv2.phaseCorrelate(
            src_patch.astype(np.float64),
            ref_patch.astype(np.float64),
        )

        return float(dx), float(dy), float(response)

    # ==================================================================
    # Batch helpers
    # ==================================================================

    def refine_from_verification(
        self,
        source_image: npt.NDArray[Any],
        reference_image: npt.NDArray[Any],
        verification_result: Any,
    ) -> RefinementResult:
        """Convenience: refine directly from a :class:`VerificationResult`.

        Parameters
        ----------
        source_image, reference_image : ndarray
        verification_result
            The output of :pymeth:`GeometricVerifier.verify`.

        Returns
        -------
        RefinementResult
        """
        return self.refine(
            source_image,
            reference_image,
            verification_result.src_pts,
            verification_result.ref_pts,
        )

    def compute_refinement_statistics(
        self,
        result: RefinementResult,
    ) -> Dict[str, Any]:
        """Compute summary statistics from a refinement result.

        Useful for pipeline dashboards and quality-control reports.

        Returns
        -------
        dict
            ``total``, ``refined``, ``rejected``, ``acceptance_rate``,
            ``mean_shift``, ``max_shift``, ``median_shift``,
            ``std_shift``, ``reject_breakdown``.
        """
        accepted = [p for p in result.per_point if p.accepted]
        rejected = [p for p in result.per_point if not p.accepted]

        shifts = np.array(
            [np.hypot(p.dx, p.dy) for p in accepted], dtype=np.float64
        )

        # Breakdown of rejection reasons
        reasons: Dict[str, int] = {}
        for p in rejected:
            reasons[p.reject_reason] = reasons.get(p.reject_reason, 0) + 1

        total = len(result.per_point)

        return {
            "total": total,
            "refined": result.num_refined,
            "rejected": result.num_rejected,
            "acceptance_rate": result.num_refined / max(total, 1),
            "mean_shift": float(shifts.mean()) if shifts.size > 0 else 0.0,
            "max_shift": float(shifts.max()) if shifts.size > 0 else 0.0,
            "median_shift": float(np.median(shifts)) if shifts.size > 0 else 0.0,
            "std_shift": float(shifts.std()) if shifts.size > 0 else 0.0,
            "reject_breakdown": reasons,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        src_pts: npt.NDArray[np.float64],
        ref_pts: npt.NDArray[np.float64],
    ) -> None:
        """Raise ``ValueError`` for clearly invalid inputs."""
        if src_pts.ndim != 2 or src_pts.shape[1] != 2:
            raise ValueError(
                f"src_pts must be shape (N, 2), got {src_pts.shape}."
            )
        if ref_pts.ndim != 2 or ref_pts.shape[1] != 2:
            raise ValueError(
                f"ref_pts must be shape (N, 2), got {ref_pts.shape}."
            )
        if src_pts.shape[0] != ref_pts.shape[0]:
            raise ValueError(
                f"Point count mismatch: src has {src_pts.shape[0]}, "
                f"ref has {ref_pts.shape[0]}."
            )
        if src_pts.shape[0] == 0:
            raise ValueError("Cannot refine an empty point set.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"patch={self._patch_size}, "
            f"min_response={self._min_response}, "
            f"max_shift={self._max_shift}, "
            f"hann={self._hann})"
        )

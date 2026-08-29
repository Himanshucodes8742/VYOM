"""
geometry.py – Geometric verification & transformation estimation for
lunar image registration (Phase 3).

Takes raw source / reference correspondences from the matching phase
and produces:

1. **Outlier rejection** via MAGSAC++ (``cv2.USAC_MAGSAC``) – handles
   the high outlier ratios typical of cross-modal lunar matching.
2. **Spatially uniform point selection** – grid-based filter that
   prevents match clustering around prominent craters and ensures
   coverage across the full image footprint.
3. **Transformation matrix estimation** – both affine (6-DOF) and
   full perspective homography (8-DOF), with re-estimation on the
   filtered uniform inlier set.

Usage::

    from src.matcher import LunarFeatureMatcher, MatchResult
    from src.geometry import GeometricVerifier

    matcher  = LunarFeatureMatcher()
    matches  = matcher.match_sift(source, reference)

    verifier = GeometricVerifier()
    result   = verifier.verify(matches["src_pts"], matches["ref_pts"],
                               image_shape=source.shape[:2])

    print(result.transform_matrix)
    print(result.num_inliers)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Enums & constants
# ═══════════════════════════════════════════════════════════════════════


class TransformType(str, Enum):
    """Supported geometric transformation models."""

    HOMOGRAPHY = "homography"
    AFFINE = "affine"


_MIN_POINTS_HOMOGRAPHY: int = 4
_MIN_POINTS_AFFINE: int = 3


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VerificationResult:
    """Output of the full geometric-verification pipeline.

    Attributes
    ----------
    src_pts : ndarray, shape (M, 2), float32
        Filtered source (moving-image) inlier points.
    ref_pts : ndarray, shape (M, 2), float32
        Corresponding reference (fixed-image) inlier points.
    transform_matrix : ndarray, shape (3, 3), float64
        Estimated transformation matrix (homography **or** affine
        embedded in a 3×3 with last row ``[0, 0, 1]``).
    transform_type : TransformType
        Which model was used (``HOMOGRAPHY`` or ``AFFINE``).
    inlier_mask : ndarray, shape (N,), bool
        Boolean mask over the *original* input correspondences
        indicating which were classified as inliers by RANSAC /
        MAGSAC++.
    num_inliers : int
        Total inliers after outlier rejection (before spatial
        filtering).
    num_uniform : int
        Number of points remaining after spatial-uniformity filtering
        (``M``).
    reprojection_error : float
        Mean symmetric reprojection error (pixels) of the final
        filtered inlier set.
    cells_populated : int
        Number of grid cells that contributed at least one point
        during the uniformity filter.
    diagnostics : dict[str, Any]
        Extra diagnostic data (per-cell counts, RANSAC iterations, etc.)
    """

    src_pts: npt.NDArray[np.float32]
    ref_pts: npt.NDArray[np.float32]
    transform_matrix: npt.NDArray[np.float64]
    transform_type: TransformType
    inlier_mask: npt.NDArray[np.bool_]
    num_inliers: int
    num_uniform: int
    reprojection_error: float
    cells_populated: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════


class GeometricVerifier:
    """Outlier rejection, spatial filtering, and transform estimation.

    Parameters
    ----------
    ransac_reproj_thresh : float
        Maximum allowed reprojection error (pixels) for a
        correspondence to be considered an inlier.  3.0–5.0 is typical
        for cross-modal lunar data.
    ransac_confidence : float
        Desired probability that the result is outlier-free
        (0 < conf < 1).  0.999 is conservative but safe for high
        outlier ratios.
    ransac_max_iters : int
        Upper bound on RANSAC / MAGSAC++ iterations.
    grid_divisions : tuple[int, int]
        ``(rows, cols)`` of the spatial uniformity grid.
    points_per_cell : int
        Maximum number of inlier points retained per grid cell.
    transform_type : TransformType
        Default transformation model.

    Examples
    --------
    >>> verifier = GeometricVerifier(grid_divisions=(12, 12))
    >>> result   = verifier.verify(src_pts, ref_pts, image_shape=(4096, 4096))
    >>> result.num_uniform
    287
    """

    def __init__(
        self,
        ransac_reproj_thresh: float = 4.0,
        ransac_confidence: float = 0.999,
        ransac_max_iters: int = 10_000,
        grid_divisions: Tuple[int, int] = (10, 10),
        points_per_cell: int = 15,
        transform_type: TransformType = TransformType.HOMOGRAPHY,
    ) -> None:
        self._reproj_thresh = ransac_reproj_thresh
        self._confidence = ransac_confidence
        self._max_iters = ransac_max_iters
        self._grid_div = grid_divisions
        self._pts_per_cell = points_per_cell
        self._default_transform = transform_type

    # ══════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════

    def verify(
        self,
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
        *,
        image_shape: Optional[Tuple[int, int]] = None,
        transform_type: Optional[TransformType] = None,
        reestimate_on_uniform: bool = True,
    ) -> VerificationResult:
        """Run the full verification pipeline.

        1. Outlier rejection (MAGSAC++ or RANSAC).
        2. Spatial-uniformity filtering on inliers.
        3. (Optional) re-estimate the transformation on the filtered set.

        Parameters
        ----------
        src_pts : ndarray (N, 2) float32
            Source matched points from the matcher.
        ref_pts : ndarray (N, 2) float32
            Reference matched points.
        image_shape : (height, width), optional
            Shape of the source image – used to define the uniformity
            grid.  If ``None``, the grid is derived from the bounding
            box of ``src_pts``.
        transform_type : TransformType, optional
            Override instance default.
        reestimate_on_uniform : bool
            If ``True``, re-compute the transformation matrix on
            the spatially-filtered subset for a more accurate final
            estimate.

        Returns
        -------
        VerificationResult
        """
        t_type = transform_type or self._default_transform
        src_pts = np.asarray(src_pts, dtype=np.float32)
        ref_pts = np.asarray(ref_pts, dtype=np.float32)

        self._validate_inputs(src_pts, ref_pts, t_type)

        # ---- Step 1: outlier rejection -----------------------------------
        matrix, inlier_mask = self.reject_outliers(
            src_pts, ref_pts, transform_type=t_type
        )

        inlier_idx = np.flatnonzero(inlier_mask)
        num_inliers = int(inlier_idx.shape[0])
        src_inliers = src_pts[inlier_idx]
        ref_inliers = ref_pts[inlier_idx]

        logger.info(
            "Outlier rejection: %d / %d inliers (%.1f %%).",
            num_inliers,
            src_pts.shape[0],
            100.0 * num_inliers / max(src_pts.shape[0], 1),
        )

        if num_inliers == 0:
            return self._empty_result(src_pts.shape[0], t_type)

        # ---- Step 2: spatial uniformity ----------------------------------
        reproj_errors = self._reprojection_errors(
            src_inliers, ref_inliers, matrix
        )
        uniform_idx, cells_populated, cell_counts = (
            self.filter_uniform(
                src_inliers,
                confidence_scores=1.0 / (reproj_errors + 1e-6),
                image_shape=image_shape,
            )
        )

        src_uniform = src_inliers[uniform_idx]
        ref_uniform = ref_inliers[uniform_idx]

        logger.info(
            "Spatial filter: %d → %d points (%d / %d cells populated).",
            num_inliers,
            src_uniform.shape[0],
            cells_populated,
            self._grid_div[0] * self._grid_div[1],
        )

        # ---- Step 3: (re-)estimate transformation -----------------------
        if reestimate_on_uniform and src_uniform.shape[0] >= (
            _MIN_POINTS_HOMOGRAPHY
            if t_type == TransformType.HOMOGRAPHY
            else _MIN_POINTS_AFFINE
        ):
            matrix, _ = self.reject_outliers(
                src_uniform, ref_uniform, transform_type=t_type
            )
            logger.debug("Transformation re-estimated on uniform set.")

        mean_err = float(
            self._reprojection_errors(
                src_uniform, ref_uniform, matrix
            ).mean()
        ) if src_uniform.shape[0] > 0 else float("inf")

        logger.info(
            "Final transform (%s): %d points, mean reproj error = %.3f px.",
            t_type.value,
            src_uniform.shape[0],
            mean_err,
        )

        return VerificationResult(
            src_pts=src_uniform,
            ref_pts=ref_uniform,
            transform_matrix=matrix,
            transform_type=t_type,
            inlier_mask=inlier_mask,
            num_inliers=num_inliers,
            num_uniform=int(src_uniform.shape[0]),
            reprojection_error=mean_err,
            cells_populated=cells_populated,
            diagnostics={"cell_counts": cell_counts},
        )

    # ==================================================================
    # Task 1 – Outlier rejection (MAGSAC++)
    # ==================================================================

    def reject_outliers(
        self,
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
        *,
        transform_type: Optional[TransformType] = None,
        reproj_thresh: Optional[float] = None,
        confidence: Optional[float] = None,
        max_iters: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Robust outlier rejection using MAGSAC++.

        MAGSAC++ is preferred over classical RANSAC because it does not
        require a hard inlier/outlier threshold — it marginalises over
        a range of σ values, making it far more tolerant of the
        heterogeneous error distribution seen in cross-modal lunar
        matches.

        Parameters
        ----------
        src_pts, ref_pts : ndarray (N, 2) float32
        transform_type : TransformType, optional
        reproj_thresh : float, optional
            Override instance ``ransac_reproj_thresh``.
        confidence : float, optional
            Override instance ``ransac_confidence``.
        max_iters : int, optional
            Override instance ``ransac_max_iters``.

        Returns
        -------
        matrix : ndarray (3, 3) float64
            Estimated transformation matrix.
        inlier_mask : ndarray (N,) bool
            ``True`` for inlier correspondences.

        Raises
        ------
        ValueError
            If fewer points than required by the model are provided.
        RuntimeError
            If OpenCV fails to find any valid model.
        """
        t_type = transform_type or self._default_transform
        thresh = reproj_thresh if reproj_thresh is not None else self._reproj_thresh
        conf = confidence if confidence is not None else self._confidence
        iters = max_iters if max_iters is not None else self._max_iters

        self._validate_inputs(src_pts, ref_pts, t_type)

        # Select the best available robust estimator.
        # USAC_MAGSAC is available from OpenCV 4.5+; fall back to RANSAC.
        usac_flag = getattr(cv2, "USAC_MAGSAC", None)
        if usac_flag is None:
            logger.warning(
                "cv2.USAC_MAGSAC not available (OpenCV < 4.5); "
                "falling back to cv2.RANSAC."
            )
            usac_flag = cv2.RANSAC

        if t_type == TransformType.HOMOGRAPHY:
            matrix, mask = cv2.findHomography(
                src_pts,
                ref_pts,
                method=usac_flag,
                ransacReprojThreshold=thresh,
                maxIters=iters,
                confidence=conf,
            )
        else:
            # Affine (6 DOF) – estimateAffine2D also accepts USAC flags
            matrix_2x3, mask = cv2.estimateAffine2D(
                src_pts,
                ref_pts,
                method=usac_flag,
                ransacReprojThreshold=thresh,
                maxIters=iters,
                confidence=conf,
            )
            if matrix_2x3 is not None:
                # Embed 2×3 into 3×3 with [0, 0, 1] last row
                matrix = np.vstack(
                    [matrix_2x3, np.array([0.0, 0.0, 1.0])]
                )
            else:
                matrix = None

        if matrix is None or mask is None:
            raise RuntimeError(
                "Geometric estimation failed — no valid model found. "
                "The match set may be too small or entirely degenerate."
            )

        inlier_mask: npt.NDArray[np.bool_] = mask.ravel().astype(bool)
        matrix = matrix.astype(np.float64)

        logger.debug(
            "MAGSAC++ (%s): %d / %d inliers, thresh=%.1f, conf=%.4f.",
            t_type.value,
            int(inlier_mask.sum()),
            src_pts.shape[0],
            thresh,
            conf,
        )

        return matrix, inlier_mask

    # ==================================================================
    # Task 2 – Spatially uniform point selection
    # ==================================================================

    def filter_uniform(
        self,
        points: npt.NDArray[np.float32],
        *,
        confidence_scores: Optional[npt.NDArray[Any]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        grid_divisions: Optional[Tuple[int, int]] = None,
        points_per_cell: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.intp], int, Dict[Tuple[int, int], int]]:
        """Grid-based spatial filter for uniform point distribution.

        Divides the coordinate space into a grid and keeps at most the
        top-*N* highest-confidence points per cell.  This prevents
        match clustering around a single large crater (e.g. Copernicus)
        from dominating the transform estimate while under-constraining
        the geometry in featureless mare regions.

        Parameters
        ----------
        points : ndarray (M, 2) float32
            2-D point coordinates (typically ``src_pts`` from inliers).
        confidence_scores : ndarray (M,), optional
            Per-point score; higher is better.  Points with the highest
            scores within each cell are kept.  If ``None``, selection
            within each cell is arbitrary (first-come).
        image_shape : (height, width), optional
            Used to define the grid extent.  If ``None``, the bounding
            box of *points* is used (with a small margin).
        grid_divisions : (rows, cols), optional
            Override instance default.
        points_per_cell : int, optional
            Override instance default.

        Returns
        -------
        selected_indices : ndarray of int
            Indices into *points* of the retained subset.
        cells_populated : int
            Number of grid cells that contain at least one point.
        cell_counts : dict[(row, col), int]
            Number of points selected from each populated cell.
        """
        n_rows, n_cols = grid_divisions or self._grid_div
        max_per_cell = points_per_cell or self._pts_per_cell

        if points.shape[0] == 0:
            return (
                np.empty(0, dtype=np.intp),
                0,
                {},
            )

        # ---- Define grid extent ------------------------------------------
        if image_shape is not None:
            h, w = image_shape
            x_min, y_min = 0.0, 0.0
            x_max, y_max = float(w), float(h)
        else:
            margin = 1.0
            x_min = float(points[:, 0].min()) - margin
            y_min = float(points[:, 1].min()) - margin
            x_max = float(points[:, 0].max()) + margin
            y_max = float(points[:, 1].max()) + margin

        cell_w = (x_max - x_min) / n_cols
        cell_h = (y_max - y_min) / n_rows

        # Guard against degenerate cases
        if cell_w < 1e-9 or cell_h < 1e-9:
            logger.warning("Grid cell size is near-zero; returning all points.")
            idx = np.arange(points.shape[0], dtype=np.intp)
            return idx, 1, {(0, 0): int(points.shape[0])}

        # ---- Assign each point to a cell ---------------------------------
        col_idx = np.clip(
            ((points[:, 0] - x_min) / cell_w).astype(int),
            0,
            n_cols - 1,
        )
        row_idx = np.clip(
            ((points[:, 1] - y_min) / cell_h).astype(int),
            0,
            n_rows - 1,
        )

        # ---- Select top-N per cell ---------------------------------------
        if confidence_scores is not None:
            scores = np.asarray(confidence_scores, dtype=np.float64)
        else:
            scores = np.ones(points.shape[0], dtype=np.float64)

        # Build cell → list of (score, global_index)
        cells: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
        for i in range(points.shape[0]):
            key = (int(row_idx[i]), int(col_idx[i]))
            cells.setdefault(key, []).append((float(scores[i]), i))

        selected: List[int] = []
        cell_counts: Dict[Tuple[int, int], int] = {}

        for key, entries in cells.items():
            # Sort descending by score and keep top-N
            entries.sort(key=lambda t: t[0], reverse=True)
            kept = entries[:max_per_cell]
            for _, idx in kept:
                selected.append(idx)
            cell_counts[key] = len(kept)

        selected_arr = np.array(sorted(selected), dtype=np.intp)
        cells_populated = len(cells)

        logger.debug(
            "Uniform filter: %d → %d points across %d cells "
            "(grid %dx%d, max %d/cell).",
            points.shape[0],
            selected_arr.shape[0],
            cells_populated,
            n_rows,
            n_cols,
            max_per_cell,
        )

        return selected_arr, cells_populated, cell_counts

    # ==================================================================
    # Task 3 – Transformation matrix computation
    # ==================================================================

    def compute_transform(
        self,
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
        *,
        transform_type: Optional[TransformType] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Compute the final transformation matrix.

        This is a thin convenience wrapper around :pymeth:`reject_outliers`
        that explicitly communicates intent.  It uses the same MAGSAC++
        estimator internally.

        Parameters
        ----------
        src_pts, ref_pts : ndarray (N, 2) float32
        transform_type : TransformType, optional

        Returns
        -------
        matrix : ndarray (3, 3) float64
            3×3 transformation matrix.  For ``AFFINE`` the last row is
            ``[0, 0, 1]``.
        inlier_mask : ndarray (N,) bool
        """
        return self.reject_outliers(
            src_pts, ref_pts, transform_type=transform_type
        )

    def compute_affine(
        self,
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Shortcut: compute a 6-DOF affine transform."""
        return self.compute_transform(
            src_pts, ref_pts, transform_type=TransformType.AFFINE
        )

    def compute_homography(
        self,
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Shortcut: compute an 8-DOF perspective homography."""
        return self.compute_transform(
            src_pts, ref_pts, transform_type=TransformType.HOMOGRAPHY
        )

    # ==================================================================
    # Diagnostic helpers
    # ==================================================================

    @staticmethod
    def _reprojection_errors(
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
        matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute per-point symmetric reprojection error.

        For homography *H*:

        .. math::

            e_i = \\frac{1}{2}\\bigl(
                \\|H\\,p_i^{\\text{src}} - p_i^{\\text{ref}}\\|
              + \\|H^{-1}\\,p_i^{\\text{ref}} - p_i^{\\text{src}}\\|
            \\bigr)

        Parameters
        ----------
        src_pts, ref_pts : ndarray (N, 2)
        matrix : ndarray (3, 3)

        Returns
        -------
        ndarray (N,) float64
        """
        n = src_pts.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64)

        # Forward: H @ src → ref_hat
        ones = np.ones((n, 1), dtype=np.float64)
        src_h = np.hstack([src_pts.astype(np.float64), ones])  # (N, 3)
        proj_fwd = (matrix @ src_h.T).T  # (N, 3)
        proj_fwd = proj_fwd[:, :2] / proj_fwd[:, 2:3]
        err_fwd = np.linalg.norm(
            proj_fwd - ref_pts.astype(np.float64), axis=1
        )

        # Backward: H⁻¹ @ ref → src_hat
        try:
            matrix_inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return err_fwd  # degenerate — return forward-only

        ref_h = np.hstack([ref_pts.astype(np.float64), ones])
        proj_bwd = (matrix_inv @ ref_h.T).T
        proj_bwd = proj_bwd[:, :2] / proj_bwd[:, 2:3]
        err_bwd = np.linalg.norm(
            proj_bwd - src_pts.astype(np.float64), axis=1
        )

        return (err_fwd + err_bwd) / 2.0

    @staticmethod
    def decompose_homography(
        matrix: npt.NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Extract human-readable geometric parameters from a 3×3 matrix.

        Uses SVD of the upper-left 2×2 block to report scale, rotation,
        and translation — useful for sanity-checking that the estimated
        transform is physically plausible for lunar imagery (e.g.
        rotation should be small, scale near unity for same-sensor
        data).

        Returns
        -------
        dict
            ``scale_x``, ``scale_y``, ``rotation_deg``,
            ``translation_x``, ``translation_y``, ``is_affine``.
        """
        m = matrix.astype(np.float64)
        tx = m[0, 2]
        ty = m[1, 2]
        is_affine = np.allclose(m[2, :], [0, 0, 1], atol=1e-8)

        upper = m[:2, :2]
        U, S, Vt = np.linalg.svd(upper)
        rotation = np.arctan2(U[1, 0], U[0, 0])

        return {
            "scale_x": float(S[0]),
            "scale_y": float(S[1]),
            "rotation_deg": float(np.degrees(rotation)),
            "translation_x": float(tx),
            "translation_y": float(ty),
            "is_affine": bool(is_affine),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        src_pts: npt.NDArray[np.float32],
        ref_pts: npt.NDArray[np.float32],
        t_type: TransformType,
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
                f"Point count mismatch: src_pts has {src_pts.shape[0]}, "
                f"ref_pts has {ref_pts.shape[0]}."
            )
        min_pts = (
            _MIN_POINTS_HOMOGRAPHY
            if t_type == TransformType.HOMOGRAPHY
            else _MIN_POINTS_AFFINE
        )
        if src_pts.shape[0] < min_pts:
            raise ValueError(
                f"{t_type.value} requires at least {min_pts} point pairs, "
                f"got {src_pts.shape[0]}."
            )

    def _empty_result(
        self,
        n_original: int,
        t_type: TransformType,
    ) -> VerificationResult:
        """Return a valid but empty :class:`VerificationResult`."""
        return VerificationResult(
            src_pts=np.empty((0, 2), dtype=np.float32),
            ref_pts=np.empty((0, 2), dtype=np.float32),
            transform_matrix=np.eye(3, dtype=np.float64),
            transform_type=t_type,
            inlier_mask=np.zeros(n_original, dtype=bool),
            num_inliers=0,
            num_uniform=0,
            reprojection_error=float("inf"),
            cells_populated=0,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"thresh={self._reproj_thresh}, "
            f"grid={self._grid_div}, "
            f"pts_per_cell={self._pts_per_cell}, "
            f"model={self._default_transform.value})"
        )

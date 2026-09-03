"""Registration accuracy metrics for lunar image alignment evaluation."""

import numpy as np
import cv2


def compute_metrics(
    good_matches: list[cv2.DMatch],
    kp_source: list[cv2.KeyPoint],
    kp_reference: list[cv2.KeyPoint],
    transform_matrix: np.ndarray,
    total_raw_matches: int | None = None,
    grid_size: int = 4,
) -> dict:
    """Compute quantitative registration quality metrics.

    Args:
        good_matches: Spatially-filtered inlier DMatch objects.
        kp_source: Source image keypoints (full list from detection).
        kp_reference: Reference image keypoints (full list from detection).
        transform_matrix: Estimated 3x3 homography.
        total_raw_matches: Total matches before RANSAC/spatial filtering (for
                           inlier_ratio calculation). If None, ratio is set to 0.
        grid_size: Grid divisions for the distribution_score (rows x cols).

    Returns:
        dict with keys:
          - rmse: Root Mean Square reprojection Error in pixels.
          - inlier_count: Number of good (spatially-filtered inlier) matches.
          - inlier_ratio: inlier_count / total_raw_matches (0.0..1.0).
          - distribution_score: Fraction of grid cells containing at least one
                                inlier match (0.0..1.0). Higher is better.
    """
    inlier_count = len(good_matches)

    # Default metrics for degenerate cases
    if inlier_count == 0:
        return {
            "rmse": float("inf"),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "distribution_score": 0.0,
        }

    # --- RMSE: reprojection error ---
    # Project source keypoints through the homography and measure distance
    # to the matched reference keypoints.
    src_pts = np.float32(
        [kp_source[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    ref_pts = np.float32(
        [kp_reference[m.trainIdx].pt for m in good_matches]
    )

    projected = cv2.perspectiveTransform(src_pts, transform_matrix).reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - ref_pts) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # --- Inlier ratio ---
    if total_raw_matches is not None and total_raw_matches > 0:
        inlier_ratio = inlier_count / total_raw_matches
    else:
        inlier_ratio = 0.0

    # --- Distribution score ---
    # What fraction of a grid_size x grid_size grid over the reference image
    # contains at least one inlier match?
    x_coords = ref_pts[:, 0]
    y_coords = ref_pts[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    eps = 1e-6
    cell_w = (x_max - x_min + eps) / grid_size
    cell_h = (y_max - y_min + eps) / grid_size

    occupied_cells: set[tuple[int, int]] = set()
    for x, y in zip(x_coords, y_coords):
        col = min(int((x - x_min) / cell_w), grid_size - 1)
        row = min(int((y - y_min) / cell_h), grid_size - 1)
        occupied_cells.add((row, col))

    total_cells = grid_size * grid_size
    distribution_score = len(occupied_cells) / total_cells

    return {
        "rmse": rmse,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "distribution_score": distribution_score,
    }

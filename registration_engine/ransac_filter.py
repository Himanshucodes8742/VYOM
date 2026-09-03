"""RANSAC-based outlier filtering with spatial uniformity enforcement."""

import numpy as np
import cv2


def filter_matches(
    matches: list[cv2.DMatch],
    kp_source: list[cv2.KeyPoint],
    kp_reference: list[cv2.KeyPoint],
    grid_size: int = 4,
) -> tuple[np.ndarray, list[cv2.DMatch]]:
    """Filter matches using RANSAC homography then enforce spatial spread.

    Args:
        matches: List of DMatch objects from the feature matching stage.
        kp_source: Keypoints detected in the source image.
        kp_reference: Keypoints detected in the reference image.
        grid_size: Number of rows and columns for the spatial uniformity grid.

    Returns:
        Tuple of (transform_matrix, good_matches).
        transform_matrix is a 3x3 homography (np.ndarray).
        good_matches is the spatially-filtered list of inlier DMatch objects.
    """
    if len(matches) < 4:
        # Not enough matches to compute a homography (minimum 4 point pairs)
        return np.eye(3, dtype=np.float64), []

    # Extract matched point coordinates
    src_pts = np.float32([kp_source[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ref_pts = np.float32([kp_reference[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography with RANSAC — ransacReprojThreshold = 5.0 pixels
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

    if H is None:
        return np.eye(3, dtype=np.float64), []

    # Keep only RANSAC inliers
    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, is_inlier in zip(matches, inlier_mask) if is_inlier]

    if len(inlier_matches) == 0:
        return H, []

    # ------------------------------------------------------------------
    # Spatial-uniformity enforcement
    #
    # The SIH problem statement specifically requires that matched points
    # are evenly distributed across the image, not clustered in one high-
    # texture corner (e.g. a single large crater rim). Without this step,
    # RANSAC happily returns a geometrically valid homography supported
    # by a dense cluster of points in one region, which may still fail to
    # represent the overall alignment quality across the full frame.
    #
    # Strategy: divide the reference image into a grid_size x grid_size
    # grid of cells. If any single cell contains more than 3x the average
    # number of inliers per cell, keep only the strongest (lowest-distance)
    # 3x-average matches in that cell and discard the rest.
    # ------------------------------------------------------------------

    # Find the bounding box of inlier reference points to define the grid
    ref_inlier_pts = np.float32(
        [kp_reference[m.trainIdx].pt for m in inlier_matches]
    )
    x_min, y_min = ref_inlier_pts.min(axis=0)
    x_max, y_max = ref_inlier_pts.max(axis=0)

    # Add a tiny epsilon to avoid division by zero on degenerate cases
    eps = 1e-6
    cell_w = (x_max - x_min + eps) / grid_size
    cell_h = (y_max - y_min + eps) / grid_size

    # Assign each inlier match to a grid cell
    cells: dict[tuple[int, int], list[cv2.DMatch]] = {}
    for m in inlier_matches:
        x, y = kp_reference[m.trainIdx].pt
        col = min(int((x - x_min) / cell_w), grid_size - 1)
        row = min(int((y - y_min) / cell_h), grid_size - 1)
        cells.setdefault((row, col), []).append(m)

    # Compute average inliers per cell (across ALL cells, including empty ones)
    total_cells = grid_size * grid_size
    avg_per_cell = len(inlier_matches) / total_cells
    cap = max(1, int(3 * avg_per_cell))  # 3x average cap

    # Cap over-represented cells, keeping the strongest (lowest distance) matches
    spread_filtered: list[cv2.DMatch] = []
    for cell_key, cell_matches in cells.items():
        if len(cell_matches) > cap:
            # Sort by match distance (ascending = best first) and keep only cap
            cell_matches_sorted = sorted(cell_matches, key=lambda dm: dm.distance)
            spread_filtered.extend(cell_matches_sorted[:cap])
        else:
            spread_filtered.extend(cell_matches)

    # Re-estimate homography on the spatially-filtered inliers for consistency
    if len(spread_filtered) >= 4:
        src_filt = np.float32(
            [kp_source[m.queryIdx].pt for m in spread_filtered]
        ).reshape(-1, 1, 2)
        ref_filt = np.float32(
            [kp_reference[m.trainIdx].pt for m in spread_filtered]
        ).reshape(-1, 1, 2)
        H_refined, mask_refined = cv2.findHomography(src_filt, ref_filt, cv2.RANSAC, 5.0)
        if H_refined is not None:
            H = H_refined
            refined_mask = mask_refined.ravel().astype(bool)
            spread_filtered = [
                m for m, keep in zip(spread_filtered, refined_mask) if keep
            ]

    return H, spread_filtered

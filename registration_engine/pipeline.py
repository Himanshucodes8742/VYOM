"""End-to-end registration pipeline orchestrating all engine modules."""

import traceback
import numpy as np

from registration_engine.io_utils import load_and_resample
from registration_engine.preprocessing import clahe
from registration_engine.matchers import detect_and_match
from registration_engine.ransac_filter import filter_matches
from registration_engine.warp import warp_image
from registration_engine.metrics import compute_metrics


def run_pipeline(
    source_path: str,
    reference_path: str,
    algorithm: str = "sift",
) -> dict:
    """Execute the full image registration pipeline.

    Stages:
      1. load_and_resample  — read both images as grayscale arrays
      2. clahe              — contrast-enhance both images
      3. detect_and_match   — find keypoints and ratio-tested matches
      4. filter_matches     — RANSAC + spatial uniformity filtering
      5. warp_image         — warp source into reference frame
      6. compute_metrics    — RMSE, inlier count/ratio, distribution score

    Args:
        source_path: Path to the source image file.
        reference_path: Path to the reference image file.
        algorithm: Feature detection algorithm ("sift" or "akaze").

    Returns:
        dict with keys:
          - success (bool): True if pipeline completed without error.
          - warped_image (np.ndarray | None): Source image warped onto reference frame.
          - matches (list): Spatially-filtered inlier DMatch objects.
        - metrics (dict | None): Registration quality metrics.
        - transform_matrix (np.ndarray | None): Estimated 3x3 homography.
        - kp_source (list | None): Source keypoints.
        - kp_reference (list | None): Reference keypoints.
        - error (str | None): Human-readable error message if success is False.
    """
    result = {
        "success": False,
        "warped_image": None,
        "matches": [],
        "metrics": None,
        "transform_matrix": None,
        "kp_source": None,
        "kp_reference": None,
        "error": None,
    }

    try:
        # Stage 1: Load images
        source_img, reference_img = load_and_resample(source_path, reference_path)

        # Stage 2: CLAHE contrast enhancement on both images
        source_enhanced = clahe(source_img)
        reference_enhanced = clahe(reference_img)

        # Stage 3: Detect features and match with ratio test
        raw_matches, kp_source, kp_reference = detect_and_match(
            source_enhanced, reference_enhanced, algorithm=algorithm
        )

        if len(raw_matches) < 4:
            result["error"] = (
                f"Insufficient matches found ({len(raw_matches)}). "
                "Need at least 4 to compute a homography. "
                "Try a different algorithm or check that images overlap."
            )
            return result

        # Stage 4: RANSAC + spatial uniformity filtering
        transform_matrix, good_matches = filter_matches(
            raw_matches, kp_source, kp_reference
        )
        result["transform_matrix"] = transform_matrix

        if len(good_matches) == 0:
            result["error"] = (
                "RANSAC found no inliers. The images may not overlap or "
                "the transform may be too extreme for the current settings."
            )
            return result

        # Stage 5: Warp source image into reference frame
        warped = warp_image(source_img, transform_matrix, reference_img.shape)
        result["warped_image"] = warped

        # Stage 6: Compute accuracy metrics
        metrics = compute_metrics(
            good_matches,
            kp_source,
            kp_reference,
            transform_matrix,
            total_raw_matches=len(raw_matches),
        )
        result["metrics"] = metrics
        result["matches"] = good_matches
        result["kp_source"] = kp_source
        result["kp_reference"] = kp_reference
        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result

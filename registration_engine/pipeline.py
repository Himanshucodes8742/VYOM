"""End-to-end registration pipeline orchestrating all engine modules."""

import traceback
import numpy as np
import time
import cv2

from registration_engine.io_utils import load_and_resample
from registration_engine.preprocessing import clahe, preprocess_image
from registration_engine.matchers import detect_and_match, filter_with_learned_verifier
from registration_engine.ransac_filter import filter_matches
from registration_engine.warp import warp_image
from registration_engine.metrics import compute_metrics


def run_pipeline(
    source_path: str,
    reference_path: str,
    algorithm: str = "sift",
    preprocessing: str = "clahe",
    source_sun_elevation: float | None = None,
    reference_sun_elevation: float | None = None,
) -> dict:
    """Execute the full image registration pipeline.

    Stages:
      1. load_and_resample  — read both images as grayscale arrays
      2. preprocessing      — contrast/photometric enhancement (CLAHE, photometric, etc.)
      3. detect_and_match   — find keypoints and ratio-tested matches (or ML verified)
      4. filter_matches     — RANSAC + spatial uniformity filtering
      5. warp_image         — warp source into reference frame
      6. compute_metrics    — RMSE, inlier count/ratio, distribution score

    Args:
        source_path: Path to the source image file.
        reference_path: Path to the reference image file.
        algorithm: Feature detection algorithm ("sift", "akaze", "rift2", "learned_verifier", "learned_descriptor", or "crater_landmarks").
        preprocessing: Preprocessing option ("clahe", "photometric", "photometric_clahe", "histogram").
        source_sun_elevation: Optional sun elevation angle for source image in degrees.
        reference_sun_elevation: Optional sun elevation angle for reference image in degrees.

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
        start_time = time.time()
        
        # Stage 1: Load images
        source_img, reference_img = load_and_resample(source_path, reference_path)

        # Stage 2: Preprocessing
        prep_key = preprocessing.strip().lower()
        if prep_key in ("photometric", "photometric_clahe", "photometric+clahe"):
            if source_sun_elevation is not None:
                source_enhanced = preprocess_image(
                    source_img, method=prep_key, sun_elevation_deg=source_sun_elevation
                )
            else:
                source_enhanced = clahe(source_img)

            if reference_sun_elevation is not None:
                reference_enhanced = preprocess_image(
                    reference_img, method=prep_key, sun_elevation_deg=reference_sun_elevation
                )
            else:
                reference_enhanced = clahe(reference_img) if "clahe" in prep_key else reference_img
        elif prep_key in ("histogram", "histogram_match"):
            source_enhanced = preprocess_image(source_img, method="histogram", reference_image=reference_img)
            reference_enhanced = reference_img
        else:
            source_enhanced = clahe(source_img)
            reference_enhanced = clahe(reference_img)

        # Stage 3: Detect features and match with ratio test
        alg_lower = algorithm.strip().lower()
        if alg_lower == "learned_verifier":
            # Run AKAZE detection and pre-filter with learned ML verifier before RANSAC
            raw_matches, kp_source, kp_reference = detect_and_match(
                source_enhanced, reference_enhanced, algorithm="akaze"
            )
            raw_matches = filter_with_learned_verifier(
                raw_matches, kp_source, kp_reference
            )
        elif alg_lower == "crater_landmarks":
            raw_matches, kp_source, kp_reference = detect_and_match(
                source_img,
                reference_img,
                algorithm="crater_landmarks",
            )
        else:
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
        
        runtime = time.time() - start_time
        metrics["runtime"] = runtime
        
        result["metrics"] = metrics
        result["matches"] = good_matches
        result["kp_source"] = kp_source
        result["kp_reference"] = kp_reference
        result["success"] = True

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return result


def detect_changes(
    source_path: str,
    reference_path: str,
    sensor_pair: str = "OHRC -> LROC NAC",
    algorithm: str = "akaze",
    threshold: float = 30.0,
    min_region_area: int = 50,
    preprocessing: str = "clahe",
    source_sun_elevation: float | None = None,
    reference_sun_elevation: float | None = None,
    morph_kernel_size: int = 5,
) -> dict:
    """Detect surface changes between co-registered lunar orbital images.

    Executes registration pipeline, normalizes illumination with preprocessing,
    warps the source image into reference coordinates, computes pixel-wise
    absolute difference on illumination-normalized rasters, applies thresholding
    and morphological opening, and extracts connected changed regions with
    spatial statistics.

    Args:
        source_path: Path to source image.
        reference_path: Path to reference image.
        sensor_pair: Descriptive sensor pair name.
        algorithm: Registration algorithm ("akaze", "sift", "crater_landmarks", etc.).
        threshold: Absolute difference intensity threshold (0-255) for change flag.
        min_region_area: Minimum connected pixel area to qualify as a valid change region.
        preprocessing: Illumination normalization method ("clahe", "photometric", etc.).
        source_sun_elevation: Sun elevation angle for source image in degrees.
        reference_sun_elevation: Sun elevation angle for reference image in degrees.
        morph_kernel_size: Kernel size for morphological noise removal opening filter.

    Returns:
        dict containing:
            - success (bool): True if registration and change detection converged.
            - change_mask (np.ndarray | None): Binary uint8 mask (255=change, 0=unchanged).
            - diff_image (np.ndarray | None): Grayscale uint8 absolute difference map.
            - warped_image (np.ndarray | None): Registered source image.
            - reference_image (np.ndarray | None): Reference image.
            - change_percentage (float): Percentage of overlapping area flagged as changed.
            - changed_pixels (int): Total count of pixels flagged as changed.
            - total_valid_pixels (int): Total overlapping valid pixel area.
            - region_count (int): Number of distinct connected change regions.
            - regions (list of dict): Details (id, area, bbox, centroid, mean_diff) of largest regions.
            - registration_metrics (dict | None): RMSE, inlier count/ratio from registration.
            - transform_matrix (np.ndarray | None): Estimated 3x3 homography.
            - error (str | None): Error description if failure occurred.
    """
    output = {
        "success": False,
        "change_mask": None,
        "diff_image": None,
        "warped_image": None,
        "reference_image": None,
        "warped_preprocessed": None,
        "reference_preprocessed": None,
        "valid_overlap_mask": None,
        "change_percentage": 0.0,
        "changed_pixels": 0,
        "total_valid_pixels": 0,
        "region_count": 0,
        "regions": [],
        "registration_metrics": None,
        "transform_matrix": None,
        "error": None,
    }

    try:
        # Step 1: Run registration pipeline to get homography and warped image
        reg_result = run_pipeline(
            source_path=source_path,
            reference_path=reference_path,
            algorithm=algorithm,
            preprocessing=preprocessing,
            source_sun_elevation=source_sun_elevation,
            reference_sun_elevation=reference_sun_elevation,
        )

        if not reg_result["success"] or reg_result["transform_matrix"] is None:
            output["error"] = reg_result.get(
                "error", "Registration step failed; cannot co-register images for change detection."
            )
            output["registration_metrics"] = reg_result.get("metrics")
            return output

        transform_matrix = reg_result["transform_matrix"]
        output["transform_matrix"] = transform_matrix
        output["registration_metrics"] = reg_result["metrics"]
        output["warped_image"] = reg_result["warped_image"]

        # Step 2: Load raw images and apply chosen preprocessing for illumination normalization
        source_img, reference_img = load_and_resample(source_path, reference_path)
        output["reference_image"] = reference_img

        prep_key = preprocessing.strip().lower()
        if prep_key in ("photometric", "photometric_clahe", "photometric+clahe"):
            if source_sun_elevation is not None:
                source_enhanced = preprocess_image(
                    source_img, method=prep_key, sun_elevation_deg=source_sun_elevation
                )
            else:
                source_enhanced = clahe(source_img)

            if reference_sun_elevation is not None:
                reference_enhanced = preprocess_image(
                    reference_img, method=prep_key, sun_elevation_deg=reference_sun_elevation
                )
            else:
                reference_enhanced = (
                    clahe(reference_img) if "clahe" in prep_key else reference_img
                )
        elif prep_key in ("histogram", "histogram_match"):
            source_enhanced = preprocess_image(
                source_img, method="histogram", reference_image=reference_img
            )
            reference_enhanced = reference_img
        else:
            source_enhanced = clahe(source_img)
            reference_enhanced = clahe(reference_img)

        # Step 3: Warp preprocessed source into reference frame
        ref_h, ref_w = reference_img.shape[:2]
        warped_enhanced = warp_image(source_enhanced, transform_matrix, (ref_h, ref_w))
        output["warped_preprocessed"] = warped_enhanced
        output["reference_preprocessed"] = reference_enhanced

        # Step 4: Compute valid overlap mask to exclude black homography borders
        src_mask = np.full(source_img.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            src_mask,
            transform_matrix,
            (ref_w, ref_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Erode border by 2px to eliminate interpolation edge ringing artifacts
        kernel_border = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        valid_overlap = cv2.erode(warped_mask, kernel_border, iterations=2) > 0

        # Valid overlap is strictly within co-registered image boundaries
        total_valid_pixels = int(np.count_nonzero(valid_overlap))
        output["valid_overlap_mask"] = valid_overlap
        output["total_valid_pixels"] = total_valid_pixels

        if total_valid_pixels == 0:
            output["error"] = "No overlapping surface area found between warped source and reference image."
            return output

        # Step 5: Compute absolute difference between illumination-normalized images
        diff = cv2.absdiff(warped_enhanced, reference_enhanced)
        diff_masked = np.where(valid_overlap, diff, 0).astype(np.uint8)
        output["diff_image"] = diff_masked

        # Step 6: Threshold difference to produce raw binary change mask
        thresh_val = float(threshold)
        _, binary_raw = cv2.threshold(diff_masked, thresh_val, 255, cv2.THRESH_BINARY)
        binary_raw[~valid_overlap] = 0

        # Step 7: Morphological opening to suppress single-pixel and salt noise
        k_size = max(3, int(morph_kernel_size))
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        change_mask = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel_morph)
        output["change_mask"] = change_mask

        # Step 8: Connected components extraction with spatial statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_mask, connectivity=8
        )

        regions = []
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_region_area:
                x = int(stats[lbl, cv2.CC_STAT_LEFT])
                y = int(stats[lbl, cv2.CC_STAT_TOP])
                w = int(stats[lbl, cv2.CC_STAT_WIDTH])
                h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[lbl]

                # Mean intensity delta in this region
                region_pixels = (labels == lbl)
                mean_diff = float(np.mean(diff_masked[region_pixels])) if np.any(region_pixels) else 0.0

                regions.append({
                    "id": len(regions) + 1,
                    "label": int(lbl),
                    "area": area,
                    "bbox": [x, y, w, h],
                    "centroid": [round(float(cx), 2), round(float(cy), 2)],
                    "mean_intensity_diff": round(mean_diff, 2),
                })

        # Sort regions largest to smallest
        regions.sort(key=lambda r: r["area"], reverse=True)
        # Re-index IDs so #1 is largest
        for idx, r in enumerate(regions):
            r["id"] = idx + 1

        changed_pixels = int(np.count_nonzero(change_mask))
        change_pct = round(float((changed_pixels / max(total_valid_pixels, 1)) * 100), 2)

        output["changed_pixels"] = changed_pixels
        output["change_percentage"] = change_pct
        output["region_count"] = len(regions)
        output["regions"] = regions
        output["success"] = True

    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    return output


def run_adaptive_pipeline(
    source_path: str,
    reference_path: str,
    sensor_pair: str = "OHRC -> LROC NAC",
    source_sun_elevation: float | None = None,
    reference_sun_elevation: float | None = None,
) -> dict:
    """Run an adaptive registration pipeline, escalating to robust methods if baseline fails."""
    # 1. Baseline attempt (fast)
    baseline_result = run_pipeline(
        source_path=source_path,
        reference_path=reference_path,
        algorithm="akaze",
        preprocessing="clahe",
        source_sun_elevation=source_sun_elevation,
        reference_sun_elevation=reference_sun_elevation,
    )
    
    baseline_metrics = baseline_result.get("metrics")
    
    escalate = True
    reason = "Registration failed completely with baseline AKAZE."
    
    if baseline_result["success"] and baseline_metrics is not None:
        inliers = baseline_metrics.get("inlier_count", 0)
        score = baseline_metrics.get("distribution_score", 0.0)
        
        if inliers >= 15 and score >= 0.5:
            escalate = False
            reason = f"Baseline AKAZE succeeded (Inliers: {inliers} >= 15, Spread: {score:.2f} >= 0.5). No escalation needed."
        else:
            reason = f"Baseline AKAZE metrics too low (Inliers: {inliers} < 15 or Spread: {score:.2f} < 0.5). Escalating to robust RIFT2."
    
    if not escalate:
        baseline_result["escalated"] = False
        baseline_result["reason"] = reason
        baseline_result["baseline_metrics"] = baseline_metrics
        return baseline_result
        
    # 2. Escalation attempt (robust)
    escalated_result = run_pipeline(
        source_path=source_path,
        reference_path=reference_path,
        algorithm="rift2",
        preprocessing="clahe",
        source_sun_elevation=source_sun_elevation,
        reference_sun_elevation=reference_sun_elevation,
    )
    
    escalated_result["escalated"] = True
    escalated_result["reason"] = reason
    escalated_result["baseline_metrics"] = baseline_metrics
    return escalated_result

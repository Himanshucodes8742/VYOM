"""Feature extraction routines for candidate match verification."""

from typing import Sequence
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree

FEATURE_COLUMNS = [
    "descriptor_distance",
    "lowes_ratio",
    "scale_ratio",
    "local_density",
    "response_strength",
]


def compute_match_features(
    matches: Sequence[cv2.DMatch],
    kp_source: Sequence[cv2.KeyPoint],
    kp_reference: Sequence[cv2.KeyPoint],
    match_ratios: dict[int, float] | None = None,
    density_radius: float = 50.0,
) -> pd.DataFrame:
    """Compute the 5 match verifier features for each candidate match.

    Features:
      1. descriptor_distance: AKAZE descriptor Hamming distance.
      2. lowes_ratio: Best distance divided by second-best distance (d1 / d2).
      3. scale_ratio: Source keypoint scale divided by reference keypoint scale.
      4. local_density: Count of other source keypoints within density_radius (50px).
      5. response_strength: Detector corner response of source keypoint.

    Args:
        matches: List of cv2.DMatch objects.
        kp_source: Keypoints detected in the source image.
        kp_reference: Keypoints detected in the reference image.
        match_ratios: Optional mapping of queryIdx -> Lowe's ratio. If None,
                      attempts to read from detect_and_match.last_match_ratios.
        density_radius: Search radius in pixels for local keypoint density (default: 50.0).

    Returns:
        pandas DataFrame with columns matching FEATURE_COLUMNS, ordered identical
        to the input matches list.
    """
    if len(matches) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS, dtype=np.float32)

    # Fallback to cached ratios on detect_and_match if available
    if match_ratios is None:
        try:
            from registration_engine.matchers import detect_and_match
            match_ratios = getattr(detect_and_match, "last_match_ratios", {})
        except ImportError:
            match_ratios = {}

    # Build KD-Tree for local keypoint density on source keypoints
    src_pts_arr = np.array([kp.pt for kp in kp_source], dtype=np.float32)
    src_tree = cKDTree(src_pts_arr)

    # Query neighbors within density_radius for the matched source keypoints
    query_src_pts = np.array(
        [kp_source[m.queryIdx].pt for m in matches], dtype=np.float32
    )
    src_densities = [
        len(neighbors) - 1
        for neighbors in src_tree.query_ball_point(query_src_pts, r=density_radius)
    ]

    records: list[dict[str, float]] = []

    for idx, m in enumerate(matches):
        q_idx = m.queryIdx
        t_idx = m.trainIdx

        kp_s = kp_source[q_idx]
        kp_r = kp_reference[t_idx]

        desc_dist = float(m.distance)
        ratio = float(match_ratios.get(q_idx, 0.70)) if match_ratios else 0.70
        scale = float(kp_s.size / (kp_r.size if kp_r.size > 0 else 1.0))
        density = float(src_densities[idx])
        response = float(kp_s.response)

        records.append(
            {
                "descriptor_distance": desc_dist,
                "lowes_ratio": round(ratio, 4),
                "scale_ratio": round(scale, 4),
                "local_density": density,
                "response_strength": round(response, 6),
            }
        )

    return pd.DataFrame(records, columns=FEATURE_COLUMNS)

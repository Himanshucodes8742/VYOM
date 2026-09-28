"""Bulk Synthetic Training Data Generator for Match Verifier.

Generates synthetic lunar image pairs in bulk with randomized geometric and
radiometric transformations, extracts AKAZE keypoint matches and features via
registration_engine.matchers.detect_and_match(), labels matches as correct (1) or
incorrect (0) using known ground-truth transforms, and saves the dataset as a CSV.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import cKDTree

# Ensure project root is in sys.path so registration_engine can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registration_engine.matchers import detect_and_match
from registration_engine.feature_extraction import compute_match_features


def discover_source_images(input_dirs: list[str]) -> list[Path]:
    """Discover base lunar images from provided directories.

    Args:
        input_dirs: List of directory paths to scan.

    Returns:
        List of Path objects for base lunar images.
    """
    found_images: list[Path] = []
    seen_paths: set[Path] = set()

    for dir_str in input_dirs:
        d = Path(dir_str).resolve()
        if not d.exists():
            print(f"[!] Warning: Directory not found: {d}")
            continue

        # Look for source images, avoiding synthetic target / derived files
        for ext in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
            for f in sorted(d.glob(ext)):
                # Avoid target.png in synthetic validation directory as it is already warped
                if f.name.lower() in ("target.png", "target.tif", "target.tiff", "warped.png"):
                    continue
                # If a PNG with the same stem exists, prefer the PNG over TIF
                if f.suffix.lower() in (".tif", ".tiff"):
                    png_equivalent = f.with_suffix(".png")
                    if png_equivalent.exists():
                        continue
                    # Also handle source_ohrc.tif vs source.png or reference_nac.tif vs reference.png
                    if (d / "source.png").exists() and "ohrc" in f.name.lower():
                        continue
                    if (d / "reference.png").exists() and "nac" in f.name.lower():
                        continue

                if f not in seen_paths:
                    seen_paths.add(f)
                    found_images.append(f)

    return found_images


def generate_synthetic_pair(
    source_img: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a single synthetic pair with randomized affine warp and illumination shift.

    Args:
        source_img: Grayscale 2D uint8 numpy array.
        rng: Initialized NumPy random generator.

    Returns:
        Tuple of (target_img, M_homogeneous_3x3, transform_params).
    """
    height, width = source_img.shape[:2]
    center = (width / 2.0, height / 2.0)

    # 1. Randomized geometric transformation:
    # - Rotation between -15 and 15 degrees
    # - Scale between 0.7 and 1.3
    angle_degrees = float(rng.uniform(-15.0, 15.0))
    scale_factor = float(rng.uniform(0.7, 1.3))

    M_affine_2x3 = cv2.getRotationMatrix2D(center, angle_degrees, scale_factor)
    M_homogeneous_3x3 = np.vstack([M_affine_2x3, [0.0, 0.0, 1.0]])

    warped_img = cv2.warpAffine(
        source_img,
        M_affine_2x3,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 2. Randomized radiometric illumination changes:
    # - Non-linear gamma shift between 0.75 and 1.35
    # - Brightness offset between -25 and +25 counts
    gamma = float(rng.uniform(0.75, 1.35))
    brightness_offset = int(rng.integers(-25, 26))

    gamma_lut = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)]
    ).astype(np.uint8)
    gamma_corrected = cv2.LUT(warped_img, gamma_lut)

    adjusted = np.clip(
        gamma_corrected.astype(np.int16) + brightness_offset, 0, 255
    ).astype(np.uint8)

    # Preserve pure black borders created by geometric warp
    valid_pixels = warped_img > 0
    target_img = np.zeros_like(warped_img)
    target_img[valid_pixels] = adjusted[valid_pixels]

    params = {
        "angle_degrees": angle_degrees,
        "scale_factor": scale_factor,
        "gamma": gamma,
        "brightness_offset": brightness_offset,
    }

    return target_img, M_homogeneous_3x3, params


def extract_features_and_labels(
    source_img: np.ndarray,
    target_img: np.ndarray,
    M_gt: np.ndarray,
    pair_id: str,
    source_name: str,
    all_candidates: bool = False,
    reproj_threshold_px: float = 3.0,
) -> list[dict]:
    """Detect candidate matches with AKAZE, extract features, and label against ground truth.

    Args:
        source_img: Source image (grayscale uint8).
        target_img: Transformed target image (grayscale uint8).
        M_gt: 3x3 ground-truth homogeneous transformation matrix.
        pair_id: Unique string identifier for the pair.
        source_name: Name or path of the base image.
        all_candidates: If True, uses all 2-NN pairs from knnMatch. If False,
                        uses matches from detect_and_match().
        reproj_threshold_px: Reprojection error threshold in pixels for label=1.

    Returns:
        List of feature dictionaries, one per candidate match.
    """
    if all_candidates:
        raw_matches, kp_source, kp_reference = detect_and_match(
            source_img, target_img, algorithm="akaze", return_raw=True
        )
        match_candidates = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                r = float(m.distance / (n.distance if n.distance > 0 else 1e-6))
                match_candidates.append((m, r))
    else:
        good_matches, kp_source, kp_reference = detect_and_match(
            source_img, target_img, algorithm="akaze"
        )
        cached_ratios = getattr(detect_and_match, "last_match_ratios", {})
        match_candidates = [
            (m, cached_ratios.get(m.queryIdx, 0.0)) for m in good_matches
        ]

    if not match_candidates or not kp_source or not kp_reference:
        return []

    matches_list = [m for m, _ in match_candidates]
    ratios_map = {m.queryIdx: r for m, r in match_candidates}

    features_df = compute_match_features(
        matches=matches_list,
        kp_source=kp_source,
        kp_reference=kp_reference,
        match_ratios=ratios_map,
    )

    records: list[dict] = []

    for idx, (m, _) in enumerate(match_candidates):
        q_idx = m.queryIdx
        t_idx = m.trainIdx

        if q_idx >= len(kp_source) or t_idx >= len(kp_reference):
            continue

        kp_s = kp_source[q_idx]
        kp_r = kp_reference[t_idx]

        feat_row = features_df.iloc[idx]

        # Ground truth projection check:
        # Project source keypoint with ground-truth matrix M_gt
        pt_src_h = np.array([kp_s.pt[0], kp_s.pt[1], 1.0], dtype=np.float64)
        pt_proj_h = M_gt @ pt_src_h
        w = pt_proj_h[2] if abs(pt_proj_h[2]) > 1e-8 else 1.0
        pt_proj = pt_proj_h[:2] / w

        reprojection_error = float(
            np.linalg.norm(pt_proj - np.array(kp_r.pt, dtype=np.float64))
        )
        label = 1 if reprojection_error <= reproj_threshold_px else 0

        records.append(
            {
                "pair_id": pair_id,
                "source_image": source_name,
                "query_idx": q_idx,
                "train_idx": t_idx,
                "src_x": round(float(kp_s.pt[0]), 2),
                "src_y": round(float(kp_s.pt[1]), 2),
                "ref_x": round(float(kp_r.pt[0]), 2),
                "ref_y": round(float(kp_r.pt[1]), 2),
                "descriptor_distance": float(feat_row["descriptor_distance"]),
                "lowes_ratio": round(float(feat_row["lowes_ratio"]), 4),
                "scale_ratio": round(float(feat_row["scale_ratio"]), 4),
                "local_density": int(feat_row["local_density"]),
                "response_strength": round(float(feat_row["response_strength"]), 6),
                "response_strength_ref": round(float(kp_r.response), 6),
                "reprojection_error": round(reprojection_error, 3),
                "label": label,
            }
        )

    return records


def generate_dataset(
    input_dirs: list[str],
    output_path: Path,
    pairs_per_image: int = 20,
    base_seed: int | None = 42,
    all_candidates: bool = False,
) -> pd.DataFrame:
    """Generate the full dataset across discovered source images and synthetic pairs.

    Args:
        input_dirs: Folders to scan for source lunar images.
        output_path: Path to the destination CSV file.
        pairs_per_image: Number of synthetic pairs to generate per source image (>= 20).
        base_seed: Base integer seed for random generator reproducibility.
        all_candidates: Whether to include all 2-NN matches or only detect_and_match() matches.

    Returns:
        Constructed pandas DataFrame.
    """
    source_images = discover_source_images(input_dirs)

    if not source_images:
        print("[!] Error: No valid base lunar images found in the specified directories:")
        for d in input_dirs:
            print(f"    - {d}")
        sys.exit(1)

    print(f"[+] Discovered {len(source_images)} base source image(s):")
    for img_path in source_images:
        print(f"    - {img_path.name} ({img_path})")

    print(f"\n[+] Configuration:")
    print(f"    - Pairs per image: {pairs_per_image}")
    print(f"    - Total pairs to generate: {len(source_images) * pairs_per_image}")
    print(f"    - Candidate mode: {'All 2-NN matches' if all_candidates else 'detect_and_match() ratio-filtered'}")
    print(f"    - Output destination: {output_path}")

    all_records: list[dict] = []
    pair_counter = 0

    for img_idx, img_path in enumerate(source_images, start=1):
        print(f"\n[{img_idx}/{len(source_images)}] Processing source: {img_path.name}", flush=True)
        src_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if src_img is None:
            print(f"[!] Warning: Could not decode {img_path}, skipping.", flush=True)
            continue

        h, w = src_img.shape[:2]
        print(f"    Dimensions: {w}x{h}", flush=True)

        for pair_i in range(1, pairs_per_image + 1):
            pair_counter += 1
            seed = (base_seed + pair_counter * 1000) if base_seed is not None else None
            rng = np.random.default_rng(seed)

            target_img, M_gt, params = generate_synthetic_pair(src_img, rng)
            pair_id = f"{img_path.stem}_pair_{pair_i:03d}"

            records = extract_features_and_labels(
                source_img=src_img,
                target_img=target_img,
                M_gt=M_gt,
                pair_id=pair_id,
                source_name=img_path.name,
                all_candidates=all_candidates,
            )

            all_records.extend(records)
            correct_in_pair = sum(1 for r in records if r["label"] == 1)
            print(
                f"    [Pair {pair_i:02d}/{pairs_per_image}] "
                f"rot={params['angle_degrees']:+5.1f}°, scale={params['scale_factor']:.2f}, "
                f"gamma={params['gamma']:.2f}, b_offset={params['brightness_offset']:+3d} -> "
                f"{len(records)} matches ({correct_in_pair} correct)",
                flush=True,
            )

    df = pd.DataFrame(all_records)

    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[+] Saved dataset to: {output_path}", flush=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print detailed summary and class balance analysis."""
    total_matches = len(df)
    if total_matches == 0:
        print("\n[!] Summary: No matches were generated.")
        return

    correct_matches = int((df["label"] == 1).sum())
    incorrect_matches = int((df["label"] == 0).sum())

    pct_correct = (correct_matches / total_matches) * 100.0
    pct_incorrect = (incorrect_matches / total_matches) * 100.0

    print("\n" + "=" * 68)
    print("       MATCH VERIFIER TRAINING DATASET: SUMMARY REPORT")
    print("=" * 68)
    print(f"  Total Pairs Processed:       {df['pair_id'].nunique()}")
    print(f"  Total Candidate Matches:     {total_matches:,}")
    print(f"  Correct Matches   (Label 1): {correct_matches:,} ({pct_correct:.2f}%)")
    print(f"  Incorrect Matches (Label 0): {incorrect_matches:,} ({pct_incorrect:.2f}%)")
    print("-" * 68)

    # Compute and display imbalance analysis
    if incorrect_matches > 0:
        ratio_str = f"{correct_matches / incorrect_matches:.2f}:1"
    else:
        ratio_str = "All Positive (no negatives)"

    print(f"  Class Balance Ratio (1:0):   {ratio_str}")

    print("\n  Class Imbalance Assessment:")
    if pct_correct >= 90.0 or pct_correct <= 10.0:
        dominant_class = 1 if pct_correct >= 90.0 else 0
        dom_pct = max(pct_correct, pct_incorrect)
        print(f"  [!] HIGH CLASS IMBALANCE DETECTED ({dom_pct:.1f}% Class {dominant_class}).")
        print("      Impact on Model Training:")
        print("      - A naive classifier predicting always Class 1 would achieve")
        print(f"        {dom_pct:.1f}% accuracy while learning zero discriminative power.")
        print("      - Recommendation for Training:")
        print("        * Use balanced class weights: class_weight='balanced' (e.g. in Scikit-Learn/LightGBM).")
        print("        * Optimize PR-AUC (Precision-Recall AUC) or F1-Score instead of raw accuracy.")
        print("        * Consider focal loss or stratified negative subsampling/oversampling (SMOTE).")
    else:
        print(f"  [OK] Class balance is moderate ({pct_correct:.1f}% Class 1 vs {pct_incorrect:.1f}% Class 0).")

    print("\n  Feature Column Statistics:")
    numeric_cols = [
        "descriptor_distance",
        "lowes_ratio",
        "scale_ratio",
        "local_density",
        "response_strength",
        "reprojection_error",
    ]
    summary_stats = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    print(summary_stats.to_string())
    print("=" * 68 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bulk Synthetic Training Data Generator for Match Verifier."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            str(PROJECT_ROOT / "data" / "demo_pairs" / "synthetic_validation"),
            str(PROJECT_ROOT / "data" / "demo_pairs" / "ohrc_nac_crater_x"),
        ],
        help="Directories containing source lunar images.",
    )
    parser.add_argument(
        "--pairs-per-image",
        type=int,
        default=20,
        help="Number of synthetic pairs to generate per source image (default: 20).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "training" / "match_verifier_dataset.csv"),
        help="Destination path for output dataset CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="If set, includes all 2-NN matches from AKAZE before ratio filtering.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_file).resolve()
    df = generate_dataset(
        input_dirs=args.input_dirs,
        output_path=output_path,
        pairs_per_image=args.pairs_per_image,
        base_seed=args.seed,
        all_candidates=args.all_candidates,
    )
    print_summary(df)


if __name__ == "__main__":
    main()

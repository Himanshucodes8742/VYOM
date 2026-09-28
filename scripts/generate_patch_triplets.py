"""Lunar Surface Patch Triplet Generator for Deep Metric Learning (Module 2).

Generates anchor, positive, and negative 32x32 grayscale patch triplets from
lunar orbital imagery for metric learning (Triplet Loss / Siamese networks):
  - Anchor: 32x32 patch centered on an AKAZE keypoint in the source image.
  - Positive: 32x32 patch centered on the same physical crater feature after
              random synthetic homography (rotation, scale, perspective jitter)
              and radiometric illumination shifts (gamma, brightness, contrast).
  - Negative: 32x32 patch centered on a different keypoint from the same image
              at least 50 pixels away to ensure non-overlapping spatial terrain.

Saves output triplets to data/training/patch_triplets.npz and visualizes samples
to data/training/sample_triplets.png.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def discover_source_images(input_dirs: list[str]) -> list[Path]:
    """Discover base lunar images from provided directories.

    Args:
        input_dirs: List of directory paths to scan.

    Returns:
        List of Path objects pointing to valid lunar source images.
    """
    found_images: list[Path] = []
    seen_paths: set[Path] = set()

    for dir_str in input_dirs:
        d = Path(dir_str).resolve()
        if not d.exists():
            print(f"[!] Warning: Directory not found: {d}")
            continue

        for ext in ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
            for f in sorted(d.rglob(ext)):
                # Avoid synthetic target / derived files
                if f.name.lower() in ("target.png", "target.tif", "target.tiff", "warped.png"):
                    continue
                # If a PNG with the same stem exists, prefer PNG over TIF to save I/O time
                if f.suffix.lower() in (".tif", ".tiff"):
                    png_equivalent = f.with_suffix(".png")
                    if png_equivalent.exists():
                        continue
                    if (f.parent / "source.png").exists() and "ohrc" in f.name.lower():
                        continue
                    if (f.parent / "reference.png").exists() and "nac" in f.name.lower():
                        continue

                if f not in seen_paths:
                    seen_paths.add(f)
                    found_images.append(f)

    return found_images


def generate_synthetic_homography_warp(
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply random synthetic homography and illumination shift to an image.

    Args:
        image: Grayscale 2D uint8 numpy array.
        rng: Initialized NumPy random generator.

    Returns:
        Tuple of (warped_image, valid_mask, H_3x3).
    """
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    # 1. Rotation and Scale:
    angle = float(rng.uniform(-15.0, 15.0))
    scale = float(rng.uniform(0.85, 1.15))
    tx = float(rng.uniform(-25.0, 25.0))
    ty = float(rng.uniform(-25.0, 25.0))

    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # 2. Perspective Jitter:
    # Transform 4 image corners with affine matrix, then add small random offsets
    src_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_corners = np.zeros_like(src_corners)
    for i, (x, y) in enumerate(src_corners):
        pt_affine = M_rot @ np.array([x, y, 1.0], dtype=np.float64)
        dst_corners[i] = pt_affine[:2]

    # Perspective jitter offset bounded by 2.5% of min dimension
    jitter_max = min(width, height) * 0.025
    corner_jitter = rng.uniform(-jitter_max, jitter_max, size=dst_corners.shape).astype(np.float32)
    dst_corners += corner_jitter

    H = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # 3. Geometric Warp:
    warped_img = cv2.warpPerspective(
        image,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Binary mask of valid interior pixels (excluding black warp borders)
    valid_mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 4. Radiometric Illumination Shift (Simulate Lunar Sun Angle Differences):
    # - Non-linear gamma adjustment
    gamma = float(rng.uniform(0.75, 1.35))
    gamma_lut = np.array(
        [((i / 255.0) ** gamma) * 255.0 for i in range(256)]
    ).astype(np.uint8)
    gamma_corrected = cv2.LUT(warped_img, gamma_lut)

    # - Contrast and brightness adjustments
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = int(rng.integers(-25, 26))

    adjusted = np.clip(
        gamma_corrected.astype(np.float32) * contrast + brightness, 0, 255
    ).astype(np.uint8)

    # Zero out out-of-bound pixels
    adjusted[valid_mask == 0] = 0

    return adjusted, valid_mask, H


def extract_triplets(
    source_images: list[Path],
    target_count: int = 5000,
    patch_size: int = 32,
    min_negative_dist: float = 50.0,
    warps_per_image: int = 35,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate anchor/positive/negative patch triplets across source images.

    Args:
        source_images: List of source lunar image paths.
        target_count: Target number of triplets to generate (>= 5000).
        patch_size: Square patch side length in pixels (default: 32).
        min_negative_dist: Minimum Euclidean distance for negative patch keypoint (px).
        warps_per_image: Number of synthetic homography warps per source image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (anchors, positives, negatives) as numpy arrays of shape [N, 32, 32].
    """
    rng = np.random.default_rng(seed)
    half_size = patch_size / 2.0
    akaze = cv2.AKAZE_create()

    anchors_list: list[np.ndarray] = []
    positives_list: list[np.ndarray] = []
    negatives_list: list[np.ndarray] = []

    print(f"[*] Discovering keypoints and generating patch triplets...")
    print(f"    Target count: {target_count} triplets")
    print(f"    Patch size:   {patch_size}x{patch_size} px")
    print(f"    Min neg dist: {min_negative_dist} px")

    for img_idx, img_path in enumerate(source_images):
        if len(anchors_list) >= target_count:
            break

        print(f"\n[{img_idx + 1}/{len(source_images)}] Processing: {img_path.name}")
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    [!] Failed to load image: {img_path}")
            continue

        h, w = image.shape[:2]
        print(f"    Image dimensions: {w}x{h} px")

        # Detect AKAZE keypoints
        kps = akaze.detect(image, None)
        print(f"    Detected AKAZE keypoints: {len(kps)}")

        # Filter keypoints that are too close to image borders for a 32x32 patch
        margin = half_size + 2.0
        valid_kps = [
            k for k in kps
            if margin <= k.pt[0] <= (w - margin) and margin <= k.pt[1] <= (h - margin)
        ]
        num_valid = len(valid_kps)
        print(f"    Valid interior keypoints: {num_valid}")

        if num_valid < 10:
            print(f"    [!] Too few interior keypoints in {img_path.name}; skipping.")
            continue

        pts_array = np.array([k.pt for k in valid_kps], dtype=np.float32)

        # Distribute triplet budget across warps
        needed = target_count - len(anchors_list)
        per_warp_target = max(15, int(np.ceil(needed / (warps_per_image * (len(source_images) - img_idx)))))

        triplets_before_img = len(anchors_list)

        for warp_idx in range(warps_per_image):
            if len(anchors_list) >= target_count:
                break

            warped_img, valid_mask, H = generate_synthetic_homography_warp(image, rng)

            # Shuffle keypoints order for this warp
            shuffled_indices = rng.permutation(num_valid)

            warp_triplets_count = 0
            for idx in shuffled_indices:
                if len(anchors_list) >= target_count:
                    break
                if warp_triplets_count >= per_warp_target:
                    break

                pt_src = pts_array[idx]

                # Project anchor keypoint to warped coordinate frame:
                # [x_proj, y_proj, w] = H @ [x, y, 1]
                p_homo = H @ np.array([pt_src[0], pt_src[1], 1.0], dtype=np.float64)
                w_coord = p_homo[2]
                if abs(w_coord) < 1e-8:
                    continue
                x_proj = float(p_homo[0] / w_coord)
                y_proj = float(p_homo[1] / w_coord)

                # Ensure positive patch is within image borders
                if not (margin <= x_proj <= (w - margin) and margin <= y_proj <= (h - margin)):
                    continue

                # Ensure positive patch does not intersect the black warp border:
                # Sample the valid_mask over the 32x32 window
                mask_patch = cv2.getRectSubPix(valid_mask, (patch_size, patch_size), (x_proj, y_proj))
                if np.min(mask_patch) < 250:
                    # Patch touches or crosses the out-of-bounds border
                    continue

                # Select a negative keypoint from the same image with distance >= min_negative_dist
                # Rapid rejection sampling:
                neg_idx = None
                for _ in range(25):
                    candidate_idx = int(rng.integers(0, num_valid))
                    if candidate_idx == idx:
                        continue
                    pt_cand = pts_array[candidate_idx]
                    dist_sq = (pt_src[0] - pt_cand[0]) ** 2 + (pt_src[1] - pt_cand[1]) ** 2
                    if dist_sq >= (min_negative_dist ** 2):
                        neg_idx = candidate_idx
                        break

                if neg_idx is None:
                    # Fallback exhaustive search
                    dists = np.linalg.norm(pts_array - pt_src, axis=1)
                    eligible = np.where(dists >= min_negative_dist)[0]
                    if len(eligible) == 0:
                        continue
                    neg_idx = int(rng.choice(eligible))

                pt_neg = pts_array[neg_idx]

                # Extract 32x32 patches using sub-pixel interpolation
                anchor_patch = cv2.getRectSubPix(
                    image, (patch_size, patch_size), (float(pt_src[0]), float(pt_src[1]))
                )
                positive_patch = cv2.getRectSubPix(
                    warped_img, (patch_size, patch_size), (x_proj, y_proj)
                )
                negative_patch = cv2.getRectSubPix(
                    image, (patch_size, patch_size), (float(pt_neg[0]), float(pt_neg[1]))
                )

                anchors_list.append(anchor_patch)
                positives_list.append(positive_patch)
                negatives_list.append(negative_patch)
                warp_triplets_count += 1

        img_triplets_added = len(anchors_list) - triplets_before_img
        print(f"    Triplets generated from this image: {img_triplets_added}")
        print(f"    Cumulative total: {len(anchors_list)} / {target_count}")

    total_achieved = len(anchors_list)
    print("\n" + "=" * 65)
    if total_achieved >= target_count:
        print(f"[OK] SUCCESS: Target achieved! Total triplets generated: {total_achieved}")
    else:
        print(f"[!] WARNING: Target of {target_count} was NOT reached!")
        print(f"    Achieved: {total_achieved} triplets.")
        print(f"    Reason: Insufficient keypoints or valid warp projections across images.")
    print("=" * 65)

    anchors_arr = np.array(anchors_list, dtype=np.uint8)
    positives_arr = np.array(positives_list, dtype=np.uint8)
    negatives_arr = np.array(negatives_list, dtype=np.uint8)

    return anchors_arr, positives_arr, negatives_arr


def save_sample_visualization(
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    output_path: Path,
    num_samples: int = 8,
) -> None:
    """Save a clean grid visualization of sample triplets for visual inspection.

    Args:
        anchors: Array of anchor patches [N, 32, 32].
        positives: Array of positive patches [N, 32, 32].
        negatives: Array of negative patches [N, 32, 32].
        output_path: File path to save the generated figure.
        num_samples: Number of triplet rows to render (default: 8).
    """
    total = len(anchors)
    if total == 0:
        print("[!] No triplets available to visualize.")
        return

    n = min(num_samples, total)
    # Select evenly spaced samples across the dataset
    indices = np.linspace(0, total - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(9.5, n * 2.4), facecolor="#090D14")
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(
        "LUNAR SURFACE PATCH TRIPLETS (32x32 GRAYSCALE)\nDeep Metric Learning Verification",
        fontsize=13,
        fontweight="bold",
        color="#FF9F43",
        y=0.998,
    )

    column_titles = [
        "Anchor (Source Image)",
        "Positive (Homography + Illum)",
        "Negative (Displaced >= 50px)",
    ]

    for col_idx in range(3):
        axes[0, col_idx].set_title(
            column_titles[col_idx],
            fontsize=10,
            fontweight="bold",
            color="#3FD0E0" if col_idx == 1 else "#E056FD" if col_idx == 2 else "#FF9F43",
            pad=12,
        )

    for row_idx, sample_idx in enumerate(indices):
        a = anchors[sample_idx]
        p = positives[sample_idx]
        neg = negatives[sample_idx]

        patches = [a, p, neg]
        border_colors = ["#FF9F43", "#3FD0E0", "#E056FD"]

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.imshow(patches[col_idx], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])

            # Styled patch border
            for spine in ax.spines.values():
                spine.set_color(border_colors[col_idx])
                spine.set_linewidth(1.5)

            # Intensity stats subtext
            min_v, max_v = int(patches[col_idx].min()), int(patches[col_idx].max())
            ax.set_xlabel(
                f"[{min_v}..{max_v}]",
                fontsize=8,
                color="#94A3B8",
                labelpad=3,
            )

        axes[row_idx, 0].set_ylabel(
            f"Triplet #{sample_idx + 1}",
            fontsize=9,
            fontweight="bold",
            color="#CBD5E1",
            rotation=0,
            labelpad=35,
            va="center",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved sample triplet visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 32x32 lunar crater patch triplets for deep metric learning."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data/demo_pairs"],
        help="Directories to search for source lunar images (default: data/demo_pairs).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/training/patch_triplets.npz",
        help="Destination path for .npz triplet arrays (default: data/training/patch_triplets.npz).",
    )
    parser.add_argument(
        "--sample-image",
        type=str,
        default="data/training/sample_triplets.png",
        help="Destination path for sample triplets visualization image (default: data/training/sample_triplets.png).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5000,
        help="Minimum target number of triplets to generate (default: 5000).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Side length of square patches in pixels (default: 32).",
    )
    parser.add_argument(
        "--min-negative-dist",
        type=float,
        default=50.0,
        help="Minimum distance in pixels for negative patch keypoints (default: 50.0).",
    )
    parser.add_argument(
        "--warps-per-image",
        type=int,
        default=35,
        help="Number of synthetic homography warps per source image (default: 35).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path = Path(args.sample_image)

    # 1. Discover lunar source images
    source_images = discover_source_images(args.input_dirs)
    if not source_images:
        print("[!] Error: No base lunar images discovered in:")
        for d in args.input_dirs:
            print(f"    - {d}")
        sys.exit(1)

    print(f"[*] Discovered {len(source_images)} lunar source images:")
    for img in source_images:
        print(f"    - {img.relative_to(PROJECT_ROOT) if PROJECT_ROOT in img.parents else img}")

    # 2. Extract triplets
    anchors, positives, negatives = extract_triplets(
        source_images=source_images,
        target_count=args.target_count,
        patch_size=args.patch_size,
        min_negative_dist=args.min_negative_dist,
        warps_per_image=args.warps_per_image,
        seed=args.seed,
    )

    # 3. Save compressed npz archive
    print(f"\n[*] Saving triplets to: {output_path}")
    np.savez_compressed(
        output_path,
        anchors=anchors,
        positives=positives,
        negatives=negatives,
    )
    file_size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
    print(f"    Archive saved successfully ({file_size_mb:.2f} MB)")
    print(f"    Anchors shape:   {anchors.shape}   dtype: {anchors.dtype}")
    print(f"    Positives shape: {positives.shape} dtype: {positives.dtype}")
    print(f"    Negatives shape: {negatives.shape} dtype: {negatives.dtype}")

    # 4. Generate visual verification plot
    save_sample_visualization(anchors, positives, negatives, sample_path, num_samples=8)


if __name__ == "__main__":
    main()

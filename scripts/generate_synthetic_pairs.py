"""Synthetic Validation Pair Generator for Lunar Image Registration.

Why this synthetic pair exists:
Real lunar orbital imagery (such as Chandrayaan-2 and NASA LRO data) lacks
empirically measured pixel-to-pixel ground-truth alignments. Consequently,
evaluating registration quality on raw pairs relies on subjective visual inspection
or imperfect proxy metrics.

By taking a single real lunar base image and applying a mathematically KNOWN
transformation (known rotation, scale, and radiometric illumination changes), we
create a controlled synthetic benchmark pair. This provides an objective ground-truth
transformation matrix against which we can rigorously compute numerical error metrics
(such as Mean Squared Error, corner transfer error, and RANSAC inlier quality) before
testing algorithms on messier, uncalibrated real mission data.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def generate_synthetic_pair() -> None:
    # Resolve directory paths relative to this script so it works from anywhere
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    validation_dir = project_root / "data" / "demo_pairs" / "synthetic_validation"
    
    source_path = validation_dir / "source.png"
    target_path = validation_dir / "target.png"
    transform_path = validation_dir / "ground_truth_transform.npy"

    # Step 1: Check if source image exists
    if not source_path.exists():
        print(f"Error: Source image file not found at:\n  {source_path}")
        print("\nPlease place a source image at that location (e.g. source.png) and run this script again.")
        sys.exit(1)

    # Step 2: Load the source image as a single-channel grayscale image
    print(f"[+] Loading source image from: {source_path}")
    source_gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)

    if source_gray is None:
        print(f"Error: Failed to decode image file at:\n  {source_path}")
        print("Please ensure the file is a valid image.")
        sys.exit(1)

    height, width = source_gray.shape[:2]
    print(f"    Image dimensions: {width}x{height} (Width x Height, Grayscale)")

    # Step 3: Define known geometric transformation parameters
    # - Rotation: 5 degrees (counter-clockwise)
    # - Scale: 80% (0.8 scale factor)
    # - Pivot: Image center (width / 2, height / 2)
    angle_degrees = 5.0
    scale_factor = 0.8
    center = (width / 2.0, height / 2.0)

    print(f"\n[+] Computing geometric transform:")
    print(f"    - Rotation: {angle_degrees} degrees")
    print(f"    - Scale: {scale_factor * 100:.1f}% (factor = {scale_factor})")
    print(f"    - Center of rotation: {center}")

    # cv2.getRotationMatrix2D builds a 2x3 affine transformation matrix:
    # [ [ alpha,  beta, (1 - alpha) * cx - beta * cy ],
    #   [ -beta, alpha, beta * cx + (1 - alpha) * cy ] ]
    # where alpha = scale * cos(angle), beta = scale * sin(angle)
    M_affine_2x3 = cv2.getRotationMatrix2D(center, angle_degrees, scale_factor)

    # Convert to 3x3 homogeneous transformation matrix for standard homography comparison
    M_homogeneous_3x3 = np.vstack([M_affine_2x3, [0.0, 0.0, 1.0]])

    print(f"\n[+] Ground-Truth 3x3 Transformation Matrix:")
    for row in M_homogeneous_3x3:
        print(f"    [ {row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f} ]")

    # Step 4: Apply geometric warp using OpenCV
    # Uses bilinear interpolation (INTER_LINEAR) and constant black padding (borderValue=0)
    print("\n[+] Applying geometric warp (rotation + scaling)...")
    warped_img = cv2.warpAffine(
        source_gray,
        M_affine_2x3,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Step 5: Simulate sun-angle difference (radiometric / illumination change)
    # On the Moon, different sun angles alter shadow depths and surface reflectance non-linearly.
    # We simulate this using two complementary adjustments:
    # 1. Non-linear Gamma correction: gamma = 1.25 darkens shadows and midtones.
    # 2. Linear brightness offset: -10 intensity counts reduces baseline brightness.
    print("[+] Applying illumination changes (gamma + brightness shift) to simulate sun angle...")
    gamma = 1.25
    brightness_offset = -10

    # Build a 256-element Look-Up Table (LUT) for fast gamma mapping:
    # new_intensity = ((intensity / 255) ^ gamma) * 255
    gamma_lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
    gamma_corrected = cv2.LUT(warped_img, gamma_lut)

    # Apply brightness offset and clip to valid uint8 range [0, 255]
    adjusted = np.clip(gamma_corrected.astype(np.int16) + brightness_offset, 0, 255).astype(np.uint8)

    # Preserve pure black border pixels created by the geometric warp
    valid_pixels = warped_img > 0
    target_img = np.zeros_like(warped_img)
    target_img[valid_pixels] = adjusted[valid_pixels]

    # Step 6: Save the transformed target image using Pillow
    print(f"\n[+] Saving target image to: {target_path}")
    target_pil = Image.fromarray(target_img)
    target_pil.save(str(target_path))

    # Step 7: Save the exact transformation matrix as a NumPy binary file (.npy)
    print(f"[+] Saving ground-truth matrix to: {transform_path}")
    np.save(str(transform_path), M_homogeneous_3x3)

    print("\n[SUCCESS] Synthetic validation pair successfully generated!")
    print(f"    - Source: {source_path}")
    print(f"    - Target: {target_path}")
    print(f"    - Ground Truth: {transform_path}")


if __name__ == "__main__":
    generate_synthetic_pair()

"""Validation script testing detect_changes on two different synthetic pairs with known artificial changes."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registration_engine.pipeline import detect_changes


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / max(union_area, 1e-6)


def run_synthetic_change_detection_tests():
    out_dir = PROJECT_ROOT / "data" / "test_outputs" / "change_detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base lunar image
    base_candidates = [
        PROJECT_ROOT / "data" / "demo_pairs" / "synthetic_validation" / "source.png",
        PROJECT_ROOT / "data" / "demo_pairs" / "ohrc_nac_crater_x" / "source.png",
    ]
    base_path = None
    for c in base_candidates:
        if c.exists():
            base_path = c
            break

    if not base_path:
        raise FileNotFoundError("Could not find a base lunar image for synthetic tests.")

    base_img = cv2.imread(str(base_path), cv2.IMREAD_GRAYSCALE)
    # Use a clean 600x600 subcrop for fast, controlled evaluation
    h, w = base_img.shape[:2]
    crop_size = 600
    start_y = max(0, (h - crop_size) // 2)
    start_x = max(0, (w - crop_size) // 2)
    clean_crop = base_img[start_y : start_y + crop_size, start_x : start_x + crop_size]
    ch, cw = clean_crop.shape[:2]

    print("================================================================================")
    print("           MODULE 6 — LUNAR SURFACE CHANGE DETECTION VALIDATION                ")
    print("================================================================================")
    print(f"Base lunar image: {base_path.name} (Evaluated Crop: {cw}x{ch} px)")

    results_summary = []

    # =========================================================================
    # TEST CASE 1: Bright Synthetic Impact Flash / Ejecta Patch
    # =========================================================================
    print("\n--- TEST CASE 1: Artificial Bright Ejecta Patch ---")
    ref1 = clean_crop.copy()

    # Known change 1: Bright square patch (simulating new impact ejecta)
    patch1_x, patch1_y, patch1_w, patch1_h = 180, 160, 36, 36
    gt_bbox_1 = [patch1_x, patch1_y, patch1_w, patch1_h]
    gt_centroid_1 = (patch1_x + patch1_w / 2.0, patch1_y + patch1_h / 2.0)
    ref1[patch1_y : patch1_y + patch1_h, patch1_x : patch1_x + patch1_w] = 245

    # Source 1: Slightly rotated (+1.2 deg) and translated (+10, -8 px)
    center = (cw / 2.0, ch / 2.0)
    M1 = cv2.getRotationMatrix2D(center, 1.2, 1.01)
    M1[0, 2] += 10.0
    M1[1, 2] -= 8.0
    src1 = cv2.warpAffine(clean_crop, M1, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    src1_path = out_dir / "test1_source.png"
    ref1_path = out_dir / "test1_reference.png"
    cv2.imwrite(str(src1_path), src1)
    cv2.imwrite(str(ref1_path), ref1)

    print(f"Ground Truth Change Injected:")
    print(f"  Type: Bright High-Albedo Ejecta (Value 245)")
    print(f"  Bounding Box: [x={patch1_x}, y={patch1_y}, w={patch1_w}, h={patch1_h}] (Area: {patch1_w*patch1_h} px²)")
    print(f"  Centroid: ({gt_centroid_1[0]:.1f}, {gt_centroid_1[1]:.1f})")

    res1 = detect_changes(
        str(src1_path),
        str(ref1_path),
        algorithm="akaze",
        threshold=35.0,
        min_region_area=30,
        preprocessing="clahe",
    )

    if not res1["success"]:
        print(f"FAILED: {res1.get('error')}")
        results_summary.append({"test": 1, "success": False, "error": res1.get("error")})
    else:
        print(f"Registration & Change Detection Succeeded:")
        print(f"  Overlap Area: {res1['total_valid_pixels']:,} px")
        print(f"  Changed Area: {res1['changed_pixels']:,} px ({res1['change_percentage']:.2f}% of surface)")
        print(f"  Significant Regions Found: {res1['region_count']}")

        top_region_1 = res1["regions"][0] if res1["regions"] else None
        if top_region_1:
            det_bbox = top_region_1["bbox"]
            det_centroid = top_region_1["centroid"]
            dist_err = np.sqrt(
                (det_centroid[0] - gt_centroid_1[0]) ** 2 + (det_centroid[1] - gt_centroid_1[1]) ** 2
            )
            iou = calculate_iou(gt_bbox_1, det_bbox)

            print(f"  Top Detected Region (#1):")
            print(f"    Bounding Box: {det_bbox} (Area: {top_region_1['area']} px²)")
            print(f"    Centroid: ({det_centroid[0]:.1f}, {det_centroid[1]:.1f})")
            print(f"    Centroid Localization Error: {dist_err:.2f} pixels")
            print(f"    Intersection over Union (IoU): {iou * 100:.1f}%")
            print(f"    Mean Intensity Delta: {top_region_1['mean_intensity_diff']:.1f}")

            # Save overlay image
            ov1 = cv2.cvtColor(ref1, cv2.COLOR_GRAY2BGR)
            ov1[res1["change_mask"] > 0] = [40, 30, 255]
            cv2.rectangle(ov1, (det_bbox[0], det_bbox[1]), (det_bbox[0] + det_bbox[2], det_bbox[1] + det_bbox[3]), (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / "test1_detected_overlay.png"), ov1)

            results_summary.append({
                "test": 1,
                "name": "Bright Ejecta Patch",
                "success": True,
                "gt_centroid": gt_centroid_1,
                "det_centroid": det_centroid,
                "dist_error": dist_err,
                "iou": iou,
                "change_pct": res1["change_percentage"],
            })

    # =========================================================================
    # TEST CASE 2: Dark Synthetic Crater Pit / Shadow Excavation
    # =========================================================================
    print("\n--- TEST CASE 2: Artificial Dark Excavation Pit ---")
    ref2 = clean_crop.copy()

    # Known change 2: Dark rectangular excavation pit at a different location
    patch2_x, patch2_y, patch2_w, patch2_h = 370, 320, 44, 30
    gt_bbox_2 = [patch2_x, patch2_y, patch2_w, patch2_h]
    gt_centroid_2 = (patch2_x + patch2_w / 2.0, patch2_y + patch2_h / 2.0)
    ref2[patch2_y : patch2_y + patch2_h, patch2_x : patch2_x + patch2_w] = 0

    # Source 2: Different warp: rotation (-1.5 deg), translation (-12, +15 px), slight scaling
    M2 = cv2.getRotationMatrix2D(center, -1.5, 0.99)
    M2[0, 2] -= 12.0
    M2[1, 2] += 15.0
    src2 = cv2.warpAffine(clean_crop, M2, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    src2_path = out_dir / "test2_source.png"
    ref2_path = out_dir / "test2_reference.png"
    cv2.imwrite(str(src2_path), src2)
    cv2.imwrite(str(ref2_path), ref2)

    print(f"Ground Truth Change Injected:")
    print(f"  Type: Dark Excavation Shadow Pit (Value 0)")
    print(f"  Bounding Box: [x={patch2_x}, y={patch2_y}, w={patch2_w}, h={patch2_h}] (Area: {patch2_w*patch2_h} px²)")
    print(f"  Centroid: ({gt_centroid_2[0]:.1f}, {gt_centroid_2[1]:.1f})")

    res2 = detect_changes(
        str(src2_path),
        str(ref2_path),
        algorithm="akaze",
        threshold=30.0,
        min_region_area=30,
        preprocessing="clahe",
    )

    if not res2["success"]:
        print(f"FAILED: {res2.get('error')}")
        results_summary.append({"test": 2, "success": False, "error": res2.get("error")})
    else:
        print(f"Registration & Change Detection Succeeded:")
        print(f"  Overlap Area: {res2['total_valid_pixels']:,} px")
        print(f"  Changed Area: {res2['changed_pixels']:,} px ({res2['change_percentage']:.2f}% of surface)")
        print(f"  Significant Regions Found: {res2['region_count']}")

        top_region_2 = res2["regions"][0] if res2["regions"] else None
        if top_region_2:
            det_bbox2 = top_region_2["bbox"]
            det_centroid2 = top_region_2["centroid"]
            dist_err2 = np.sqrt(
                (det_centroid2[0] - gt_centroid_2[0]) ** 2 + (det_centroid2[1] - gt_centroid_2[1]) ** 2
            )
            iou2 = calculate_iou(gt_bbox_2, det_bbox2)

            print(f"  Top Detected Region (#1):")
            print(f"    Bounding Box: {det_bbox2} (Area: {top_region_2['area']} px²)")
            print(f"    Centroid: ({det_centroid2[0]:.1f}, {det_centroid2[1]:.1f})")
            print(f"    Centroid Localization Error: {dist_err2:.2f} pixels")
            print(f"    Intersection over Union (IoU): {iou2 * 100:.1f}%")
            print(f"    Mean Intensity Delta: {top_region_2['mean_intensity_diff']:.1f}")

            # Save overlay image
            ov2 = cv2.cvtColor(ref2, cv2.COLOR_GRAY2BGR)
            ov2[res2["change_mask"] > 0] = [40, 30, 255]
            cv2.rectangle(ov2, (det_bbox2[0], det_bbox2[1]), (det_bbox2[0] + det_bbox2[2], det_bbox2[1] + det_bbox2[3]), (0, 255, 255), 2)
            cv2.imwrite(str(out_dir / "test2_detected_overlay.png"), ov2)

            results_summary.append({
                "test": 2,
                "name": "Dark Excavation Pit",
                "success": True,
                "gt_centroid": gt_centroid_2,
                "det_centroid": det_centroid2,
                "dist_error": dist_err2,
                "iou": iou2,
                "change_pct": res2["change_percentage"],
            })

    print("\n================================================================================")
    print("                            FINAL VERIFICATION SUMMARY                          ")
    print("================================================================================")
    for r in results_summary:
        if r["success"]:
            print(f"Test {r['test']} ({r['name']}): PASS")
            print(f"  GT Centroid: ({r['gt_centroid'][0]:.1f}, {r['gt_centroid'][1]:.1f}) -> Detected: ({r['det_centroid'][0]:.1f}, {r['det_centroid'][1]:.1f})")
            print(f"  Offset: {r['dist_error']:.2f} px | Bounding Box IoU: {r['iou']*100:.1f}% | Change: {r['change_pct']}%")
        else:
            print(f"Test {r['test']}: FAIL ({r['error']})")
    print("================================================================================")


if __name__ == "__main__":
    run_synthetic_change_detection_tests()

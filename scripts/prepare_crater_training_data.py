"""Prepare labeled crater patch training dataset from Robbins Lunar Crater Database and real lunar image crops.

Workflow:
1. Inspect and confirm column names from Robbins crater catalog CSV.
2. Extract footprint coordinates from PDS4 XML labels (falling back to / merging with image_footprints.json).
3. Find catalog craters falling within each image's geographic bounds.
4. Calculate expected pixel (x, y) coordinates and diameter, then crop positive crater patches.
5. Sample non-crater background patches (negatives) from locations away from crater extents.
6. Normalize/resize all patches to 64x64 uint8 and save to crater_patches.npz.
7. Render sample grid image to crater_sample_patches.png for visual verification.
"""

import os
import sys
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_pds4_footprint(xml_path: Path) -> dict | None:
    """Extract geographic bounds and ground resolution from a PDS4 XML label."""
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lats, lons = [], []
        res_m = 0.26  # default OHRC resolution

        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            tag_lower = tag.lower()
            if "latitude" in tag_lower and elem.text:
                try:
                    lats.append(float(elem.text.strip()))
                except ValueError:
                    pass
            elif "longitude" in tag_lower and elem.text:
                try:
                    lons.append(float(elem.text.strip()))
                except ValueError:
                    pass
            elif "pixel_resolution" in tag_lower and elem.text:
                try:
                    res_m = float(elem.text.strip())
                except ValueError:
                    pass

        if lats and lons:
            return {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons),
                "gsd_m": res_m,
                "source": f"PDS4 XML label: {xml_path.name}",
            }
    except Exception as e:
        print(f"Warning: Failed to parse XML label {xml_path}: {e}")
    return None


def load_image_footprints() -> dict:
    """Load footprints from PDS4 XML labels and/or image_footprints.json config."""
    footprints = {}

    # 1. First, check PDS4 labels in data/
    xml_files = list((PROJECT_ROOT / "data").glob("*.xml"))
    for xml_f in xml_files:
        fp = extract_pds4_footprint(xml_f)
        if fp:
            print(f"Extracted footprint from PDS4 XML '{xml_f.name}':")
            print(f"  Lat: [{fp['min_lat']:.4f}, {fp['max_lat']:.4f}], Lon: [{fp['min_lon']:.4f}, {fp['max_lon']:.4f}], Base GSD: {fp['gsd_m']} m/px")
            # For our demo pair source.png (1200x9014), the 10x downsampling gives GSD ~ 2.6 m/px
            footprints["ohrc_nac_crater_x/source.png"] = {
                **fp,
                "gsd_m": fp["gsd_m"] * 10.0,
            }
            footprints["ohrc_nac_crater_x/source_ohrc.tif"] = {
                **fp,
                "gsd_m": fp["gsd_m"] * 10.0,
            }

    # 2. Check and merge with data/training/image_footprints.json if present
    json_path = PROJECT_ROOT / "data" / "training" / "image_footprints.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                user_fps = json.load(f)
                for k, v in user_fps.items():
                    footprints[k] = v
            print(f"Loaded {len(user_fps)} footprint entries from {json_path.name}")
        except Exception as e:
            print(f"Warning reading {json_path}: {e}")

    return footprints


def find_crater_catalog() -> Path:
    """Locate the Robbins lunar crater database CSV file."""
    candidates = [
        PROJECT_ROOT / "data" / "training" / "robbins_crater_catalog.csv",
        PROJECT_ROOT / "data" / "training" / "lunar_crater_database_robbins_2018.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Robbins crater catalog CSV not found in data/training/. Expected candidates: {[str(c) for c in candidates]}"
    )


def lat_lon_to_pixel(lat: float, lon: float, fp: dict, img_width: int, img_height: int) -> tuple[int, int]:
    """Map selenographic latitude/longitude to pixel (x, y) coordinates within image."""
    min_lat, max_lat = fp["min_lat"], fp["max_lat"]
    min_lon, max_lon = fp["min_lon"], fp["max_lon"]

    # In typical map projection: top is North (max_lat), bottom is South (min_lat)
    # Left is West (min_lon), right is East (max_lon)
    norm_y = (max_lat - lat) / (max_lat - min_lat)
    norm_x = (lon - min_lon) / (max_lon - min_lon)

    px_x = int(round(norm_x * img_width))
    px_y = int(round(norm_y * img_height))
    return px_x, px_y


def main():
    print("=" * 70)
    print("VYOM LUNAR CRATER TRAINING DATA PREPARATION")
    print("=" * 70)

    # 1. Locate and inspect crater catalog CSV
    catalog_path = find_crater_catalog()
    print(f"\n[Step 1] Loading crater database from: {catalog_path.name}")
    print(f"File size: {catalog_path.stat().st_size / (1024 * 1024):.2f} MB")

    # Read header and first few rows to verify actual column names
    df_sample = pd.read_csv(catalog_path, nrows=5)
    print("\nActual column names verified in catalog:")
    print("  " + ", ".join(df_sample.columns.tolist()[:10]) + " ...")

    # Determine latitude, longitude, and diameter column names
    lat_col = "LAT_CIRC_IMG" if "LAT_CIRC_IMG" in df_sample.columns else "lat"
    lon_col = "LON_CIRC_IMG" if "LON_CIRC_IMG" in df_sample.columns else "lon"
    diam_col = "DIAM_CIRC_IMG" if "DIAM_CIRC_IMG" in df_sample.columns else "diamkm"
    id_col = "CRATER_ID" if "CRATER_ID" in df_sample.columns else "craterid"

    print(f"Using mapped columns:")
    print(f"  ID: {id_col}, Latitude: {lat_col}, Longitude: {lon_col}, Diameter: {diam_col}")

    # 2. Load image footprints
    print("\n[Step 2] Resolving image geographic footprints...")
    footprints = load_image_footprints()
    if not footprints:
        print("Error: No image footprints found from PDS4 XML or image_footprints.json.")
        sys.exit(1)

    # 3. For each image, query catalog craters within bounds
    print("\n[Step 3 & 4] Matching craters and cropping patches...")
    positive_patches = []
    crater_records = []
    crater_regions = {}  # image_key -> list of (cx, cy, radius_px)

    demo_pairs_dir = PROJECT_ROOT / "data" / "demo_pairs"

    for img_rel_path, fp in footprints.items():
        img_full_path = demo_pairs_dir / img_rel_path
        if not img_full_path.exists():
            continue

        print(f"\nProcessing image: {img_rel_path}")
        img = cv2.imread(str(img_full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load image at {img_full_path}")
            continue

        h, w = img.shape
        print(f"  Dimensions: {w}x{h} px")
        print(f"  Footprint Lat: [{fp['min_lat']:.4f}, {fp['max_lat']:.4f}], Lon: [{fp['min_lon']:.4f}, {fp['max_lon']:.4f}], GSD: {fp['gsd_m']} m/px")

        # Query catalog in chunks to conserve memory
        matching_craters = []
        for chunk in pd.read_csv(catalog_path, chunksize=150000):
            mask = (
                (chunk[lat_col] >= fp["min_lat"] - 0.02)
                & (chunk[lat_col] <= fp["max_lat"] + 0.02)
                & (chunk[lon_col] >= fp["min_lon"] - 0.02)
                & (chunk[lon_col] <= fp["max_lon"] + 0.02)
            )
            sub = chunk[mask]
            if len(sub) > 0:
                matching_craters.append(sub)

        if not matching_craters:
            print("  No craters found in this footprint.")
            continue

        df_matched = pd.concat(matching_craters).drop_duplicates(subset=[id_col])
        print(f"  Found {len(df_matched)} catalog crater(s) within / near footprint:")

        crater_regions[img_rel_path] = []

        for _, row in df_matched.iterrows():
            cid = row[id_col]
            clat = float(row[lat_col])
            clon = float(row[lon_col])
            cdiam_km = float(row[diam_col])
            cdiam_m = cdiam_km * 1000.0

            px_x, px_y = lat_lon_to_pixel(clat, clon, fp, w, h)
            diam_px = cdiam_m / fp["gsd_m"]
            radius_px = diam_px / 2.0

            print(f"    - Crater {cid}: Lat={clat:.4f}°, Lon={clon:.4f}°, Diam={cdiam_km:.2f} km -> Pixel (x={px_x}, y={px_y}), Diam={diam_px:.1f} px")

            # Check if crater center falls within image
            if 0 <= px_x < w and 0 <= px_y < h:
                crater_regions[img_rel_path].append((px_x, px_y, radius_px))
                crater_records.append({
                    "image": img_rel_path,
                    "id": cid,
                    "lat": clat,
                    "lon": clon,
                    "diam_km": cdiam_km,
                    "px_x": px_x,
                    "px_y": px_y,
                    "diam_px": diam_px,
                })

                # Size patch proportionally to crater diameter, bounded by [32, 128] pixels
                base_patch_size = int(np.clip(round(diam_px * 0.4), 32, 128))

                # Extract canonical center crop and diverse scale/jitter augmentations
                variations = [
                    (0, 0, 1.0),
                    (-3, 3, 0.9),
                    (3, -3, 1.1),
                    (4, 4, 0.95),
                    (-4, -4, 1.05),
                    (0, 2, 0.85),
                    (2, 0, 1.15),
                    (-2, 0, 1.0),
                    (0, -2, 1.0),
                    (5, -2, 0.9),
                    (-5, 2, 1.1),
                    (2, -5, 0.95),
                    (-2, 5, 1.05),
                    (3, 3, 1.0),
                    (-3, -3, 1.0),
                ]

                for dx, dy, scale in variations:
                    psize = int(round(base_patch_size * scale))
                    half_p = psize // 2
                    x1 = px_x + dx - half_p
                    x2 = x1 + psize
                    y1 = px_y + dy - half_p
                    y2 = y1 + psize

                    if 0 <= x1 and x2 <= w and 0 <= y1 and y2 <= h:
                        patch = img[y1:y2, x1:x2]
                        # Discard patches that touch nodata / black padding
                        zero_ratio = np.mean(patch <= 5)
                        if zero_ratio < 0.10 and np.mean(patch) > 20:
                            # Resize to standard 64x64
                            patch_64 = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                            positive_patches.append(patch_64)

                            # Add flips
                            positive_patches.append(cv2.flip(patch_64, 1))

    total_pos = len(positive_patches)
    print(f"\n[Summary] Total positive crater patches extracted: {total_pos}")

    if total_pos == 0:
        print("Error: No positive crater patches could be extracted.")
        sys.exit(1)

    # 5. Generate equal number of negative background patches
    print(f"\n[Step 5] Sampling {total_pos} negative background patches...")
    negative_patches = []
    random.seed(42)

    image_entries = []
    for img_rel_path, regions in crater_regions.items():
        img_full_path = demo_pairs_dir / img_rel_path
        img = cv2.imread(str(img_full_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_entries.append((img, regions))

    attempts = 0
    max_attempts = total_pos * 2000
    while len(negative_patches) < total_pos and attempts < max_attempts:
        attempts += 1
        img, regions = random.choice(image_entries)
        h, w = img.shape
        psize = random.randint(32, 128)
        if w <= psize or h <= psize:
            continue
        rx = random.randint(0, w - psize)
        ry = random.randint(0, h - psize)
        rcx = rx + psize // 2
        rcy = ry + psize // 2

        # Check if this patch overlaps with any cataloged crater
        in_crater = False
        for (cx, cy, r_px) in regions:
            dist = np.hypot(rcx - cx, rcy - cy)
            if dist < (r_px + psize * 0.75):
                in_crater = True
                break

        if not in_crater:
            neg_patch = img[ry : ry + psize, rx : rx + psize]
            zero_ratio = np.mean(neg_patch <= 5)
            if zero_ratio < 0.05 and np.mean(neg_patch) > 20 and np.std(neg_patch) > 6:
                neg_64 = cv2.resize(neg_patch, (64, 64), interpolation=cv2.INTER_AREA)
                negative_patches.append(neg_64)

    total_neg = len(negative_patches)
    print(f"Total negative background patches extracted: {total_neg}")

    # 6. Save labeled dataset to data/training/crater_patches.npz
    print("\n[Step 6] Saving labeled dataset to data/training/crater_patches.npz...")
    all_patches = np.array(positive_patches + negative_patches, dtype=np.uint8)
    all_labels = np.array([1] * total_pos + [0] * total_neg, dtype=np.int64)

    # Shuffle dataset
    indices = np.arange(len(all_patches))
    np.random.seed(42)
    np.random.shuffle(indices)
    all_patches = all_patches[indices]
    all_labels = all_labels[indices]

    out_npz = PROJECT_ROOT / "data" / "training" / "crater_patches.npz"
    np.savez_compressed(out_npz, patches=all_patches, labels=all_labels)
    print(f"Saved: {out_npz}")
    print(f"  Patches shape: {all_patches.shape}, dtype: {all_patches.dtype}")
    print(f"  Labels shape:  {all_labels.shape}, dtype: {all_labels.dtype}")
    print(f"  Class balance: {np.sum(all_labels == 1)} Positives (1), {np.sum(all_labels == 0)} Negatives (0)")

    # 7. Render sample grid image to crater_sample_patches.png
    print("\n[Step 7] Generating sample verification grid: data/training/crater_sample_patches.png...")
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    plt.suptitle("Lunar Crater Dataset Verification: Positive (Top 2 Rows) vs Negative (Bottom 2 Rows)", fontsize=13, fontweight="bold")

    # Show 16 positive samples (first 2 rows)
    pos_samples = [p for p, l in zip(all_patches, all_labels) if l == 1][:16]
    for i, ax in enumerate(axes[:2].flat):
        if i < len(pos_samples):
            ax.imshow(pos_samples[i], cmap="gray")
            ax.set_title(f"Pos #{i+1}", fontsize=8, color="darkgreen", fontweight="bold")
        ax.axis("off")

    # Show 16 negative samples (bottom 2 rows)
    neg_samples = [p for p, l in zip(all_patches, all_labels) if l == 0][:16]
    for i, ax in enumerate(axes[2:].flat):
        if i < len(neg_samples):
            ax.imshow(neg_samples[i], cmap="gray")
            ax.set_title(f"Neg #{i+1}", fontsize=8, color="navy", fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out_png = PROJECT_ROOT / "data" / "training" / "crater_sample_patches.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample verification image saved to: {out_png}")
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()

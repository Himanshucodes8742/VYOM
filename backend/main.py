import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="phasepack")

import sys
from pathlib import Path
import tempfile
import base64
import time
import shutil
import cv2
cv2.setNumThreads(0)

import torch
torch.set_num_threads(1)

import numpy as np
import concurrent.futures

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Ensure project root is on sys.path so registration_engine is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registration_engine.pipeline import run_pipeline, detect_changes, run_adaptive_pipeline
from registration_engine.matchers import SUPPORTED_ALGORITHMS
from registration_engine.metadata import get_sun_elevation


def get_default_source_sun_elevation() -> float | None:
    """Extract real sun elevation from known PDS4 labels in data directory if available."""
    label_candidates = [
        PROJECT_ROOT / "data" / "ch2_ohr_ncp_20210401T2357376656_d_img_d18.xml",
        PROJECT_ROOT / "data" / "demo_pairs" / "ch2_ohr_ncp_20210401T2357376656_d_img_d18.xml",
    ]
    for lbl in label_candidates:
        if lbl.exists():
            try:
                return get_sun_elevation(str(lbl))
            except Exception:
                pass
    return None

app = FastAPI(
    title="VYOM Lunar Image Registration API",
    description="Backend API wrapping registration_engine for multi-modal lunar image alignment.",
    version="1.0.0",
)

# Dedicated persistent threadpool for computer vision and PyTorch operations
# (prevents Python 3.14 TLS cleanup crashes on transient AnyIO worker threads)
_ENGINE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine_worker")

# Enable CORS for frontend development server (Vite on port 3000, 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = PROJECT_ROOT / "data" / "demo_pairs"

DEMO_METADATA = {
    "ohrc_nac_crater_x": {
        "title": "OHRC / LROC NAC Crater X (South Pole)",
        "source_sensor": "OHRC",
        "reference_sensor": "LROC NAC",
        "source_file": "source.png",
        "reference_file": "reference.png",
        "description": "High-latitude crater rim with extreme solar phase disparity (CH-2 OHRC 0.25m/px vs LRO NAC 0.50m/px).",
    },
    "synthetic_validation": {
        "title": "Synthetic Validation Pair",
        "source_sensor": "OHRC (Simulated)",
        "reference_sensor": "LROC NAC (Simulated)",
        "source_file": "source.png",
        "reference_file": "target.png",
        "description": "Controlled affine perspective transform with ground-truth verification.",
    },
}


def parse_optional_float(val: any) -> float | None:
    """Safely parse float from form field, string, or default object."""
    if val is None or hasattr(val, "default"):
        return None
    try:
        s = str(val).strip()
        if not s or s.lower() in ("none", "null", "undefined"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_form_str(val: any, default: str) -> str:
    """Safely parse string from form field or default object."""
    if hasattr(val, "default"):
        return str(val.default) if val.default is not None else default
    if val is None:
        return default
    return str(val)


def normalize_algorithm(algorithm: any) -> str:
    """Normalize human-readable algorithm name or ID into supported engine identifier."""
    raw = parse_form_str(algorithm, "sift")
    alg_lower = raw.strip().lower()
    if "adaptive" in alg_lower:
        return "adaptive"
    if "crater" in alg_lower or "landmark" in alg_lower:
        return "crater_landmarks"
    if "descriptor" in alg_lower:
        return "learned_descriptor"
    if "learned" in alg_lower or "verifier" in alg_lower:
        return "learned_verifier"
    if "rift" in alg_lower:
        return "rift2"
    if "akaze" in alg_lower:
        return "akaze"
    if "sift" in alg_lower:
        return "sift"
    if alg_lower in SUPPORTED_ALGORITHMS:
        return alg_lower
    return "sift"


@app.get("/health")
def health_check():
    """Health check endpoint confirming API availability."""
    return {"status": "ok", "service": "vyom-registration-backend", "timestamp": time.time()}


@app.get("/demo-pairs")
def get_demo_pairs():
    """Return available demo pair folders with sensor metadata and file information."""
    pairs = []
    if DATA_DIR.exists():
        for d in sorted(DATA_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                meta = DEMO_METADATA.get(d.name, {})
                files = [f.name for f in d.iterdir() if f.is_file() and not f.name.startswith(".")]

                source_file = meta.get("source_file")
                if not source_file:
                    for candidate in ["source.png", "source_ohrc.tif", "source.tif"]:
                        if (d / candidate).exists():
                            source_file = candidate
                            break

                ref_file = meta.get("reference_file")
                if not ref_file:
                    for candidate in ["reference.png", "target.png", "reference_nac.tif"]:
                        if (d / candidate).exists():
                            ref_file = candidate
                            break

                pairs.append({
                    "id": d.name,
                    "title": meta.get("title", d.name.replace("_", " ").title()),
                    "source_sensor": meta.get("source_sensor", "OHRC"),
                    "reference_sensor": meta.get("reference_sensor", "LROC NAC"),
                    "source_file": source_file,
                    "reference_file": ref_file,
                    "description": meta.get("description", "Multimodal lunar orbiter image pair."),
                    "files": files,
                    "source_url": f"/demo-pairs/{d.name}/source" if source_file else None,
                    "reference_url": f"/demo-pairs/{d.name}/reference" if ref_file else None,
                })
    return pairs


@app.get("/demo-pairs/{pair_id}/source")
def get_demo_source_image(pair_id: str):
    """Serve the source image file for a given demo pair."""
    pair_dir = DATA_DIR / pair_id
    if not pair_dir.exists():
        raise HTTPException(status_code=404, detail=f"Demo pair '{pair_id}' not found.")
    meta = DEMO_METADATA.get(pair_id, {})
    candidates = [meta.get("source_file"), "source.png", "source_ohrc.tif", "source.tif"]
    for c in candidates:
        if c and (pair_dir / c).exists():
            media_type = "image/tiff" if c.endswith((".tif", ".tiff")) else "image/png"
            return FileResponse(pair_dir / c, media_type=media_type)
    raise HTTPException(status_code=404, detail="Source image file not found in demo pair.")


@app.get("/demo-pairs/{pair_id}/reference")
def get_demo_reference_image(pair_id: str):
    """Serve the reference image file for a given demo pair."""
    pair_dir = DATA_DIR / pair_id
    if not pair_dir.exists():
        raise HTTPException(status_code=404, detail=f"Demo pair '{pair_id}' not found.")
    meta = DEMO_METADATA.get(pair_id, {})
    candidates = [meta.get("reference_file"), "reference.png", "target.png", "reference_nac.tif", "ref.png"]
    for c in candidates:
        if c and (pair_dir / c).exists():
            media_type = "image/tiff" if c.endswith((".tif", ".tiff")) else "image/png"
            return FileResponse(pair_dir / c, media_type=media_type)
    raise HTTPException(status_code=404, detail="Reference image file not found in demo pair.")


@app.post("/register")
def register_images(
    source_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    sensor_pair: str = Form("OHRC -> LROC NAC"),
    algorithm: str = Form("sift"),
    preprocessing: str = Form("clahe"),
    source_sun_elevation: float | None = Form(None),
):
    """Register source image onto reference image using registration_engine pipeline.

    Returns JSON containing:
      - success (bool)
      - registered_image (base64 data URL)
      - registered_image_base64 (raw base64 string)
      - matches (list of tie-points with coordinates and residuals)
      - metrics (dict of RMSE, inliers, ratio, distribution score, runtime)
      - transform_matrix (3x3 homography matrix)
      - algorithm (normalized algorithm used)
      - preprocessing (preprocessing mode applied)
      - error (error message if failed)
    """
    sensor_pair_str = parse_form_str(sensor_pair, "OHRC -> LROC NAC")
    normalized_alg = normalize_algorithm(algorithm)
    prep_key = parse_form_str(preprocessing, "clahe").strip().lower()
    if prep_key not in ("clahe", "photometric", "photometric_clahe", "photometric+clahe", "histogram", "none"):
        prep_key = "clahe"

    print(f"[API] register_images started: algorithm={normalized_alg}, prep={prep_key}", flush=True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src_ext = Path(source_image.filename or "source.png").suffix or ".png"
        ref_ext = Path(reference_image.filename or "reference.png").suffix or ".png"

        src_path = os.path.join(tmpdir, f"source{src_ext}")
        ref_path = os.path.join(tmpdir, f"reference{ref_ext}")

        with open(src_path, "wb") as f_src:
            shutil.copyfileobj(source_image.file, f_src)

        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(reference_image.file, f_ref)

        print(f"[API] Images saved: src={os.path.getsize(src_path)}B, ref={os.path.getsize(ref_path)}B", flush=True)

        parsed_elev = parse_optional_float(source_sun_elevation)
        source_sun_elev = (
            parsed_elev
            if parsed_elev is not None
            else get_default_source_sun_elevation()
        )

        if normalized_alg == "adaptive":
            future = _ENGINE_EXECUTOR.submit(
                run_adaptive_pipeline,
                src_path,
                ref_path,
                sensor_pair=sensor_pair_str,
                source_sun_elevation=source_sun_elev,
            )
        else:
            future = _ENGINE_EXECUTOR.submit(
                run_pipeline,
                src_path,
                ref_path,
                algorithm=normalized_alg,
                preprocessing=prep_key,
                source_sun_elevation=source_sun_elev,
            )
        result = future.result()

        print(f"[API] run_pipeline finished: success={result['success']}", flush=True)

        if not result["success"]:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": False,
                    "algorithm": normalized_alg,
                    "preprocessing": prep_key,
                    "sensor_pair": sensor_pair_str,
                    "registered_image": None,
                    "registered_image_base64": None,
                    "matches": [],
                    "metrics": None,
                    "transform_matrix": None,
                    "error": result.get("error", "Registration pipeline failed to align images."),
                },
            )

        # Encode warped image to PNG base64
        warped_img = result["warped_image"]
        registered_b64 = None
        registered_data_url = None
        if warped_img is not None and isinstance(warped_img, np.ndarray):
            encode_success, buffer = cv2.imencode(".png", warped_img)
            if encode_success:
                registered_b64 = base64.b64encode(buffer).decode("utf-8")
                registered_data_url = f"data:image/png;base64,{registered_b64}"

        # Format matches with coordinate mappings and reprojection residual
        good_matches = result["matches"] or []
        kp_source = result["kp_source"] or []
        kp_reference = result["kp_reference"] or []
        transform_matrix = result["transform_matrix"]

        formatted_matches = []
        if good_matches and kp_source and kp_reference:
            H = None
            if transform_matrix is not None and isinstance(transform_matrix, (np.ndarray, list)):
                try:
                    H = np.asarray(transform_matrix, dtype=np.float64)
                    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
                        H = None
                except Exception:
                    H = None

            residuals = [0.0] * len(good_matches)
            if H is not None:
                try:
                    src_pts_h = np.array([[kp_source[m.queryIdx].pt[0], kp_source[m.queryIdx].pt[1], 1.0] for m in good_matches], dtype=np.float64)
                    ref_pts_xy = np.array([[kp_reference[m.trainIdx].pt[0], kp_reference[m.trainIdx].pt[1]] for m in good_matches], dtype=np.float64)
                    proj_h = (H @ src_pts_h.T).T
                    denom = proj_h[:, 2]
                    denom_safe = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
                    proj_xy = proj_h[:, :2] / denom_safe[:, None]
                    residuals = np.sqrt(np.sum((proj_xy - ref_pts_xy) ** 2, axis=1)).tolist()
                except Exception as proj_err:
                    print(f"[API] Warning: point projection failed: {proj_err}", flush=True)

            for i, m in enumerate(good_matches):
                sx, sy = kp_source[m.queryIdx].pt
                rx, ry = kp_reference[m.trainIdx].pt
                res_val = residuals[i] if i < len(residuals) else 0.0
                formatted_matches.append({
                    "id": f"KP-{1000 + i}",
                    "x": round(float(sx), 2),
                    "y": round(float(sy), 2),
                    "refX": round(float(rx), 2),
                    "refY": round(float(ry), 2),
                    "dx": round(float(rx - sx), 2),
                    "dy": round(float(ry - sy), 2),
                    "residual": round(float(res_val), 2),
                    "confidence": round(float(max(0.05, min(0.99, 1.0 - (m.distance / 250.0)))), 2)
                    if getattr(m, "distance", 0) > 0 else 0.88,
                })

        matrix_list = (
            transform_matrix.tolist() if isinstance(transform_matrix, np.ndarray) else None
        )

        return {
            "success": True,
            "algorithm": normalized_alg,
            "preprocessing": prep_key,
            "sensor_pair": sensor_pair_str,
            "registered_image": registered_data_url,
            "registered_image_base64": registered_b64,
            "matches": formatted_matches,
            "metrics": result["metrics"],
            "transform_matrix": matrix_list,
            "escalated": result.get("escalated"),
            "reason": result.get("reason"),
            "baseline_metrics": result.get("baseline_metrics"),
            "error": None,
        }


@app.post("/detect-changes")
def detect_surface_changes(
    source_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    sensor_pair: str = Form("OHRC -> LROC NAC"),
    algorithm: str = Form("akaze"),
    preprocessing: str = Form("clahe"),
    threshold: float = Form(30.0),
    min_region_area: int = Form(50),
    source_sun_elevation: float | None = Form(None),
):
    """Detect surface changes between co-registered lunar orbital images.

    Executes registration pipeline, normalizes illumination with preprocessing,
    warps the source image into reference coordinates, computes pixel-wise
    absolute difference on illumination-normalized rasters, applies thresholding
    and morphological opening, and extracts connected changed regions with
    spatial statistics.
    """
    sensor_pair_str = parse_form_str(sensor_pair, "OHRC -> LROC NAC")
    normalized_alg = normalize_algorithm(algorithm)
    prep_key = parse_form_str(preprocessing, "clahe").strip().lower()
    if prep_key not in ("clahe", "photometric", "photometric_clahe", "photometric+clahe", "histogram", "none"):
        prep_key = "clahe"

    thresh_val = parse_optional_float(threshold)
    if thresh_val is None:
        thresh_val = 30.0

    parsed_area = parse_optional_float(min_region_area)
    min_area_val = int(parsed_area) if parsed_area is not None else 50

    print(
        f"[API] detect_surface_changes started: algorithm={normalized_alg}, prep={prep_key}, thresh={thresh_val}, min_area={min_area_val}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        src_ext = Path(source_image.filename or "source.png").suffix or ".png"
        ref_ext = Path(reference_image.filename or "reference.png").suffix or ".png"

        src_path = os.path.join(tmpdir, f"source{src_ext}")
        ref_path = os.path.join(tmpdir, f"reference{ref_ext}")

        with open(src_path, "wb") as f_src:
            shutil.copyfileobj(source_image.file, f_src)

        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(reference_image.file, f_ref)

        parsed_elev = parse_optional_float(source_sun_elevation)
        source_sun_elev = (
            parsed_elev
            if parsed_elev is not None
            else get_default_source_sun_elevation()
        )

        future = _ENGINE_EXECUTOR.submit(
            detect_changes,
            src_path,
            ref_path,
            sensor_pair=sensor_pair_str,
            algorithm=normalized_alg,
            threshold=thresh_val,
            min_region_area=min_area_val,
            preprocessing=prep_key,
            source_sun_elevation=source_sun_elev,
        )
        result = future.result()

        if not result["success"]:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": False,
                    "algorithm": normalized_alg,
                    "preprocessing": prep_key,
                    "sensor_pair": sensor_pair_str,
                    "change_percentage": 0.0,
                    "changed_pixels": 0,
                    "total_valid_pixels": 0,
                    "region_count": 0,
                    "regions": [],
                    "change_mask": None,
                    "change_overlay": None,
                    "diff_heatmap": None,
                    "registered_image": None,
                    "metrics": result.get("registration_metrics"),
                    "transform_matrix": None,
                    "error": result.get("error", "Change detection failed."),
                },
            )

        # 1. Encode binary change mask
        mask_data_url = None
        mask_raw = result["change_mask"]
        if mask_raw is not None and isinstance(mask_raw, np.ndarray):
            ok, buf = cv2.imencode(".png", mask_raw)
            if ok:
                mask_b64 = base64.b64encode(buf).decode("utf-8")
                mask_data_url = f"data:image/png;base64,{mask_b64}"

        # 2. Encode difference heatmap
        diff_data_url = None
        diff_raw = result["diff_image"]
        if diff_raw is not None and isinstance(diff_raw, np.ndarray):
            heatmap = cv2.applyColorMap(diff_raw, cv2.COLORMAP_INFERNO)
            valid_ov = result.get("valid_overlap_mask")
            if valid_ov is not None and isinstance(valid_ov, np.ndarray):
                heatmap[~valid_ov] = 0
            ok, buf = cv2.imencode(".png", heatmap)
            if ok:
                diff_b64 = base64.b64encode(buf).decode("utf-8")
                diff_data_url = f"data:image/png;base64,{diff_b64}"

        # 3. Create change overlay on reference image with distinct highlight color and region bounding boxes
        overlay_data_url = None
        ref_raw = result["reference_image"]
        if ref_raw is not None and mask_raw is not None and isinstance(ref_raw, np.ndarray):
            if len(ref_raw.shape) == 2:
                overlay = cv2.cvtColor(ref_raw, cv2.COLOR_GRAY2BGR)
            else:
                overlay = ref_raw.copy()

            # Distinct highlight color (vibrant neon coral/magenta [B=40, G=30, R=255])
            highlight_color = np.array([40, 30, 255], dtype=np.uint8)
            mask_bool = mask_raw > 0
            if np.any(mask_bool):
                overlay[mask_bool] = cv2.addWeighted(
                    overlay[mask_bool], 0.35, np.full_like(overlay[mask_bool], highlight_color), 0.65, 0
                )

            # Draw bounding boxes and region labels for top detected regions
            for reg in result["regions"][:15]:
                bx, by, bw, bh = reg["bbox"]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 240, 255), 2)
                lbl_text = f"#{reg['id']} ({reg['area']}px)"
                cv2.putText(
                    overlay,
                    lbl_text,
                    (bx, max(14, by - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 240, 255),
                    1,
                    cv2.LINE_AA,
                )

            ok, buf = cv2.imencode(".png", overlay)
            if ok:
                ov_b64 = base64.b64encode(buf).decode("utf-8")
                overlay_data_url = f"data:image/png;base64,{ov_b64}"

        # 4. Registered warped image
        warped_data_url = None
        warped_raw = result.get("warped_image")
        if warped_raw is not None and isinstance(warped_raw, np.ndarray):
            ok, buf = cv2.imencode(".png", warped_raw)
            if ok:
                warped_b64 = base64.b64encode(buf).decode("utf-8")
                warped_data_url = f"data:image/png;base64,{warped_b64}"

        matrix_list = (
            result["transform_matrix"].tolist()
            if isinstance(result.get("transform_matrix"), np.ndarray)
            else None
        )

        return {
            "success": True,
            "algorithm": normalized_alg,
            "preprocessing": prep_key,
            "sensor_pair": sensor_pair_str,
            "threshold": thresh_val,
            "min_region_area": min_area_val,
            "change_percentage": result["change_percentage"],
            "changed_pixels": result["changed_pixels"],
            "total_valid_pixels": result["total_valid_pixels"],
            "region_count": result["region_count"],
            "regions": result["regions"],
            "change_mask": mask_data_url,
            "change_overlay": overlay_data_url,
            "diff_heatmap": diff_data_url,
            "registered_image": warped_data_url,
            "metrics": result["registration_metrics"],
            "transform_matrix": matrix_list,
            "error": None,
        }


@app.post("/compare")
def compare_algorithms(
    source_image: UploadFile = File(...),
    reference_image: UploadFile = File(...),
    sensor_pair: str = Form("OHRC -> LROC NAC"),
    source_sun_elevation: float | None = Form(None),
):
    """Run registration pipeline across multiple algorithms and preprocessing modes for comparative benchmarks."""
    parsed_elev = parse_optional_float(source_sun_elevation)
    source_sun_elev = (
        parsed_elev
        if parsed_elev is not None
        else get_default_source_sun_elevation()
    )
    sensor_pair_str = parse_form_str(sensor_pair, "OHRC -> LROC NAC")

    eval_algorithms = [
        {"id": "rift2", "name": "RIFT2-style (Phase Congruency)", "tag": "RECOMMENDED", "engine": "Phase Congruency + Max Moments", "algorithm": "rift2", "preprocessing": "clahe"},
        {"id": "akaze_clahe", "name": "AKAZE (CLAHE only)", "tag": "BASELINE", "engine": "Fast Explicit Diffusion", "algorithm": "akaze", "preprocessing": "clahe"},
        {"id": "akaze_photometric", "name": "AKAZE (Photometric)", "tag": "PHOTOMETRIC", "engine": "Lommel-Seeliger Norm", "algorithm": "akaze", "preprocessing": "photometric"},
        {"id": "sift_clahe", "name": "SIFT (CLAHE only)", "tag": "BASELINE", "engine": "Difference of Gaussians", "algorithm": "sift", "preprocessing": "clahe"},
        {"id": "sift_photometric", "name": "SIFT (Photometric)", "tag": "PHOTOMETRIC", "engine": "Lommel-Seeliger Norm", "algorithm": "sift", "preprocessing": "photometric"},
        {"id": "learned_verifier", "name": "Learned Match Verifier (trained)", "tag": "TRAINED ML", "engine": "Random Forest Verifier + AKAZE", "algorithm": "learned_verifier", "preprocessing": "clahe"},
        {"id": "learned_descriptor", "name": "Learned Descriptor (trained)", "tag": "TRAINED CNN", "engine": "CNN Descriptor + AKAZE", "algorithm": "learned_descriptor", "preprocessing": "clahe"},
        {"id": "crater_landmarks", "name": "Crater Landmarks (trained CNN)", "tag": "LANDMARKS", "engine": "Crater Classifier CNN + AKAZE", "algorithm": "crater_landmarks", "preprocessing": "clahe"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        src_ext = Path(source_image.filename or "source.png").suffix or ".png"
        ref_ext = Path(reference_image.filename or "reference.png").suffix or ".png"

        src_path = os.path.join(tmpdir, f"source{src_ext}")
        ref_path = os.path.join(tmpdir, f"reference{ref_ext}")

        with open(src_path, "wb") as f_src:
            shutil.copyfileobj(source_image.file, f_src)

        with open(ref_path, "wb") as f_ref:
            shutil.copyfileobj(reference_image.file, f_ref)

        def eval_single(item):
            alg_id = item.get("algorithm", item["id"])
            prep = item.get("preprocessing", "clahe")
            try:
                pipe_res = run_pipeline(
                    src_path,
                    ref_path,
                    algorithm=alg_id,
                    preprocessing=prep,
                    source_sun_elevation=source_sun_elev,
                )
            except Exception as e:
                pipe_res = {"success": False, "error": str(e), "metrics": None}

            if pipe_res.get("success") and pipe_res.get("metrics"):
                m = pipe_res["metrics"]
                return {
                    "id": item["id"],
                    "algorithm": alg_id,
                    "preprocessing": prep,
                    "name": item["name"],
                    "tag": item["tag"],
                    "engine": item["engine"],
                    "success": True,
                    "rmse": round(float(m["rmse"]), 2),
                    "inliers": int(m["inlier_count"]),
                    "ratio": round(float(m["inlier_ratio"] * 100), 1),
                    "score": round(float(m["distribution_score"]), 2),
                    "runtime": round(float(m.get("runtime", 0.0)), 2),
                    "metrics": m,
                    "error": None,
                }
            else:
                return {
                    "id": item["id"],
                    "algorithm": alg_id,
                    "preprocessing": prep,
                    "name": item["name"],
                    "tag": item["tag"],
                    "engine": item["engine"],
                    "success": False,
                    "rmse": None,
                    "inliers": 0,
                    "ratio": 0.0,
                    "score": 0.0,
                    "runtime": 0.0,
                    "metrics": None,
                    "error": pipe_res.get("error", "Failed to compute inliers."),
                }

        worker_count = min(4, os.cpu_count() or 2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(eval_single, eval_algorithms))

    # Flag best performance attributes among successful algorithms
    successful = [r for r in results if r["success"] and r["rmse"] is not None]
    if successful:
        best_rmse = min(r["rmse"] for r in successful)
        best_inliers = max(r["inliers"] for r in successful)
        best_ratio = max(r["ratio"] for r in successful)
        best_score = max(r["score"] for r in successful)
        best_runtime = min(r["runtime"] for r in successful)

        for r in results:
            if r["success"]:
                r["isBestRmse"] = (r["rmse"] == best_rmse)
                r["isBestInliers"] = (r["inliers"] == best_inliers)
                r["isBestRatio"] = (r["ratio"] == best_ratio)
                r["isBestScore"] = (r["score"] == best_score)
                r["isBestRuntime"] = (r["runtime"] == best_runtime)

    return {
        "sensor_pair": sensor_pair_str,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

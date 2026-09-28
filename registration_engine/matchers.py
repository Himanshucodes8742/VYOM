from pathlib import Path
import numpy as np
import cv2
import joblib

from registration_engine.preprocessing import phase_congruency_map
from registration_engine.feature_extraction import compute_match_features


SUPPORTED_ALGORITHMS = ("sift", "akaze", "rift2", "learned_verifier", "learned_descriptor", "crater_landmarks")


class CraterKeyPoint(cv2.KeyPoint):
    """Subclass of cv2.KeyPoint that also supports unpacking as (x, y) and indexing."""

    def __getitem__(self, idx: int) -> float:
        return (self.pt[0], self.pt[1])[idx]

    def __iter__(self):
        return iter((self.pt[0], self.pt[1]))

    def __len__(self) -> int:
        return 2


def detect_and_match(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    algorithm: str = "sift",
    ratio_threshold: float | None = 0.75,
    return_raw: bool = False,
    raw_source: np.ndarray | None = None,
    raw_reference: np.ndarray | None = None,
) -> tuple[list, list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    """Detect keypoints and compute descriptor matches between two grayscale images.

    Args:
        source_img: Grayscale source image (2D uint8 numpy array).
        reference_img: Grayscale reference image (2D uint8 numpy array).
        algorithm: Feature detection algorithm — "sift", "akaze", "rift2", "learned_verifier", "learned_descriptor", or "crater_landmarks".
        ratio_threshold: Threshold for Lowe's ratio test (default 0.75). If None, all matches kept.
        return_raw: If True, returns raw KNN match pairs list[list[cv2.DMatch]].
        raw_source: Optional unenhanced raw source image (used for CNN crater detection).
        raw_reference: Optional unenhanced raw reference image (used for CNN crater detection).

    Returns:
        Tuple of (matches, keypoints_source, keypoints_reference).
        By default, matches is the list of cv2.DMatch objects passing Lowe's ratio test.

    Raises:
        ValueError: If algorithm is not one of the supported values.
    """
    algorithm_lower = algorithm.strip().lower()
    if algorithm_lower not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Choose one of: {', '.join(SUPPORTED_ALGORITHMS)}"
        )

    if algorithm_lower == "crater_landmarks":
        return detect_and_match_crater_landmarks(
            source_img,
            reference_img,
            ratio_threshold=ratio_threshold if (ratio_threshold is not None and ratio_threshold > 0.85) else 0.90,
            return_raw=return_raw,
            raw_source=raw_source,
            raw_reference=raw_reference,
        )

    if algorithm_lower == "learned_verifier":
        # Run AKAZE detection and matching, then pre-filter with learned match verifier
        matches, kp_source, kp_reference = detect_and_match(
            source_img, reference_img, algorithm="akaze", ratio_threshold=ratio_threshold, return_raw=return_raw
        )
        if return_raw:
            return matches, kp_source, kp_reference
        verified_matches = filter_with_learned_verifier(matches, kp_source, kp_reference)
        return verified_matches, kp_source, kp_reference

    if algorithm_lower == "learned_descriptor":
        return detect_and_match_learned_descriptor(
            source_img, reference_img, ratio_threshold=ratio_threshold, return_raw=return_raw
        )

    if algorithm_lower == "rift2":
        return _detect_and_match_rift2(
            source_img, reference_img, ratio_threshold=ratio_threshold, return_raw=return_raw
        )

    # --- Standard SIFT / AKAZE path ---

    # Create the feature detector/descriptor
    if algorithm_lower == "sift":
        detector = cv2.SIFT_create()
    else:  # akaze
        detector = cv2.AKAZE_create()

    # Detect keypoints and compute descriptors
    kp_source, desc_source = detector.detectAndCompute(source_img, None)
    kp_reference, desc_reference = detector.detectAndCompute(reference_img, None)

    # Guard against empty descriptor sets
    if desc_source is None or desc_reference is None:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)
    if len(desc_source) < 2 or len(desc_reference) < 2:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)

    # Choose the appropriate norm for the descriptor type:
    # SIFT uses float32 descriptors -> L2 norm
    # AKAZE (default) uses binary descriptors -> Hamming norm
    if algorithm_lower == "sift":
        norm_type = cv2.NORM_L2
    else:
        norm_type = cv2.NORM_HAMMING

    # Brute-force matcher with k=2 nearest neighbours for ratio test
    bf = cv2.BFMatcher(norm_type)
    raw_matches = bf.knnMatch(desc_source, desc_reference, k=2)

    detect_and_match.last_raw_matches = raw_matches

    if return_raw:
        return raw_matches, list(kp_source), list(kp_reference)

    # Apply Lowe's ratio test to discard ambiguous matches
    good_matches: list[cv2.DMatch] = []
    good_ratios: dict[int, float] = {}
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            r = float(m.distance / (n.distance if n.distance > 0 else 1e-6))
            if ratio_threshold is None or m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                good_ratios[m.queryIdx] = r

    detect_and_match.last_match_ratios = good_ratios

    return good_matches, list(kp_source), list(kp_reference)


def _detect_and_match_rift2(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    ratio_threshold: float | None = 0.75,
    return_raw: bool = False,
) -> tuple[list, list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    """Simplified RIFT2-inspired matching via phase congruency + ORB.

    This is a simplified reimplementation inspired by:
      - Li, J., Hu, Q., & Ai, M. (2020). "RIFT: Multi-Modal Image Matching
        Based on Radiation-Invariant Feature Transform." IEEE Transactions on
        Image Processing, 29, 3296-3310.
      - Li, J., Hu, Q., & Ai, M. (2023). "RIFT2: Speeding-Up RIFT with A New
        Rotation-Invariance Technique." ISPRS Journal of Photogrammetry and
        Remote Sensing, 199, 55-70.
    It is NOT a byte-for-byte reproduction of either published algorithm.

    The key insight from RIFT/RIFT2 is that phase congruency maps are
    inherently invariant to illumination / radiometric differences between
    multi-modal images (e.g. different sun angles, different sensors). By
    first computing phase congruency maps and then running a standard fast
    detector (ORB) on those maps, we approximate the illumination-invariant
    feature detection behaviour of the full RIFT2 algorithm at much lower
    implementation complexity. ORB is chosen over AKAZE here because the
    phase congruency maps are already edge-normalised uint8 images, and ORB's
    FAST corner detection + BRIEF descriptors run efficiently on them.
    """
    # Step 1: Compute illumination-invariant phase congruency maps for both images
    pc_source = phase_congruency_map(source_img)
    pc_reference = phase_congruency_map(reference_img)

    # Step 2: Detect keypoints and compute descriptors on the phase congruency maps
    # using ORB (fast binary detector suitable for the normalised PC maps)
    detector = cv2.ORB_create(nfeatures=10000)
    kp_source, desc_source = detector.detectAndCompute(pc_source, None)
    kp_reference, desc_reference = detector.detectAndCompute(pc_reference, None)

    # Guard against empty descriptor sets
    if desc_source is None or desc_reference is None:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source or []), list(kp_reference or [])
    if len(desc_source) < 2 or len(desc_reference) < 2:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)

    # Step 3: Brute-force match with Hamming norm (binary ORB descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(desc_source, desc_reference, k=2)

    detect_and_match.last_raw_matches = raw_matches

    if return_raw:
        return raw_matches, list(kp_source), list(kp_reference)

    # Step 4: Lowe's ratio test to filter ambiguous matches
    good_matches: list[cv2.DMatch] = []
    good_ratios: dict[int, float] = {}
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            r = float(m.distance / (n.distance if n.distance > 0 else 1e-6))
            if ratio_threshold is None or m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                good_ratios[m.queryIdx] = r

    detect_and_match.last_match_ratios = good_ratios

    return good_matches, list(kp_source), list(kp_reference)


# Module-level defaults for metadata tracking
detect_and_match.last_raw_matches = []
detect_and_match.last_match_ratios = {}

_MODEL_CACHE: dict[str, object] = {}


def filter_with_learned_verifier(
    matches: list[cv2.DMatch],
    kp_source: list[cv2.KeyPoint],
    kp_reference: list[cv2.KeyPoint],
    model_path: str | Path = "registration_engine/models/match_verifier.pkl",
    probability_threshold: float = 0.5,
) -> list[cv2.DMatch]:
    """Filter candidate matches using the trained Random Forest match verifier model.

    Computes the 5 match verifier features (descriptor_distance, lowes_ratio,
    scale_ratio, local_density, response_strength) and keeps matches predicted as
    inlier with probability >= probability_threshold.

    Args:
        matches: List of cv2.DMatch objects.
        kp_source: Keypoints detected in the source image.
        kp_reference: Keypoints detected in the reference image.
        model_path: Path to serialized match_verifier.pkl model.
        probability_threshold: Probability threshold for inlier classification (default 0.5).

    Returns:
        Filtered list of cv2.DMatch objects predicted as correct inliers.
    """
    if len(matches) == 0:
        return []

    # Resolve model path
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / resolved_path
        if candidate.exists():
            resolved_path = candidate

    resolved_str = str(resolved_path)
    if resolved_str not in _MODEL_CACHE:
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Match verifier model not found at {resolved_path}. "
                "Please run scripts/train_match_verifier.py first to generate it."
            )
        _MODEL_CACHE[resolved_str] = joblib.load(resolved_path)

    model = _MODEL_CACHE[resolved_str]

    # Compute the 5 exact features
    features_df = compute_match_features(
        matches=matches,
        kp_source=kp_source,
        kp_reference=kp_reference,
    )

    # Predict inlier probabilities (class 1)
    probabilities = model.predict_proba(features_df)
    classes = list(model.classes_)
    inlier_col_idx = classes.index(1) if 1 in classes else -1
    inlier_probs = probabilities[:, inlier_col_idx]

    filtered_matches = [
        m for m, p in zip(matches, inlier_probs) if p >= probability_threshold
    ]

    return filtered_matches


def detect_and_match_learned_descriptor(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    model_path: str | Path = "registration_engine/models/patch_descriptor.pt",
    ratio_threshold: float | None = 0.75,
    return_raw: bool = False,
) -> tuple[list, list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    import torch
    from registration_engine.learned_descriptor import PatchDescriptorNet

    # Resolve model path
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / resolved_path
        if candidate.exists():
            resolved_path = candidate

    resolved_str = str(resolved_path)
    if resolved_str not in _MODEL_CACHE:
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Patch descriptor model not found at {resolved_path}. "
                "Please run scripts/train_descriptor.py first."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PatchDescriptorNet().to(device)
        model.load_state_dict(torch.load(resolved_path, map_location=device))
        model.eval()
        _MODEL_CACHE[resolved_str] = model

    model = _MODEL_CACHE[resolved_str]
    device = next(model.parameters()).device

    detector = cv2.AKAZE_create()
    kp_source = detector.detect(source_img, None)
    kp_reference = detector.detect(reference_img, None)

    def compute_descriptors(img, keypoints):
        valid_kp = []
        patches = []
        h, w = img.shape
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # Check if 32x32 patch can be extracted
            if x - 16 >= 0 and x + 16 <= w and y - 16 >= 0 and y + 16 <= h:
                patch = img[y - 16:y + 16, x - 16:x + 16]
                patches.append(patch)
                valid_kp.append(kp)
        
        if len(patches) == 0:
            return valid_kp, None
        
        patches_arr = np.array(patches, dtype=np.float32) / 255.0
        patches_tensor = torch.tensor(patches_arr).unsqueeze(1).to(device)
        
        with torch.no_grad():
            descriptors = model(patches_tensor).cpu().numpy()
            
        return valid_kp, descriptors

    kp_source, desc_source = compute_descriptors(source_img, kp_source)
    kp_reference, desc_reference = compute_descriptors(reference_img, kp_reference)

    if desc_source is None or desc_reference is None or len(desc_source) < 2 or len(desc_reference) < 2:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)

    # Brute-force matcher with L2 norm for our continuous descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc_source, desc_reference, k=2)

    detect_and_match.last_raw_matches = raw_matches

    if return_raw:
        return raw_matches, list(kp_source), list(kp_reference)

    good_matches = []
    good_ratios = {}
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            r = float(m.distance / (n.distance if n.distance > 0 else 1e-6))
            if ratio_threshold is None or m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                good_ratios[m.queryIdx] = r

    detect_and_match.last_match_ratios = good_ratios

    return good_matches, list(kp_source), list(kp_reference)


def detect_crater_landmarks(
    image: np.ndarray,
    model_path: str | Path = "registration_engine/models/crater_classifier.pt",
    window_sizes: list[int] | tuple[int, ...] = (64, 96),
    stride: int = 32,
    conf_threshold: float = 0.8,
    batch_size: int = 128,
) -> list[cv2.KeyPoint]:
    """Detect crater landmarks across an image using a CNN classifier sliding-window scan.

    Scans the image at the given window sizes with a sliding-window stride, scores each
    window patch with the trained crater classifier, filters by confidence threshold,
    and applies non-maximum suppression (NMS) to eliminate duplicate/overlapping detections.

    Args:
        image: 2D grayscale image as numpy array.
        model_path: Path to trained crater_classifier.pt model weights.
        window_sizes: Sizes of sliding windows in pixels (default [64, 96]).
        stride: Step size in pixels between adjacent windows (default 32).
        conf_threshold: Minimum crater classification probability (default 0.8).
        batch_size: Mini-batch size for model inference (default 128).

    Returns:
        List of cv2.KeyPoint objects positioned at the centers of detected craters.
    """
    import torch
    from registration_engine.crater_detector import CraterClassifierNet

    # Ensure grayscale 2D uint8
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    # Resolve model path
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / resolved_path
        if candidate.exists():
            resolved_path = candidate

    resolved_str = str(resolved_path)
    if resolved_str not in _MODEL_CACHE:
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Crater classifier model not found at {resolved_path}. "
                "Please run scripts/train_crater_classifier or registration_engine/crater_detector.py first."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CraterClassifierNet().to(device)
        model.load_state_dict(torch.load(resolved_path, map_location=device))
        model.eval()
        _MODEL_CACHE[resolved_str] = model

    model = _MODEL_CACHE[resolved_str]
    device = next(model.parameters()).device

    h, w = img_u8.shape
    candidates: list[tuple[float, float, float, float]] = []

    for ws in window_sizes:
        if h < ws or w < ws:
            continue
        batch_crops = []
        batch_meta = []

        with torch.no_grad():
            for y in range(0, h - ws + 1, stride):
                for x in range(0, w - ws + 1, stride):
                    crop = img_u8[y : y + ws, x : x + ws]
                    if ws != 64:
                        crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
                    batch_crops.append(crop)
                    batch_meta.append((x + ws / 2.0, y + ws / 2.0, ws / 2.0))

                    if len(batch_crops) == batch_size:
                        t = torch.from_numpy(np.array(batch_crops, dtype=np.float32) / 255.0).unsqueeze(1).to(device)
                        out = model(t).cpu().numpy().flatten()
                        for (cx, cy, r), p in zip(batch_meta, out):
                            if p >= conf_threshold:
                                candidates.append((cx, cy, r, float(p)))
                        batch_crops = []
                        batch_meta = []

            if batch_crops:
                t = torch.from_numpy(np.array(batch_crops, dtype=np.float32) / 255.0).unsqueeze(1).to(device)
                out = model(t).cpu().numpy().flatten()
                for (cx, cy, r), p in zip(batch_meta, out):
                    if p >= conf_threshold:
                        candidates.append((cx, cy, r, float(p)))

    # Non-Maximum Suppression (NMS) to eliminate overlapping detections
    candidates.sort(key=lambda c: c[3], reverse=True)
    kept: list[tuple[float, float, float, float]] = []
    for c in candidates:
        cx, cy, r, p = c
        suppressed = False
        for kx, ky, kr, kp in kept:
            dist = np.hypot(cx - kx, cy - ky)
            if dist < max(r, kr):
                suppressed = True
                break
        if not suppressed:
            kept.append(c)

    keypoints = [
        CraterKeyPoint(
            x=float(c[0]),
            y=float(c[1]),
            size=float(c[2] * 2),  # diameter as keypoint size
            response=float(c[3]),  # confidence probability
            octave=0,
            class_id=0,
        )
        for c in kept
    ]
    return keypoints


def detect_and_match_crater_landmarks(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    model_path: str | Path = "registration_engine/models/crater_classifier.pt",
    window_sizes: list[int] = [64, 96],
    ratio_threshold: float | None = 0.90,
    return_raw: bool = False,
    raw_source: np.ndarray | None = None,
    raw_reference: np.ndarray | None = None,
    descriptor_type: str = "akaze",
) -> tuple[list, list[cv2.KeyPoint], list[cv2.KeyPoint]]:
    """Detect crater landmarks via CNN classifier and compute descriptor matches.

    Uses crater centers detected by detect_crater_landmarks as keypoints, computes
    feature descriptors (AKAZE or learned descriptor) at those positions, and matches
    them using brute-force matching and Lowe's ratio test.

    Args:
        source_img: Preprocessed source image for descriptor computation.
        reference_img: Preprocessed reference image for descriptor computation.
        model_path: Path to trained crater classifier model weights.
        window_sizes: Sliding-window scan patch dimensions.
        ratio_threshold: Lowe's ratio test threshold (default 0.85).
        return_raw: If True, returns raw KNN matches list.
        raw_source: Optional raw unenhanced source image for CNN crater detection.
        raw_reference: Optional raw unenhanced reference image for CNN crater detection.
        descriptor_type: Feature descriptor to compute ("akaze" or "learned").

    Returns:
        Tuple of (matches, keypoints_source, keypoints_reference).
    """
    # 1. Detect crater landmarks (using raw images if provided to maintain CNN training distribution)
    det_source = raw_source if raw_source is not None else source_img
    det_reference = raw_reference if raw_reference is not None else reference_img

    kp_source = detect_crater_landmarks(
        det_source, model_path=model_path, window_sizes=window_sizes
    )
    kp_reference = detect_crater_landmarks(
        det_reference, model_path=model_path, window_sizes=window_sizes
    )

    if len(kp_source) == 0 or len(kp_reference) == 0:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)

    # 2. Compute descriptors at crater locations
    if descriptor_type == "learned":
        import torch
        from registration_engine.learned_descriptor import PatchDescriptorNet
        ld_path = Path(__file__).resolve().parent / "models" / "patch_descriptor.pt"
        if ld_path.exists():
            if str(ld_path) not in _MODEL_CACHE:
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                net = PatchDescriptorNet().to(dev)
                net.load_state_dict(torch.load(ld_path, map_location=dev))
                net.eval()
                _MODEL_CACHE[str(ld_path)] = net
            net = _MODEL_CACHE[str(ld_path)]
            dev = next(net.parameters()).device

            def _get_learned_desc(img, keypoints):
                val_kps, patches = [], []
                h, w = img.shape
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    if x - 16 >= 0 and x + 16 <= w and y - 16 >= 0 and y + 16 <= h:
                        patches.append(img[y - 16 : y + 16, x - 16 : x + 16])
                        val_kps.append(kp)
                if not patches:
                    return val_kps, None
                t = torch.tensor(np.array(patches, dtype=np.float32) / 255.0).unsqueeze(1).to(dev)
                with torch.no_grad():
                    desc = net(t).cpu().numpy()
                return val_kps, desc

            kp_source, desc_source = _get_learned_desc(source_img, kp_source)
            kp_reference, desc_reference = _get_learned_desc(reference_img, kp_reference)
            norm_type = cv2.NORM_L2
        else:
            detector = cv2.AKAZE_create()
            native_src = [cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), size=float(k.size), response=float(k.response), octave=int(k.octave), class_id=int(k.class_id)) for k in kp_source]
            native_ref = [cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), size=float(k.size), response=float(k.response), octave=int(k.octave), class_id=int(k.class_id)) for k in kp_reference]
            kp_source, desc_source = detector.compute(source_img, native_src)
            kp_reference, desc_reference = detector.compute(reference_img, native_ref)
            norm_type = cv2.NORM_HAMMING
    else:
        # Default: AKAZE descriptor computed at crater landmark locations
        detector = cv2.AKAZE_create()
        native_src = [cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), size=float(k.size), response=float(k.response), octave=int(k.octave), class_id=int(k.class_id)) for k in kp_source]
        native_ref = [cv2.KeyPoint(x=float(k.pt[0]), y=float(k.pt[1]), size=float(k.size), response=float(k.response), octave=int(k.octave), class_id=int(k.class_id)) for k in kp_reference]
        kp_source, desc_source = detector.compute(source_img, native_src)
        kp_reference, desc_reference = detector.compute(reference_img, native_ref)
        norm_type = cv2.NORM_HAMMING

    if desc_source is None or desc_reference is None or len(desc_source) < 2 or len(desc_reference) < 2:
        detect_and_match.last_raw_matches = []
        detect_and_match.last_match_ratios = {}
        return [], list(kp_source), list(kp_reference)

    # 3. Match descriptors via Brute-Force Matcher with KNN (k=2)
    bf = cv2.BFMatcher(norm_type)
    raw_matches = bf.knnMatch(desc_source, desc_reference, k=2)

    detect_and_match.last_raw_matches = raw_matches

    if return_raw:
        return raw_matches, list(kp_source), list(kp_reference)

    # 4. Filter with Lowe's ratio test and enforce 1-to-1 matching
    candidate_matches: list[cv2.DMatch] = []
    good_ratios: dict[int, float] = {}
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            r = float(m.distance / (n.distance if n.distance > 0 else 1e-6))
            if ratio_threshold is None or m.distance < ratio_threshold * n.distance:
                candidate_matches.append(m)
                good_ratios[m.queryIdx] = r

    # Enforce 1-to-1 matching: if multiple source craters match the same reference crater,
    # retain only the one with the smallest descriptor distance
    best_for_train: dict[int, cv2.DMatch] = {}
    for m in candidate_matches:
        if m.trainIdx not in best_for_train or m.distance < best_for_train[m.trainIdx].distance:
            best_for_train[m.trainIdx] = m

    good_matches = list(best_for_train.values())
    detect_and_match.last_match_ratios = good_ratios
    return good_matches, list(kp_source), list(kp_reference)


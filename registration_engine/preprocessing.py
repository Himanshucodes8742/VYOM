"""Preprocessing routines for lunar image normalization and feature enhancement."""

import warnings
import numpy as np
import cv2

try:
    from skimage.exposure import match_histograms
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import phasepack


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast across lunar surface regions, balancing deep crater shadows
    and overexposed illuminated ridges without amplifying noise.

    Args:
        image: Grayscale input image (2D numpy array).
        clip_limit: Contrast limit threshold for OpenCV CLAHE.
        tile_grid_size: Tile grid dimensions (rows, cols) for adaptive equalization.

    Returns:
        Processed grayscale image with identical shape and dtype uint8.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    # Ensure input image is uint8
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    clahe_filter = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_filter.apply(img_u8)


def _manual_histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Manual CDF-based histogram matching fallback when scikit-image is unavailable."""
    s_values, bin_idx, s_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference.ravel(), return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    interp_values = np.interp(s_quantiles, r_quantiles, r_values)
    matched = interp_values[bin_idx].reshape(source.shape)
    return np.clip(np.round(matched), 0, 255).astype(np.uint8)


def histogram_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of the input image to the reference image.

    Normalizes global photometric distribution across images captured under differing
    sun angles or sensor sensitivities.

    Args:
        image: Source grayscale image (2D numpy array).
        reference: Reference grayscale image to match against (2D numpy array).

    Returns:
        Histogram-matched image with identical shape as image and dtype uint8.
    """
    if image.ndim != 2 or reference.ndim != 2:
        raise ValueError("Both image and reference must be 2D grayscale arrays")

    img_u8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    ref_u8 = reference if reference.dtype == np.uint8 else np.clip(reference, 0, 255).astype(np.uint8)

    if HAS_SKIMAGE:
        matched = match_histograms(img_u8, ref_u8)
        return np.clip(np.round(matched), 0, 255).astype(np.uint8)
    else:
        return _manual_histogram_match(img_u8, ref_u8)


def phase_congruency_map(image: np.ndarray) -> np.ndarray:
    """Compute illumination-invariant phase congruency feature map.

    Detects salient visual features (edges, crater rims, rilles) based on maximal order
    in Fourier phase components. Phase congruency is inherently invariant to intensity
    shifts and contrast variations, making it well-suited for multi-modal lunar matching.

    Args:
        image: Grayscale input image (2D numpy array).

    Returns:
        Combined edge-strength/moment map normalized to [0, 255] as dtype uint8.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    # Filter non-critical pyfftw fallback warning from phasepack
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="phasepack")
        # phasecong returns: M (max moment), m (min moment), ori, ft, PC, EO, T
        M, m, _, _, _, _, _ = phasepack.phasecong(image)

    # Combined edge strength and corner moment response
    combined = M + m

    min_val = np.nanmin(combined)
    max_val = np.nanmax(combined)

    if max_val > min_val:
        norm = ((combined - min_val) / (max_val - min_val) * 255.0)
        return np.clip(np.nan_to_num(norm), 0, 255).astype(np.uint8)
    else:
        return np.zeros(image.shape, dtype=np.uint8)


def photometric_normalize(
    image: np.ndarray,
    sun_elevation_deg: float,
    min_factor: float = 0.05
) -> np.ndarray:
    """Photometric normalization using the Lommel-Seeliger scattering model.

    Compensates for illumination-driven brightness variation across differing
    solar elevations before further local contrast enhancement is applied.

    Args:
        image: Grayscale input image (2D numpy array).
        sun_elevation_deg: Solar elevation angle above the local horizon in degrees.
        min_factor: Lower bound for the Lommel-Seeliger factor to prevent division by
                    near-zero values at grazing sun angles.

    Returns:
        Photometrically normalized 2D grayscale image with dtype uint8 in [0, 255].
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    # Ensure float precision for radiometric calculations
    img_float = image.astype(np.float32)

    # 1. Compute incidence angle as (90 - sun_elevation_deg) in radians.
    # NOTE: This assumes a flat local surface normal across the scene (scene-level approximation).
    # It does not perform a per-pixel terrain-aware topographic correction, which requires a
    # high-resolution Digital Elevation Model (DEM) and will be added separately in a later module.
    incidence_deg = max(0.0, min(90.0, 90.0 - float(sun_elevation_deg)))
    incidence_rad = np.radians(incidence_deg)

    # 2. Assume emission angle near 0 (nadir viewing geometry), so mu = cos(0) = 1.0.
    mu = 1.0
    mu0 = float(np.cos(incidence_rad))

    # 3. Compute the Lommel-Seeliger reflectance factor: mu0 / (mu0 + mu)
    ls_factor = mu0 / (mu0 + mu)

    # 4. Handle edge case: very low sun elevation (near-grazing illumination, where mu0 -> 0).
    # NOTE: As sun elevation approaches 0, the denominator approaches 0, which would amplify
    # sensor readout noise and cause extreme, blown-out, unusable pixel values. We clip the
    # correction factor to min_factor to maintain numerical stability and visual fidelity.
    effective_factor = max(ls_factor, float(min_factor))

    # 5. Divide pixel values by factor to compensate for predicted illumination disparity,
    # then clip and convert back to valid 0-255 uint8 range.
    normalized = img_float / effective_factor
    return np.clip(normalized, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    method: str = "clahe",
    sun_elevation_deg: float | None = None,
    reference_image: np.ndarray | None = None,
) -> np.ndarray:
    """Execute selected preprocessing routine or chained pipeline.

    Supported methods:
      - "clahe": Contrast Limited Adaptive Histogram Equalization.
      - "photometric": Lommel-Seeliger illumination normalization.
      - "photometric_clahe" (or "photometric+clahe"): Lommel-Seeliger normalization
        followed by CLAHE contrast equalization.
      - "histogram": Match histogram to reference_image.
      - "phase_congruency": Phase congruency moment map.
      - "none": Return unadjusted uint8 image.

    Args:
        image: Grayscale 2D array.
        method: Preprocessing pipeline option string.
        sun_elevation_deg: Sun elevation angle in degrees (required for photometric modes).
        reference_image: Reference 2D array (required for histogram matching).

    Returns:
        Preprocessed 2D uint8 image.
    """
    key = method.strip().lower()
    if key == "none":
        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    elif key == "clahe":
        return clahe(image)
    elif key == "photometric":
        if sun_elevation_deg is None:
            raise ValueError("sun_elevation_deg is required for photometric preprocessing")
        return photometric_normalize(image, sun_elevation_deg)
    elif key in ("photometric_clahe", "photometric+clahe"):
        if sun_elevation_deg is None:
            raise ValueError("sun_elevation_deg is required for photometric+clahe preprocessing")
        photo = photometric_normalize(image, sun_elevation_deg)
        return clahe(photo)
    elif key in ("histogram", "histogram_match"):
        if reference_image is None:
            raise ValueError("reference_image is required for histogram matching")
        return histogram_match(image, reference_image)
    elif key in ("phase_congruency", "phase_congruency_map"):
        return phase_congruency_map(image)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


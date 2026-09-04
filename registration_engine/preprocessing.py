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

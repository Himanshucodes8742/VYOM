"""Unit tests for registration_engine preprocessing routines."""

import numpy as np
import pytest

from registration_engine.preprocessing import clahe, histogram_match, phase_congruency_map


@pytest.fixture
def sample_synthetic_image() -> np.ndarray:
    """Create a 50x50 synthetic grayscale image with geometric shapes and gradients."""
    img = np.zeros((50, 50), dtype=np.uint8)
    # Add a gradient across rows
    for y in range(50):
        img[y, :] = int((y / 49.0) * 160)
    # Add contrasting square features simulating crater structure
    img[15:35, 15:35] = 230
    img[22:28, 22:28] = 40
    return img


@pytest.fixture
def reference_synthetic_image() -> np.ndarray:
    """Create a 50x50 synthetic reference image with different lighting/intensity."""
    ref = np.full((50, 50), 90, dtype=np.uint8)
    ref[10:40, 10:40] = 180
    return ref


def test_clahe(sample_synthetic_image: np.ndarray) -> None:
    """Assert CLAHE runs without error and returns array of identical shape and dtype uint8."""
    result = clahe(sample_synthetic_image)

    assert isinstance(result, np.ndarray)
    assert result.shape == sample_synthetic_image.shape
    assert result.dtype == np.uint8


def test_histogram_match(
    sample_synthetic_image: np.ndarray,
    reference_synthetic_image: np.ndarray
) -> None:
    """Assert histogram matching runs without error and returns array of identical shape and dtype uint8."""
    result = histogram_match(sample_synthetic_image, reference_synthetic_image)

    assert isinstance(result, np.ndarray)
    assert result.shape == sample_synthetic_image.shape
    assert result.dtype == np.uint8


def test_phase_congruency_map(sample_synthetic_image: np.ndarray) -> None:
    """Assert phase congruency runs without error and returns normalized array of identical shape and dtype uint8."""
    result = phase_congruency_map(sample_synthetic_image)

    assert isinstance(result, np.ndarray)
    assert result.shape == sample_synthetic_image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255

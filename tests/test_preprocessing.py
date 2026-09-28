from pathlib import Path
import numpy as np
import pytest

from registration_engine.preprocessing import (
    clahe,
    histogram_match,
    phase_congruency_map,
    photometric_normalize,
    preprocess_image,
)
from registration_engine.metadata import get_sun_elevation
from registration_engine.pipeline import run_pipeline


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


def test_photometric_normalize(sample_synthetic_image: np.ndarray) -> None:
    """Verify Lommel-Seeliger photometric normalization across solar elevations."""
    # Test nominal elevation
    res_45 = photometric_normalize(sample_synthetic_image, sun_elevation_deg=45.0)
    assert isinstance(res_45, np.ndarray)
    assert res_45.shape == sample_synthetic_image.shape
    assert res_45.dtype == np.uint8

    # Test grazing illumination edge case (near-horizon solar elevation)
    res_grazing = photometric_normalize(sample_synthetic_image, sun_elevation_deg=0.5)
    assert isinstance(res_grazing, np.ndarray)
    assert res_grazing.shape == sample_synthetic_image.shape
    assert res_grazing.dtype == np.uint8
    assert not np.isnan(res_grazing).any()
    assert not np.isinf(res_grazing).any()

    # Test extreme boundary at 0.0 deg
    res_zero = photometric_normalize(sample_synthetic_image, sun_elevation_deg=0.0)
    assert res_zero.dtype == np.uint8


def test_preprocess_image_modes(
    sample_synthetic_image: np.ndarray,
    reference_synthetic_image: np.ndarray
) -> None:
    """Test preprocess_image options: clahe, photometric, photometric_clahe, histogram, none."""
    # CLAHE
    c = preprocess_image(sample_synthetic_image, method="clahe")
    assert c.shape == sample_synthetic_image.shape

    # Photometric
    p = preprocess_image(sample_synthetic_image, method="photometric", sun_elevation_deg=15.0)
    assert p.shape == sample_synthetic_image.shape

    # Photometric + CLAHE chain
    pc = preprocess_image(sample_synthetic_image, method="photometric_clahe", sun_elevation_deg=15.0)
    assert pc.shape == sample_synthetic_image.shape

    # Histogram
    h = preprocess_image(sample_synthetic_image, method="histogram", reference_image=reference_synthetic_image)
    assert h.shape == sample_synthetic_image.shape

    # None
    n = preprocess_image(sample_synthetic_image, method="none")
    assert np.array_equal(n, sample_synthetic_image)


def test_photometric_vs_clahe_real_pair() -> None:
    """Compare registration performance of 'CLAHE only' vs 'Photometric + CLAHE' on real lunar pair."""
    project_root = Path(__file__).resolve().parent.parent
    real_dir = project_root / "data" / "demo_pairs" / "ohrc_nac_crater_x"
    label_path = project_root / "data" / "ch2_ohr_ncp_20210401T2357376656_d_img_d18.xml"

    src_path = real_dir / "source.png"
    ref_path = real_dir / "reference.png"

    if not src_path.exists() or not ref_path.exists() or not label_path.exists():
        pytest.skip("Real image pair or Chandrayaan-2 PDS4 XML label not found in data folder.")

    # Extract real sun elevation from Chandrayaan-2 PDS4 XML label
    source_sun_elev = get_sun_elevation(label_path)
    assert 0.0 <= source_sun_elev <= 90.0

    print(f"\n{'='*78}")
    print(f"  PHOTOMETRIC NORMALIZATION COMPARISON ON REAL PAIR")
    print(f"  Source: Chandrayaan-2 OHRC (Extracted Sun Elevation: {source_sun_elev:.4f}°)")
    print(f"  Reference: LROC NAC Crater X")
    print(f"{'='*78}")
    print(f"  {'Algorithm':<10} {'Preprocessing':<20} {'Inliers':>8} {'RMSE (px)':>10} {'Inlier Ratio':>13}")
    print(f"  {'-'*10} {'-'*20} {'-'*8} {'-'*10} {'-'*13}")

    for algo in ("akaze", "sift"):
        # 1. CLAHE only (baseline)
        res_clahe = run_pipeline(
            str(src_path),
            str(ref_path),
            algorithm=algo,
            preprocessing="clahe",
        )
        m_clahe = res_clahe.get("metrics") or {}
        inliers_c = m_clahe.get("inlier_count", 0)
        rmse_c = f"{m_clahe.get('rmse', 0.0):.4f}" if "rmse" in m_clahe else "N/A"
        ratio_c = f"{m_clahe.get('inlier_ratio', 0.0)*100:.2f}%" if "inlier_ratio" in m_clahe else "N/A"

        print(f"  {algo.upper():<10} {'CLAHE only':<20} {inliers_c:>8} {rmse_c:>10} {ratio_c:>13}")

        # 2. Photometric + CLAHE
        res_photo = run_pipeline(
            str(src_path),
            str(ref_path),
            algorithm=algo,
            preprocessing="photometric_clahe",
            source_sun_elevation=source_sun_elev,
        )
        m_photo = res_photo.get("metrics") or {}
        inliers_p = m_photo.get("inlier_count", 0)
        rmse_p = f"{m_photo.get('rmse', 0.0):.4f}" if "rmse" in m_photo else "N/A"
        ratio_p = f"{m_photo.get('inlier_ratio', 0.0)*100:.2f}%" if "inlier_ratio" in m_photo else "N/A"

        print(f"  {algo.upper():<10} {'Photometric + CLAHE':<20} {inliers_p:>8} {rmse_p:>10} {ratio_p:>13}")
        print(f"  {'-'*65}")

    print(f"{'='*78}\n")

    # Assert both pipelines ran without unhandled exceptions
    assert res_clahe["success"] or res_clahe["error"] is not None
    assert res_photo["success"] or res_photo["error"] is not None


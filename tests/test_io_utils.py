"""Unit tests for registration_engine I/O and resampling utilities."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import tifffile

from registration_engine.io_utils import RegistrationInputError, load_and_resample


@pytest.fixture
def temp_image_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary PNG and TIFF test images."""
    source_path = tmp_path / "source_test.png"
    ref_path = tmp_path / "ref_test.tif"

    source_arr = np.arange(100, dtype=np.uint8).reshape((10, 10))
    ref_arr = np.full((10, 10), 128, dtype=np.uint8)

    Image.fromarray(source_arr).save(source_path)
    tifffile.imwrite(str(ref_path), ref_arr)

    return source_path, ref_path


def test_load_and_resample_success(temp_image_pair: tuple[Path, Path]) -> None:
    """Test successful loading of PNG and TIFF pair without resampling."""
    src_path, ref_path = temp_image_pair
    src, ref = load_and_resample(str(src_path), str(ref_path))

    assert isinstance(src, np.ndarray)
    assert isinstance(ref, np.ndarray)
    assert src.shape == (10, 10)
    assert ref.shape == (10, 10)
    assert src.dtype == np.uint8
    assert ref.dtype == np.uint8


def test_load_and_resample_with_ratio(temp_image_pair: tuple[Path, Path]) -> None:
    """Test loading and resampling with target_gsd scale ratio."""
    src_path, ref_path = temp_image_pair
    src, ref = load_and_resample(str(src_path), str(ref_path), target_gsd=2.0)

    assert src.shape == (20, 20)
    assert ref.shape == (20, 20)


def test_missing_file_raises_custom_error(tmp_path: Path) -> None:
    """Test that missing image files raise RegistrationInputError with descriptive message."""
    missing = tmp_path / "non_existent.png"
    valid = tmp_path / "valid.png"
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(valid)

    with pytest.raises(RegistrationInputError, match="Image file does not exist"):
        load_and_resample(str(missing), str(valid))

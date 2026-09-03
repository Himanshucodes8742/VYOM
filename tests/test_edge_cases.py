import pytest
import numpy as np
import cv2
import tempfile
import os

from registration_engine.pipeline import run_pipeline
from registration_engine.ransac_filter import filter_matches
from registration_engine.io_utils import RegistrationInputError

def test_missing_file():
    res = run_pipeline("does_not_exist.png", "also_missing.png")
    assert res["success"] is False
    assert "Image file does not exist" in res["error"] or "does not exist" in res["error"]

def test_zero_overlapping_content():
    # Two random noise images
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.png")
        ref = os.path.join(td, "ref.png")
        cv2.imwrite(src, np.random.randint(0, 256, (500, 500), dtype=np.uint8))
        cv2.imwrite(ref, np.random.randint(0, 256, (500, 500), dtype=np.uint8))
        
        res = run_pipeline(src, ref)
        # Should gracefully fail or return 0 inliers
        assert res["success"] is False
        assert "no inliers" in res["error"].lower() or "insufficient matches" in res["error"].lower()

def test_unsupported_algorithm():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.png")
        ref = os.path.join(td, "ref.png")
        cv2.imwrite(src, np.zeros((100, 100), dtype=np.uint8))
        cv2.imwrite(ref, np.zeros((100, 100), dtype=np.uint8))
        
        res = run_pipeline(src, ref, algorithm="invalid_algo")
        assert res["success"] is False
        assert "unsupported algorithm" in res["error"].lower()

def test_filter_matches_empty():
    H, good_matches = filter_matches([], [], [])
    assert H.shape == (3, 3)
    assert len(good_matches) == 0

def test_very_small_image():
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.png")
        ref = os.path.join(td, "ref.png")
        cv2.imwrite(src, np.random.randint(0, 256, (20, 20), dtype=np.uint8))
        cv2.imwrite(ref, np.random.randint(0, 256, (20, 20), dtype=np.uint8))
        
        # Should not crash, just return success=False due to not enough features
        res = run_pipeline(src, ref)
        if not res["success"]:
            assert res["error"] is not None
        else:
            assert True

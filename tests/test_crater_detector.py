"""Unit tests for crater classification CNN model and inference pipeline."""

import pytest
import numpy as np
import torch
from pathlib import Path

from registration_engine.crater_detector import CraterClassifierNet, predict_crater


def test_crater_classifier_architecture():
    """Verify input/output shapes and parameter range for CraterClassifierNet."""
    model = CraterClassifierNet()
    model.eval()

    # Test with standard 64x64 batch
    dummy_input_64 = torch.randn(4, 1, 64, 64)
    out_64 = model(dummy_input_64)
    assert out_64.shape == (4, 1)
    assert torch.all(out_64 >= 0.0) and torch.all(out_64 <= 1.0)

    # Test with 32x32 batch (adaptive pooling support)
    dummy_input_32 = torch.randn(2, 1, 32, 32)
    out_32 = model(dummy_input_32)
    assert out_32.shape == (2, 1)
    assert torch.all(out_32 >= 0.0) and torch.all(out_32 <= 1.0)


def test_saved_model_inference():
    """Verify inference on saved checkpoint if present."""
    model_path = Path("registration_engine/models/crater_classifier.pt")
    if not model_path.exists():
        pytest.skip("Model checkpoint not yet trained/saved.")

    # Synthetic test patch
    test_patch = np.zeros((64, 64), dtype=np.uint8)
    prob = predict_crater(test_patch, model_path=str(model_path))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_detect_crater_landmarks():
    """Verify sliding-window landmark detection, NMS, and KeyPoint attributes."""
    from registration_engine.matchers import detect_crater_landmarks, CraterKeyPoint
    import cv2

    model_path = Path("registration_engine/models/crater_classifier.pt")
    if not model_path.exists():
        pytest.skip("Model checkpoint not yet trained/saved.")

    # Create synthetic image large enough for sliding windows
    img = np.zeros((200, 200), dtype=np.uint8)
    # Draw synthetic circular crater feature
    cv2.circle(img, (100, 100), 20, 180, -1)
    cv2.circle(img, (100, 100), 15, 60, -1)

    kps = detect_crater_landmarks(
        img,
        model_path=str(model_path),
        window_sizes=[64],
        stride=32,
        conf_threshold=0.5,
    )
    assert isinstance(kps, list)
    for kp in kps:
        assert isinstance(kp, cv2.KeyPoint)
        assert isinstance(kp, CraterKeyPoint)
        assert hasattr(kp, "pt")
        # Check unpackability
        x, y = kp
        assert isinstance(x, float) and isinstance(y, float)


def test_crater_landmarks_pipeline_registration():
    """Verify crater_landmarks algorithm option in detect_and_match and pipeline."""
    from registration_engine.matchers import detect_and_match, SUPPORTED_ALGORITHMS

    assert "crater_landmarks" in SUPPORTED_ALGORITHMS

    model_path = Path("registration_engine/models/crater_classifier.pt")
    if not model_path.exists():
        pytest.skip("Model checkpoint not yet trained/saved.")

    img = np.zeros((150, 150), dtype=np.uint8)
    matches, kp_s, kp_r = detect_and_match(img, img, algorithm="crater_landmarks")
    assert isinstance(matches, list)
    assert isinstance(kp_s, list)
    assert isinstance(kp_r, list)


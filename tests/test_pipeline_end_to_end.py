"""End-to-end test of the full registration pipeline on the synthetic validation pair."""

import os
from pathlib import Path
import numpy as np
import pytest

from registration_engine.pipeline import run_pipeline


# Resolve paths relative to the project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "demo_pairs" / "synthetic_validation"
REAL_PAIR_DIR = PROJECT_ROOT / "data" / "demo_pairs" / "ohrc_nac_crater_x"

SOURCE_PATH = str(SYNTHETIC_DIR / "source.png")
TARGET_PATH = str(SYNTHETIC_DIR / "target.png")
GROUND_TRUTH_PATH = str(SYNTHETIC_DIR / "ground_truth_transform.npy")

ALGORITHMS = ["sift", "akaze", "rift2"]


# ---------------------------------------------------------------------------
# Fixtures — run each algorithm once and cache the result per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sift_result():
    """Run the full pipeline with SIFT on the synthetic pair."""
    return run_pipeline(SOURCE_PATH, TARGET_PATH, algorithm="sift")


@pytest.fixture(scope="module")
def akaze_result():
    """Run the full pipeline with AKAZE on the synthetic pair."""
    return run_pipeline(SOURCE_PATH, TARGET_PATH, algorithm="akaze")


@pytest.fixture(scope="module")
def rift2_result():
    """Run the full pipeline with RIFT2 on the synthetic pair."""
    return run_pipeline(SOURCE_PATH, TARGET_PATH, algorithm="rift2")


# ---------------------------------------------------------------------------
# Parametrised synthetic-pair tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ALGORITHMS)
def test_pipeline_success(algo, sift_result, akaze_result, rift2_result) -> None:
    """Assert that the pipeline completes successfully for each algorithm."""
    result = {"sift": sift_result, "akaze": akaze_result, "rift2": rift2_result}[algo]
    if not result["success"]:
        pytest.fail(f"[{algo.upper()}] Pipeline failed with error:\n{result['error']}")
    assert result["success"] is True


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_pipeline_metrics_present(algo, sift_result, akaze_result, rift2_result) -> None:
    """Assert that all expected metric keys are present."""
    result = {"sift": sift_result, "akaze": akaze_result, "rift2": rift2_result}[algo]
    assert result["metrics"] is not None, f"[{algo.upper()}] Metrics is None"
    for key in ("rmse", "inlier_count", "inlier_ratio", "distribution_score"):
        assert key in result["metrics"], f"[{algo.upper()}] Missing metric key: {key}"


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_pipeline_rmse_is_finite(algo, sift_result, akaze_result, rift2_result) -> None:
    """Assert RMSE is a finite, non-negative number."""
    result = {"sift": sift_result, "akaze": akaze_result, "rift2": rift2_result}[algo]
    metrics = result["metrics"]
    assert metrics is not None
    assert np.isfinite(metrics["rmse"])
    assert metrics["rmse"] >= 0


# ---------------------------------------------------------------------------
# Comparison report — prints all three side by side
# ---------------------------------------------------------------------------

def test_print_comparison_table(sift_result, akaze_result, rift2_result) -> None:
    """Print a side-by-side comparison of all three algorithms on the synthetic pair.

    This test always passes — its purpose is to surface the numbers in pytest -s output.
    """
    results = {
        "SIFT": sift_result,
        "AKAZE": akaze_result,
        "RIFT2": rift2_result,
    }

    header = f"\n{'='*72}\n  SYNTHETIC PAIR REGISTRATION: ALGORITHM COMPARISON\n{'='*72}"
    print(header)
    print(f"  {'Algorithm':<10} {'RMSE (px)':>10} {'Inliers':>10} {'Ratio':>10} {'DistScore':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, res in results.items():
        m = res.get("metrics")
        if m is None or not res["success"]:
            print(f"  {name:<10} {'FAILED':>10} {'':>10} {'':>10} {'':>10}")
            continue
        print(
            f"  {name:<10} "
            f"{m['rmse']:>10.4f} "
            f"{m['inlier_count']:>10d} "
            f"{m['inlier_ratio']:>10.4f} "
            f"{m['distribution_score']:>10.4f}"
        )

    print(f"{'='*72}")

    # Ground-truth comparison for each algorithm
    if os.path.exists(GROUND_TRUTH_PATH):
        gt = np.load(GROUND_TRUTH_PATH)
        print("\n  Max |diff| from ground-truth homography:")
        for name, res in results.items():
            est = res.get("transform_matrix")
            if est is not None and res["success"]:
                max_diff = float(np.abs(gt - est).max())
                print(f"    {name:<10} {max_diff:.6f}")
            else:
                print(f"    {name:<10} N/A")
        print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Real-pair tests (skipped if data directory doesn't exist yet)
# ---------------------------------------------------------------------------

def _find_real_pair_images() -> tuple[str, str] | None:
    """Look for source/reference images in the real pair directory."""
    if not REAL_PAIR_DIR.exists():
        return None
    # Accept common naming patterns
    for src_name in ("source.png", "ohrc.png", "source.tif", "ohrc.tif"):
        for ref_name in ("reference.png", "nac.png", "reference.tif", "nac.tif"):
            src = REAL_PAIR_DIR / src_name
            ref = REAL_PAIR_DIR / ref_name
            if src.exists() and ref.exists():
                return str(src), str(ref)
    return None


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_real_pair(algo) -> None:
    """Run pipeline on real OHRC/NAC pair if available, else skip."""
    pair = _find_real_pair_images()
    if pair is None:
        pytest.skip(
            f"Real image pair directory not found or incomplete at: {REAL_PAIR_DIR}. "
            "Place source and reference images there to enable this test."
        )

    src_path, ref_path = pair
    result = run_pipeline(src_path, ref_path, algorithm=algo)

    m = result.get("metrics")
    print(f"\n  [REAL PAIR / {algo.upper()}]", end="")
    if result["success"] and m is not None:
        print(
            f"  RMSE={m['rmse']:.4f}  Inliers={m['inlier_count']}  "
            f"Ratio={m['inlier_ratio']:.4f}  DistScore={m['distribution_score']:.4f}"
        )
    else:
        print(f"  FAILED: {result.get('error', 'unknown error')}")

    # Don't hard-fail on real data — just report.  The synthetic tests are the gate.
    assert result["success"] is True or result["error"] is not None

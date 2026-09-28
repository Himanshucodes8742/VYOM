"""Match Verifier Model Training Script.

Trains a Random Forest classifier on synthetic lunar match features to classify
matches as correct (1) or incorrect (0). Evaluates performance on a stratified
held-out test set, outputs accuracy, precision, recall, F1, confusion matrix,
and feature importances, and saves the trained model to registration_engine/models/match_verifier.pkl.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The 5 primary candidate match features
FEATURE_COLUMNS = [
    "descriptor_distance",
    "lowes_ratio",
    "scale_ratio",
    "local_density",
    "response_strength",
]
TARGET_COLUMN = "label"

FEATURE_DESCRIPTIONS = {
    "scale_ratio": "Keypoint scale consistency between matched pairs (src_size / ref_size). Inliers align closely with the true affine scale.",
    "lowes_ratio": "Lowe's ratio test (d1 / d2). Quantifies descriptor distinctiveness over the second-nearest neighbor.",
    "response_strength": "Detector corner response magnitude. Saliency of lunar crater rims and high-contrast surface ridges.",
    "descriptor_distance": "AKAZE descriptor Hamming distance. Bitwise difference between 486-bit binary descriptors.",
    "local_density": "Local keypoint density within 50px radius. Measures whether features are in clustered vs isolated surface areas.",
}


def load_and_preprocess_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV and extract features and labels.

    Args:
        dataset_path: Path to match_verifier_dataset.csv.

    Returns:
        Tuple of (X, y) as DataFrame and Series.
    """
    if not dataset_path.exists():
        print(f"[!] Error: Dataset not found at: {dataset_path}")
        print("    Please run scripts/generate_training_data.py first to create the dataset.")
        sys.exit(1)

    print(f"[+] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    Total samples loaded: {len(df):,}")
    print(f"    Available columns: {list(df.columns)}")

    # Check for missing feature columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        print(f"[!] Error: Missing required columns in dataset: {missing_cols}")
        sys.exit(1)

    # Check for NaN / null values
    nan_count = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    if nan_count > 0:
        print(f"[!] Warning: Found {nan_count} null values. Dropping rows with nulls.")
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    class_counts = y.value_counts().to_dict()
    total = len(y)
    print(f"\n[+] Class distribution:")
    print(f"    Class 1 (Correct Inlier)   : {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.2f}%)")
    print(f"    Class 0 (Incorrect Outlier): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.2f}%)")

    return X, y


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    test_size: float = 0.20,
    class_weight: str | None = None,
) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest classifier and compute evaluation metrics.

    Args:
        X: Feature DataFrame (5 features).
        y: Target label Series (0 or 1).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_state: Seed for reproducibility.
        test_size: Fraction of samples reserved for testing.
        class_weight: Class weighting strategy (None or 'balanced').

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    print(f"\n[+] Splitting dataset into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"    Training samples : {len(X_train):,}")
    print(f"    Test samples     : {len(X_test):,}")

    print(f"\n[+] Initializing RandomForestClassifier:")
    print(f"    - n_estimators : {n_estimators}")
    print(f"    - max_depth    : {max_depth}")
    print(f"    - class_weight : {class_weight}")
    print(f"    - random_state : {random_state}")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    print("\n[+] Training model on training split...")
    clf.fit(X_train, y_train)
    print("    Training completed.")

    print("\n[+] Evaluating on held-out test split...")
    y_pred = clf.predict(X_test)

    # Compute quantitative evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec_binary = precision_score(y_test, y_pred, zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_binary = recall_score(y_test, y_pred, zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_bin = f1_score(y_test, y_pred, zero_division=0)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_binary": prec_binary,
        "precision_macro": prec_macro,
        "recall_binary": rec_binary,
        "recall_macro": rec_macro,
        "f1_binary": f1_bin,
        "f1_macro": f1_mac,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "test_size": len(y_test),
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return clf, metrics


def print_evaluation_report(clf: RandomForestClassifier, metrics: dict) -> None:
    """Print clean, detailed evaluation metrics and feature importances."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print("\n" + "=" * 70)
    print("           MATCH VERIFIER MODEL: EVALUATION REPORT")
    print("=" * 70)
    print(f"  Test Samples Evaluated:      {metrics['test_size']:,}")
    print(f"  Accuracy:                    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Class 1, Inlier): {metrics['precision_binary']:.4f}")
    print(f"  Recall    (Class 1, Inlier): {metrics['recall_binary']:.4f}")
    print(f"  F1-Score  (Class 1, Inlier): {metrics['f1_binary']:.4f}")
    print(f"  Macro Precision:             {metrics['precision_macro']:.4f}")
    print(f"  Macro Recall:                {metrics['recall_macro']:.4f}")
    print(f"  Macro F1-Score:              {metrics['f1_macro']:.4f}")

    print("\n" + "-" * 70)
    print("  CONFUSION MATRIX:")
    print("-" * 70)
    print(f"                    Predicted Outlier (0)   Predicted Inlier (1)")
    print(f"  Actual Outlier (0):      {tn:>6d} (TN)               {fp:>6d} (FP)")
    print(f"  Actual Inlier  (1):      {fn:>6d} (FN)               {tp:>6d} (TP)")
    print("-" * 70)

    print("\n" + "-" * 70)
    print("  FEATURE IMPORTANCES (Ranked by Influence):")
    print("-" * 70)
    importances = clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_indices, start=1):
        feat = FEATURE_COLUMNS[idx]
        imp = importances[idx]
        pct = imp * 100.0
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        print(f"  {rank}. {feat:<22} : {imp:.4f} ({pct:5.1f}%)")
        print(f"     Explanation: {desc}")

    print("=" * 70 + "\n")


def save_model(clf: RandomForestClassifier, output_path: Path) -> None:
    """Save trained model to disk using joblib.

    Args:
        clf: Trained scikit-learn model.
        output_path: Destination .pkl path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"[+] Successfully saved trained model to: {output_path}")

    # Verify reloadability
    loaded_clf = joblib.load(output_path)
    assert hasattr(loaded_clf, "predict"), "Loaded model missing predict method"
    print(f"[+] Model reload verified successfully from: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest Match Verifier on lunar image features."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "data" / "training" / "match_verifier_dataset.csv"),
        help="Path to training dataset CSV.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(PROJECT_ROOT / "registration_engine" / "models" / "match_verifier.pkl"),
        help="Destination path for serialized model pickle.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of trees (default: 10).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced", "balanced_subsample", "none"],
        help="Class weighting mode (default: None).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cw = None if args.class_weight == "none" else args.class_weight
    dataset_path = Path(args.dataset).resolve()
    output_model_path = Path(args.output_model).resolve()

    X, y = load_and_preprocess_data(dataset_path)
    clf, metrics = train_and_evaluate(
        X=X,
        y=y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight=cw,
    )
    print_evaluation_report(clf, metrics)
    save_model(clf, output_model_path)


if __name__ == "__main__":
    main()

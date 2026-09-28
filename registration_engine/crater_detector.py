"""Crater classification CNN and training pipeline for lunar feature verification."""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CraterClassifierNet(nn.Module):
    """Convolutional Neural Network for binary crater patch classification.

    Architecture mirrors PatchDescriptorNet (Conv-BN-ReLU-Pool blocks),
    projecting through an adaptive pooling layer to a single Sigmoid probability output.
    """

    def __init__(self):
        super(CraterClassifierNet, self).__init__()
        # Conv block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        # Adaptive pooling ensures consistent 4x4 spatial dimensions regardless of input patch resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected binary classification head
        self.fc1 = nn.Linear(128 * 4 * 4, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects input shape (Batch, 1, H, W). Returns shape (Batch, 1)."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def train_crater_classifier(
    data_path: str = "data/training/crater_patches.npz",
    save_path: str = "registration_engine/models/crater_classifier.pt",
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    random_seed: int = 42,
) -> dict:
    """Train the crater classification CNN using BCE loss and Adam optimizer.

    Args:
        data_path: Path to crater_patches.npz containing 'patches' and 'labels'.
        save_path: Output file path for best-validation-accuracy model weights.
        epochs: Maximum number of training epochs (default 30).
        batch_size: Mini-batch size for DataLoader (default 16).
        learning_rate: Adam optimizer learning rate (default 0.001).
        random_seed: Random seed for reproducible 80/20 train/validation split.

    Returns:
        dict containing final validation metrics:
            - accuracy, precision, recall, f1_score, val_loss, best_epoch
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on compute device: {device}")

    # 1. Load dataset
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    print(f"Loading crater patches from: {data_file}")
    data = np.load(data_file)
    raw_patches = data["patches"]  # shape (N, 64, 64), uint8
    raw_labels = data["labels"]    # shape (N,), int64

    total_samples = len(raw_patches)
    print(f"Total samples loaded: {total_samples} (Pos: {np.sum(raw_labels == 1)}, Neg: {np.sum(raw_labels == 0)})")

    # Normalize pixels to [0.0, 1.0] and add channel dimension (N, 1, 64, 64)
    patches_norm = (raw_patches.astype(np.float32) / 255.0)[:, np.newaxis, :, :]
    labels_float = raw_labels.astype(np.float32)[:, np.newaxis]

    # 2. Stratified 80/20 Train/Validation Split
    pos_indices = np.where(raw_labels == 1)[0]
    neg_indices = np.where(raw_labels == 0)[0]

    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    n_pos_train = int(round(0.8 * len(pos_indices)))
    n_neg_train = int(round(0.8 * len(neg_indices)))

    train_indices = np.concatenate([pos_indices[:n_pos_train], neg_indices[:n_neg_train]])
    val_indices = np.concatenate([pos_indices[n_pos_train:], neg_indices[n_neg_train:]])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    print(f"Train/Val Split (80/20): {len(train_indices)} train ({n_pos_train} pos, {n_neg_train} neg) | {len(val_indices)} val ({len(pos_indices) - n_pos_train} pos, {len(neg_indices) - n_neg_train} neg)")

    # 3. Create PyTorch Datasets and DataLoaders
    train_x = torch.tensor(patches_norm[train_indices], dtype=torch.float32)
    train_y = torch.tensor(labels_float[train_indices], dtype=torch.float32)

    val_x = torch.tensor(patches_norm[val_indices], dtype=torch.float32)
    val_y = torch.tensor(labels_float[val_indices], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )

    # 4. Initialize Model, Loss Function, and Optimizer
    model = CraterClassifierNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"{'EPOCH':<8} | {'TRAIN LOSS':<12} | {'TRAIN ACC':<11} | {'VAL LOSS':<11} | {'VAL ACC':<10} | {'STATUS'}")
    print("=" * 80)

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * bx.size(0)
            predicted_class = (preds >= 0.5).float()
            train_correct += (predicted_class == by).sum().item()
            train_total += bx.size(0)

        epoch_train_loss = train_loss_accum / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_preds = model(vx)
                v_loss = criterion(v_preds, vy)

                val_loss_accum += v_loss.item() * vx.size(0)
                v_predicted = (v_preds >= 0.5).float()
                val_correct += (v_predicted == vy).sum().item()
                val_total += vx.size(0)

        epoch_val_loss = val_loss_accum / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        # Checkpointing criteria: highest validation accuracy (secondary tie-breaker: lower val loss)
        is_best = False
        if (epoch_val_acc > best_val_acc) or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            is_best = True
            torch.save(model.state_dict(), str(save_path_obj))

        status_str = "-> BEST (SAVED)" if is_best else ""
        print(f"Epoch {epoch:02d}/{epochs:02d} | {epoch_train_loss:<12.4f} | {epoch_train_acc:<10.1f}% | {epoch_val_loss:<11.4f} | {epoch_val_acc:<9.1f}% | {status_str}")

    print("=" * 80)
    print(f"Training completed. Best model checkpoint from Epoch {best_epoch} saved to: {save_path_obj}")

    # 6. Final Evaluation on Validation Set using Best Saved Model
    best_model = CraterClassifierNet().to(device)
    best_model.load_state_dict(torch.load(str(save_path_obj), map_location=device))
    best_model.eval()

    all_val_preds = []
    all_val_targets = []

    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device)
            preds = best_model(vx).cpu().numpy().flatten()
            targets = vy.numpy().flatten()
            all_val_preds.extend(preds)
            all_val_targets.extend(targets)

    val_preds_arr = np.array(all_val_preds)
    val_targets_arr = np.array(all_val_targets)

    pred_binary = (val_preds_arr >= 0.5).astype(int)
    targets_binary = val_targets_arr.astype(int)

    tp = int(np.sum((pred_binary == 1) & (targets_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (targets_binary == 0)))
    tn = int(np.sum((pred_binary == 0) & (targets_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (targets_binary == 1)))

    val_accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    val_precision = tp / max(1, (tp + fp))
    val_recall = tp / max(1, (tp + fn))
    val_f1 = (
        2 * (val_precision * val_recall) / max(1e-6, (val_precision + val_recall))
        if (val_precision + val_recall) > 0
        else 0.0
    )

    print("\n" + "=" * 50)
    print("FINAL VALIDATION EVALUATION RESULTS")
    print("=" * 50)
    print(f"Best Model Epoch:       {best_epoch}")
    print(f"Validation Accuracy:    {val_accuracy * 100:.2f}% ({tp + tn}/{len(targets_binary)})")
    print(f"Validation Precision:   {val_precision:.4f} ({val_precision * 100:.1f}%)")
    print(f"Validation Recall:      {val_recall:.4f} ({val_recall * 100:.1f}%)")
    print(f"Validation F1-Score:    {val_f1:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  True Positives  (TP): {tp:<4} | False Positives (FP): {fp:<4}")
    print(f"  False Negatives (FN): {fn:<4} | True Negatives  (TN): {tn:<4}")
    print("=" * 50)

    return {
        "best_epoch": best_epoch,
        "accuracy": float(val_accuracy),
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1_score": float(val_f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "best_val_loss": float(best_val_loss),
    }


def predict_crater(
    patch: np.ndarray,
    model: nn.Module | None = None,
    model_path: str = "registration_engine/models/crater_classifier.pt",
) -> float:
    """Predict the probability that an input image patch is a lunar crater.

    Args:
        patch: 2D numpy array (grayscale image patch).
        model: Optional pre-loaded CraterClassifierNet instance.
        model_path: Path to saved model weights if model is not provided.

    Returns:
        float: Probability in [0.0, 1.0] that the patch contains a crater.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = CraterClassifierNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

    # Preprocess patch
    patch_u8 = patch if patch.dtype == np.uint8 else np.clip(patch, 0, 255).astype(np.uint8)
    if patch_u8.shape != (64, 64):
        import cv2
        patch_u8 = cv2.resize(patch_u8, (64, 64), interpolation=cv2.INTER_AREA)

    patch_tensor = torch.tensor(
        patch_u8.astype(np.float32) / 255.0, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(patch_tensor).item()

    return float(prob)


if __name__ == "__main__":
    train_crater_classifier()

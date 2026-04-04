import os
import math
import copy
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# =========================================================
# CNN-LSTM for Human Activity Recognition using a real dataset
# Target dataset structure:
# UCI HAR Dataset/
#   train/
#     X_train.txt
#     y_train.txt
#     subject_train.txt
#     Inertial Signals/
#       body_acc_x_train.txt
#       body_acc_y_train.txt
#       body_acc_z_train.txt
#       body_gyro_x_train.txt
#       body_gyro_y_train.txt
#       body_gyro_z_train.txt
#   test/
#     X_test.txt
#     y_test.txt
#     subject_test.txt
#     Inertial Signals/
#       body_acc_x_test.txt
#       body_acc_y_test.txt
#       body_acc_z_test.txt
#       body_gyro_x_test.txt
#       body_gyro_y_test.txt
#       body_gyro_z_test.txt
#
# This script uses the REAL raw inertial signal files, not the engineered 561 features.
# =========================================================


@dataclass
class Config:
    dataset_root: Optional[str] = None
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    num_classes: int = 6
    num_workers: int = 0
    patience: int = 8
    seed: int = 42
    augment: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_signal_file(path: str) -> np.ndarray:
    # Each row is one sample, each column is one timestep
    return np.loadtxt(path, dtype=np.float32)


def resolve_dataset_root(explicit_root: Optional[str]) -> str:
    if explicit_root and os.path.isdir(explicit_root):
        return explicit_root

    candidates = [
        "./UCI HAR Dataset",
        "./UCI_HAR_Dataset",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find UCI HAR dataset. Expected one of: "
        "'./UCI HAR Dataset' or './UCI_HAR_Dataset', or pass --dataset-root."
    )


def load_split(root: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signals_dir = os.path.join(root, split, "Inertial Signals")
    signal_arrays = []

    for signal_name in SIGNAL_FILES:
        filename = f"{signal_name}_{split}.txt"
        full_path = os.path.join(signals_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing file: {full_path}")
        arr = load_signal_file(full_path)  # [N, T]
        signal_arrays.append(arr)

    # Stack into [N, C, T]
    x = np.stack(signal_arrays, axis=1)

    y_path = os.path.join(root, split, f"y_{split}.txt")
    s_path = os.path.join(root, split, f"subject_{split}.txt")

    y = np.loadtxt(y_path, dtype=np.int64) - 1  # labels become 0..5
    subjects = np.loadtxt(s_path, dtype=np.int64)

    return x, y, subjects


def channelwise_standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x shape: [N, C, T]
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_val, x_test


def sanitize_signals(x: np.ndarray) -> np.ndarray:
    # Protect training from malformed values and heavy outliers.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(x, 1.0, axis=(0, 2), keepdims=True)
    high = np.percentile(x, 99.0, axis=(0, 2), keepdims=True)
    return np.clip(x, low, high)


class HARDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.augment:
            x = self.apply_augmentation(x)
        return x, self.y[idx]

    @staticmethod
    def apply_augmentation(x: torch.Tensor) -> torch.Tensor:
        # x shape: [C, T]
        if torch.rand(1).item() < 0.7:
            x = x + 0.01 * torch.randn_like(x)
        if torch.rand(1).item() < 0.5:
            scale = torch.empty(x.size(0), 1).uniform_(0.9, 1.1)
            x = x * scale
        if torch.rand(1).item() < 0.3:
            t = x.size(1)
            mask_len = max(4, t // 10)
            start = torch.randint(0, max(1, t - mask_len + 1), (1,)).item()
            x[:, start : start + mask_len] = 0.0
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.features(x)  # [B, 256, T_reduced]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, H]
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)  # [B, S]
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, H]
        return context, weights


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        in_channels: int = 9,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = AttentionLayer(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T]
        features = self.encoder(x)              # [B, 256, T_reduced]
        sequence = features.transpose(1, 2)    # [B, T_reduced, 256]
        lstm_out, _ = self.lstm(sequence)      # [B, T_reduced, 2H]
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float, float]:
    model.train()
    losses = []
    preds_all = []
    targets_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.detach().cpu().numpy())
        targets_all.extend(yb.detach().cpu().numpy())

    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average="macro")
    return float(np.mean(losses)), acc, f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds_all = []
    targets_all = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits, _ = model(xb)
        loss = criterion(logits, yb)

        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.cpu().numpy())
        targets_all.extend(yb.cpu().numpy())

    acc = accuracy_score(targets_all, preds_all)
    f1 = f1_score(targets_all, preds_all, average="macro")
    return float(np.mean(losses)), acc, f1, np.array(targets_all), np.array(preds_all)


class EarlyStopping:
    def __init__(self, patience: int = 8):
        self.patience = patience
        self.best_score = -math.inf
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def prepare_data(config: Config):
    x_train_full, y_train_full, subjects_train = load_split(config.dataset_root, "train")
    x_test, y_test, subjects_test = load_split(config.dataset_root, "test")

    # Subject-aware validation split
    unique_subjects = np.unique(subjects_train)
    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=0.2,
        random_state=config.seed,
        shuffle=True,
    )

    train_mask = np.isin(subjects_train, train_subjects)
    val_mask = np.isin(subjects_train, val_subjects)

    x_train = x_train_full[train_mask]
    y_train = y_train_full[train_mask]
    x_val = x_train_full[val_mask]
    y_val = y_train_full[val_mask]

    x_train = sanitize_signals(x_train)
    x_val = sanitize_signals(x_val)
    x_test = sanitize_signals(x_test)
    x_train, x_val, x_test = channelwise_standardize(x_train, x_val, x_test)

    train_ds = HARDataset(x_train, y_train, augment=config.augment)
    val_ds = HARDataset(x_val, y_val)
    test_ds = HARDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = compute_class_weights(y_train, config.num_classes)
    num_channels = x_train.shape[1]
    return train_loader, val_loader, test_loader, class_weights, subjects_train, subjects_test, num_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN-LSTM HAR training on real UCI HAR raw signals")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to UCI HAR dataset root")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--augment", action="store_true", help="Enable training-time augmentation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.augment:
        config.augment = True
    config.dataset_root = resolve_dataset_root(args.dataset_root)

    set_seed(config.seed)

    print(f"Using device: {config.device}")
    print(f"Loading dataset from: {config.dataset_root}")

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        subjects_train,
        subjects_test,
        num_channels,
    ) = prepare_data(config)

    print(f"Train subjects: {sorted(np.unique(subjects_train).tolist())}")
    print(f"Test subjects:  {sorted(np.unique(subjects_test).tolist())}")

    model = CNNLSTM(
        num_classes=config.num_classes,
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
        in_channels=num_channels,
    ).to(config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    early_stopper = EarlyStopping(patience=config.patience)

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, config.device
        )

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"LR {current_lr:.6f} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}"
        )

        should_stop = early_stopper.step(val_f1, model)
        if should_stop:
            print("Early stopping triggered.")
            break

    if early_stopper.best_state is None:
        raise RuntimeError("Training finished without saving a best model state.")

    model.load_state_dict(early_stopper.best_state)
    torch.save(model.state_dict(), "best_cnn_lstm_har.pt")
    print("Saved best model to best_cnn_lstm_har.pt")

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, config.device
    )

    print("\n===== Final Test Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")

    print("\n===== Classification Report =====")
    report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3, 4, 5],
    target_names=CLASS_NAMES,
    digits=4
)
    print(report)

    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

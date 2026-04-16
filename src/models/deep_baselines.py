from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.io import read_image
    from torchvision.transforms import v2
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
    raise SystemExit(
        "A9 requires torch and torchvision in the experiment .venv. "
        "Install them before running run_a9_deep_baselines.py."
    ) from exc

from evaluation.metrics import compute_binary_metrics, optimize_threshold


IMAGE_SIZE = 128


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CropDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_column: str,
        feature_columns: list[str] | None = None,
        train_mode: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_column = image_column
        self.feature_columns = feature_columns or []
        self.train_mode = train_mode
        self.transform = v2.Compose(
            [
                v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = read_image(str(row[self.image_column]))
        image = self.transform(image)
        sample = {
            "image": image,
            "target": torch.tensor(int(row["target_binary"]), dtype=torch.float32),
        }
        if self.feature_columns:
            tabular_values = row[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            tabular_values = np.nan_to_num(tabular_values, nan=0.0, posinf=0.0, neginf=0.0)
            sample["tabular"] = torch.tensor(tabular_values, dtype=torch.float32)
        return sample


class SmallCNN(nn.Module):
    def __init__(self, tabular_dim: int = 0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.tabular_dim = tabular_dim
        fusion_dim = 64 + tabular_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        image_embedding = self.image_head(self.features(image))
        if self.tabular_dim > 0:
            assert tabular is not None
            logits = self.classifier(torch.cat([image_embedding, tabular], dim=1))
        else:
            logits = self.classifier(image_embedding)
        return logits.squeeze(1)


@dataclass
class TrainingResult:
    model_path: str
    history: list[dict[str, float]]
    threshold: float
    test_predictions: pd.DataFrame
    test_metrics: dict[str, float]


def _make_loader(frame: pd.DataFrame, image_column: str, feature_columns: list[str] | None, shuffle: bool) -> DataLoader:
    dataset = CropDataset(frame, image_column=image_column, feature_columns=feature_columns, train_mode=shuffle)
    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    if optimizer is None:
        model.eval()
    else:
        model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        tabular = batch.get("tabular")
        if tabular is not None:
            tabular = tabular.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(image, tabular)
        loss = criterion(logits, target)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_targets.append(target.detach().cpu().numpy())
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(all_targets), np.concatenate(all_scores)


def train_deep_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_column: str,
    feature_columns: list[str] | None,
    checkpoint_path: Path,
    max_epochs: int = 12,
    patience: int = 3,
) -> TrainingResult:
    device = choose_device()
    model = SmallCNN(tabular_dim=len(feature_columns or [])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = _make_loader(train_df, image_column=image_column, feature_columns=feature_columns, shuffle=True)
    val_loader = _make_loader(val_df, image_column=image_column, feature_columns=feature_columns, shuffle=False)
    test_loader = _make_loader(test_df, image_column=image_column, feature_columns=feature_columns, shuffle=False)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_pr_auc = -np.inf
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch_idx in range(1, max_epochs + 1):
        train_loss, train_true, train_score = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_true, val_score = _run_epoch(model, val_loader, device, optimizer=None)
        threshold = optimize_threshold(val_true, val_score)
        train_metrics = compute_binary_metrics(train_true, train_score, 0.5)
        val_metrics = compute_binary_metrics(val_true, val_score, threshold)
        history.append(
            {
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pr_auc": train_metrics.pr_auc,
                "val_pr_auc": val_metrics.pr_auc,
                "val_recall_blast": val_metrics.recall_blast,
            }
        )
        if val_metrics.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            best_threshold = threshold
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise ValueError("Deep model did not produce a valid checkpoint.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)

    _, test_true, test_score = _run_epoch(model, test_loader, device, optimizer=None)
    test_metrics = compute_binary_metrics(test_true, test_score, best_threshold)
    test_predictions = test_df.copy()
    test_predictions["y_true"] = test_true
    test_predictions["y_score"] = test_score
    test_predictions["y_pred"] = (test_score >= best_threshold).astype(int)

    return TrainingResult(
        model_path=str(checkpoint_path),
        history=history,
        threshold=best_threshold,
        test_predictions=test_predictions,
        test_metrics={
            "pr_auc": test_metrics.pr_auc,
            "roc_auc": test_metrics.roc_auc,
            "balanced_accuracy": test_metrics.balanced_accuracy,
            "macro_f1": test_metrics.macro_f1,
            "recall_blast": test_metrics.recall_blast,
            "specificity": test_metrics.specificity,
            "matthews_corrcoef": test_metrics.matthews_corrcoef,
            "brier_score": test_metrics.brier_score,
            "expected_calibration_error": test_metrics.expected_calibration_error,
        },
    )


def save_training_history(history_payload: dict[str, list[dict[str, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history_payload, ensure_ascii=False, indent=2), encoding="utf-8")

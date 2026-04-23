from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset
from clwl_experiments_module6_metrics import evaluate_scores_on_dataset


Array = np.ndarray


@dataclass
class CLPLTrainConfig:
    model_type: Literal["linear", "mlp"] = "linear"
    hidden_dim: int = 128
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    seed: int = 0
    log_every: int = 1


@dataclass
class CLPLEpochLog:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_clean_accuracy: float
    train_max_preservation_rate: float
    train_pairwise_order_rate: float
    val_clean_accuracy: Optional[float]
    val_max_preservation_rate: Optional[float]
    val_pairwise_order_rate: Optional[float]


@dataclass
class CLPLTrainResult:
    model: nn.Module
    config: CLPLTrainConfig
    logs: list[CLPLEpochLog]
    final_train_metrics: dict[str, Any]
    final_val_metrics: Optional[dict[str, Any]]


class CLPLTrainingError(ValueError):
    pass


class TorchPartialLabelDataset(Dataset):
    def __init__(self, dataset: WeakLabelDataset) -> None:
        if dataset.weak_label_vectors is None:
            raise CLPLTrainingError(
                "CLPL requires dataset.weak_label_vectors. Use a partial-label family from module 2."
            )
        self.X = torch.tensor(np.asarray(dataset.X, dtype=np.float32), dtype=torch.float32)
        self.b = torch.tensor(np.asarray(dataset.weak_label_vectors, dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.b[idx]


class LinearScoreModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPScoreModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _validate_partial_dataset(dataset: WeakLabelDataset) -> tuple[int, int]:
    X = np.asarray(dataset.X, dtype=np.float64)
    eta = np.asarray(dataset.eta, dtype=np.float64)
    y = np.asarray(dataset.y, dtype=np.intp)
    B = dataset.weak_label_vectors

    if X.ndim != 2:
        raise CLPLTrainingError(f"dataset.X must be 2D, got shape {X.shape}.")
    if eta.ndim != 2:
        raise CLPLTrainingError(f"dataset.eta must be 2D, got shape {eta.shape}.")
    if y.ndim != 1:
        raise CLPLTrainingError(f"dataset.y must be 1D, got shape {y.shape}.")
    if B is None:
        raise CLPLTrainingError("dataset.weak_label_vectors is required for CLPL.")

    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 2:
        raise CLPLTrainingError(f"dataset.weak_label_vectors must be 2D, got shape {B.shape}.")

    n, input_dim = X.shape
    n_eta, c = eta.shape
    if y.shape[0] != n or n_eta != n or B.shape != (n, c):
        raise CLPLTrainingError(
            f"Dataset arrays are inconsistent: X={X.shape}, y={y.shape}, eta={eta.shape}, B={B.shape}."
        )

    if np.min(B) < -1e-8 or np.max(B) > 1.0 + 1e-8:
        raise CLPLTrainingError("dataset.weak_label_vectors must lie in [0, 1].")
    if np.any(B.sum(axis=1) <= 0):
        raise CLPLTrainingError("Each weak-label vector must contain at least one candidate label.")

    return input_dim, c



def build_clpl_model(input_dim: int, num_classes: int, config: CLPLTrainConfig) -> nn.Module:
    if config.model_type == "linear":
        return LinearScoreModel(input_dim=input_dim, num_classes=num_classes)
    if config.model_type == "mlp":
        if config.hidden_dim <= 0:
            raise CLPLTrainingError(f"hidden_dim must be positive, got {config.hidden_dim}.")
        return MLPScoreModel(input_dim=input_dim, hidden_dim=config.hidden_dim, num_classes=num_classes)
    raise CLPLTrainingError(f"Unsupported model_type={config.model_type}.")



def clpl_torch_loss(scores: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise CLPLTrainingError(f"scores must be 2D, got shape {tuple(scores.shape)}.")
    if b.ndim != 2:
        raise CLPLTrainingError(f"b must be 2D, got shape {tuple(b.shape)}.")
    if scores.shape != b.shape:
        raise CLPLTrainingError(f"scores shape {tuple(scores.shape)} must match b shape {tuple(b.shape)}.")

    candidate_sizes = b.sum(dim=1)
    if torch.any(candidate_sizes <= 0):
        raise CLPLTrainingError("Each row of b must contain at least one candidate label.")

    mean_candidate_scores = (b * scores).sum(dim=1) / candidate_sizes
    pos_term = torch.nn.functional.softplus(-mean_candidate_scores)
    neg_term = ((1.0 - b) * torch.nn.functional.softplus(scores)).sum(dim=1)
    return (pos_term + neg_term).mean()



def make_dataloader(dataset: WeakLabelDataset, batch_size: int, *, shuffle: bool) -> DataLoader:
    if batch_size <= 0:
        raise CLPLTrainingError(f"batch_size must be positive, got {batch_size}.")
    torch_dataset = TorchPartialLabelDataset(dataset)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=shuffle)



def scores_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> Array:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[Array] = []
    device_obj = torch.device(device)

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device_obj)
            scores = model(x_batch).detach().cpu().numpy().astype(np.float64)
            outputs.append(scores)

    if not outputs:
        raise CLPLTrainingError("scores_from_model received an empty dataset.")
    return np.concatenate(outputs, axis=0)



def empirical_risk_from_model(
    model: nn.Module,
    dataset: WeakLabelDataset,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> float:
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_count = 0
    device_obj = torch.device(device)

    with torch.no_grad():
        for x_batch, b_batch in loader:
            x_batch = x_batch.to(device_obj)
            b_batch = b_batch.to(device_obj)
            loss = clpl_torch_loss(model(x_batch), b_batch)
            batch_n = int(x_batch.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n

    if total_count == 0:
        raise CLPLTrainingError("empirical_risk_from_model received an empty dataset.")
    return total_loss / total_count



def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: WeakLabelDataset,
    *,
    batch_size: int = 512,
    device: str = "cpu",
) -> dict[str, Any]:
    scores = scores_from_model(model, dataset, batch_size=batch_size, device=device)
    metrics = evaluate_scores_on_dataset(scores, dataset)
    empirical_risk = empirical_risk_from_model(model, dataset, batch_size=batch_size, device=device)
    return {
        "clean_accuracy": metrics.clean_accuracy,
        "max_preservation_rate": metrics.max_preservation_rate,
        "pairwise_order_rate": metrics.pairwise_order_rate,
        "pairwise_total": metrics.pairwise_total,
        "pairwise_correct": metrics.pairwise_correct,
        "mean_margin_on_ordered_pairs": metrics.mean_margin_on_ordered_pairs,
        "empirical_risk": empirical_risk,
    }



def train_clpl_model(
    train_dataset: WeakLabelDataset,
    *,
    val_dataset: Optional[WeakLabelDataset] = None,
    config: Optional[CLPLTrainConfig] = None,
) -> CLPLTrainResult:
    if config is None:
        config = CLPLTrainConfig()

    input_dim, num_classes = _validate_partial_dataset(train_dataset)
    if val_dataset is not None:
        _validate_partial_dataset(val_dataset)

    set_torch_seed(config.seed)
    device_obj = torch.device(config.device)
    model = build_clpl_model(input_dim=input_dim, num_classes=num_classes, config=config).to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)

    logs: list[CLPLEpochLog] = []

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_count = 0

        for x_batch, b_batch in train_loader:
            x_batch = x_batch.to(device_obj)
            b_batch = b_batch.to(device_obj)

            optimizer.zero_grad()
            scores = model(x_batch)
            loss = clpl_torch_loss(scores, b_batch)
            loss.backward()
            optimizer.step()

            batch_n = int(x_batch.shape[0])
            running_loss += float(loss.item()) * batch_n
            total_count += batch_n

        if total_count == 0:
            raise CLPLTrainingError("Training loader is empty.")
        train_loss = running_loss / total_count

        if epoch % config.log_every == 0 or epoch == config.num_epochs:
            train_metrics = evaluate_model_on_dataset(
                model,
                train_dataset,
                batch_size=max(config.batch_size, 512),
                device=config.device,
            )
            val_metrics = None
            if val_dataset is not None:
                val_metrics = evaluate_model_on_dataset(
                    model,
                    val_dataset,
                    batch_size=max(config.batch_size, 512),
                    device=config.device,
                )

            logs.append(
                CLPLEpochLog(
                    epoch=epoch,
                    train_loss=float(train_loss),
                    val_loss=None if val_metrics is None else float(val_metrics["empirical_risk"]),
                    train_clean_accuracy=float(train_metrics["clean_accuracy"]),
                    train_max_preservation_rate=float(train_metrics["max_preservation_rate"]),
                    train_pairwise_order_rate=float(train_metrics["pairwise_order_rate"]),
                    val_clean_accuracy=None if val_metrics is None else float(val_metrics["clean_accuracy"]),
                    val_max_preservation_rate=None if val_metrics is None else float(val_metrics["max_preservation_rate"]),
                    val_pairwise_order_rate=None if val_metrics is None else float(val_metrics["pairwise_order_rate"]),
                )
            )

    final_train_metrics = evaluate_model_on_dataset(
        model,
        train_dataset,
        batch_size=max(config.batch_size, 512),
        device=config.device,
    )
    final_val_metrics = None
    if val_dataset is not None:
        final_val_metrics = evaluate_model_on_dataset(
            model,
            val_dataset,
            batch_size=max(config.batch_size, 512),
            device=config.device,
        )

    return CLPLTrainResult(
        model=model,
        config=config,
        logs=logs,
        final_train_metrics=final_train_metrics,
        final_val_metrics=final_val_metrics,
    )


if __name__ == "__main__":
    from clwl_experiments_module2_weak_label_generators import make_uniform_partial_label_family
    from clwl_experiments_module3_synthetic_clean_data import generate_linear_softmax_dataset, train_val_test_split
    from clwl_experiments_module4_weak_label_dataset import build_weak_label_splits

    ds = generate_linear_softmax_dataset(
        n=1000,
        input_dim=8,
        num_classes=4,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
    )
    splits = train_val_test_split(ds, train_frac=0.6, val_frac=0.2, seed=42)
    family = make_uniform_partial_label_family(c=4, candidate_size=2)
    weak_splits = build_weak_label_splits(splits, family, seed=10)

    config = CLPLTrainConfig(
        model_type="linear",
        batch_size=128,
        num_epochs=10,
        learning_rate=1e-2,
        device="cpu",
        seed=0,
        log_every=1,
    )
    result = train_clpl_model(
        train_dataset=weak_splits["train"],
        val_dataset=weak_splits["val"],
        config=config,
    )

    print("=== Final train metrics ===")
    print(result.final_train_metrics)
    print("\n=== Final val metrics ===")
    print(result.final_val_metrics)
    print("\n=== Last epoch log ===")
    print(asdict(result.logs[-1]))

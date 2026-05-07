from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from clwl_experiments_module7_clwl_training import CLWLTrainConfig
from clwl_experiments_module17_forward_vs_clwl_imperfect_M import (
    make_confusion_map_noisy_M,
    make_structured_misspecified_M,
    run_clwl_vs_forward_imperfect_M,
    plot_clwl_vs_forward_imperfect_M,
    plot_gap_to_forward,
)


Array = np.ndarray


@dataclass
class RealCleanDataset:
    X: Array
    y: Array
    eta: Array
    logits: Array
    metadata: dict


def one_hot(y: Array, c: int) -> Array:
    y = np.asarray(y, dtype=np.intp)
    eta = np.zeros((len(y), c), dtype=np.float64)
    eta[np.arange(len(y)), y] = 1.0
    return eta


def _safe_logits_from_one_hot(eta: Array, eps: float = 1e-12) -> Array:
    return np.log(np.clip(np.asarray(eta, dtype=np.float64), eps, 1.0))


def _summarize_split(name: str, y: Array, c: int) -> None:
    counts = np.bincount(np.asarray(y, dtype=np.intp), minlength=c)
    print(f"[{name}] n={len(y)}, class_counts={counts.tolist()}")


def _extract_mnist_subset(
    ds: MNIST,
    *,
    classes: Sequence[int],
    max_samples: Optional[int],
    seed: int,
) -> tuple[Array, Array]:
    class_to_new = {int(old): int(new) for new, old in enumerate(classes)}
    X_list: list[Array] = []
    y_list: list[int] = []

    for img, label in ds:
        label = int(label)
        if label in class_to_new:
            x = img.numpy().reshape(-1).astype(np.float64)
            X_list.append(x)
            y_list.append(class_to_new[label])

    if not X_list:
        raise RuntimeError(f"No MNIST samples found for classes={classes}.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.intp)

    if max_samples is not None and max_samples < len(y):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y


def _standardize_splits(
    X_train: Array,
    X_val: Array,
    X_test: Array,
    *,
    eps: float = 1e-8,
) -> tuple[Array, Array, Array]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def load_mnist_subset_clean_splits(
    *,
    root: str = "./data",
    classes: Sequence[int] = (0, 1, 2, 3),
    val_frac: float = 0.2,
    seed: int = 42,
    download: bool = True,
    standardize: bool = True,
    max_trainval_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> dict[str, RealCleanDataset]:
    """
    Load a real MNIST subset and convert it to the clean-dataset interface used by module17.

    Important:
    - MNIST only gives clean labels y, not the true posterior eta(x).
    - We therefore set eta(x)=one_hot(y), so real-data evaluation should emphasize clean accuracy.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must lie in (0,1), got {val_frac}.")
    if len(classes) < 2:
        raise ValueError("Need at least two classes.")

    classes = tuple(int(k) for k in classes)
    c = len(classes)

    transform = transforms.ToTensor()
    train_raw = MNIST(root=root, train=True, download=download, transform=transform)
    test_raw = MNIST(root=root, train=False, download=download, transform=transform)

    X_trainval, y_trainval = _extract_mnist_subset(
        train_raw,
        classes=classes,
        max_samples=max_trainval_samples,
        seed=seed,
    )
    X_test, y_test = _extract_mnist_subset(
        test_raw,
        classes=classes,
        max_samples=max_test_samples,
        seed=seed + 1,
    )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y_trainval))
    n_val = int(round(val_frac * len(idx)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X_trainval[train_idx]
    y_train = y_trainval[train_idx]
    X_val = X_trainval[val_idx]
    y_val = y_trainval[val_idx]

    if standardize:
        X_train, X_val, X_test = _standardize_splits(X_train, X_val, X_test)

    def make_ds(X: Array, y: Array, split: str) -> RealCleanDataset:
        eta = one_hot(y, c)
        logits = _safe_logits_from_one_hot(eta)
        return RealCleanDataset(
            X=np.asarray(X, dtype=np.float64),
            y=np.asarray(y, dtype=np.intp),
            eta=eta,
            logits=logits,
            metadata={
                "dataset_name": "MNIST",
                "classes_original": list(classes),
                "num_classes": c,
                "split": split,
                "eta_source": "one_hot_clean_label",
                "oracle_metrics": False,
                "input_kind": "flattened_image_28x28",
                "standardize": bool(standardize),
            },
        )

    splits = {
        "train": make_ds(X_train, y_train, "train"),
        "val": make_ds(X_val, y_val, "val"),
        "test": make_ds(X_test, y_test, "test"),
    }

    print("=== MNIST subset summary ===")
    print(f"classes_original={list(classes)} mapped_to={list(range(c))}")
    _summarize_split("train", y_train, c)
    _summarize_split("val", y_val, c)
    _summarize_split("test", y_test, c)

    return splits


def build_default_mnist4_transition_matrices(
    *,
    total_noise: float = 0.50,
    special_noise: float = 0.42,
) -> tuple[Array, Array]:
    """
    Default 4-class structured-noise setting.

    True confusion:
        0 <-> 1, 2 <-> 3
    Wrong confusion:
        0 -> 2, 1 -> 3, 2 -> 0, 3 -> 1
    """
    true_confusion_map = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
    }
    wrong_confusion_map = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
    }

    M_true = make_confusion_map_noisy_M(
        c=4,
        confusion_map=true_confusion_map,
        total_noise=total_noise,
        special_noise=special_noise,
    )
    M_wrong = make_confusion_map_noisy_M(
        c=4,
        confusion_map=wrong_confusion_map,
        total_noise=total_noise,
        special_noise=special_noise,
    )
    return M_true, M_wrong


def print_transition_diagnostics(M_true: Array, M_wrong: Array) -> None:
    print("\n=== M_true ===")
    print(M_true)
    print("\n=== M_wrong ===")
    print(M_wrong)

    for gamma in [0.0, 0.2, 0.5, 0.8, 1.0]:
        M_est = make_structured_misspecified_M(M_true, M_wrong, gamma)
        print(f"\n=== gamma={gamma:.2f} ===")
        print(M_est)
        print("fro_error =", np.linalg.norm(M_est - M_true, ord="fro"))


def main() -> None:
    out_dir = "artifacts_mnist4_forward_vs_clwl_structured_imperfect_M"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    clean_splits = load_mnist_subset_clean_splits(
        root="./data",
        classes=(0, 1, 2, 3),
        val_frac=0.2,
        seed=42,
        download=True,
        standardize=True,
        max_trainval_samples=None,
        max_test_samples=None,
    )

    M_true, M_wrong = build_default_mnist4_transition_matrices(
        total_noise=0.50,
        special_noise=0.42,
    )
    print_transition_diagnostics(M_true, M_wrong)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device={device}")

    common_cfg = CLWLTrainConfig(
        model_type="mlp",
        hidden_dim=256,
        batch_size=256,
        num_epochs=80,
        learning_rate=3e-4,
        weight_decay=1e-4,
        device=device,
        seed=0,
        log_every=1,
        gradient_clip_norm=5.0,
        scheduler_factor=0.5,
        scheduler_patience=5,
        min_learning_rate=1e-5,
        early_stop_patience=12,
        early_stop_min_delta=1e-8,
    )

    raw_df, summary_df = run_clwl_vs_forward_imperfect_M(
        clean_splits=clean_splits,
        M_true=M_true,
        M_wrong=M_wrong,
        mismatch_grid=[0.0, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
        seeds=[0, 1, 2, 3, 4],
        clwl_config=common_cfg,
        forward_config=common_cfg,
        out_dir=out_dir,
        family_name="mnist4_confusion_aware_noisy_true_M",
    )

    print("\n=== Raw results head ===")
    print(raw_df.head())

    print("\n=== Summary results ===")
    print(summary_df)

    raw_df.to_csv(Path(out_dir) / "raw_results.csv", index=False)
    summary_df.to_csv(Path(out_dir) / "summary_results.csv", index=False)

    plot_clwl_vs_forward_imperfect_M(
        summary_df,
        out_path=str(Path(out_dir) / "clwl_vs_forward_mnist4_structured_imperfect_M.png"),
    )
    plot_gap_to_forward(
        summary_df,
        out_path=str(Path(out_dir) / "gap_to_forward_mnist4.png"),
    )

    print("\nSaved outputs to:")
    print(Path(out_dir).resolve())


if __name__ == "__main__":
    main()

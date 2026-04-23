from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional

import numpy as np


Array = np.ndarray


@dataclass
class SyntheticDataset:
    X: Array
    y: Array
    eta: Array
    logits: Array
    teacher_name: str
    metadata: dict[str, Any]


class SyntheticDataError(ValueError):
    pass


def set_seed(seed: int) -> None:
    np.random.seed(seed)



def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, Integral) or int(value) <= 0:
        raise SyntheticDataError(f"{name} must be a positive integer, got {value}.")



def _as_2d_float_array(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise SyntheticDataError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise SyntheticDataError(f"{name} must contain only finite values.")
    return arr



def _as_1d_float_array(name: str, x: Array, expected_len: Optional[int] = None) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise SyntheticDataError(f"{name} must be 1D, got shape {arr.shape}.")
    if expected_len is not None and arr.shape[0] != expected_len:
        raise SyntheticDataError(f"{name} must have length {expected_len}, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise SyntheticDataError(f"{name} must contain only finite values.")
    return arr



def softmax(logits: Array) -> Array:
    logits = _as_2d_float_array("logits", logits)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    return probs



def sample_labels_from_eta(eta: Array, *, seed: int = 0) -> Array:
    eta = _as_2d_float_array("eta", eta)
    n, c = eta.shape
    if c < 2:
        raise SyntheticDataError(f"eta must have at least 2 classes, got shape {eta.shape}.")
    if np.min(eta) < -1e-12:
        raise SyntheticDataError("eta contains negative entries.")
    row_sums = eta.sum(axis=1)
    if not np.allclose(row_sums, np.ones(n), atol=1e-8, rtol=0.0):
        raise SyntheticDataError("Each row of eta must sum to 1.")
    eta = np.clip(eta, 0.0, None)
    eta = eta / eta.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    classes = np.arange(c, dtype=np.intp)
    y = np.empty(n, dtype=np.intp)
    for i in range(n):
        y[i] = rng.choice(classes, p=eta[i])
    return y



def generate_gaussian_features(
    n: int,
    d: int,
    *,
    mean: Optional[Array] = None,
    cov: Optional[Array] = None,
    seed: int = 0,
) -> Array:
    _check_positive_int("n", n)
    _check_positive_int("d", d)
    n = int(n)
    d = int(d)

    rng = np.random.default_rng(seed)
    if mean is None:
        mean = np.zeros(d, dtype=np.float64)
    else:
        mean = _as_1d_float_array("mean", mean, expected_len=d)

    if cov is None:
        cov = np.eye(d, dtype=np.float64)
    else:
        cov = _as_2d_float_array("cov", cov)
        if cov.shape != (d, d):
            raise SyntheticDataError(f"cov must have shape {(d, d)}, got {cov.shape}.")
        if not np.allclose(cov, cov.T, atol=1e-8, rtol=0.0):
            raise SyntheticDataError("cov must be symmetric.")

    return rng.multivariate_normal(mean=mean, cov=cov, size=n)



def build_linear_teacher(
    input_dim: int,
    num_classes: int,
    *,
    seed: int = 0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.5,
) -> tuple[Array, Array]:
    _check_positive_int("input_dim", input_dim)
    _check_positive_int("num_classes", num_classes)
    input_dim = int(input_dim)
    num_classes = int(num_classes)
    if weight_scale <= 0 or bias_scale < 0:
        raise SyntheticDataError(
            f"Need weight_scale > 0 and bias_scale >= 0, got {weight_scale}, {bias_scale}."
        )

    rng = np.random.default_rng(seed)
    W = weight_scale * rng.standard_normal((input_dim, num_classes))
    b = bias_scale * rng.standard_normal(num_classes)
    return W.astype(np.float64), b.astype(np.float64)



def linear_teacher_logits(X: Array, W: Array, b: Array) -> Array:
    X = _as_2d_float_array("X", X)
    W = _as_2d_float_array("W", W)
    b = _as_1d_float_array("b", b)

    _, d = X.shape
    if W.shape[0] != d:
        raise SyntheticDataError(
            f"W shape {W.shape} is incompatible with X shape {X.shape}."
        )
    if b.shape != (W.shape[1],):
        raise SyntheticDataError(f"b must have shape {(W.shape[1],)}, got {b.shape}.")

    return X @ W + b



def build_mlp_teacher(
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    *,
    seed: int = 0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.2,
) -> dict[str, Array]:
    _check_positive_int("input_dim", input_dim)
    _check_positive_int("hidden_dim", hidden_dim)
    _check_positive_int("num_classes", num_classes)
    input_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    num_classes = int(num_classes)
    if weight_scale <= 0 or bias_scale < 0:
        raise SyntheticDataError(
            f"Need weight_scale > 0 and bias_scale >= 0, got {weight_scale}, {bias_scale}."
        )

    rng = np.random.default_rng(seed)
    return {
        "W1": (weight_scale * rng.standard_normal((input_dim, hidden_dim))).astype(np.float64),
        "b1": (bias_scale * rng.standard_normal(hidden_dim)).astype(np.float64),
        "W2": (weight_scale * rng.standard_normal((hidden_dim, num_classes))).astype(np.float64),
        "b2": (bias_scale * rng.standard_normal(num_classes)).astype(np.float64),
    }



def mlp_teacher_logits(X: Array, params: dict[str, Array]) -> Array:
    X = _as_2d_float_array("X", X)
    try:
        W1 = _as_2d_float_array("W1", params["W1"])
        b1 = _as_1d_float_array("b1", params["b1"])
        W2 = _as_2d_float_array("W2", params["W2"])
        b2 = _as_1d_float_array("b2", params["b2"])
    except KeyError as e:
        raise SyntheticDataError(f"Missing teacher parameter: {e.args[0]}") from e

    if W1.shape[0] != X.shape[1]:
        raise SyntheticDataError(
            f"W1 shape {W1.shape} is incompatible with X shape {X.shape}."
        )
    if b1.shape != (W1.shape[1],):
        raise SyntheticDataError(f"b1 must have shape {(W1.shape[1],)}, got {b1.shape}.")
    if W2.shape[0] != W1.shape[1]:
        raise SyntheticDataError("W2 input dimension must match hidden dimension.")
    if b2.shape != (W2.shape[1],):
        raise SyntheticDataError(f"b2 must have shape {(W2.shape[1],)}, got {b2.shape}.")

    hidden = np.tanh(X @ W1 + b1)
    return hidden @ W2 + b2



def generate_linear_softmax_dataset(
    n: int,
    input_dim: int,
    num_classes: int,
    *,
    feature_seed: int = 0,
    teacher_seed: int = 1,
    label_seed: int = 2,
    feature_cov_scale: float = 1.0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.5,
    logit_temperature: float = 1.0,
) -> SyntheticDataset:
    _check_positive_int("n", n)
    _check_positive_int("input_dim", input_dim)
    _check_positive_int("num_classes", num_classes)
    n = int(n)
    input_dim = int(input_dim)
    num_classes = int(num_classes)
    if num_classes < 2:
        raise SyntheticDataError(f"num_classes must be at least 2, got {num_classes}.")
    if feature_cov_scale <= 0:
        raise SyntheticDataError(f"feature_cov_scale must be > 0, got {feature_cov_scale}.")
    if logit_temperature <= 0:
        raise SyntheticDataError(f"logit_temperature must be > 0, got {logit_temperature}.")

    cov = feature_cov_scale * np.eye(input_dim, dtype=np.float64)
    X = generate_gaussian_features(n=n, d=input_dim, cov=cov, seed=feature_seed)
    W, b = build_linear_teacher(
        input_dim=input_dim,
        num_classes=num_classes,
        seed=teacher_seed,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
    )
    logits = linear_teacher_logits(X, W, b) / logit_temperature
    eta = softmax(logits)
    y = sample_labels_from_eta(eta, seed=label_seed)

    return SyntheticDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        teacher_name="linear_softmax",
        metadata={
            "n": n,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "feature_seed": feature_seed,
            "teacher_seed": teacher_seed,
            "label_seed": label_seed,
            "feature_cov_scale": feature_cov_scale,
            "weight_scale": weight_scale,
            "bias_scale": bias_scale,
            "logit_temperature": logit_temperature,
        },
    )



def generate_mlp_softmax_dataset(
    n: int,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    *,
    feature_seed: int = 0,
    teacher_seed: int = 1,
    label_seed: int = 2,
    feature_cov_scale: float = 1.0,
    weight_scale: float = 1.0,
    bias_scale: float = 0.2,
    logit_temperature: float = 1.0,
) -> SyntheticDataset:
    _check_positive_int("n", n)
    _check_positive_int("input_dim", input_dim)
    _check_positive_int("hidden_dim", hidden_dim)
    _check_positive_int("num_classes", num_classes)
    n = int(n)
    input_dim = int(input_dim)
    hidden_dim = int(hidden_dim)
    num_classes = int(num_classes)
    if num_classes < 2:
        raise SyntheticDataError(f"num_classes must be at least 2, got {num_classes}.")
    if feature_cov_scale <= 0:
        raise SyntheticDataError(f"feature_cov_scale must be > 0, got {feature_cov_scale}.")
    if logit_temperature <= 0:
        raise SyntheticDataError(f"logit_temperature must be > 0, got {logit_temperature}.")

    cov = feature_cov_scale * np.eye(input_dim, dtype=np.float64)
    X = generate_gaussian_features(n=n, d=input_dim, cov=cov, seed=feature_seed)
    params = build_mlp_teacher(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        seed=teacher_seed,
        weight_scale=weight_scale,
        bias_scale=bias_scale,
    )
    logits = mlp_teacher_logits(X, params) / logit_temperature
    eta = softmax(logits)
    y = sample_labels_from_eta(eta, seed=label_seed)

    metadata = {
        "n": n,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "feature_seed": feature_seed,
        "teacher_seed": teacher_seed,
        "label_seed": label_seed,
        "feature_cov_scale": feature_cov_scale,
        "weight_scale": weight_scale,
        "bias_scale": bias_scale,
        "logit_temperature": logit_temperature,
        "teacher_params": params,
    }

    return SyntheticDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        teacher_name="mlp_softmax",
        metadata=metadata,
    )



def train_val_test_split(
    dataset: SyntheticDataset,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, SyntheticDataset]:
    if not (0 < train_frac < 1):
        raise SyntheticDataError(f"train_frac must be in (0, 1), got {train_frac}.")
    if not (0 <= val_frac < 1):
        raise SyntheticDataError(f"val_frac must be in [0, 1), got {val_frac}.")
    if train_frac + val_frac >= 1:
        raise SyntheticDataError(
            f"Need train_frac + val_frac < 1, got {train_frac + val_frac}."
        )

    n = dataset.X.shape[0]
    if n < 3:
        raise SyntheticDataError(f"Need at least 3 samples to split train/val/test, got {n}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise SyntheticDataError(
            f"Split sizes must all be positive, got train={n_train}, val={n_val}, test={n_test}."
        )

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    def _slice(idxs: Array) -> SyntheticDataset:
        return SyntheticDataset(
            X=dataset.X[idxs].copy(),
            y=np.asarray(dataset.y[idxs], dtype=np.intp).copy(),
            eta=dataset.eta[idxs].copy(),
            logits=dataset.logits[idxs].copy(),
            teacher_name=dataset.teacher_name,
            metadata={**dataset.metadata, "subset_size": int(len(idxs))},
        )

    return {"train": _slice(idx_train), "val": _slice(idx_val), "test": _slice(idx_test)}



def dataset_summary(dataset: SyntheticDataset) -> dict[str, object]:
    X = _as_2d_float_array("dataset.X", dataset.X)
    y = np.asarray(dataset.y, dtype=np.intp)
    eta = _as_2d_float_array("dataset.eta", dataset.eta)
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise SyntheticDataError(
            f"dataset.y must have shape ({X.shape[0]},), got {y.shape}."
        )
    if eta.shape[0] != X.shape[0]:
        raise SyntheticDataError(
            f"dataset.eta must have {X.shape[0]} rows, got {eta.shape[0]}."
        )

    n, d = X.shape
    c = int(eta.shape[1])
    if np.min(y) < 0 or np.max(y) >= c:
        raise SyntheticDataError(f"dataset.y entries must lie in [0, {c - 1}].")

    label_hist = np.bincount(y, minlength=c)
    max_probs = np.max(eta, axis=1)

    return {
        "teacher_name": dataset.teacher_name,
        "n": int(n),
        "input_dim": int(d),
        "num_classes": int(c),
        "label_hist": label_hist.tolist(),
        "mean_max_eta": float(np.mean(max_probs)),
        "min_max_eta": float(np.min(max_probs)),
        "max_max_eta": float(np.max(max_probs)),
    }


if __name__ == "__main__":
    print("=== Linear softmax synthetic dataset ===")
    ds_linear = generate_linear_softmax_dataset(
        n=2000,
        input_dim=8,
        num_classes=4,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
        logit_temperature=1.0,
    )
    print(dataset_summary(ds_linear))

    splits_linear = train_val_test_split(ds_linear, train_frac=0.6, val_frac=0.2, seed=42)
    print({k: dataset_summary(v)["n"] for k, v in splits_linear.items()})

    print("\n=== MLP softmax synthetic dataset ===")
    ds_mlp = generate_mlp_softmax_dataset(
        n=2000,
        input_dim=8,
        hidden_dim=16,
        num_classes=4,
        feature_seed=3,
        teacher_seed=4,
        label_seed=5,
        logit_temperature=1.0,
    )
    print(dataset_summary(ds_mlp))

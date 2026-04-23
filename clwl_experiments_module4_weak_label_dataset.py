from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from clwl_experiments_module2_weak_label_generators import (
    WeakLabelFamily,
    sample_weak_labels_from_M,
)
from clwl_experiments_module3_synthetic_clean_data import SyntheticDataset


Array = np.ndarray


@dataclass
class WeakLabelDataset:
    X: Array
    y: Array
    eta: Array
    logits: Array
    z: Array
    M: Array
    family_name: str
    weak_label_matrix: Optional[Array]
    weak_label_vectors: Optional[Array]
    weak_label_names: Optional[list[str]]
    metadata: dict[str, Any]


class WeakLabelDatasetError(ValueError):
    pass



def _as_float_2d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise WeakLabelDatasetError(f"{name} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise WeakLabelDatasetError(f"{name} must contain only finite values.")
    return arr



def _as_index_1d(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=np.intp)
    if arr.ndim != 1:
        raise WeakLabelDatasetError(f"{name} must be 1D, got shape {arr.shape}.")
    return arr



def _validate_clean_dataset(dataset: SyntheticDataset) -> tuple[int, int, int]:
    X = _as_float_2d("dataset.X", dataset.X)
    eta = _as_float_2d("dataset.eta", dataset.eta)
    logits = _as_float_2d("dataset.logits", dataset.logits)
    y = _as_index_1d("dataset.y", dataset.y)

    n, d = X.shape
    if y.shape[0] != n:
        raise WeakLabelDatasetError(
            f"dataset.y must have length {n}, got {y.shape[0]}."
        )
    if eta.shape[0] != n:
        raise WeakLabelDatasetError(
            f"dataset.eta must have {n} rows, got {eta.shape[0]}."
        )
    if logits.shape != eta.shape:
        raise WeakLabelDatasetError(
            f"dataset.logits shape {logits.shape} must match dataset.eta shape {eta.shape}."
        )

    c = eta.shape[1]
    if np.min(y) < 0 or np.max(y) >= c:
        raise WeakLabelDatasetError(
            f"dataset.y entries must lie in [0, {c - 1}]."
        )

    row_sums = eta.sum(axis=1)
    if not np.allclose(row_sums, np.ones(n), atol=1e-8, rtol=0.0):
        raise WeakLabelDatasetError("Each row of dataset.eta must sum to 1.")
    if np.min(eta) < -1e-12:
        raise WeakLabelDatasetError("dataset.eta contains negative entries.")

    return n, d, c



def _validate_family(family: WeakLabelFamily, c: int) -> int:
    M = _as_float_2d("family.M", family.M)
    d, c_family = M.shape
    if c_family != c:
        raise WeakLabelDatasetError(
            f"family.M has {c_family} clean classes, but dataset has {c}."
        )
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, np.ones(c), atol=1e-8, rtol=0.0):
        raise WeakLabelDatasetError(
            f"family.M must be column-stochastic. Column sums are {col_sums}."
        )
    if np.min(M) < -1e-12:
        raise WeakLabelDatasetError("family.M contains negative entries.")

    if family.weak_label_matrix is not None:
        Z = _as_float_2d("family.weak_label_matrix", family.weak_label_matrix)
        if Z.shape != (c, d):
            raise WeakLabelDatasetError(
                f"family.weak_label_matrix must have shape {(c, d)}, got {Z.shape}."
            )

    if family.weak_label_names is not None and len(family.weak_label_names) != d:
        raise WeakLabelDatasetError(
            f"family.weak_label_names must have length {d}, got {len(family.weak_label_names)}."
        )

    return d



def build_weak_label_dataset(
    dataset: SyntheticDataset,
    family: WeakLabelFamily,
    *,
    seed: int = 0,
) -> WeakLabelDataset:
    n, _, c = _validate_clean_dataset(dataset)
    d = _validate_family(family, c)

    X = np.asarray(dataset.X, dtype=np.float64).copy()
    y = np.asarray(dataset.y, dtype=np.intp).copy()
    eta = np.asarray(dataset.eta, dtype=np.float64).copy()
    logits = np.asarray(dataset.logits, dtype=np.float64).copy()
    M = np.asarray(family.M, dtype=np.float64).copy()

    z = np.asarray(sample_weak_labels_from_M(y, M, seed=seed), dtype=np.intp)
    if z.shape != (n,):
        raise WeakLabelDatasetError(f"Sampled weak labels must have shape {(n,)}, got {z.shape}.")
    if np.min(z) < 0 or np.max(z) >= d:
        raise WeakLabelDatasetError(f"Sampled weak labels must lie in [0, {d - 1}].")

    weak_label_matrix = None
    weak_label_vectors = None
    if family.weak_label_matrix is not None:
        weak_label_matrix = np.asarray(family.weak_label_matrix, dtype=np.float64).copy()
        weak_label_vectors = weak_label_matrix[:, z].T.copy()
        if weak_label_vectors.shape != (n, c):
            raise WeakLabelDatasetError(
                f"weak_label_vectors must have shape {(n, c)}, got {weak_label_vectors.shape}."
            )

    metadata = {
        "teacher_name": dataset.teacher_name,
        "family_name": family.name,
        "seed": seed,
        "num_samples": n,
        "num_classes": c,
        "num_weak_labels": d,
        "eta_source": dict(dataset.metadata).get("eta_source", "oracle"),
        "oracle_metrics": bool(dict(dataset.metadata).get("oracle_metrics", True)),
        "dataset_name": dict(dataset.metadata).get("dataset_name", "synthetic"),
        "input_kind": dict(dataset.metadata).get("input_kind", "vector"),
        "dataset_metadata": dict(dataset.metadata),
        "family_metadata": dict(family.metadata or {}),
    }
    

    return WeakLabelDataset(
        X=X,
        y=y,
        eta=eta,
        logits=logits,
        z=z,
        M=M,
        family_name=family.name,
        weak_label_matrix=weak_label_matrix,
        weak_label_vectors=weak_label_vectors,
        weak_label_names=list(family.weak_label_names) if family.weak_label_names is not None else None,
        metadata=metadata,
    )



def build_weak_label_splits(
    splits: dict[str, SyntheticDataset],
    family: WeakLabelFamily,
    *,
    seed: int = 0,
    seed_stride: int = 1000,
) -> dict[str, WeakLabelDataset]:
    out: dict[str, WeakLabelDataset] = {}
    for i, split_name in enumerate(sorted(splits.keys())):
        out[split_name] = build_weak_label_dataset(
            splits[split_name],
            family,
            seed=seed + i * seed_stride,
        )
    return out



def weak_dataset_summary(dataset: WeakLabelDataset) -> dict[str, object]:
    X = _as_float_2d("dataset.X", dataset.X)
    y = _as_index_1d("dataset.y", dataset.y)
    eta = _as_float_2d("dataset.eta", dataset.eta)
    z = _as_index_1d("dataset.z", dataset.z)
    M = _as_float_2d("dataset.M", dataset.M)

    n, d_x = X.shape
    d, c = M.shape
    if y.shape[0] != n or eta.shape[0] != n or z.shape[0] != n:
        raise WeakLabelDatasetError("X, y, eta, and z must agree on sample size.")
    if eta.shape[1] != c:
        raise WeakLabelDatasetError(
            f"eta has {eta.shape[1]} classes but M has {c}."
        )

    clean_hist = np.bincount(y, minlength=c)
    weak_hist = np.bincount(z, minlength=d)

    summary = {
        "family_name": dataset.family_name,
        "num_samples": int(n),
        "input_dim": int(d_x),
        "num_classes": int(c),
        "num_weak_labels": int(d),
        "clean_hist": clean_hist.tolist(),
        "weak_hist": weak_hist.tolist(),
        "has_weak_vectors": dataset.weak_label_vectors is not None,
    }

    if dataset.weak_label_vectors is not None:
        B = _as_float_2d("dataset.weak_label_vectors", dataset.weak_label_vectors)
        summary["weak_vector_density"] = float(np.mean(B > 0))
        summary["weak_vector_mean_size"] = float(np.mean(B.sum(axis=1)))

    return summary



def subset_weak_label_dataset(dataset: WeakLabelDataset, indices: Array) -> WeakLabelDataset:
    idx = _as_index_1d("indices", indices)
    n = dataset.X.shape[0]
    if np.min(idx) < 0 or np.max(idx) >= n:
        raise WeakLabelDatasetError(f"indices must lie in [0, {n - 1}].")

    weak_label_vectors = None
    if dataset.weak_label_vectors is not None:
        weak_label_vectors = np.asarray(dataset.weak_label_vectors[idx], dtype=np.float64).copy()

    return WeakLabelDataset(
        X=np.asarray(dataset.X[idx], dtype=np.float64).copy(),
        y=np.asarray(dataset.y[idx], dtype=np.intp).copy(),
        eta=np.asarray(dataset.eta[idx], dtype=np.float64).copy(),
        logits=np.asarray(dataset.logits[idx], dtype=np.float64).copy(),
        z=np.asarray(dataset.z[idx], dtype=np.intp).copy(),
        M=np.asarray(dataset.M, dtype=np.float64).copy(),
        family_name=dataset.family_name,
        weak_label_matrix=None if dataset.weak_label_matrix is None else np.asarray(dataset.weak_label_matrix, dtype=np.float64).copy(),
        weak_label_vectors=weak_label_vectors,
        weak_label_names=None if dataset.weak_label_names is None else list(dataset.weak_label_names),
        metadata=dict(dataset.metadata),
    )


if __name__ == "__main__":
    from clwl_experiments_module2_weak_label_generators import (
        make_uniform_complementary_family,
        make_uniform_partial_label_family,
    )
    from clwl_experiments_module3_synthetic_clean_data import (
        generate_linear_softmax_dataset,
        train_val_test_split,
    )

    ds = generate_linear_softmax_dataset(
        n=1000,
        input_dim=8,
        num_classes=4,
        feature_seed=0,
        teacher_seed=1,
        label_seed=2,
    )
    splits = train_val_test_split(ds, train_frac=0.6, val_frac=0.2, seed=42)

    print("=== Weak dataset: partial family ===")
    partial_family = make_uniform_partial_label_family(c=4, candidate_size=2)
    weak_partial = build_weak_label_splits(splits, partial_family, seed=10)
    for k, v in weak_partial.items():
        print(k, weak_dataset_summary(v))

    print("\n=== Weak dataset: complementary family ===")
    comp_family = make_uniform_complementary_family(c=4)
    weak_comp = build_weak_label_splits(splits, comp_family, seed=20)
    for k, v in weak_comp.items():
        print(k, weak_dataset_summary(v))

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from clwl_experiments_module1_t_construction import construct_clwl_T
from clwl_experiments_module2_weak_label_generators import (
    WeakLabelFamily,
    make_biased_partial_label_family,
    make_nonuniform_complementary_family,
    make_random_general_weak_label_family,
    make_uniform_complementary_family,
    make_uniform_partial_label_family,
    make_manual_mnist_quad_partial_label_family,  # stronger CLPL-hard size-4 family
)
from clwl_experiments_module3_synthetic_clean_data import (
    SyntheticDataset,
    generate_linear_softmax_dataset,
    generate_mlp_softmax_dataset,
    train_val_test_split,
)
from clwl_experiments_module3_real_mnist_data import (
    MNISTRealDataConfig,
    build_mnist_real_splits,
)
from clwl_experiments_module4_weak_label_dataset import WeakLabelDataset, build_weak_label_splits
from clwl_experiments_module6_metrics import (
    evaluate_scores_on_dataset,
    scores_from_logits,
    zero_scores_like_dataset,
)
from clwl_experiments_module7_clwl_training import (
    CLWLTrainConfig,
    evaluate_model_on_dataset as eval_clwl_model,
    train_clwl_model,
)
from clwl_experiments_module8_clpl_training import (
    CLPLTrainConfig,
    evaluate_model_on_dataset as eval_clpl_model,
    train_clpl_model,
)
from clwl_experiments_module9_clcl_training import (
    CLCLTrainConfig,
    evaluate_model_on_dataset as eval_clcl_model,
    train_clcl_model,
)




FORMAL_GROUPS = [
    "g1_clwl_transition_sensitivity",
    "g2_clpl_vs_clwl_order_preserving_dominance",
    "g3_clpl_vs_clwl_order_preserving",
    "g4_clpl_vs_clwl_arbitrary_transition",
    "g5_clcl_vs_clwl_order_preserving",
    "g6_clcl_vs_clwl_non_complementary",
]


TRANSITION_REGIMES = [
    "uniform_dominance_friendly",
    "biased_non_dominance",
    "arbitrary_transition",
    "complementary_uniform",
    "non_complementary_noisy",
]


@dataclass
class SyntheticDataConfig:
    # source selector
    source: Literal["synthetic", "mnist"] = "synthetic"

    # synthetic-only fields
    teacher_type: Literal["linear", "mlp"] = "linear"
    n: int = 4000
    input_dim: int = 8
    num_classes: int = 4
    hidden_dim: int = 32
    train_frac: float = 0.6
    val_frac: float = 0.2
    feature_cov_scale: float = 1.0
    weight_scale: float = 1.0
    bias_scale: float = 0.5
    logit_temperature: float = 1.0

    # real-data-only fields (used when source == "mnist")
    root: str = "data"
    download: bool = True
    real_teacher_hidden_dim: int = 256
    real_teacher_num_epochs: int = 12
    real_teacher_batch_size: int = 256
    real_teacher_learning_rate: float = 1e-3
    real_teacher_weight_decay: float = 1e-4
    real_teacher_device: str = "cpu"
    max_train_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class FormalComparisonConfig:
    suite_name: str
    seeds: list[int]
    data: SyntheticDataConfig
    clwl_config: CLWLTrainConfig = field(default_factory=CLWLTrainConfig)
    clpl_config: CLPLTrainConfig = field(default_factory=CLPLTrainConfig)
    clcl_config: CLCLTrainConfig = field(default_factory=CLCLTrainConfig)
    partial_candidate_size: int = 2
    biased_rho: float = 5.0
    arbitrary_d: int = 6
    arbitrary_fixed_matrix: Optional[np.ndarray] = None
    non_complementary_q: float = 0.1
    run_teacher_reference: bool = True
    run_zero_reference: bool = True
    output_dir: str = "formal_group_results"
    groups: Optional[list[str]] = None


@dataclass
class FormalComparisonResult:
    suite_name: str
    group_name: str
    regime_name: str
    seed: int
    method: str
    split: str
    clean_accuracy: Optional[float]
    max_preservation_rate: Optional[float]
    pairwise_order_rate: Optional[float]
    pairwise_total: Optional[int]
    pairwise_correct: Optional[int]
    mean_margin_on_ordered_pairs: Optional[float]
    empirical_risk: Optional[float]
    teacher_mean_pairwise_margin: Optional[float]
    conditional_risk: Optional[float]
    is_applicable: bool
    metric_available: bool
    is_order_preserving: Optional[bool]
    is_dominance_satisfied: Optional[bool]
    is_uniform_transition: Optional[bool]
    is_complementary_setting: Optional[bool]
    is_native_for_clpl: Optional[bool]
    is_native_for_clcl: Optional[bool]
    extra: dict[str, Any] = field(default_factory=dict)


class FormalComparisonRunnerError(ValueError):
    pass


def _default_cyclic_affinity(c: int, rho: float) -> np.ndarray:
    A = np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    for y in range(c):
        A[y, (y + 1) % c] = rho
    return A



def _default_arbitrary_matrix(c: int, d: int) -> np.ndarray:
    if c == 4 and d == 6:
        return np.array(
            [
                [0.42, 0.08, 0.11, 0.17],
                [0.18, 0.21, 0.09, 0.13],
                [0.11, 0.24, 0.18, 0.08],
                [0.09, 0.14, 0.27, 0.19],
                [0.12, 0.18, 0.16, 0.24],
                [0.08, 0.15, 0.19, 0.19],
            ],
            dtype=np.float64,
        )
    raise FormalComparisonRunnerError(
        f"No built-in explicit arbitrary matrix for shape (d={d}, c={c}). Provide arbitrary_fixed_matrix or use random general family."
    )



def _validate_column_stochastic(M: np.ndarray, atol: float = 1e-8) -> None:
    if M.ndim != 2:
        raise FormalComparisonRunnerError(f"M must be 2D, got shape {M.shape}.")
    if np.min(M) < -atol:
        raise FormalComparisonRunnerError(f"M has negative entries: min={np.min(M):.3e}")
    if not np.allclose(M.sum(axis=0), np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise FormalComparisonRunnerError("M must be column-stochastic.")



def _build_dataset_splits(cfg: SyntheticDataConfig, seed: int) -> dict[str, SyntheticDataset]:
    if cfg.source == "synthetic":
        if cfg.teacher_type == "linear":
            ds = generate_linear_softmax_dataset(
                n=cfg.n,
                input_dim=cfg.input_dim,
                num_classes=cfg.num_classes,
                feature_seed=10 * seed + 1,
                teacher_seed=10 * seed + 2,
                label_seed=10 * seed + 3,
                feature_cov_scale=cfg.feature_cov_scale,
                weight_scale=cfg.weight_scale,
                bias_scale=cfg.bias_scale,
                logit_temperature=cfg.logit_temperature,
            )
        elif cfg.teacher_type == "mlp":
            ds = generate_mlp_softmax_dataset(
                n=cfg.n,
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                num_classes=cfg.num_classes,
                feature_seed=10 * seed + 1,
                teacher_seed=10 * seed + 2,
                label_seed=10 * seed + 3,
                feature_cov_scale=cfg.feature_cov_scale,
                weight_scale=cfg.weight_scale,
                bias_scale=min(cfg.bias_scale, 0.2),
                logit_temperature=cfg.logit_temperature,
            )
        else:
            raise FormalComparisonRunnerError(f"Unsupported teacher_type={cfg.teacher_type}.")

        return train_val_test_split(
            ds,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
            seed=10 * seed + 4,
        )

    if cfg.source == "mnist":
        if cfg.num_classes != 10:
            raise FormalComparisonRunnerError(
                f"MNIST requires num_classes=10, got {cfg.num_classes}."
            )

        real_cfg = MNISTRealDataConfig(
            root=cfg.root,
            val_frac=cfg.val_frac,
            teacher_hidden_dim=cfg.real_teacher_hidden_dim,
            teacher_num_epochs=cfg.real_teacher_num_epochs,
            teacher_batch_size=cfg.real_teacher_batch_size,
            teacher_learning_rate=cfg.real_teacher_learning_rate,
            teacher_weight_decay=cfg.real_teacher_weight_decay,
            device=cfg.real_teacher_device,
            seed=10 * seed + 1,
            flatten=True,
            download=cfg.download,
            max_train_samples=cfg.max_train_samples,
            max_test_samples=cfg.max_test_samples,
        )
        return build_mnist_real_splits(real_cfg)

    raise FormalComparisonRunnerError(f"Unsupported data source={cfg.source}.")



def make_noisy_noncomplementary_family(c: int, q: float) -> WeakLabelFamily:
    if not (0.0 < q <= 1.0 / (c - 1)):
        raise FormalComparisonRunnerError(
            f"q must lie in (0, 1/(c-1)], got q={q} for c={c}."
        )
    off_diag = (1.0 - q) / (c - 1)
    M = off_diag * np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(M, q)
    _validate_column_stochastic(M)
    return WeakLabelFamily(
        name=f"non_complementary_noisy_q_{q:.3f}",
        M=M,
        weak_label_matrix=None,
        weak_label_names=[f"obs_{i}" for i in range(c)],
        metadata={"type": "non_complementary_noisy", "q": q},
    )



def make_arbitrary_fixed_family(
    c: int,
    d: int,
    matrix: Optional[np.ndarray] = None,
    *,
    seed: int = 0,
) -> WeakLabelFamily:
    if matrix is None:
        if c == 4 and d == 6:
            M = _default_arbitrary_matrix(c, d)
        else:
            return make_random_general_weak_label_family(d=d, c=c, seed=seed)
    else:
        M = np.asarray(matrix, dtype=np.float64)

    if M.shape != (d, c):
        raise FormalComparisonRunnerError(f"Arbitrary matrix must have shape {(d, c)}, got {M.shape}.")
    _validate_column_stochastic(M)
    if np.linalg.matrix_rank(M) != c:
        raise FormalComparisonRunnerError("Arbitrary matrix must have full column rank.")
    return WeakLabelFamily(
        name=f"arbitrary_fixed_d_{d}_c_{c}",
        M=M,
        weak_label_matrix=None,
        weak_label_names=[f"w_{i}" for i in range(d)],
        metadata={"type": "arbitrary_fixed"},
    )



def _build_regime_family(regime_name: str, cfg: FormalComparisonConfig, *, seed: int) -> WeakLabelFamily:
    c = cfg.data.num_classes
    if regime_name == "uniform_dominance_friendly":
        return make_uniform_partial_label_family(c=c, candidate_size=cfg.partial_candidate_size)
    
    

    if regime_name == "biased_non_dominance":
        if cfg.data.source == "mnist" and c == 10:
            return make_manual_mnist_quad_partial_label_family(
                lambda3=48.0,
                lambda2=12.0,
                lambda1=3.0,
            )
        return make_biased_partial_label_family(
            c=c,
            candidate_size=cfg.partial_candidate_size,
            distractor_affinity=_default_cyclic_affinity(c=c, rho=cfg.biased_rho),
        )






    
    if regime_name == "arbitrary_transition":
        return make_arbitrary_fixed_family(
            c=c,
            d=cfg.arbitrary_d,
            matrix=cfg.arbitrary_fixed_matrix,
            seed=1000 + seed,
        )
    if regime_name == "complementary_uniform":
        return make_uniform_complementary_family(c=c)
    if regime_name == "non_complementary_noisy":
        return make_noisy_noncomplementary_family(c=c, q=cfg.non_complementary_q)
    raise FormalComparisonRunnerError(f"Unsupported regime_name={regime_name}.")



def _group_to_regime(group_name: str) -> tuple[str, dict[str, Any]]:
    if group_name == "g1_clwl_transition_sensitivity":
        raise FormalComparisonRunnerError("Use run_group_g1_transition_sensitivity for G1.")
    if group_name == "g2_clpl_vs_clwl_order_preserving_dominance":
        return "uniform_dominance_friendly", {
            "is_order_preserving": True,
            "is_dominance_satisfied": True,
            "is_uniform_transition": True,
            "is_complementary_setting": False,
        }
    if group_name == "g3_clpl_vs_clwl_order_preserving":
        return "biased_non_dominance", {
            "is_order_preserving": True,
            "is_dominance_satisfied": False,
            "is_uniform_transition": False,
            "is_complementary_setting": False,
        }
    if group_name == "g4_clpl_vs_clwl_arbitrary_transition":
        return "arbitrary_transition", {
            "is_order_preserving": None,
            "is_dominance_satisfied": False,
            "is_uniform_transition": False,
            "is_complementary_setting": False,
        }
    if group_name == "g5_clcl_vs_clwl_order_preserving":
        return "complementary_uniform", {
            "is_order_preserving": True,
            "is_dominance_satisfied": None,
            "is_uniform_transition": True,
            "is_complementary_setting": True,
        }
    if group_name == "g6_clcl_vs_clwl_non_complementary":
        return "non_complementary_noisy", {
            "is_order_preserving": None,
            "is_dominance_satisfied": None,
            "is_uniform_transition": False,
            "is_complementary_setting": False,
        }
    raise FormalComparisonRunnerError(f"Unsupported group_name={group_name}.")



def _teacher_or_zero_result(
    *,
    suite_name: str,
    group_name: str,
    regime_name: str,
    seed: int,
    split: str,
    dataset: WeakLabelDataset,
    use_teacher: bool,
    common_flags: dict[str, Any],
) -> FormalComparisonResult:
    scores = scores_from_logits(dataset.logits) if use_teacher else zero_scores_like_dataset(dataset)
    metrics = evaluate_scores_on_dataset(scores, dataset)
    metric_available = bool(metrics.metadata.get("order_metrics_available", True))

    return FormalComparisonResult(
        suite_name=suite_name,
        group_name=group_name,
        regime_name=regime_name,
        seed=seed,
        method="teacher_reference" if use_teacher else "zero_reference",
        split=split,
        clean_accuracy=float(metrics.clean_accuracy),
        max_preservation_rate=float(metrics.max_preservation_rate),
        pairwise_order_rate=float(metrics.pairwise_order_rate),
        pairwise_total=int(metrics.pairwise_total),
        pairwise_correct=int(metrics.pairwise_correct),
        mean_margin_on_ordered_pairs=float(metrics.mean_margin_on_ordered_pairs),
        empirical_risk=None,
        teacher_mean_pairwise_margin=(
            float(metrics.mean_margin_on_ordered_pairs) if use_teacher and metric_available else None
        ),
        conditional_risk=None,
        is_applicable=True,
        metric_available=metric_available,
        is_native_for_clpl=None,
        is_native_for_clcl=None,
        extra={"eta_source": metrics.metadata.get("eta_source", "unknown")},
        **common_flags,
    )



def _construction_to_extra(construction_result: Any) -> dict[str, Any]:
    return {
        "alpha": float(construction_result.alpha),
        "delta_max": float(construction_result.delta_max),
        "reconstruction_error": float(construction_result.reconstruction_error),
        "t_min": float(construction_result.t_min),
        "t_max": float(construction_result.t_max),
    }



def _record_from_metrics(
    *,
    suite_name: str,
    group_name: str,
    regime_name: str,
    seed: int,
    method: str,
    split: str,
    metrics: Optional[dict[str, Any]],
    is_applicable: bool,
    metric_available: bool,
    common_flags: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> FormalComparisonResult:
    if metrics is None:
        return FormalComparisonResult(
            suite_name=suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method=method,
            split=split,
            clean_accuracy=None,
            max_preservation_rate=None,
            pairwise_order_rate=None,
            pairwise_total=None,
            pairwise_correct=None,
            mean_margin_on_ordered_pairs=None,
            empirical_risk=None,
            teacher_mean_pairwise_margin=None,
            conditional_risk=None,
            is_applicable=is_applicable,
            metric_available=metric_available,
            is_native_for_clpl=(method == "CLPL") if is_applicable else False if method == "CLPL" else None,
            is_native_for_clcl=(method in {"CLCL_OR", "CLCL_ORW"}) if is_applicable else False if method in {"CLCL_OR", "CLCL_ORW"} else None,
            extra={} if extra is None else dict(extra),
            **common_flags,
        )

    derived_metric_available = bool(metric_available)
    try:
        pairwise_val = float(metrics["pairwise_order_rate"])
        if not np.isfinite(pairwise_val):
            derived_metric_available = False
    except Exception:
        derived_metric_available = False

    return FormalComparisonResult(
        suite_name=suite_name,
        group_name=group_name,
        regime_name=regime_name,
        seed=seed,
        method=method,
        split=split,
        clean_accuracy=float(metrics["clean_accuracy"]),
        max_preservation_rate=float(metrics["max_preservation_rate"]),
        pairwise_order_rate=float(metrics["pairwise_order_rate"]),
        pairwise_total=int(metrics["pairwise_total"]),
        pairwise_correct=int(metrics["pairwise_correct"]),
        mean_margin_on_ordered_pairs=float(metrics["mean_margin_on_ordered_pairs"]),
        empirical_risk=None if metrics.get("empirical_risk") is None else float(metrics["empirical_risk"]),
        teacher_mean_pairwise_margin=None,
        conditional_risk=None,
        is_applicable=is_applicable,
        metric_available=derived_metric_available,
        is_native_for_clpl=(method == "CLPL") if is_applicable else False if method == "CLPL" else None,
        is_native_for_clcl=(method in {"CLCL_OR", "CLCL_ORW"}) if is_applicable else False if method in {"CLCL_OR", "CLCL_ORW"} else None,
        extra={} if extra is None else dict(extra),
        **common_flags,
    )



def _build_weak_splits_for_regime(
    cfg: FormalComparisonConfig,
    regime_name: str,
    seed: int,
) -> tuple[WeakLabelFamily, dict[str, WeakLabelDataset]]:
    clean_splits = _build_dataset_splits(cfg.data, seed)
    family = _build_regime_family(regime_name, cfg, seed=seed)
    weak_splits = build_weak_label_splits(clean_splits, family, seed=100 * seed + 7)
    return family, weak_splits



def _run_clwl_for_regime(
    cfg: FormalComparisonConfig,
    group_name: str,
    regime_name: str,
    seed: int,
    train_ds: WeakLabelDataset,
    val_ds: WeakLabelDataset,
    test_ds: WeakLabelDataset,
    common_flags: dict[str, Any],
) -> list[FormalComparisonResult]:
    if regime_name == "complementary_uniform":
        c = train_ds.eta.shape[1]
        T = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
        extra = {"T_source": "complementary_special_case"}
    else:
        construction = construct_clwl_T(train_ds.M)
        T = construction.T
        extra = {"T_source": "generic_construction", **_construction_to_extra(construction)}

    train_cfg = replace(cfg.clwl_config, seed=seed)
    result = train_clwl_model(train_dataset=train_ds, val_dataset=val_ds, T=T, config=train_cfg)

    out = [
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLWL",
            split="train",
            metrics=result.final_train_metrics,
            is_applicable=True,
            metric_available=True,
            common_flags=common_flags,
            extra=extra,
        ),
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLWL",
            split="val",
            metrics=result.final_val_metrics,
            is_applicable=True,
            metric_available=result.final_val_metrics is not None,
            common_flags=common_flags,
            extra=extra,
        ),
    ]
    test_metrics = eval_clwl_model(result.model, test_ds, T, batch_size=max(train_cfg.batch_size, 512), device=train_cfg.device)
    out.append(
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLWL",
            split="test",
            metrics=test_metrics,
            is_applicable=True,
            metric_available=True,
            common_flags=common_flags,
            extra=extra,
        )
    )
    return out



def _run_clpl_for_regime(
    cfg: FormalComparisonConfig,
    group_name: str,
    regime_name: str,
    seed: int,
    train_ds: WeakLabelDataset,
    val_ds: WeakLabelDataset,
    test_ds: WeakLabelDataset,
    common_flags: dict[str, Any],
    *,
    applicable: bool,
) -> list[FormalComparisonResult]:
    if not applicable:
        return [
            _record_from_metrics(
                suite_name=cfg.suite_name,
                group_name=group_name,
                regime_name=regime_name,
                seed=seed,
                method="CLPL",
                split=split,
                metrics=None,
                is_applicable=False,
                metric_available=False,
                common_flags=common_flags,
            )
            for split in ["train", "val", "test"]
        ]

    train_cfg = replace(cfg.clpl_config, seed=seed)
    result = train_clpl_model(train_dataset=train_ds, val_dataset=val_ds, config=train_cfg)
    out = [
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLPL",
            split="train",
            metrics=result.final_train_metrics,
            is_applicable=True,
            metric_available=True,
            common_flags=common_flags,
        ),
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLPL",
            split="val",
            metrics=result.final_val_metrics,
            is_applicable=True,
            metric_available=result.final_val_metrics is not None,
            common_flags=common_flags,
        ),
    ]
    test_metrics = eval_clpl_model(result.model, test_ds, batch_size=max(train_cfg.batch_size, 512), device=train_cfg.device)
    out.append(
        _record_from_metrics(
            suite_name=cfg.suite_name,
            group_name=group_name,
            regime_name=regime_name,
            seed=seed,
            method="CLPL",
            split="test",
            metrics=test_metrics,
            is_applicable=True,
            metric_available=True,
            common_flags=common_flags,
        )
    )
    return out




def _run_clcl_for_regime(
    cfg: FormalComparisonConfig,
    group_name: str,
    regime_name: str,
    seed: int,
    train_ds: WeakLabelDataset,
    val_ds: WeakLabelDataset,
    test_ds: WeakLabelDataset,
    common_flags: dict[str, Any],
    *,
    applicable: bool,
) -> list[FormalComparisonResult]:
    variants_to_run: list[tuple[str, str]] = [
        ("or", "CLCL_OR"),
        ("or_w", "CLCL_ORW"),
    ]

    out: list[FormalComparisonResult] = []

    for variant, method_name in variants_to_run:
        if not applicable:
            out.extend(
                [
                    _record_from_metrics(
                        suite_name=cfg.suite_name,
                        group_name=group_name,
                        regime_name=regime_name,
                        seed=seed,
                        method=method_name,
                        split=split,
                        metrics=None,
                        is_applicable=False,
                        metric_available=False,
                        common_flags=common_flags,
                        extra={"clcl_variant": variant},
                    )
                    for split in ["train", "val", "test"]
                ]
            )
            continue

        train_cfg = replace(cfg.clcl_config, seed=seed, variant=variant)
        result = train_clcl_model(train_dataset=train_ds, val_dataset=val_ds, config=train_cfg)

        out.extend(
            [
                _record_from_metrics(
                    suite_name=cfg.suite_name,
                    group_name=group_name,
                    regime_name=regime_name,
                    seed=seed,
                    method=method_name,
                    split="train",
                    metrics=result.final_train_metrics,
                    is_applicable=True,
                    metric_available=True,
                    common_flags=common_flags,
                    extra={"clcl_variant": variant},
                ),
                _record_from_metrics(
                    suite_name=cfg.suite_name,
                    group_name=group_name,
                    regime_name=regime_name,
                    seed=seed,
                    method=method_name,
                    split="val",
                    metrics=result.final_val_metrics,
                    is_applicable=True,
                    metric_available=result.final_val_metrics is not None,
                    common_flags=common_flags,
                    extra={"clcl_variant": variant},
                ),
            ]
        )

        test_metrics = eval_clcl_model(
            result.model,
            test_ds,
            train_cfg,
            batch_size=max(train_cfg.batch_size, 512),
            device=train_cfg.device,
        )
        out.append(
            _record_from_metrics(
                suite_name=cfg.suite_name,
                group_name=group_name,
                regime_name=regime_name,
                seed=seed,
                method=method_name,
                split="test",
                metrics=test_metrics,
                is_applicable=True,
                metric_available=True,
                common_flags=common_flags,
                extra={"clcl_variant": variant},
            )
        )

    return out




def run_group_g1_transition_sensitivity(cfg: FormalComparisonConfig, seed: int) -> list[FormalComparisonResult]:
    group_name = "g1_clwl_transition_sensitivity"
    results: list[FormalComparisonResult] = []
    for regime_name, flags in [
        ("uniform_dominance_friendly", {"is_order_preserving": True, "is_dominance_satisfied": True, "is_uniform_transition": True, "is_complementary_setting": False}),
        ("biased_non_dominance", {"is_order_preserving": True, "is_dominance_satisfied": False, "is_uniform_transition": False, "is_complementary_setting": False}),
        ("arbitrary_transition", {"is_order_preserving": None, "is_dominance_satisfied": False, "is_uniform_transition": False, "is_complementary_setting": False}),
    ]:
        _, weak_splits = _build_weak_splits_for_regime(cfg, regime_name, seed)
        train_ds, val_ds, test_ds = weak_splits["train"], weak_splits["val"], weak_splits["test"]
        common_flags = {**flags}

        if cfg.run_teacher_reference:
            for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
                results.append(
                    _teacher_or_zero_result(
                        suite_name=cfg.suite_name,
                        group_name=group_name,
                        regime_name=regime_name,
                        seed=seed,
                        split=split_name,
                        dataset=ds,
                        use_teacher=True,
                        common_flags=common_flags,
                    )
                )
        if cfg.run_zero_reference:
            for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
                results.append(
                    _teacher_or_zero_result(
                        suite_name=cfg.suite_name,
                        group_name=group_name,
                        regime_name=regime_name,
                        seed=seed,
                        split=split_name,
                        dataset=ds,
                        use_teacher=False,
                        common_flags=common_flags,
                    )
                )
        results.extend(_run_clwl_for_regime(cfg, group_name, regime_name, seed, train_ds, val_ds, test_ds, common_flags))
    return results



def run_group(cfg: FormalComparisonConfig, group_name: str, seed: int) -> list[FormalComparisonResult]:
    if group_name == "g1_clwl_transition_sensitivity":
        return run_group_g1_transition_sensitivity(cfg, seed)

    regime_name, flags = _group_to_regime(group_name)
    _, weak_splits = _build_weak_splits_for_regime(cfg, regime_name, seed)
    train_ds, val_ds, test_ds = weak_splits["train"], weak_splits["val"], weak_splits["test"]
    common_flags = {**flags}

    results: list[FormalComparisonResult] = []
    if cfg.run_teacher_reference:
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            results.append(
                _teacher_or_zero_result(
                    suite_name=cfg.suite_name,
                    group_name=group_name,
                    regime_name=regime_name,
                    seed=seed,
                    split=split_name,
                    dataset=ds,
                    use_teacher=True,
                    common_flags=common_flags,
                )
            )
    if cfg.run_zero_reference:
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            results.append(
                _teacher_or_zero_result(
                    suite_name=cfg.suite_name,
                    group_name=group_name,
                    regime_name=regime_name,
                    seed=seed,
                    split=split_name,
                    dataset=ds,
                    use_teacher=False,
                    common_flags=common_flags,
                )
            )

    results.extend(_run_clwl_for_regime(cfg, group_name, regime_name, seed, train_ds, val_ds, test_ds, common_flags))

    if group_name in {
        "g2_clpl_vs_clwl_order_preserving_dominance",
        "g3_clpl_vs_clwl_order_preserving",
        "g4_clpl_vs_clwl_arbitrary_transition",
    }:
        results.extend(
            _run_clpl_for_regime(
                cfg,
                group_name,
                regime_name,
                seed,
                train_ds,
                val_ds,
                test_ds,
                common_flags,
                applicable=group_name != "g4_clpl_vs_clwl_arbitrary_transition",
            )
        )

    if group_name in {
        "g5_clcl_vs_clwl_order_preserving",
        "g6_clcl_vs_clwl_non_complementary",
    }:
        results.extend(
            _run_clcl_for_regime(
                cfg,
                group_name,
                regime_name,
                seed,
                train_ds,
                val_ds,
                test_ds,
                common_flags,
                applicable=group_name == "g5_clcl_vs_clwl_order_preserving",
            )
        )

    return results



def run_formal_comparison_suite(cfg: FormalComparisonConfig) -> list[FormalComparisonResult]:
    results: list[FormalComparisonResult] = []
    groups = FORMAL_GROUPS if cfg.groups is None else list(cfg.groups)

    for seed in cfg.seeds:
        for group_name in groups:
            results.extend(run_group(cfg, group_name, seed))
    return results



def results_to_dataframe(results: list[FormalComparisonResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        row = asdict(r)
        extra = row.pop("extra", {})
        for k, v in extra.items():
            row[f"extra__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)



def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_cols = ["suite_name", "group_name", "regime_name", "method", "split"]
    passthrough_cols = [
        "is_applicable",
        "metric_available",
        "is_order_preserving",
        "is_dominance_satisfied",
        "is_uniform_transition",
        "is_complementary_setting",
        "is_native_for_clpl",
        "is_native_for_clcl",
    ]
    metric_cols = [
        "clean_accuracy",
        "max_preservation_rate",
        "pairwise_order_rate",
        "mean_margin_on_ordered_pairs",
        "empirical_risk",
        "teacher_mean_pairwise_margin",
        "conditional_risk",
    ]
    extra_cols = [c for c in df.columns if c.startswith("extra__")]

    grouped = df.groupby(group_cols, dropna=False, observed=True)
    out = grouped[metric_cols + extra_cols].mean(numeric_only=True).reset_index()
    std = grouped[metric_cols].std(ddof=0).reset_index()
    counts = grouped.size().reset_index(name="num_runs")

    for col in metric_cols:
        out[f"{col}__std"] = std[col]
    out = out.merge(counts, on=group_cols, how="left")

    passthrough = grouped[passthrough_cols].first().reset_index()
    out = out.merge(passthrough, on=group_cols, how="left")
    return out



def default_formal_comparison_config() -> FormalComparisonConfig:
    return FormalComparisonConfig(
        suite_name="formal_comparison_suite",
        seeds=[0, 1, 2, 3, 4],
        data=SyntheticDataConfig(
            source="synthetic",
            teacher_type="linear",
            n=4000,
            input_dim=8,
            num_classes=4,
            hidden_dim=32,
            train_frac=0.6,
            val_frac=0.2,
            feature_cov_scale=1.0,
            weight_scale=1.0,
            bias_scale=0.5,
            logit_temperature=1.0,
        ),
        clwl_config=CLWLTrainConfig(
            model_type="linear",
            hidden_dim=128,
            batch_size=128,
            num_epochs=60,
            learning_rate=1e-2,
            weight_decay=0.0,
            device="cpu",
            seed=0,
            log_every=10,
        ),
        clpl_config=CLPLTrainConfig(
            model_type="linear",
            hidden_dim=128,
            batch_size=128,
            num_epochs=60,
            learning_rate=1e-2,
            weight_decay=0.0,
            device="cpu",
            seed=0,
            log_every=10,
        ),
        clcl_config=CLCLTrainConfig(
            model_type="linear",
            variant="or_w",
            hidden_dim=128,
            batch_size=128,
            num_epochs=60,
            learning_rate=1e-2,
            weight_decay=0.0,
            device="cpu",
            seed=0,
            log_every=10,
            weight_power=1.0,
            min_weight=1e-3,
        ),
        partial_candidate_size=2,
        biased_rho=5.0,
        arbitrary_d=6,
        arbitrary_fixed_matrix=None,
        non_complementary_q=0.1,
        run_teacher_reference=True,
        run_zero_reference=True,
        output_dir="formal_group_results",
        groups=None,
    )



def default_mnist_real_config() -> FormalComparisonConfig:
    return FormalComparisonConfig(
        suite_name="formal_mnist_real_suite",


        seeds=[0, 1, 2, 3, 4, 5, 6],


        groups=[
            "g2_clpl_vs_clwl_order_preserving_dominance",
            "g3_clpl_vs_clwl_order_preserving",
            "g5_clcl_vs_clwl_order_preserving",
        ],
        data=SyntheticDataConfig(
            source="mnist",
            num_classes=10,
            input_dim=784,
            val_frac=0.1,
            root="data",
            download=True,
            real_teacher_hidden_dim=256,
            real_teacher_num_epochs=12,
            real_teacher_batch_size=256,
            real_teacher_learning_rate=1e-3,
            real_teacher_weight_decay=1e-4,
            real_teacher_device="cpu",
            max_train_samples=None,
            max_test_samples=None,
        ),
        clwl_config=CLWLTrainConfig(
            model_type="mlp",
            hidden_dim=256,
            batch_size=256,
            num_epochs=20,
            learning_rate=1e-3,
            weight_decay=1e-4,
            device="cpu",
            seed=0,
            log_every=5,
        ),
        clpl_config=CLPLTrainConfig(
            model_type="mlp",
            hidden_dim=256,
            batch_size=256,
            num_epochs=20,
            learning_rate=1e-3,
            weight_decay=1e-4,
            device="cpu",
            seed=0,
            log_every=5,
        ),
        clcl_config=CLCLTrainConfig(
            model_type="mlp",
            variant="or_w",   # runner will now override this and run both OR and OR-W
            hidden_dim=256,
            batch_size=256,
            num_epochs=20,
            learning_rate=5e-5, #closer to the Liu2023 MNIST-family optimization setup.
            weight_decay=1e-4,
            device="cpu",
            seed=0,
            log_every=5,
            weight_power=1.0,
            min_weight=1e-3,
            weight_eps=1e-6,
            detach_weight=True,
        ),
        partial_candidate_size=2,
        biased_rho=5.0,
        arbitrary_d=12,
        arbitrary_fixed_matrix=None,
        non_complementary_q=0.05,
        run_teacher_reference=True,
        run_zero_reference=True,
        output_dir="formal_comparison_results_mnist_real",
    )



def save_suite_outputs(cfg: FormalComparisonConfig) -> dict[str, Path]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_formal_comparison_suite(cfg)
    raw_df = results_to_dataframe(results)
    agg_df = aggregate_results(raw_df)

    raw_path = output_dir / f"{cfg.suite_name}_raw_results.csv"
    agg_path = output_dir / f"{cfg.suite_name}_aggregated_results.csv"
    cfg_path = output_dir / f"{cfg.suite_name}_config.txt"

    raw_df.to_csv(raw_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    cfg_path.write_text(repr(cfg), encoding="utf-8")
    return {"raw": raw_path, "aggregated": agg_path, "config": cfg_path}


if __name__ == "__main__":
    cfg = default_mnist_real_config()
    files = save_suite_outputs(cfg)
    print("=== Saved formal comparison outputs ===")
    for k, v in files.items():
        print(k, v)

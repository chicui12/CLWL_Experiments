from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import pandas as pd

from clwl_experiments_module13_formal_comparison_runner import (
    FORMAL_GROUPS,
    FormalComparisonConfig,
    SyntheticDataConfig,
    results_to_dataframe,
    run_formal_comparison_suite,
    aggregate_results,
    default_mnist_real_config,
)

from clwl_experiments_module7_clwl_training import CLWLTrainConfig
from clwl_experiments_module8_clpl_training import CLPLTrainConfig
from clwl_experiments_module9_clcl_training import CLCLTrainConfig
from clwl_experiments_module13_formal_comparison_runner import (
    FORMAL_GROUPS,
    FormalComparisonConfig,
    SyntheticDataConfig,
    results_to_dataframe,
    run_formal_comparison_suite,
    aggregate_results,
)


GROUP_ORDER = list(FORMAL_GROUPS)
METHOD_ORDER = [
    "teacher_reference",
    "zero_reference",
    "CLWL",
    "CLPL",
    "CLCL_OR",
    "CLCL_ORW",
]
SPLIT_ORDER = ["train", "val", "test"]



def _ordered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "group_name" in out.columns:
        out["group_name"] = pd.Categorical(out["group_name"], categories=GROUP_ORDER, ordered=True)
    if "method" in out.columns:
        out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    if "split" in out.columns:
        out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)

    sort_cols = [c for c in ["suite_name", "group_name", "regime_name", "method", "split", "seed"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out



def build_formal_comparison_suite_configs() -> list[FormalComparisonConfig]:
    seeds = [0, 1, 2, 3, 4]

    base_linear_data = SyntheticDataConfig(
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
    )

    base_mlp_data = SyntheticDataConfig(
        teacher_type="mlp",
        n=4000,
        input_dim=8,
        num_classes=4,
        hidden_dim=32,
        train_frac=0.6,
        val_frac=0.2,
        feature_cov_scale=1.0,
        weight_scale=1.0,
        bias_scale=0.2,
        logit_temperature=1.0,
    )

    clwl_linear = CLWLTrainConfig(
        model_type="linear",
        hidden_dim=128,
        batch_size=128,
        num_epochs=60,
        learning_rate=1e-2,
        weight_decay=0.0,
        device="cpu",
        seed=0,
        log_every=10,
    )
    clpl_linear = CLPLTrainConfig(
        model_type="linear",
        hidden_dim=128,
        batch_size=128,
        num_epochs=60,
        learning_rate=1e-2,
        weight_decay=0.0,
        device="cpu",
        seed=0,
        log_every=10,
    )
    clcl_linear = CLCLTrainConfig(
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
    )

    clwl_mlp = replace(clwl_linear, model_type="mlp", hidden_dim=128, num_epochs=80)
    clpl_mlp = replace(clpl_linear, model_type="mlp", hidden_dim=128, num_epochs=80)
    clcl_mlp = replace(clcl_linear, model_type="mlp", hidden_dim=128, num_epochs=80)

    suite_linear = FormalComparisonConfig(
        suite_name="formal_comparison_linear_suite",
        seeds=seeds,
        data=base_linear_data,
        clwl_config=clwl_linear,
        clpl_config=clpl_linear,
        clcl_config=clcl_linear,
        partial_candidate_size=2,
        biased_rho=5.0,
        arbitrary_d=6,
        arbitrary_fixed_matrix=None,
        non_complementary_q=0.1,
        run_teacher_reference=True,
        run_zero_reference=True,
        output_dir="formal_comparison_results",
    )

    suite_mlp = FormalComparisonConfig(
        suite_name="formal_comparison_mlp_suite",
        seeds=seeds,
        data=base_mlp_data,
        clwl_config=clwl_mlp,
        clpl_config=clpl_mlp,
        clcl_config=clcl_mlp,
        partial_candidate_size=2,
        biased_rho=5.0,
        arbitrary_d=6,
        arbitrary_fixed_matrix=None,
        non_complementary_q=0.1,
        run_teacher_reference=True,
        run_zero_reference=True,
        output_dir="formal_comparison_results",
    )

    return [suite_linear, suite_mlp]



def run_and_save_suite(cfg: FormalComparisonConfig, output_dir: str | Path | None = None) -> dict[str, Path]:
    output_path = Path(cfg.output_dir if output_dir is None else output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = run_formal_comparison_suite(cfg)
    raw_df = _ordered_dataframe(results_to_dataframe(results))
    agg_df = _ordered_dataframe(aggregate_results(raw_df))

    raw_file = output_path / f"{cfg.suite_name}_raw_results.csv"
    agg_file = output_path / f"{cfg.suite_name}_aggregated_results.csv"
    config_file = output_path / f"{cfg.suite_name}_config.txt"

    raw_df.to_csv(raw_file, index=False)
    agg_df.to_csv(agg_file, index=False)
    config_file.write_text(repr(cfg), encoding="utf-8")

    return {
        "raw_results": raw_file,
        "aggregated_results": agg_file,
        "config": config_file,
    }



def run_and_save_all_formal_comparison_suites(output_dir: str | Path) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for cfg in build_formal_comparison_suite_configs():
        files = run_and_save_suite(cfg, output_path)
        manifest_rows.append(
            {
                "suite_name": cfg.suite_name,
                "raw_results": str(files["raw_results"]),
                "aggregated_results": str(files["aggregated_results"]),
                "config": str(files["config"]),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = output_path / "formal_comparison_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    return manifest



def collect_all_aggregated_results(output_dir: str | Path) -> pd.DataFrame:
    output_path = Path(output_dir)
    agg_files = sorted(output_path.glob("*_aggregated_results.csv"))
    if not agg_files:
        raise FileNotFoundError(f"No aggregated result files found in {output_path}.")

    frames: list[pd.DataFrame] = []
    for file in agg_files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    return _ordered_dataframe(out)



def collect_test_split_table(output_dir: str | Path) -> pd.DataFrame:
    df = collect_all_aggregated_results(output_dir)
    if "split" not in df.columns:
        raise ValueError("Aggregated result table does not contain a 'split' column.")
    test_df = df[df["split"] == "test"].copy()
    return _ordered_dataframe(test_df)



def collect_group_tables(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    df = collect_test_split_table(output_dir)
    tables: dict[str, pd.DataFrame] = {}
    for group_name in GROUP_ORDER:
        tables[group_name] = df[df["group_name"] == group_name].copy().reset_index(drop=True)
    return tables


if __name__ == "__main__":
    output_dir = Path("formal_comparison_results")
    manifest = run_and_save_all_formal_comparison_suites(output_dir)
    print("=== Formal comparison manifest ===")
    print(manifest)

    test_table = collect_test_split_table(output_dir)
    print("\n=== Test split aggregate table ===")
    print(test_table)

def run_and_save_mnist_real_suite(output_dir: str | Path) -> dict[str, Path]:
    cfg = default_mnist_real_config()
    return run_and_save_suite(cfg, output_dir)

def build_real_data_suite_configs() -> list[FormalComparisonConfig]:
    return [default_mnist_real_config()]
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


GROUP_LABELS = {
    "g1_clwl_transition_sensitivity": "G1 CLWL transition sensitivity",
    "g2_clpl_vs_clwl_order_preserving_dominance": "G2 CLPL vs CLWL (order preserving + dominance)",
    "g3_clpl_vs_clwl_order_preserving": "G3 CLPL vs CLWL (order preserving)",
    "g4_clpl_vs_clwl_arbitrary_transition": "G4 CLPL vs CLWL (arbitrary transition)",
    "g5_clcl_vs_clwl_order_preserving": "G5 CLCL vs CLWL (order preserving)",
    "g6_clcl_vs_clwl_non_complementary": "G6 CLCL vs CLWL (non-complementary)",
}

METHOD_LABELS = {
    "teacher_reference": "Teacher",
    "zero_reference": "Zero",
    "CLWL": "CLWL",
    "CLPL": "CLPL",
    "CLCL_OR": "CLCL-OR",
    "CLCL_ORW": "CLCL-ORW",
}

SUITE_LABELS = {
    "formal_comparison_linear_suite": "Linear",
    "formal_comparison_mlp_suite": "MLP",
}

GROUP_ORDER = list(GROUP_LABELS.keys())
METHOD_ORDER = [
    "teacher_reference",
    "zero_reference",
    "CLWL",
    "CLPL",
    "CLCL_OR",
    "CLCL_ORW",
]
SPLIT_ORDER = ["train", "val", "test"]

GROUP_METHODS = {
    "g1_clwl_transition_sensitivity": ["teacher_reference", "zero_reference", "CLWL"],
    "g2_clpl_vs_clwl_order_preserving_dominance": ["teacher_reference", "zero_reference", "CLWL", "CLPL"],
    "g3_clpl_vs_clwl_order_preserving": ["teacher_reference", "zero_reference", "CLWL", "CLPL"],
    "g4_clpl_vs_clwl_arbitrary_transition": ["teacher_reference", "zero_reference", "CLWL", "CLPL"],
    "g5_clcl_vs_clwl_order_preserving": ["teacher_reference", "zero_reference", "CLWL", "CLCL_OR", "CLCL_ORW"],
    "g6_clcl_vs_clwl_non_complementary": ["teacher_reference", "zero_reference", "CLWL", "CLCL_OR", "CLCL_ORW"],
}

DEFAULT_METRICS = [
    "clean_accuracy",
    "max_preservation_rate",
    "pairwise_order_rate",
    "mean_margin_on_ordered_pairs",
    "empirical_risk",
    "teacher_mean_pairwise_margin",
    "conditional_risk",
]


@dataclass
class ComparisonTableBundle:
    name: str
    dataframe: pd.DataFrame
    latex: str
    csv_path: Optional[Path] = None
    latex_path: Optional[Path] = None


class FormalComparisonSummarizerError(ValueError):
    pass



def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)



def load_formal_comparison_manifest(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    return _safe_read_csv(output_dir / "formal_comparison_manifest.csv")



def load_all_aggregated_results(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("*_aggregated_results.csv"))
    if not files:
        raise FileNotFoundError(f"No aggregated result files found in {output_dir}")
    frames = []
    for file in files:
        df = _safe_read_csv(file)
        df["source_file"] = file.name
        frames.append(df)
    out = pd.concat(frames, axis=0, ignore_index=True)
    required = ["suite_name", "group_name", "regime_name", "method", "split"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise FormalComparisonSummarizerError(f"Aggregated results missing required columns: {missing}")
    return out



def normalize_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = out["group_name"].map(GROUP_LABELS).fillna(out["group_name"])
    out["method_label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    out["suite_label"] = out["suite_name"].map(SUITE_LABELS).fillna(out["suite_name"])

    if "is_applicable" not in out.columns:
        out["is_applicable"] = True
    if "metric_available" not in out.columns:
        out["metric_available"] = True
    if "teacher_mean_pairwise_margin" not in out.columns:
        out["teacher_mean_pairwise_margin"] = pd.NA
    if "conditional_risk" not in out.columns:
        out["conditional_risk"] = pd.NA

    out["group_name"] = pd.Categorical(out["group_name"], categories=GROUP_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    return out.sort_values(["suite_name", "group_name", "regime_name", "method", "split"]).reset_index(drop=True)



def filter_test_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        raise FormalComparisonSummarizerError("Missing split column.")
    return df[df["split"] == "test"].copy()



def _metric_std_column(metric: str) -> str:
    return f"{metric}__std"



def _format_mean_std(mean: object, std: object, decimals: int = 4, na_token: str = "NA") -> str:
    if pd.isna(mean):
        return na_token
    if pd.isna(std):
        return f"{float(mean):.{decimals}f}"
    return f"{float(mean):.{decimals}f} ± {float(std):.{decimals}f}"



def _format_metric_column(
    df: pd.DataFrame,
    metric: str,
    *,
    decimals: int = 4,
    applicability_col: str = "is_applicable",
    availability_col: str = "metric_available",
) -> pd.Series:
    std_col = _metric_std_column(metric)
    if metric not in df.columns:
        return pd.Series(["NA"] * len(df), index=df.index)

    values: list[str] = []
    for i in df.index:
        applicable = bool(df.loc[i, applicability_col]) if applicability_col in df.columns else True
        available = bool(df.loc[i, availability_col]) if availability_col in df.columns else True
        if not applicable:
            values.append("N/A")
            continue
        if not available:
            values.append("NA")
            continue
        mean = df.loc[i, metric]
        std = df.loc[i, std_col] if std_col in df.columns else pd.NA
        values.append(_format_mean_std(mean, std, decimals=decimals))
    return pd.Series(values, index=df.index)



def build_group_summary_table(
    df: pd.DataFrame,
    *,
    group_name: str,
    metrics: Optional[list[str]] = None,
    decimals: int = 4,
) -> pd.DataFrame:
    work = normalize_result_frame(df)
    work = filter_test_split(work)
    work = work[work["group_name"] == group_name].copy()
    if work.empty:
        return pd.DataFrame()

    allowed_methods = GROUP_METHODS.get(group_name, METHOD_ORDER)
    work = work[work["method"].isin(allowed_methods)].copy()
    if metrics is None:
        metrics = [m for m in DEFAULT_METRICS if m in work.columns]

    base_cols = [
        "suite_name",
        "suite_label",
        "group_name",
        "group_label",
        "regime_name",
        "method",
        "method_label",
        "is_applicable",
        "metric_available",
        "is_order_preserving",
        "is_dominance_satisfied",
        "is_uniform_transition",
        "is_complementary_setting",
        "is_native_for_clpl",
        "is_native_for_clcl",
    ]
    base_cols = [c for c in base_cols if c in work.columns]
    out = work[base_cols].copy()

    for metric in metrics:
        out[metric] = _format_metric_column(work, metric, decimals=decimals)

    out = out.sort_values(["suite_label", "regime_name", "method_label"]).reset_index(drop=True)
    return out



def build_group_metric_pivot(
    df: pd.DataFrame,
    *,
    group_name: str,
    metric: str,
    decimals: int = 4,
) -> pd.DataFrame:
    work = normalize_result_frame(df)
    work = filter_test_split(work)
    work = work[work["group_name"] == group_name].copy()
    if work.empty:
        return pd.DataFrame()

    allowed_methods = GROUP_METHODS.get(group_name, METHOD_ORDER)
    work = work[work["method"].isin(allowed_methods)].copy()
    work[metric] = _format_metric_column(work, metric, decimals=decimals)

    pivot = work.pivot_table(
        index=["regime_name", "method_label"],
        columns="suite_label",
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot



def _table_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        return "% Empty table"
    latex = df.to_latex(index=False, escape=False)
    latex += f"\n% caption: {caption}\n% label: {label}\n"
    return latex



def build_all_group_tables(df: pd.DataFrame) -> dict[str, ComparisonTableBundle]:
    bundles: dict[str, ComparisonTableBundle] = {}
    for group_name in GROUP_ORDER:
        summary = build_group_summary_table(df, group_name=group_name)
        clean_pivot = build_group_metric_pivot(df, group_name=group_name, metric="clean_accuracy")
        order_pivot = build_group_metric_pivot(df, group_name=group_name, metric="pairwise_order_rate")

        bundles[f"{group_name}_summary"] = ComparisonTableBundle(
            name=f"{group_name}_summary",
            dataframe=summary,
            latex=_table_to_latex(summary, f"Summary table for {GROUP_LABELS[group_name]}.", f"tab:{group_name}_summary"),
        )
        bundles[f"{group_name}_clean_accuracy"] = ComparisonTableBundle(
            name=f"{group_name}_clean_accuracy",
            dataframe=clean_pivot,
            latex=_table_to_latex(clean_pivot, f"Clean accuracy for {GROUP_LABELS[group_name]}.", f"tab:{group_name}_clean_accuracy"),
        )
        bundles[f"{group_name}_pairwise_order"] = ComparisonTableBundle(
            name=f"{group_name}_pairwise_order",
            dataframe=order_pivot,
            latex=_table_to_latex(order_pivot, f"Pairwise order rate for {GROUP_LABELS[group_name]}.", f"tab:{group_name}_pairwise_order"),
        )
    return bundles



def build_plot_ready_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = normalize_result_frame(df)
    work = filter_test_split(work)
    plot_metrics = [m for m in DEFAULT_METRICS if m in work.columns]
    frames: dict[str, pd.DataFrame] = {}
    for metric in plot_metrics:
        std_col = _metric_std_column(metric)
        cols = [
            c for c in [
                "suite_name",
                "suite_label",
                "group_name",
                "group_label",
                "regime_name",
                "method",
                "method_label",
                metric,
                std_col,
                "is_applicable",
                "metric_available",
                "is_order_preserving",
                "is_dominance_satisfied",
                "is_uniform_transition",
                "is_complementary_setting",
                "is_native_for_clpl",
                "is_native_for_clcl",
            ] if c in work.columns
        ]
        frames[metric] = work[cols].copy().sort_values(["suite_label", "group_name", "regime_name", "method_label"]).reset_index(drop=True)
    return frames



def save_bundle(bundle: ComparisonTableBundle, output_dir: str | Path) -> ComparisonTableBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{bundle.name}.csv"
    tex_path = output_dir / f"{bundle.name}.tex"
    bundle.dataframe.to_csv(csv_path, index=False)
    tex_path.write_text(bundle.latex, encoding="utf-8")
    bundle.csv_path = csv_path
    bundle.latex_path = tex_path
    return bundle



def export_formal_comparison_materials(
    formal_results_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, ComparisonTableBundle]:
    df = load_all_aggregated_results(formal_results_dir)
    bundles = build_all_group_tables(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in list(bundles.keys()):
        bundles[name] = save_bundle(bundles[name], output_dir)

    plot_frames = build_plot_ready_frames(df)
    for metric, frame in plot_frames.items():
        frame.to_csv(output_dir / f"plot_ready_{metric}.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "name": name,
                "csv_path": str(bundle.csv_path) if bundle.csv_path is not None else "",
                "latex_path": str(bundle.latex_path) if bundle.latex_path is not None else "",
            }
            for name, bundle in bundles.items()
        ]
    )
    manifest.to_csv(output_dir / "formal_comparison_materials_manifest.csv", index=False)
    return bundles


if __name__ == "__main__":
    formal_results_dir = Path("formal_comparison_results")
    output_dir = Path("formal_comparison_tables")
    bundles = export_formal_comparison_materials(formal_results_dir, output_dir)
    print("=== Exported formal comparison tables ===")
    for name, bundle in bundles.items():
        print(name, bundle.csv_path, bundle.latex_path)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_LABELS = {
    "g1_clwl_transition_sensitivity": "G1 CLWL transition sensitivity",
    "g2_clpl_vs_clwl_order_preserving_dominance": "G2 CLPL vs CLWL (OP + dominance)",
    "g3_clpl_vs_clwl_order_preserving": "G3 CLPL vs CLWL (OP)",
    "g4_clpl_vs_clwl_arbitrary_transition": "G4 CLPL vs CLWL (arbitrary M)",
    "g5_clcl_vs_clwl_order_preserving": "G5 CLCL vs CLWL (OP)",
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
SUITE_ORDER = [
    "formal_comparison_linear_suite",
    "formal_comparison_mlp_suite",
]
SPLIT_ORDER = ["train", "val", "test"]

DEFAULT_METRICS = [
    "clean_accuracy",
    "pairwise_order_rate",
    "max_preservation_rate",
    "empirical_risk",
]


@dataclass
class FigureBuildResult:
    name: str
    path: Path


class FormalFigureBuilderError(ValueError):
    pass



def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)



def load_aggregated_results(formal_results_dir: str | Path) -> pd.DataFrame:
    formal_results_dir = Path(formal_results_dir)
    files = sorted(formal_results_dir.glob("*_aggregated_results.csv"))
    if not files:
        raise FileNotFoundError(f"No aggregated result files found in {formal_results_dir}")
    frames = []
    for file in files:
        df = _safe_read_csv(file)
        df["source_file"] = file.name
        frames.append(df)
    out = pd.concat(frames, axis=0, ignore_index=True)
    required = ["suite_name", "group_name", "regime_name", "method", "split"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise FormalFigureBuilderError(f"Aggregated result files missing required columns: {missing}")
    return out



def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_label"] = out["group_name"].map(GROUP_LABELS).fillna(out["group_name"])
    out["method_label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    out["suite_label"] = out["suite_name"].map(SUITE_LABELS).fillna(out["suite_name"])
    if "is_applicable" not in out.columns:
        out["is_applicable"] = True
    if "metric_available" not in out.columns:
        out["metric_available"] = True
    out["group_name"] = pd.Categorical(out["group_name"], categories=GROUP_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out["suite_name"] = pd.Categorical(out["suite_name"], categories=SUITE_ORDER, ordered=True)
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    return out.sort_values(["suite_name", "group_name", "regime_name", "method", "split"]).reset_index(drop=True)



def filter_plot_frame(
    df: pd.DataFrame,
    *,
    split: str = "test",
    require_applicable: bool = True,
    require_metric_available: bool = True,
) -> pd.DataFrame:
    work = normalize_frame(df)
    work = work[work["split"] == split].copy()
    if require_applicable and "is_applicable" in work.columns:
        work = work[work["is_applicable"] == True].copy()
    if require_metric_available and "metric_available" in work.columns:
        work = work[work["metric_available"] == True].copy()
    return work.reset_index(drop=True)



def _std_col(metric: str) -> str:
    return f"{metric}__std"



def _sanitize_filename(text: str) -> str:
    out = text.lower()
    for ch in [" ", "/", "(", ")", ",", ":", "="]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")



def _plot_grouped_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_col: str,
    hue_col: str,
    y_col: str,
    yerr_col: Optional[str] = None,
    title: str,
    ylabel: str,
) -> None:
    x_values = list(data[x_col].drop_duplicates())
    hue_values = list(data[hue_col].drop_duplicates())
    if not x_values or not hue_values:
        raise FormalFigureBuilderError("No data available for grouped bar plot.")

    x = np.arange(len(x_values))
    width = 0.8 / max(len(hue_values), 1)

    for j, hue in enumerate(hue_values):
        subset = data[data[hue_col] == hue]
        heights = []
        errors = []
        for xv in x_values:
            row = subset[subset[x_col] == xv]
            if row.empty:
                heights.append(np.nan)
                errors.append(np.nan)
            else:
                heights.append(float(row.iloc[0][y_col]))
                if yerr_col is not None and yerr_col in row.columns and pd.notna(row.iloc[0][yerr_col]):
                    errors.append(float(row.iloc[0][yerr_col]))
                else:
                    errors.append(0.0)
        offsets = x - 0.4 + width / 2 + j * width
        ax.bar(offsets, heights, width=width, label=str(hue))
        if yerr_col is not None:
            ax.errorbar(offsets, heights, yerr=errors, fmt="none", capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_values], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)



def make_group_metric_figure(
    df: pd.DataFrame,
    *,
    group_name: str,
    metric: str,
    output_dir: str | Path,
) -> FigureBuildResult:
    work = filter_plot_frame(df, split="test")
    work = work[work["group_name"] == group_name].copy()
    if work.empty:
        raise FormalFigureBuilderError(f"No data for group {group_name}.")
    if metric not in work.columns:
        raise FormalFigureBuilderError(f"Metric {metric} not found in frame.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    suites_present = [s for s in SUITE_ORDER if s in set(work["suite_name"].astype(str))]
    if len(suites_present) == 1:
        axes = [axes[0]]
    for ax, suite_name in zip(axes, suites_present):
        suite_df = work[work["suite_name"].astype(str) == suite_name].copy()
        suite_df = suite_df.sort_values(["regime_name", "method"])
        _plot_grouped_bars(
            ax,
            suite_df,
            x_col="regime_name",
            hue_col="method_label",
            y_col=metric,
            yerr_col=_std_col(metric) if _std_col(metric) in suite_df.columns else None,
            title=f"{GROUP_LABELS.get(group_name, group_name)} — {SUITE_LABELS.get(suite_name, suite_name)}",
            ylabel=metric,
        )
    if len(suites_present) == 1:
        fig.delaxes(plt.gcf().axes[-1]) if len(fig.axes) > 1 else None

    filename = f"{_sanitize_filename(group_name)}_{_sanitize_filename(metric)}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FigureBuildResult(name=filename[:-4], path=path)



def make_overview_metric_figure(
    df: pd.DataFrame,
    *,
    metric: str,
    output_dir: str | Path,
) -> FigureBuildResult:
    work = filter_plot_frame(df, split="test")
    if work.empty:
        raise FormalFigureBuilderError("No data available for overview figure.")
    if metric not in work.columns:
        raise FormalFigureBuilderError(f"Metric {metric} not found in frame.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = work.copy()
    plot_df = plot_df.groupby(["suite_name", "group_name", "method_label"], dropna=False)[metric].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    suites_present = [s for s in SUITE_ORDER if s in set(plot_df["suite_name"].astype(str))]
    if len(suites_present) == 1:
        axes = [axes[0]]

    for ax, suite_name in zip(axes, suites_present):
        suite_df = plot_df[plot_df["suite_name"].astype(str) == suite_name].copy()
        suite_df["group_label_short"] = suite_df["group_name"].astype(str).map(GROUP_LABELS)
        _plot_grouped_bars(
            ax,
            suite_df,
            x_col="group_label_short",
            hue_col="method_label",
            y_col=metric,
            yerr_col=None,
            title=f"Overview — {SUITE_LABELS.get(suite_name, suite_name)}",
            ylabel=metric,
        )
    if len(suites_present) == 1:
        fig.delaxes(plt.gcf().axes[-1]) if len(fig.axes) > 1 else None

    filename = f"overview_{_sanitize_filename(metric)}.png"
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FigureBuildResult(name=filename[:-4], path=path)



def make_transition_sensitivity_figure(
    df: pd.DataFrame,
    *,
    metric: str,
    output_dir: str | Path,
) -> FigureBuildResult:
    return make_group_metric_figure(
        df,
        group_name="g1_clwl_transition_sensitivity",
        metric=metric,
        output_dir=output_dir,
    )



def build_all_default_figures(
    formal_results_dir: str | Path,
    output_dir: str | Path,
    *,
    metrics: Optional[list[str]] = None,
) -> list[FigureBuildResult]:
    df = load_aggregated_results(formal_results_dir)
    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[FigureBuildResult] = []
    for metric in metrics:
        figures.append(make_overview_metric_figure(df, metric=metric, output_dir=output_dir))
        for group_name in GROUP_ORDER:
            try:
                figures.append(make_group_metric_figure(df, group_name=group_name, metric=metric, output_dir=output_dir))
            except FormalFigureBuilderError:
                continue

    manifest = pd.DataFrame([{"name": fig.name, "path": str(fig.path)} for fig in figures])
    manifest.to_csv(output_dir / "formal_comparison_figures_manifest.csv", index=False)
    return figures


if __name__ == "__main__":
    formal_results_dir = Path("formal_comparison_results")
    output_dir = Path("formal_comparison_figures")
    figures = build_all_default_figures(formal_results_dir, output_dir)
    print("=== Exported formal comparison figures ===")
    for fig in figures:
        print(fig.name, fig.path)

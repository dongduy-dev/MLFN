"""
figures.py
==========
Checkpoint 2A — Figure generation for static feature EDA.

All figures are saved as PNG and then closed immediately to avoid memory leaks.
No modelling, splitting, or transformation is performed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from credit_default.eda.static_features import (
    DOCUMENTED_VALUES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DPI = 150
FIGSIZE_SINGLE = (7, 5)
FIGSIZE_WIDE = (9, 5)
FIGSIZE_CAT = (8, 5)

COLOR_DEFAULT = "#D64045"      # red-ish  — positive class
COLOR_NO_DEFAULT = "#4472C4"   # blue     — negative class
COLOR_BAR = "#4C72B0"          # generic bar
COLOR_UNDOC = "#E89B35"        # orange   — undocumented category

LABEL_DEFAULT = f"Default ({POSITIVE_CLASS})"
LABEL_NO_DEFAULT = f"No default ({NEGATIVE_CLASS})"


# ---------------------------------------------------------------------------
# 1. Target distribution bar chart
# ---------------------------------------------------------------------------
def plot_target_distribution(
    target_summary: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Bar chart of target class counts with percentage labels."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [COLOR_NO_DEFAULT, COLOR_DEFAULT]
    labels = target_summary["class_label"].str.replace("_", " ").str.title()
    counts = target_summary["count"]
    pcts = target_summary["percentage"]

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{pct:.2f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Target Class Distribution\n(Positive class = Default next month)", fontsize=12)
    ax.set_ylabel("Count (number of clients)", fontsize=10)
    ax.set_xlabel("Target class", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "target_distribution.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2 & 3. LIMIT_BAL and AGE — distribution by target + boxplot by target
# ---------------------------------------------------------------------------
def plot_numeric_distribution_by_target(
    df: pd.DataFrame,
    col: str,
    out_dir: Path,
    xlabel: str | None = None,
    n_bins: int = 40,
) -> Path:
    """Overlapping histogram of a numeric column split by target class."""
    _require_col(df, col)
    xlabel = xlabel or col

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    bins, pct_negative, pct_positive = compute_class_histogram_percentages(df, col, n_bins)
    
    width = np.diff(bins)
    centers = bins[:-1] + width / 2
    
    n_neg = len(df.loc[df[TARGET_COL] == NEGATIVE_CLASS, col].dropna())
    n_pos = len(df.loc[df[TARGET_COL] == POSITIVE_CLASS, col].dropna())

    ax.bar(centers, pct_negative, width=width, alpha=0.55, color=COLOR_NO_DEFAULT, label=f"{LABEL_NO_DEFAULT} (n={n_neg:,})")
    ax.bar(centers, pct_positive, width=width, alpha=0.55, color=COLOR_DEFAULT, label=f"{LABEL_DEFAULT} (n={n_pos:,})")

    ax.set_title(f"{col} Distribution by Default Status", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Percentage within class (%)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"{col.lower()}_distribution_by_target.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_numeric_boxplot_by_target(
    df: pd.DataFrame,
    col: str,
    out_dir: Path,
    ylabel: str | None = None,
) -> Path:
    """Side-by-side boxplot of a numeric column split by target class."""
    _require_col(df, col)
    ylabel = ylabel or col

    groups = [
        df.loc[df[TARGET_COL] == NEGATIVE_CLASS, col].values,
        df.loc[df[TARGET_COL] == POSITIVE_CLASS, col].values,
    ]
    group_labels = [
        f"No default\n(n={len(groups[0]):,})",
        f"Default\n(n={len(groups[1]):,})",
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bp = ax.boxplot(
        groups,
        labels=group_labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    colors = [COLOR_NO_DEFAULT, COLOR_DEFAULT]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title(f"{col} by Default Status", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"{col.lower()}_boxplot_by_target.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4–6. Categorical — count bar chart and default-rate bar chart
# ---------------------------------------------------------------------------
def plot_categorical_counts(
    cat_dist: pd.DataFrame,
    col: str,
    out_dir: Path,
    small_n_threshold: int = 200,
) -> Path:
    """
    Horizontal bar chart of category counts for a categorical feature.
    Undocumented categories are coloured distinctly.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_CAT)
    data = cat_dist.sort_values("raw_value")

    bar_colors = [
        COLOR_UNDOC if s == "not_explicitly_defined_in_uci_docs" else COLOR_BAR
        for s in data["documentation_status"]
    ]
    x_labels = _make_labels(data, col)

    bars = ax.bar(x_labels, data["count"], color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, data.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + data["count"].max() * 0.01,
            f"n={row.count:,}\n({row.percentage:.1f}%)",
            ha="center", va="bottom", fontsize=8,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_BAR, label="Documented")]
    if any(s == "not_explicitly_defined_in_uci_docs" for s in data["documentation_status"]):
        legend_elements.append(Patch(facecolor=COLOR_UNDOC, label="Not explicitly defined in UCI docs"))
    ax.legend(handles=legend_elements, fontsize=8)

    ax.set_title(f"{col} — Category Counts (n = count per category)", fontsize=12)
    ax.set_xlabel(f"{col} raw code", fontsize=10)
    ax.set_ylabel("Count (number of clients)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, data["count"].max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"{col.lower()}_category_counts.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_categorical_default_rates(
    cat_rate: pd.DataFrame,
    col: str,
    out_dir: Path,
    small_n_threshold: int = 200,
) -> Path:
    """
    Bar chart of default rate per category code.
    Categories below small_n_threshold get an asterisk and note.
    Undocumented categories are coloured distinctly.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_CAT)
    data = cat_rate.sort_values("raw_value")

    bar_colors = [
        COLOR_UNDOC if s == "not_explicitly_defined_in_uci_docs" else COLOR_DEFAULT
        for s in data["documentation_status"]
    ]
    x_labels = _make_labels(data, col)

    bars = ax.bar(x_labels, data["default_rate"] * 100, color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.85)

    overall_rate = (data["default_count"].sum() / data["total_count"].sum()) * 100
    ref_line = ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1.2)

    for bar, row in zip(bars, data.itertuples()):
        label = f"{row.default_rate * 100:.1f}%"
        if row.total_count < small_n_threshold:
            label += "*"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            label,
            ha="center", va="bottom", fontsize=8,
        )

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLOR_DEFAULT, alpha=0.85, label="Documented"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label=f"Overall rate ({overall_rate:.1f}%)"),
    ]
    if any(s == "not_explicitly_defined_in_uci_docs" for s in data["documentation_status"]):
        legend_elements.append(Patch(facecolor=COLOR_UNDOC, alpha=0.85, label="Not explicitly defined in UCI docs"))
    if any(data["small_sample_warning"]):
        legend_elements.append(
            Line2D([], [], marker="*", color="black", linestyle="none", markersize=8, label=f"* n < {small_n_threshold} — interpret cautiously")
        )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax.set_title(f"{col} — Default Rate by Category\n(Positive class = default next month)", fontsize=12)
    ax.set_xlabel(f"{col} raw code", fontsize=10)
    ax.set_ylabel("Default rate (%)", fontsize=10)
    ax.set_ylim(*compute_categorical_axis_limits(data["default_rate"].max()))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"{col.lower()}_default_rates.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_histogram_percentages(
    df: pd.DataFrame, col: str, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate shared bin edges and within-class percentages for a numeric column.
    
    Returns:
        bins: Shared bin edges array
        pct_negative: Percentage array for the negative class
        pct_positive: Percentage array for the positive class
    """
    _require_col(df, col)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
        
    full_data = df[col].dropna()
    _, bins = np.histogram(full_data, bins=n_bins)
    
    neg_data = df.loc[df[TARGET_COL] == NEGATIVE_CLASS, col].dropna()
    pos_data = df.loc[df[TARGET_COL] == POSITIVE_CLASS, col].dropna()
    
    neg_weights = np.ones_like(neg_data) / len(neg_data) * 100 if len(neg_data) > 0 else []
    pos_weights = np.ones_like(pos_data) / len(pos_data) * 100 if len(pos_data) > 0 else []
    
    pct_negative, _ = np.histogram(neg_data, bins=bins, weights=neg_weights)
    pct_positive, _ = np.histogram(pos_data, bins=bins, weights=pos_weights)
    
    return bins, pct_negative, pct_positive

def compute_categorical_axis_limits(data_max_rate: float) -> tuple[float, float]:
    """Determine axis limits for categorical default rate charts."""
    upper = min(100.0, data_max_rate * 100 * 1.35 + 8)
    return 0.0, upper
def _make_labels(data: pd.DataFrame, col: str) -> list[str]:
    """Build x-axis tick labels; mark undocumented codes."""
    labels = []
    for row in data.itertuples():
        lbl = str(row.raw_value)
        lbl += f"\n(n={row.total_count if hasattr(row, 'total_count') else row.count:,})"
        if row.documentation_status == "not_explicitly_defined_in_uci_docs":
            lbl += "\n[undoc]"
        labels.append(lbl)
    return labels


def _require_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame. Available: {list(df.columns)}")

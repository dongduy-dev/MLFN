"""
repayment_status_figures.py
===========================
Figure generation for Checkpoint 2B1 temporal EDA.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from credit_default.eda.static_features import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
)
from credit_default.eda.repayment_status import MONTH_NAMES, KNOWN_STATUS_CODES

DPI = 150
FIGSIZE_WIDE = (10, 6)
FIGSIZE_HEATMAP = (8, 6)
COLOR_DEFAULT = "#D64045"
COLOR_NO_DEFAULT = "#4472C4"

# Deterministic style mapping for the 12 raw status codes
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
    "#bcbd22", "#17becf", "#333333", "#a0a0a0"
]
_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X", "d", "h"]

def get_status_style(code: int) -> dict:
    """Return deterministic styling for a given status code."""
    codes = sorted(KNOWN_STATUS_CODES)
    idx = codes.index(code)
    return {"color": _COLORS[idx], "marker": _MARKERS[idx]}


def plot_status_distribution_lines(
    dist_month: pd.DataFrame,
    out_dir: Path
) -> Path:
    """1. Raw status-code percentage by month, overall (line plot)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # We want a line for each status code across months
    for code in sorted(KNOWN_STATUS_CODES):
        sub = dist_month[dist_month["raw_status_value"] == code].sort_values("chronological_month_index")
        style = get_status_style(code)
        ax.plot(
            sub["month"], 
            sub["population_percentage"], 
            marker=style["marker"], 
            color=style["color"],
            label=f"Status {code}",
            alpha=0.7
        )
        
    ax.set_title("Overall Raw Status-Code Percentage by Month", fontsize=12)
    ax.set_ylabel("Percentage of Total Clients (%)")
    ax.set_ylim(bottom=0)
    # Put legend outside
    ax.legend(title="Raw Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "1_status_dist_overall.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_status_distribution_target(
    dist_target: pd.DataFrame,
    target_class: int,
    title: str,
    filename: str,
    out_dir: Path
) -> Path:
    """2 & 3. Raw status-code percentage by month for a specific target class."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    data = dist_target[dist_target["target_class"] == target_class]
    
    for code in sorted(KNOWN_STATUS_CODES):
        sub = data[data["raw_status_value"] == code].sort_values("chronological_month_index")
        style = get_status_style(code)
        ax.plot(
            sub["month"], 
            sub["percentage_within_target_class"], 
            marker=style["marker"], 
            color=style["color"],
            label=f"Status {code}",
            alpha=0.7
        )
        
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Percentage within Target Class (%)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Raw Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_default_rate_by_status_and_month(
    dist_month: pd.DataFrame,
    out_dir: Path
) -> Path:
    """4. Default rate by status code and month (12 small multiples)."""
    codes = sorted(KNOWN_STATUS_CODES)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, code in enumerate(codes):
        ax = axes[idx]
        sub = dist_month[dist_month["raw_status_value"] == code].sort_values("chronological_month_index")
        style = get_status_style(code)
        
        # Plot observed combination
        sub_obs = sub[sub["observed_combination"]]
        if not sub_obs.empty:
            ax.plot(
                sub_obs["month"], 
                sub_obs["default_rate"] * 100, 
                marker=style["marker"], 
                color=style["color"],
                label=f"Status {code}",
                alpha=0.7
            )
            
            # Annotate small n
            for _, row in sub_obs.iterrows():
                if row["small_sample_warning"]:
                    ax.plot(
                        row["month"], 
                        row["default_rate"] * 100, 
                        marker='o', 
                        markersize=14,
                        markerfacecolor='none',
                        markeredgecolor='red',
                        alpha=0.8
                    )
                    
                    # Prevent clipping by placing text slightly above or below bounds dynamically
                    y_val = row["default_rate"] * 100
                    y_offset = -15 if y_val > 80 else 15
                    
                    ax.annotate(
                        f"n={int(row['total_count'])}",
                        (row["month"], y_val),
                        xytext=(0, y_offset),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='red',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
                    )
        
        # Mark unobserved with X
        sub_unobs = sub[~sub["observed_combination"]]
        for _, row in sub_unobs.iterrows():
            ax.plot(row["month"], 0, marker='x', color='gray', markersize=8)
            ax.annotate("X", (row["month"], 5), ha="center", color="gray", fontsize=8)
            
        ax.set_title(f"Status {code}", fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis='x', rotation=45)
            
    fig.suptitle("Default Rate by Status Code and Month", fontsize=14, y=0.98)
    fig.text(0.5, -0.02, "n < 200 indicates the project caution threshold; interpret cautiously. Gray X = Unobserved.", 
             ha="center", fontsize=10, color="red")
    
    fig.tight_layout()
    path = out_dir / "4_default_rate_by_status.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distribution_heatmap(
    dist_month: pd.DataFrame,
    out_dir: Path
) -> Path:
    """5. Status-code distribution heatmap across months."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    
    codes = sorted(KNOWN_STATUS_CODES)
    matrix = np.zeros((len(codes), len(MONTH_NAMES)))
    
    # Fill matrix
    for i, code in enumerate(codes):
        for j, month in enumerate(MONTH_NAMES):
            row = dist_month[(dist_month["raw_status_value"] == code) & (dist_month["month"] == month)]
            if not row.empty and row.iloc[0]["observed_combination"]:
                matrix[i, j] = row.iloc[0]["population_percentage"]
            else:
                matrix[i, j] = np.nan
                
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    
    ax.set_xticks(np.arange(len(MONTH_NAMES)))
    ax.set_yticks(np.arange(len(codes)))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_yticklabels(codes)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Raw Status Code")
    ax.set_title("Status-Code Distribution Heatmap (% of total)", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Population Percentage (%)")
    
    # Annotate missing/unobserved
    for i in range(len(codes)):
        for j in range(len(MONTH_NAMES)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "X", ha="center", va="center", color="red", fontsize=8)
                
    fig.tight_layout()
    path = out_dir / "5_distribution_heatmap.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_default_rate_heatmap(
    dist_month: pd.DataFrame,
    out_dir: Path
) -> Path:
    """6. Default-rate heatmap across month/status combinations."""
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    
    codes = sorted(KNOWN_STATUS_CODES)
    matrix = np.zeros((len(codes), len(MONTH_NAMES)))
    
    for i, code in enumerate(codes):
        for j, month in enumerate(MONTH_NAMES):
            row = dist_month[(dist_month["raw_status_value"] == code) & (dist_month["month"] == month)]
            if not row.empty and row.iloc[0]["observed_combination"]:
                matrix[i, j] = row.iloc[0]["default_rate"] * 100
            else:
                matrix[i, j] = np.nan
                
    im = ax.imshow(matrix, cmap="Oranges", aspect="auto")
    
    ax.set_xticks(np.arange(len(MONTH_NAMES)))
    ax.set_yticks(np.arange(len(codes)))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_yticklabels(codes)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Raw Status Code")
    ax.set_title("Default-Rate Heatmap (%)", fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Default Rate (%)")
    
    # Annotate missing and small n
    for i, code in enumerate(codes):
        for j, month in enumerate(MONTH_NAMES):
            row = dist_month[(dist_month["raw_status_value"] == code) & (dist_month["month"] == month)]
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "X", ha="center", va="center", color="gray", fontsize=8)
            elif not row.empty and row.iloc[0]["small_sample_warning"]:
                # Indicate small n
                ax.text(j, i, "n<200", ha="center", va="center", color="red", fontsize=8, fontweight="bold")
                
    fig.text(0.5, 0.02, "X = unobserved combination. Red 'n<200' indicates the project caution threshold; interpret cautiously.", 
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = out_dir / "6_default_rate_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_transition_heatmap(
    transitions: pd.DataFrame,
    pair_idx: int,
    title: str,
    filename: str,
    out_dir: Path
) -> Path:
    """7 & 8. Transition heatmap for a specific adjacent month pair."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sub = transitions[transitions["chronological_pair_index"] == pair_idx]
    codes = sorted(KNOWN_STATUS_CODES)
    matrix = np.zeros((len(codes), len(codes)))
    
    source_totals = {}
    
    for i, src in enumerate(codes):
        for j, dst in enumerate(codes):
            row = sub[(sub["source_raw_status"] == src) & (sub["destination_raw_status"] == dst)]
            if not row.empty and row.iloc[0]["observed_combination"]:
                matrix[i, j] = row.iloc[0]["percentage_within_source_status"]
                source_totals[src] = int(row.iloc[0]["source_status_total"])
            else:
                matrix[i, j] = np.nan
                if src not in source_totals and not row.empty:
                    source_totals[src] = int(row.iloc[0]["source_status_total"])
                    
    # Fill any missing source totals with 0 just in case
    for src in codes:
        if src not in source_totals:
            source_totals[src] = 0
                
    im = ax.imshow(matrix, cmap="Purples", aspect="auto")
    
    ax.set_xticks(np.arange(len(codes)))
    ax.set_yticks(np.arange(len(codes)))
    ax.set_xticklabels(codes)
    
    # Create y-tick labels with source denominators
    yticklabels = [f"{src} (n={source_totals[src]})" for src in codes]
    ax.set_yticklabels(yticklabels)
    
    ax.set_xlabel("Destination Status Code")
    ax.set_ylabel("Source Status Code")
    ax.set_title(title, fontsize=12)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of Source Population")
    
    for i in range(len(codes)):
        for j in range(len(codes)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "X", ha="center", va="center", color="gray", fontsize=8)
                
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_top_sequence_patterns(
    patterns: pd.DataFrame,
    sort_col: str,
    title: str,
    filename: str,
    out_dir: Path
) -> Path:
    """9 & 10. Top 10 six-month sequence patterns (bar chart)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # Sort deterministically
    top10 = patterns.sort_values(by=[sort_col, "sequence_pattern"], ascending=[False, True]).head(10)
    
    # Reverse so largest is at the top of the horizontal bar chart
    top10 = top10.iloc[::-1]
    
    bars = ax.barh(top10["sequence_pattern"], top10[sort_col], color="#4C72B0")
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (top10[sort_col].max() * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f"{int(width):,}", 
                va='center')
        
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("Sequence Pattern (Apr | May | Jun | Jul | Aug | Sep)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, top10[sort_col].max() * 1.15)
    
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COL
from credit_default.eda.bill_amount import CHRONOLOGICAL_BILL_COLS, MONTH_MAPPING, compute_histogram_percentages

DPI = 150
FIGSIZE_WIDE = (12, 6)
COLOR_DEFAULT = "#D64045"
COLOR_NO_DEFAULT = "#4472C4"

def _plot_display_range_disclosure(fig: plt.Figure) -> None:
    fig.text(0.5, 0.01, "Display limited to the global 1st-99th percentile range. Values outside the range are not moved into boundary bins and remain included in all calculations.", ha="center", fontsize=9, color="red")

def plot_monthly_median_iqr_overall(monthly: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(monthly["month"], monthly["median"], marker="o", color="black", label="Median")
    ax.fill_between(monthly["month"], monthly["p25"], monthly["p75"], color="gray", alpha=0.3, label="IQR (Q1 to Q3)")
    
    ax.set_title("Monthly Median and IQR Trajectory (Overall)", fontsize=12)
    ax.set_ylabel("Raw BILL_AMT")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "1_median_iqr_trajectory.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

def plot_monthly_median_iqr_by_target(by_target: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    colors = {NEGATIVE_CLASS: COLOR_NO_DEFAULT, POSITIVE_CLASS: COLOR_DEFAULT}
    labels = {NEGATIVE_CLASS: "No Default", POSITIVE_CLASS: "Default"}
    
    for tgt in [NEGATIVE_CLASS, POSITIVE_CLASS]:
        sub = by_target[by_target["target"] == tgt]
        ax.plot(sub["month"], sub["median"], marker="o", color=colors[tgt], label=f"Median ({labels[tgt]})")
        ax.fill_between(sub["month"], sub["q1"], sub["q3"], color=colors[tgt], alpha=0.2, label=f"IQR ({labels[tgt]})")
        
    ax.set_title("Monthly Median and IQR Trajectory (By Target)", fontsize=12)
    ax.set_ylabel("Raw BILL_AMT")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "2_median_iqr_trajectory_by_target.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_six_panel_distributions(df: pd.DataFrame, disp: pd.DataFrame, out_dir: Path) -> Path:
    vmin = disp.iloc[0]["global_1st_percentile"]
    vmax = disp.iloc[0]["global_99th_percentile"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS):
        ax = axes[idx]
        month = MONTH_MAPPING[col]
        vals = df[col].dropna().values
        
        from credit_default.eda.bill_amount import compute_histogram_stats
        edges, counts, below, above = compute_histogram_stats(vals, 40, vmin, vmax)
        
        ax.hist(edges[:-1], bins=edges, weights=counts, color="gray", edgecolor="black")
        
        ax.annotate(f"<{vmin:,.0f}: {below:,}\n>{vmax:,.0f}: {above:,}", xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=8, color="red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        ax.set_title(month, fontsize=10)
        ax.set_ylabel("Count")
        if idx >= 3:
            ax.set_xlabel("Raw BILL_AMT — display range only")
        ax.spines[["top", "right"]].set_visible(False)
        
    fig.suptitle("BILL_AMT Distributions", fontsize=14, y=0.98)
    _plot_display_range_disclosure(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    path = out_dir / "3_distributions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_six_panel_distributions_by_target(df: pd.DataFrame, disp: pd.DataFrame, out_dir: Path) -> Path:
    vmin = disp.iloc[0]["global_1st_percentile"]
    vmax = disp.iloc[0]["global_99th_percentile"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    colors = {NEGATIVE_CLASS: COLOR_NO_DEFAULT, POSITIVE_CLASS: COLOR_DEFAULT}
    labels = {NEGATIVE_CLASS: "No Default", POSITIVE_CLASS: "Default"}
    
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS):
        ax = axes[idx]
        month = MONTH_MAPPING[col]
        
        edges, pct_dict = compute_histogram_percentages(df, col, n_bins=40, vmin=vmin, vmax=vmax)
        
        # Plot as step histograms
        width = np.diff(edges)
        text_lines = []
        for tgt in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            ax.bar(edges[:-1], pct_dict[tgt]["percentages"], width=width, align="edge", alpha=0.5, color=colors[tgt], label=labels[tgt])
            bp = pct_dict[tgt]["below_percentage"]
            ap = pct_dict[tgt]["above_percentage"]
            text_lines.append(f"{labels[tgt]} <{vmin:,.0f}: {bp:.1f}% | >{vmax:,.0f}: {ap:.1f}%")
            
        ax.annotate("\n".join(text_lines), xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=7, color="red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            
        ax.set_title(month, fontsize=10)
        ax.set_ylabel("Within-Class %")
        if idx >= 3:
            ax.set_xlabel("Raw BILL_AMT — display range only")
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend()
            
    fig.suptitle("BILL_AMT Distributions by Target", fontsize=14, y=0.98)
    _plot_display_range_disclosure(fig)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    path = out_dir / "4_distributions_by_target.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_six_panel_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    
    colors = [COLOR_NO_DEFAULT, COLOR_DEFAULT]
    labels = ["No Default", "Default"]
    
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS):
        ax = axes[idx]
        month = MONTH_MAPPING[col]
        
        data_to_plot = [
            df[df[TARGET_COL] == NEGATIVE_CLASS][col].dropna().values,
            df[df[TARGET_COL] == POSITIVE_CLASS][col].dropna().values
        ]
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False, tick_labels=labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        for median in bp['medians']:
            median.set_color("black")
            median.set_linewidth(2)
            
        ax.set_title(month, fontsize=10)
        ax.set_ylabel("Raw BILL_AMT")
        ax.spines[["top", "right"]].set_visible(False)
        
    fig.suptitle("BILL_AMT Boxplots by Target and Month", fontsize=14, y=0.98)
    fig.text(0.5, 0.01, "Individual fliers hidden for visual clarity; potential-extreme counts are reported separately.", ha="center", fontsize=9, color="red")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    path = out_dir / "5_boxplots_by_target.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_sign_proportions(sign_over: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    months = sign_over["month"].unique()
    neg = sign_over[sign_over["sign_category"] == "negative"]["population_percentage"].values
    zero = sign_over[sign_over["sign_category"] == "zero"]["population_percentage"].values
    pos = sign_over[sign_over["sign_category"] == "positive"]["population_percentage"].values
    
    ax.bar(months, neg, label="Negative", color="#ff7f0e")
    ax.bar(months, zero, bottom=neg, label="Zero", color="#7f7f7f")
    ax.bar(months, pos, bottom=neg + zero, label="Positive", color="#1f77b4")
    
    ax.set_title("Proportion of Negative, Zero, and Positive Bill Amounts", fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "6_sign_proportions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_sign_default_rates(sign_over: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    months = sign_over["month"].unique()
    styles = {"negative": ("#ff7f0e", "o"), "zero": ("#7f7f7f", "s"), "positive": ("#1f77b4", "^")}
    
    needs_note = False
    for sign in ["negative", "zero", "positive"]:
        sub = sign_over[sign_over["sign_category"] == sign]
        color, marker = styles[sign]
        
        ax.plot(sub["month"], sub["default_rate"] * 100, marker=marker, color=color, label=sign.capitalize())
        
        # Small sample warnings
        for _, row in sub.iterrows():
            if row["caution_flag"]:
                needs_note = True
                y_val = row["default_rate"] * 100
                ax.plot(row["month"], y_val, marker='o', markersize=14, markerfacecolor='none', markeredgecolor='red')
                ax.annotate(f"n={int(row['total_count'])}", (row["month"], y_val), xytext=(0, 15), textcoords='offset points', ha='center', va='center', fontsize=8, color='red', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                
    ax.set_title("Default Rate by Sign Category", fontsize=12)
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Sign Category")
    if needs_note:
        ax.text(1.05, 0.5, "Red hollow markers & labels\nindicate n < 200 (interpret cautiously)", transform=ax.transAxes, color="red", fontsize=9, va='center')
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "7_sign_default_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_adjacent_change_medians(change_over: pd.DataFrame, change_tgt: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3))
    
    labels = [f"{r.source_month}->{r.destination_month}" for _, r in change_over.iterrows()]
    
    # By target
    sub0 = change_tgt[change_tgt["target"] == NEGATIVE_CLASS]
    sub1 = change_tgt[change_tgt["target"] == POSITIVE_CLASS]
    
    data = [
        change_over["median_change"].values,
        sub0["median_change"].values,
        sub1["median_change"].values
    ]
    
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [[f"{v:,.2f}" for v in row] for row in data]
    row_labels = ["Overall", "No Default", "Default"]
    
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    ax.set_title("Adjacent-Month Median Raw Change Matrix", fontsize=12, pad=20)
    
    fig.tight_layout()
    path = out_dir / "8_adjacent_change_medians.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

def plot_adjacent_change_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
    axes = axes.flatten()
    
    pairs = [
        ("April->May", "BILL_AMT6", "BILL_AMT5"),
        ("May->June", "BILL_AMT5", "BILL_AMT4"),
        ("June->July", "BILL_AMT4", "BILL_AMT3"),
        ("July->Aug", "BILL_AMT3", "BILL_AMT2"),
        ("Aug->Sep", "BILL_AMT2", "BILL_AMT1")
    ]
    
    all_diffs = []
    for _, src, dst in pairs:
        all_diffs.append((df[dst] - df[src]).dropna().values)
        
    flat_diffs = np.concatenate(all_diffs)
    vmin = np.percentile(flat_diffs, 5)
    vmax = np.percentile(flat_diffs, 95)
    
    for idx, (label, src, dst) in enumerate(pairs):
        ax = axes[idx]
        diffs = (df[dst] - df[src]).dropna().values
        
        from credit_default.eda.bill_amount import compute_histogram_stats
        edges, counts, below, above = compute_histogram_stats(diffs, 40, vmin, vmax)
        
        ax.hist(edges[:-1], bins=edges, weights=counts, color="gray", edgecolor="black")
        
        pct_below = (below / len(diffs)) * 100
        pct_above = (above / len(diffs)) * 100
        ax.annotate(f"<{vmin:,.0f}: {pct_below:.1f}%\n>{vmax:,.0f}: {pct_above:.1f}%", xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=8, color="red", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Count")
        if idx >= 2:
            ax.set_xlabel("Raw destination-minus-source change — display range only")
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        
    axes[-1].axis("off")
    fig.suptitle("Adjacent-Month Raw Change Distributions", fontsize=14, y=0.98)
    fig.text(0.5, 0.01, "Display limited for visual clarity. Values outside the visible interval remain included in all calculated summaries.", ha="center", fontsize=9, color="red")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    path = out_dir / "9_adjacent_change_distributions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if -0.5 < corr.iloc[i,j] < 0.5 else "white")
            
    ax.set_title("Six-Month BILL_AMT Pearson Correlation Matrix", fontsize=12)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    
    path = out_dir / "10_correlation_heatmap.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

def plot_potential_extreme_counts(monthly: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(monthly["month"], monthly["total_potential_extreme_count"], color="orange", edgecolor="black")
    
    ax.set_title("Potential Extreme Counts by Month (1.5 × IQR Rule)", fontsize=12)
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "11_potential_extremes.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

def plot_outside_display_range(disp: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    sub = disp[disp["month"] != "Global"]
    ax.plot(sub["month"], sub["outside_range_percentage"], marker="o", color="purple")
    
    ax.set_title("Percentage of Observations Outside Display Range by Month", fontsize=12)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "12_outside_display_range.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

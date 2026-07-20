import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS
from credit_default.eda.bill_payment_relationship import PAIRS

DPI = 300
FIGSIZE_WIDE = (12, 5)
FIGSIZE_LARGE = (14, 8)

def plot_monthly_medians(monthly: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(monthly["month"], monthly["bill_median"], marker='o', label="Bill Median", color="#1f77b4")
    ax.plot(monthly["month"], monthly["payment_median"], marker='s', label="Payment Median", color="#ff7f0e")
    
    ax.set_title("Same-index Monthly Bill vs. Payment Medians\n(no accounting-settlement interpretation is assumed)")
    ax.set_ylabel("NT Dollars")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "1_monthly_medians.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_six_panel_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Calculate global 1st and 99th for bills and payments for consistent display
    bills = df[[p[1] for p in PAIRS]].values.flatten()
    pays = df[[p[2] for p in PAIRS]].values.flatten()
    
    b_vmin, b_vmax = np.percentile(bills[~np.isnan(bills)], 1), np.percentile(bills[~np.isnan(bills)], 99)
    p_vmin, p_vmax = np.percentile(pays[~np.isnan(pays)], 1), np.percentile(pays[~np.isnan(pays)], 99)
    
    for i, (month, bill, pay) in enumerate(PAIRS):
        ax = axes[i]
        b_vals = df[bill]
        p_vals = df[pay]
        
        # Hexbin plot with log scale for density
        hb = ax.hexbin(b_vals, p_vals, gridsize=50, cmap="Blues", bins="log", extent=(b_vmin, b_vmax, p_vmin, p_vmax))
        ax.plot([max(b_vmin, p_vmin), min(b_vmax, p_vmax)], [max(b_vmin, p_vmin), min(b_vmax, p_vmax)], color='red', linestyle='--', alpha=0.5, label='y=x')
        
        ax.set_title(month)
        if i >= 3:
            ax.set_xlabel("Bill Amount")
        if i % 3 == 0:
            ax.set_ylabel("Payment Amount")
            
        ax.set_xlim(b_vmin, b_vmax)
        ax.set_ylim(p_vmin, p_vmax)
        
    fig.suptitle("Same-index Monthly Comparison (Display range limited to 1st-99th percentiles)\nAll values remain in calculations; no accounting-settlement interpretation is assumed.", fontsize=12)
    fig.tight_layout()
    path = out_dir / "2_scatter_density.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_monthly_correlations(monthly: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(monthly["month"], monthly["pearson_correlation"], color="#2ca02c")
    
    ax.set_title("Pearson Correlation Between Raw Bill and Payment Amounts\n(Same-index pairs)")
    ax.set_ylabel("Pearson Correlation (r)")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    
    for i, val in enumerate(monthly["pearson_correlation"]):
        if pd.notna(val):
            ax.text(i, val + 0.02, f"{val:.3f}", ha='center')
            
    fig.tight_layout()
    path = out_dir / "3_monthly_correlations.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_ratio_median_iqr(ratio_over: pd.DataFrame, ratio_tgt: pd.DataFrame, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Overall
    ax1.plot(ratio_over["month"], ratio_over["ratio_median"], marker='o', color="#1f77b4", label="Overall Median")
    ax1.fill_between(ratio_over["month"], ratio_over["ratio_p25"], ratio_over["ratio_p75"], color="#1f77b4", alpha=0.2, label="Overall IQR")
    ax1.set_title("Eligible Positive-Bill Ratio Median and IQR (Overall)")
    ax1.set_ylabel("Ratio (PAY_AMT / BILL_AMT)")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    
    # By Target
    t0 = ratio_tgt[ratio_tgt["target"] == NEGATIVE_CLASS]
    t1 = ratio_tgt[ratio_tgt["target"] == POSITIVE_CLASS]
    
    ax2.plot(t0["month"], t0["ratio_median"], marker='o', color="#1f77b4", label="Target 0 Median")
    ax2.plot(t1["month"], t1["ratio_median"], marker='s', color="#ff7f0e", label="Target 1 Median")
    ax2.set_title("Eligible Positive-Bill Ratio Median (By Target)")
    ax2.set_ylabel("Ratio (PAY_AMT / BILL_AMT)")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    
    fig.suptitle("Ratio calculated only where BILL_AMT > 0", fontsize=12, color="red")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    path = out_dir / "4_positive_bill_ratio_trajectory.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_category_proportions(cats: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    months = cats["month"].unique()
    categories = [
        "bill_positive_payment_positive",
        "bill_positive_payment_zero",
        "bill_nonpositive_payment_positive",
        "bill_nonpositive_payment_zero"
    ]
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    bottoms = np.zeros(len(months))
    for i, cat in enumerate(categories):
        sub = cats[cats["category"] == cat]
        vals = sub["population_percentage"].values
        ax.bar(months, vals, bottom=bottoms, label=cat, color=colors[i])
        bottoms += vals
        
    ax.set_title("Relationship-Category Proportions by Month")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "5_category_proportions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_category_default_rates(cats: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    categories = [
        "bill_positive_payment_positive",
        "bill_positive_payment_zero",
        "bill_nonpositive_payment_positive",
        "bill_nonpositive_payment_zero"
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]
    
    needs_note = False
    
    for i, cat in enumerate(categories):
        sub = cats[cats["category"] == cat]
        if not sub["total_count"].any() > 0:
            continue
            
        ax.plot(sub["month"], sub["default_rate"] * 100, marker=markers[i], color=colors[i], label=cat)
        
        for _, row in sub.iterrows():
            if row["caution_flag"] and pd.notna(row["default_rate"]):
                needs_note = True
                y_val = row["default_rate"] * 100
                ax.plot(row["month"], y_val, marker='o', markersize=14, markerfacecolor='none', markeredgecolor='red')
                ax.annotate(f"n={int(row['total_count'])}", (row["month"], y_val), xytext=(0, 15), textcoords='offset points', ha='center', va='center', fontsize=8, color='red', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                
    ax.set_title("Relationship-Category Observed Default Rates by Month")
    ax.set_ylabel("Default Rate (%)")
    ax.set_ylim(bottom=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if needs_note:
        ax.text(1.05, 0.5, "Red hollow markers & labels\nindicate n < 200 (interpret cautiously)", transform=ax.transAxes, color="red", fontsize=9, va='center')
        
    ax.spines[["top", "right"]].set_visible(False)
    
    fig.tight_layout()
    path = out_dir / "6_category_default_rates.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path

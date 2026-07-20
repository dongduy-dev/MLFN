"""
repayment_status_findings.py
============================
Generate programmatic Markdown report for Checkpoint 2B1 temporal EDA.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS
from credit_default.eda.repayment_status import CHRONOLOGICAL_COLS, MONTH_MAPPING

def generate_repayment_findings_markdown(
    dist_month: pd.DataFrame,
    dist_target: pd.DataFrame,
    transitions: pd.DataFrame,
    patterns: pd.DataFrame,
    out_path: Path
) -> None:
    """Generate reports/eda_repayment_status_findings.md dynamically."""
    
    # 1. Undocumented code frequencies
    undoc = dist_month[dist_month["raw_status_value"].isin([0, -2])]
    undoc_0 = undoc[undoc["raw_status_value"] == 0]["total_count"].sum()
    undoc_m2 = undoc[undoc["raw_status_value"] == -2]["total_count"].sum()
    
    # 2. Sequence stats
    total_patterns = len(patterns)
    top10_patterns = patterns.head(10)
    top10_count = top10_patterns["total_count"].sum()
    top10_pct = (top10_count / 30000) * 100
    
    md = [
        "# Checkpoint 2B1 — Repayment Status Temporal EDA Findings\n\n",
        "**Scope**: Descriptive analysis of repayment statuses over 6 months.\n",
        "**Features Deferred**: `BILL_AMTx` and `PAY_AMTx` are deferred to Checkpoint 2B2.\n\n",
        "## Chronological Mapping\n",
        "The analysis follows this exact chronological mapping based on dataset documentation:\n"
    ]
    
    for col in CHRONOLOGICAL_COLS:
        md.append(f"- **{col}**: {MONTH_MAPPING[col]}\n")
        
    md.extend([
        "\n**Note on PAY_1**: The dataset schema skips `PAY_1`, transitioning directly from `PAY_2` to `PAY_0`.\n",
        "This is a known quirk of the UCI dataset structure and has been preserved programmatically.\n\n",
        "**Chronological sequence**: `PAY_6|PAY_5|PAY_4|PAY_3|PAY_2|PAY_0`\n",
        "**Months**: April → May → June → July → August → September\n\n",
        "## Undocumented Categories\n",
        "Raw code 0 is the most frequent observed repayment-status value. It is present\n",
        "in the raw dataset but is not explicitly defined by the official UCI\n",
        "documentation.\n\n",
        f"Raw code -2 (observed {undoc_m2:,} times in total) is also frequently present\n",
        "but not explicitly defined in the UCI documentation. These codes have been retained\n",
        "without recoding or assumptions of linear progression.\n\n",
        "## Target Class Distributions\n",
        "There are observable associations between status distributions and the target variable:\n"
    ])
    
    # Extract some basic facts for default vs non-default
    for tgt, tgt_lbl in [(NEGATIVE_CLASS, "No Default"), (POSITIVE_CLASS, "Default")]:
        sub = dist_target[(dist_target["target_class"] == tgt) & (dist_target["month"] == "September")]
        top_code = sub.sort_values("percentage_within_target_class", ascending=False).iloc[0]
        md.append(f"- **{tgt_lbl} (September)**: Most common status is `{int(top_code.raw_status_value)}` "
                  f"({top_code.percentage_within_target_class:.1f}%).\n")
                  
    md.extend([
        "\n## Selected Default Rates\n",
        "The following examples highlight the association between raw status codes and default rate:\n\n"
    ])
    
    # Extract examples from the table
    for example_code in [0, 2, 3]:
        sub = dist_month[(dist_month["raw_status_value"] == example_code) & (dist_month["month"] == "September") & (dist_month["observed_combination"])]
        if not sub.empty:
            row = sub.iloc[0]
            warn = " (Caution: n < 200)" if row.small_sample_warning else ""
            md.append(f"- **September, Code {example_code}**: {row.default_count:,} defaults / {row.total_count:,} total = **{row.default_rate*100:.1f}%**{warn}\n")
                  
    md.extend([
        "\n## Sequence Patterns\n",
        f"There are **{total_patterns:,} unique 6-month sequence patterns** observed in the dataset. ",
        f"The top 10 most common patterns account for **{top10_count:,} customers** ({top10_pct:.2f}% of all clients).\n\n",
        "### Most Common Patterns\n"
    ])
    
    overall_top = patterns.sort_values(by=["total_count", "sequence_pattern"], ascending=[False, True]).iloc[0]
    md.append(f"- **Overall**: `{overall_top.sequence_pattern}` (Total: {overall_top.total_count:,}, Defaults: {overall_top.default_count:,}, Rate: {overall_top.default_rate*100:.1f}%)\n")
    
    default_top = patterns.sort_values(by=["default_count", "sequence_pattern"], ascending=[False, True]).iloc[0]
    md.append(f"- **Among Defaults**: `{default_top.sequence_pattern}` (Total: {default_top.total_count:,}, Defaults: {default_top.default_count:,}, Rate: {default_top.default_rate*100:.1f}%)\n")

    nondef_top = patterns.sort_values(by=["non_default_count", "sequence_pattern"], ascending=[False, True]).iloc[0]
    md.append(f"- **Among Non-Defaults**: `{nondef_top.sequence_pattern}` (Total: {nondef_top.total_count:,}, Non-Defaults: {nondef_top.non_default_count:,})\n")
        
    md.extend([
        "\n## Methodological Caveats & Unresolved Concerns\n",
        "- **Caution on Small Samples**: Categories, transitions, or patterns with small sample counts (n < 200) should be interpreted cautiously due to their sensitivity to individual records.\n",
        "- **Unresolved Codes 0 and -2**: The treatment of raw codes 0 and -2 during preprocessing remains unresolved. Future checkpoints should compare defensible representations while preserving the raw values and documenting any transformation.\n",
        "- **Association $\\neq$ Causation**: Changes over time and their correlation with the target are observational. No causal inferences are made.\n",
        "- **Model Expectations**: While this temporal structure supports experimenting with sequence models (e.g., GRU, LSTM, CNN), ",
        "this descriptive analysis does not guarantee they will outperform tabular baselines.\n",
        "- **Deferred Work**: `BILL_AMT` and `PAY_AMT` analysis is deferred to Checkpoint 2B2.\n"
    ])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")

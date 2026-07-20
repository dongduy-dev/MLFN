"""
findings.py
===========
Checkpoint 2A — Programmatic generation of the static EDA findings document.
"""

from pathlib import Path
import pandas as pd

from credit_default.eda.static_features import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    POSITIVE_CLASS,
    NEGATIVE_CLASS,
)

def generate_findings_markdown(
    target_summary: pd.DataFrame,
    numeric_overall: pd.DataFrame,
    numeric_by_target: pd.DataFrame,
    cat_dist: pd.DataFrame,
    cat_rates: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate reports/eda_static_findings.md dynamically from the calculated DataFrames."""
    
    # Target values
    default_row = target_summary[target_summary["class_value"] == POSITIVE_CLASS].iloc[0]
    no_default_row = target_summary[target_summary["class_value"] == NEGATIVE_CLASS].iloc[0]
    imbalance = target_summary["imbalance_ratio_majority_to_minority"].iloc[0]

    md = [
        "# Checkpoint 2A — Static Feature EDA Findings\n",
        "**Scope**: Descriptive analysis of `default payment next month`, `LIMIT_BAL`,\n",
        "`AGE`, `SEX`, `EDUCATION`, and `MARRIAGE`.  \n",
        f"**Dataset**: Complete raw dataset — {int(target_summary['count'].sum()):,} clients, no modifications.  \n",
        "**Temporal features deferred**: PAY_x, BILL_AMTx, and PAY_AMTx columns are\n",
        "intentionally excluded from this checkpoint and will be analyzed in Checkpoint 2B.  \n",
        "**No model evaluation has occurred.** These findings describe observed associations\n",
        "in the raw data only.\n\n",
        "All numerical values below are taken directly from the generated CSV tables\n",
        "in `reports/tables/eda/static/`. Those CSVs are the authoritative\n",
        "machine-readable source.\n\n",
        "---\n\n",
        "## 1. Target Distribution\n\n",
        "| Class value | Class label | Count | Percentage |\n",
        "|:-----------:|:-----------:|------:|----------:|\n",
        f"| {NEGATIVE_CLASS} | No default | {int(no_default_row['count']):,} | {no_default_row['percentage']:.2f}% |\n",
        f"| {POSITIVE_CLASS} | **Default** | {int(default_row['count']):,} | {default_row['percentage']:.2f}% |\n\n",
        f"- **Positive class**: value `{POSITIVE_CLASS}` (default next month).\n",
        f"- **Imbalance ratio** (majority : minority): **{imbalance:.2f} : 1**.\n",
        "- The dataset is moderately imbalanced. Any future classifier evaluation should\n",
        "  account for this imbalance; metrics such as accuracy alone will be misleading.\n",
        "  (*Note: no resampling is applied in this checkpoint.*)\n\n",
        "---\n\n",
    ]

    for col in NUMERIC_COLS:
        col_overall = numeric_overall[numeric_overall["feature"] == col].iloc[0]
        col_tgt = numeric_by_target[numeric_by_target["feature"] == col]
        
        md.extend([
            f"## {col}\n\n",
            "### Overall statistics\n",
            "| Statistic | Value |\n",
            "|-----------|------:|\n",
            f"| Count | {int(col_overall['count']):,} |\n",
            f"| Mean | {col_overall['mean']:,.4f} |\n",
            f"| Std | {col_overall['std']:,.4f} |\n",
            f"| Min | {col_overall['min']:,.4f} |\n",
            f"| Q25 | {col_overall['q25']:,.4f} |\n",
            f"| Median | {col_overall['median']:,.4f} |\n",
            f"| Q75 | {col_overall['q75']:,.4f} |\n",
            f"| Max | {col_overall['max']:,.4f} |\n",
            f"| IQR | {col_overall['iqr']:,.4f} |\n\n",
            "### By target group\n",
            "| Group | Count | Mean | Median | Std |\n",
            "|-------|------:|-----:|-------:|----:|\n"
        ])
        
        no_def = col_tgt[col_tgt["target_value"] == NEGATIVE_CLASS].iloc[0]
        def_row = col_tgt[col_tgt["target_value"] == POSITIVE_CLASS].iloc[0]
        
        md.append(f"| No default | {int(no_def['count']):,} | {no_def['mean']:,.4f} | {no_def['median']:,.4f} | {no_def['std']:,.4f} |\n")
        md.append(f"| Default | {int(def_row['count']):,} | {def_row['mean']:,.4f} | {def_row['median']:,.4f} | {def_row['std']:,.4f} |\n\n")

        # Specific notes for LIMIT_BAL and AGE
        if col == "LIMIT_BAL":
            extreme_no_def = int(no_def['potential_extreme_note'].split()[0])
            extreme_def = int(def_row['potential_extreme_note'].split()[0])
            md.extend([
                f"- Clients who **defaulted** had a **lower mean credit limit** ({def_row['mean']:,.0f}) compared\n",
                f"  to those who did not ({no_def['mean']:,.0f}). The medians differ by {no_def['median'] - def_row['median']:,.0f} ({def_row['median']:,.0f} vs {no_def['median']:,.0f}).\n",
                "- Both groups show right-skewed distributions. The 3×IQR fence analysis found\n",
                f"  **{extreme_no_def} potential extreme values** in the No Default group and **{extreme_def}** in the\n",
                "  Default group; all are retained as-is.\n",
                "- This is a descriptive association only; lower credit limits do not necessarily\n",
                "  cause default.\n\n",
            ])
        elif col == "AGE":
            md.extend([
                f"- The age distributions of defaulters and non-defaulters are very similar.\n",
                f"  Mean difference is approximately **{abs(def_row['mean'] - no_def['mean']):.2f} years** ({def_row['mean']:.2f} vs {no_def['mean']:.2f}), with\n",
                f"  identical medians ({def_row['median']:,.0f}) for both groups.\n",
                "- No extreme values were flagged by the 3×IQR check.\n",
                f"- The dataset's recorded age range is {col_overall['min']:,.0f}–{col_overall['max']:,.0f} years.\n\n",
            ])
        md.append("---\n\n")

    for idx, col in enumerate(CATEGORICAL_COLS):
        col_dist = cat_dist[cat_dist["feature"] == col]
        col_rate = cat_rates[cat_rates["feature"] == col]
        
        md.extend([
            f"## {idx + 4}. {col}\n\n",
            "| Raw code | Count | % of total | Default count | Default rate | N | Note |\n",
            "|:--------:|------:|----------:|:-------------:|:------------:|--:|:-----|\n"
        ])
        
        for _, row in col_rate.iterrows():
            note = []
            if row['documentation_status'] == 'not_explicitly_defined_in_uci_docs':
                note.append("Not explicitly defined in UCI docs")
            if row['small_sample_warning']:
                note.append("n < 200")
            note_str = "; ".join(note)
            if note_str:
                note_str = f"**{note_str}**"
                
            md.append(f"| {row['raw_value']} | {int(row['total_count']):,} | {row['population_percentage']:.2f}% | {int(row['default_count']):,} | **{row['default_rate']*100:.1f}%** | {int(row['total_count']):,} | {note_str} |\n")
        
        md.append("\n")
        
        # Add descriptive text for categorical features
        if col == "SEX":
            md.extend([
                "- Both categories contain many observations, so their observed rates are less\n",
                "  sensitive to individual records than those of the flagged small categories.\n",
                "- Male clients show a slightly higher observed default rate than female\n",
                "  clients. This is a descriptive association; no causal interpretation\n",
                "  is made.\n",
                "- Both codes are explicitly documented in the UCI description.\n\n",
            ])
        elif col == "EDUCATION":
            md.extend([
                "**Undocumented categories**: values are present in the\n",
                "raw data but are not explicitly defined by the official UCI documentation. They\n",
                "are retained as-is; no recoding has been performed.\n\n",
                "Categories with fewer than 200 observations are flagged under the project's\n",
                "caution threshold. Their rates should be interpreted carefully because they\n",
                "are more sensitive to individual observations.\n\n",
                "Among the large documented categories, there is a visible gradient. However,\n",
                "this pattern should not be interpreted causally, and feature utility should not\n",
                "be assessed before modelling.\n\n",
            ])
        elif col == "MARRIAGE":
            md.extend([
                "**Undocumented category**: values are not explicitly defined in the\n",
                "UCI documentation. They are retained and annotated.\n\n",
                "Categories with fewer than 200 observations are flagged under the project's\n",
                "caution threshold. Their rates should be interpreted carefully because they\n",
                "are more sensitive to individual observations.\n\n",
                "These are associations only.\n\n"
            ])
        md.append("---\n\n")

    md.extend([
        "## Summary of Undocumented and Small-Sample Categories\n\n",
        "| Feature | Value | Count | Note |\n",
        "|---------|------:|------:|------|\n"
    ])
    
    for col in CATEGORICAL_COLS:
        col_rate = cat_rates[cat_rates["feature"] == col]
        for _, row in col_rate.iterrows():
            if row['documentation_status'] == 'not_explicitly_defined_in_uci_docs' or row['small_sample_warning']:
                note = []
                if row['documentation_status'] == 'not_explicitly_defined_in_uci_docs':
                    note.append("Not explicitly defined in UCI docs")
                if row['small_sample_warning']:
                    note.append("small sample")
                note_str = "; ".join(note)
                md.append(f"| {col} | {row['raw_value']} | {int(row['total_count']):,} | {note_str} |\n")

    md.extend([
        "\nNone of these values have been removed or merged. Their handling must be\n",
        "explicitly decided in Checkpoint 3 (preprocessing).\n\n",
        "---\n\n",
        "## Methodological Notes\n\n",
        f"- This EDA uses the **complete raw dataset** (n={int(target_summary['count'].sum()):,}) for descriptive\n",
        "  understanding. No train/validation/test split has been applied.\n",
        "- No model evaluation has occurred.\n",
        "- Temporal features (PAY_x, BILL_AMTx, PAY_AMTx) are **intentionally deferred**\n",
        "  to Checkpoint 2B.\n",
        "- Associations between features and the target are reported, not causal\n",
        "  relationships.\n",
        "- Feature importance or utility for prediction has **not** been assessed here.\n",
        "  That requires model-based evaluation in later checkpoints.\n",
        "- No preprocessing, encoding, scaling, imputation, or resampling was performed.\n"
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")

import pandas as pd
from pathlib import Path
from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS
from credit_default.eda.payment_amount import CHRONOLOGICAL_PAY_COLS, MONTH_MAPPING

def generate_payment_amount_findings(
    monthly: pd.DataFrame,
    by_target: pd.DataFrame,
    sign_over: pd.DataFrame,
    sign_tgt: pd.DataFrame,
    change_over: pd.DataFrame,
    change_tgt: pd.DataFrame,
    corr: pd.DataFrame,
    disp: pd.DataFrame,
    anchors: pd.DataFrame,
    out_path: Path
) -> None:
    
    md = [
        "# Checkpoint 2B2B1 — Temporal EDA Findings (Payment Amounts)\n\n",
        "**Scope**: Descriptive analysis of the six previous-payment amount columns (`PAY_AMT1` to `PAY_AMT6`) over 6 months.\n",
        "**Features Deferred**: `BILL_AMT`/`PAY_AMT` relationship analysis is deferred to Checkpoint 2B2B2.\n\n",
        
        "## Chronological Mapping\n",
        "The analysis follows this exact chronological mapping based on dataset documentation:\n"
    ]
    
    for col in CHRONOLOGICAL_PAY_COLS:
        md.append(f"- **{col}**: {MONTH_MAPPING[col]}\n")
        
    md.append("\n**Chronological sequence**: `PAY_AMT6` → `PAY_AMT5` → `PAY_AMT4` → `PAY_AMT3` → `PAY_AMT2` → `PAY_AMT1`\n")
    md.append("**Months**: April → May → June → July → August → September\n\n")
    
    md.extend([
        "## Overall Monthly Medians and Spread\n",
        "Payment amounts exhibit right skewness, as evidenced by means consistently exceeding medians. ",
        "All calculations maintain the 30,000 raw rows without filtering or outlier removal.\n\n"
    ])
    
    for _, row in monthly.iterrows():
        md.append(f"- **{row['month']}**: Mean = {row['mean']:,.4f}, Q1 = {row['p25']:,.2f}, Median = {row['median']:,.2f}, Q3 = {row['p75']:,.2f}, IQR = {row['iqr']:,.2f}\n")
    md.append("\n")
    
    md.extend([
        "## Target Group Comparisons\n"
    ])
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        def_row = by_target[(by_target["month"] == month) & (by_target["target"] == POSITIVE_CLASS)].iloc[0]
        nodef_row = by_target[(by_target["month"] == month) & (by_target["target"] == NEGATIVE_CLASS)].iloc[0]
        md.append(f"- **{month}**: Target 0 (n={int(nodef_row['count']):,}) Median = {nodef_row['median']:,.2f} | Target 1 (n={int(def_row['count']):,}) Median = {def_row['median']:,.2f}\n")
    md.append("\n")
    
    md.extend([
        "## Sign Category Findings\n",
        "No negative PAY_AMT values were observed in any month. Zero and positive values are reported descriptively without assigning business interpretations to zero payments.\n\n"
    ])
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        s_neg = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "negative")]
        s_zero = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "zero")]
        s_pos = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "positive")]
        
        md.extend([
            f"**{month}**:\n",
            f"- Negative: {int(s_neg.iloc[0]['total_count']):,} (Default rate: {s_neg.iloc[0]['default_rate']*100:.1f}%)\n" if not s_neg.empty and pd.notna(s_neg.iloc[0]['default_rate']) else f"- Negative: 0\n",
            f"- Zero: {int(s_zero.iloc[0]['total_count']):,} (Default rate: {s_zero.iloc[0]['default_rate']*100:.1f}%)\n" if not s_zero.empty and pd.notna(s_zero.iloc[0]['default_rate']) else f"- Zero: 0\n",
            f"- Positive: {int(s_pos.iloc[0]['total_count']):,} (Default rate: {s_pos.iloc[0]['default_rate']*100:.1f}%)\n\n" if not s_pos.empty and pd.notna(s_pos.iloc[0]['default_rate']) else f"- Positive: 0\n\n"
        ])
        
    md.extend([
        "## Potential Extremes\n",
        "Values falling outside $Q1 - 1.5 \\times IQR$ or $Q3 + 1.5 \\times IQR$ are flagged as potential extreme under the 1.5 × IQR rule:\n"
    ])
    
    for _, row in monthly.iterrows():
        md.append(f"- **{row['month']}**: {int(row['total_potential_extreme_count']):,} potential extremes\n")
    md.append("These observations are not dropped.\n\n")
    
    md.extend([
        "## Adjacent-Month Raw Changes\n",
        "Changes are computed as $Destination - Source$.\n"
    ])
    
    for _, row in change_over.iterrows():
        sm = row["source_month"]
        dm = row["destination_month"]
        
        tgt0 = change_tgt[(change_tgt["source_month"] == sm) & (change_tgt["destination_month"] == dm) & (change_tgt["target"] == NEGATIVE_CLASS)].iloc[0]
        tgt1 = change_tgt[(change_tgt["source_month"] == sm) & (change_tgt["destination_month"] == dm) & (change_tgt["target"] == POSITIVE_CLASS)].iloc[0]
        
        md.append(f"- **{sm} → {dm}**: Overall Median = {row['median_change']:,.2f} | Target 0 = {tgt0['median_change']:,.2f} | Target 1 = {tgt1['median_change']:,.2f}\n")
        
    md.append("\nThis is a descriptive property of the observed change distributions and does not by itself establish a business explanation.\n\n")
    
    md.extend([
        "## Correlation matrix\n",
        "Cross-month correlations between payment amounts:\n\n"
    ])
    
    apr_may = corr.loc["April", "May"]
    aug_sep = corr.loc["August", "September"]
    apr_sep = corr.loc["April", "September"]
    
    md.extend([
        f"- **April–May**: {apr_may:.4f}\n",
        f"- **August–September**: {aug_sep:.4f}\n",
        f"- **April–September**: {apr_sep:.4f}\n\n"
    ])
    
    glob = disp[disp["month"] == "Global"].iloc[0]
    vmin = glob["global_1st_percentile"]
    vmax = glob["global_99th_percentile"]
    
    md.extend([
        "## Display-Only Range Disclosure\n",
        f"Some distribution figures limit the displayed range to the global 1st ({vmin:,.2f}) and 99th ({vmax:,.2f}) percentiles ",
        f"for visual legibility. There are {int(glob['outside_range_count']):,} of {int(glob['total_count']):,} observations outside this range (approximately {glob['outside_range_percentage']:.4f}%).\n\n"
    ])
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        m_disp = disp[disp["month"] == month].iloc[0]
        md.append(f"- **{month}**: {int(m_disp['outside_range_count']):,} outside range ({m_disp['outside_range_percentage']:.4f}%)\n")
    md.append("\nValues outside the range are not moved into boundary bins and remain included in all calculations.\n\n")
    
    n_anchors = len(anchors)
    n_passed = anchors["passed"].sum()
    
    md.extend([
        "## Methodological Caveats\n",
        "- **Association vs Causation**: Association does not imply causation.\n",
        "- **Modelling Implications**: Payment history provides ordered information suitable for sequence-model experiments but does not prove sequence models will outperform tabular baselines.\n",
        f"- **Validation Anchors**: {n_passed}/{n_anchors} anchors passed independent validation tests.\n"
    ])
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(md)

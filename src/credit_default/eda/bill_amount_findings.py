import pandas as pd
from pathlib import Path

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS
from credit_default.eda.bill_amount import CHRONOLOGICAL_BILL_COLS, MONTH_MAPPING

def generate_bill_amount_findings(
    monthly: pd.DataFrame,
    by_target: pd.DataFrame,
    sign_over: pd.DataFrame,
    sign_tgt: pd.DataFrame,
    change_over: pd.DataFrame,
    change_tgt: pd.DataFrame,
    corr: pd.DataFrame,
    disp: pd.DataFrame,
    out_path: Path
) -> None:
    
    md = [
        "# Checkpoint 2B2A — Temporal EDA Findings (Bill Amounts)\n\n",
        "**Scope**: Descriptive analysis of the six bill-statement amount columns (`BILL_AMT1` to `BILL_AMT6`) over 6 months.\n",
        "**Features Deferred**: `PAY_AMTx` analysis and bill/payment relationships are deferred to Checkpoint 2B2B.\n\n",
        
        "## Chronological Mapping\n",
        "The analysis follows this exact chronological mapping based on dataset documentation:\n"
    ]
    
    for col in CHRONOLOGICAL_BILL_COLS:
        md.append(f"- **{col}**: {MONTH_MAPPING[col]}\n")
        
    md.append("\n**Chronological sequence**: `BILL_AMT6` → `BILL_AMT5` → `BILL_AMT4` → `BILL_AMT3` → `BILL_AMT2` → `BILL_AMT1`\n")
    md.append("**Months**: April → May → June → July → August → September\n\n")
    
    md.extend([
        "## Overall Monthly Medians and Spread\n",
        "Bill amounts exhibit right skewness, as evidenced by means consistently exceeding medians. ",
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
        "Negative and zero values exist as raw entries and are recorded descriptively without inferring business definitions.\n\n"
    ])
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        s_neg = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "negative")].iloc[0]
        s_zero = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "zero")].iloc[0]
        s_pos = sign_over[(sign_over["month"] == month) & (sign_over["sign_category"] == "positive")].iloc[0]
        
        md.extend([
            f"**{month}**:\n",
            f"- Negative: {int(s_neg['total_count']):,} (Default rate: {s_neg['default_rate']*100:.1f}%)\n",
            f"- Zero: {int(s_zero['total_count']):,} (Default rate: {s_zero['default_rate']*100:.1f}%)\n",
            f"- Positive: {int(s_pos['total_count']):,} (Default rate: {s_pos['default_rate']*100:.1f}%)\n\n"
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
        
    md.append("\nEvery adjacent-month overall median raw change is zero. This is a descriptive property of the observed change distributions and does not by itself establish a business explanation.\n\n")
    
    md.extend([
        "## Correlation matrix\n",
        "Adjacent months typically exhibit strong positive correlations, decaying gradually over longer temporal distances.\n\n"
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
        f"Some distribution figures clip the axes between the global 1st ({vmin:,.2f}) and 99th ({vmax:,.2f}) percentiles ",
        f"for visual legibility. There are {int(glob['outside_range_count']):,} of {int(glob['total_count']):,} observations outside this range (approximately {glob['outside_range_percentage']:.4f}%).\n\n"
    ])
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        m_disp = disp[disp["month"] == month].iloc[0]
        md.append(f"- **{month}**: {int(m_disp['outside_range_count']):,} outside range ({m_disp['outside_range_percentage']:.4f}%)\n")
    md.append("\nValues outside the range are not moved into boundary bins and remain included in all calculations.\n\n")
    
    md.extend([
        "## Methodological Caveats\n",
        "- **Association $\\neq$ Causation**: Changes over time and correlations are observational.\n",
        "- **Model Expectations**: Bill history provides ordered monthly information suitable for later sequence-model experiments ",
        "but does not prove sequence models will outperform tabular baselines.\n"
    ])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")

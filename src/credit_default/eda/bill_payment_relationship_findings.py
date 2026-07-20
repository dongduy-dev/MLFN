import pandas as pd
from pathlib import Path
from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS

def generate_relationship_findings(
    monthly: pd.DataFrame,
    ratio_over: pd.DataFrame,
    ratio_tgt: pd.DataFrame,
    cats: pd.DataFrame,
    out_path: Path
) -> None:
    md = []
    
    md.extend([
        "# Accelerated Phase 1 \u2014 Bill/Payment Relationship EDA\n\n",
        "## Scope and Constraints\n",
        "This phase analyzes same-index monthly bill and previous-payment amount pairs (e.g., BILL_AMT6 paired with PAY_AMT6 for April).\n",
        "**Important Caveats:**\n",
        "- This is a purely numerical same-index comparison.\n",
        "- No causal or accounting-settlement interpretation is assumed. We do not claim that PAY_AMTx necessarily settles BILL_AMTx.\n",
        "- All EDA is strictly descriptive. No train/validation/test splitting, no preprocessing for modelling, and no model training has begun.\n\n"
    ])
    
    md.append("## Chronological Same-Index Pairs\n")
    md.append("- **April**: BILL_AMT6 with PAY_AMT6\n")
    md.append("- **May**: BILL_AMT5 with PAY_AMT5\n")
    md.append("- **June**: BILL_AMT4 with PAY_AMT4\n")
    md.append("- **July**: BILL_AMT3 with PAY_AMT3\n")
    md.append("- **August**: BILL_AMT2 with PAY_AMT2\n")
    md.append("- **September**: BILL_AMT1 with PAY_AMT1\n\n")
    
    md.append("## Monthly Pearson Correlations\n")
    for _, row in monthly.iterrows():
        md.append(f"- **{row['month']}**: {row['pearson_correlation']:.4f}\n")
    md.append("\n")
    
    md.append("## Eligible Positive-Bill Ratios\n")
    md.append("> **Note:** The ratio (PAY_AMT / BILL_AMT) is calculated *only* where BILL_AMT > 0. Ratios strictly exclude nonpositive bills to avoid infinite or undefined values. Because very small positive bills can create extreme ratios, the IQR and percentiles provide a more robust summary than the mean.\n\n")
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        r_row = ratio_over[ratio_over["month"] == month].iloc[0]
        t0_row = ratio_tgt[(ratio_tgt["month"] == month) & (ratio_tgt["target"] == NEGATIVE_CLASS)].iloc[0]
        t1_row = ratio_tgt[(ratio_tgt["month"] == month) & (ratio_tgt["target"] == POSITIVE_CLASS)].iloc[0]
        
        md.append(f"### {month}\n")
        md.append(f"- **Positive-Bill Eligibility:** {int(r_row['eligible_count']):,} of 30,000 ({r_row['eligible_percentage']:.2f}%)\n")
        
        iqr = r_row['ratio_p75'] - r_row['ratio_p25']
        md.append(f"- **Overall Ratio:** Median = {r_row['ratio_median']:.4f} (IQR: {iqr:.4f})\n")
        
        t0_iqr = t0_row['ratio_p75'] - t0_row['ratio_p25']
        t1_iqr = t1_row['ratio_p75'] - t1_row['ratio_p25']
        md.append(f"- **By Target:** Target 0 Median = {t0_row['ratio_median']:.4f} (IQR: {t0_iqr:.4f}) | Target 1 Median = {t1_row['ratio_median']:.4f} (IQR: {t1_iqr:.4f})\n\n")
        
    md.append("## Relationship Categories and Default Rates\n")
    md.append("Observations are assigned to mutually exclusive raw numerical categories. The sum of categories across each month is exactly 30,000.\n\n")
    
    categories = [
        "bill_positive_payment_positive",
        "bill_positive_payment_zero",
        "bill_nonpositive_payment_positive",
        "bill_nonpositive_payment_zero"
    ]
    
    for month in ["April", "May", "June", "July", "August", "September"]:
        md.append(f"### {month}\n")
        m_cats = cats[cats["month"] == month]
        
        for cat in categories:
            row = m_cats[m_cats["category"] == cat]
            if not row.empty:
                r = row.iloc[0]
                cnt = int(r["total_count"])
                if cnt > 0:
                    d_rate = r["default_rate"] * 100
                    warn = " *(Warning: n < 200)*" if cnt < 200 else ""
                    md.append(f"- **{cat}**: n = {cnt:,} (Default rate: {d_rate:.2f}%){warn}\n")
                else:
                    md.append(f"- **{cat}**: n = 0\n")
        
        # Zero-payment findings with denominators
        m_summary = monthly[monthly["month"] == month].iloc[0]
        z_cnt = int(m_summary["zero_payment_count"])
        md.append(f"- **Total Zero-Payment Occurrences**: {z_cnt:,} of 30,000\n\n")

    md.append("## Conclusion\n")
    md.append("Descriptive Exploratory Data Analysis (EDA) is now entirely complete for static features, repayment statuses, bill amounts, previous-payment amounts, and same-index relationships.\n")
    md.append("Leakage-safe dataset splitting and preprocessing will be conducted in subsequent checkpoints. **No splitting, preprocessing, or modelling has occurred.**\n")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")

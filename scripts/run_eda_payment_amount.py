import pandas as pd
import sys
from pathlib import Path

from credit_default.data_loader import load_raw_dataset, compute_file_sha256
from credit_default.eda.payment_amount import (
    validate_chronological_schema,
    generate_schema_table,
    compute_monthly_summary,
    compute_by_target_summary,
    compute_sign_summary,
    compute_change_summary,
    compute_correlation_matrix,
    compute_display_range,
    generate_regression_anchors,
    run_quality_checks
)
from credit_default.eda.payment_amount_figures import (
    plot_monthly_median_iqr_overall,
    plot_monthly_median_iqr_by_target,
    plot_six_panel_distributions,
    plot_six_panel_distributions_by_target,
    plot_six_panel_boxplots,
    plot_sign_proportions,
    plot_sign_default_rates,
    plot_adjacent_change_medians,
    plot_adjacent_change_distributions,
    plot_correlation_heatmap,
    plot_potential_extreme_counts,
    plot_outside_display_range
)
from credit_default.eda.payment_amount_findings import generate_payment_amount_findings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables" / "eda" / "payment_amount"
FIGURES_DIR = REPORTS_DIR / "figures" / "eda" / "payment_amount"

def section(title: str):
    print("\n" + "="*72)
    print(f"  {title}")
    print("="*72)

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    section("1. LOAD & VALIDATE SCHEMA")
    df = load_raw_dataset()
    sha256 = compute_file_sha256()
    print(f"  Raw SHA-256: {sha256}")
    
    validate_chronological_schema(df)
    schema = generate_schema_table(df)
    schema.to_csv(TABLES_DIR / "payment_amount_schema.csv", index=False)
    print("  Saved: payment_amount_schema.csv")
    print("  Chronological schema validated.")
    
    section("2. COMPUTE SUMMARIES")
    disp = compute_display_range(df)
    disp.to_csv(TABLES_DIR / "payment_amount_display_range.csv", index=False)
    print("  Saved: payment_amount_display_range.csv")
    
    monthly = compute_monthly_summary(df)
    monthly.to_csv(TABLES_DIR / "payment_amount_monthly_summary.csv", index=False)
    print("  Saved: payment_amount_monthly_summary.csv")
    
    by_target = compute_by_target_summary(df)
    by_target.to_csv(TABLES_DIR / "payment_amount_by_target_summary.csv", index=False)
    print("  Saved: payment_amount_by_target_summary.csv")
    
    sign_over, sign_tgt = compute_sign_summary(df)
    sign_over.to_csv(TABLES_DIR / "payment_amount_sign_summary.csv", index=False)
    print("  Saved: payment_amount_sign_summary.csv")
    
    sign_tgt.to_csv(TABLES_DIR / "payment_amount_sign_by_target.csv", index=False)
    print("  Saved: payment_amount_sign_by_target.csv")
    
    section("3. COMPUTE CHANGES")
    change_over, change_tgt = compute_change_summary(df)
    change_over.to_csv(TABLES_DIR / "payment_amount_change_summary.csv", index=False)
    print("  Saved: payment_amount_change_summary.csv")
    
    change_tgt.to_csv(TABLES_DIR / "payment_amount_change_by_target.csv", index=False)
    print("  Saved: payment_amount_change_by_target.csv")
    
    section("4. COMPUTE CORRELATIONS")
    corr = compute_correlation_matrix(df)
    corr.to_csv(TABLES_DIR / "payment_amount_correlation_matrix.csv")
    print("  Saved: payment_amount_correlation_matrix.csv")
    
    section("5. REGRESSION ANCHORS")
    anchors = generate_regression_anchors(df, monthly, change_over, disp)
    anchors.to_csv(TABLES_DIR / "payment_amount_regression_anchors.csv", index=False)
    print("  Saved: payment_amount_regression_anchors.csv")
    
    section("6. QUALITY CHECKS")
    checks = run_quality_checks(
        df, schema, monthly, by_target, sign_over, sign_tgt, 
        change_over, change_tgt, corr, disp, anchors
    )
    checks.to_csv(TABLES_DIR / "payment_amount_quality_checks.csv", index=False)
    print("  Saved: payment_amount_quality_checks.csv\n")
    
    passed = checks["passed"].sum()
    total = len(checks)
    print(f"  Quality checks: {passed}/{total} passed")
    
    if passed < total:
        print("\n[!] QUALITY CHECK FAILED. Exiting.")
        sys.exit(1)
        
    section("7. GENERATE FIGURES")
    paths = [
        plot_monthly_median_iqr_overall(monthly, FIGURES_DIR),
        plot_monthly_median_iqr_by_target(by_target, FIGURES_DIR),
        plot_six_panel_distributions(df, disp, FIGURES_DIR),
        plot_six_panel_distributions_by_target(df, disp, FIGURES_DIR),
        plot_six_panel_boxplots(df, FIGURES_DIR),
        plot_sign_proportions(sign_over, FIGURES_DIR),
        plot_sign_default_rates(sign_over, FIGURES_DIR),
        plot_adjacent_change_medians(change_over, change_tgt, FIGURES_DIR),
        plot_adjacent_change_distributions(df, FIGURES_DIR),
        plot_correlation_heatmap(corr, FIGURES_DIR),
        plot_potential_extreme_counts(monthly, FIGURES_DIR),
        plot_outside_display_range(disp, FIGURES_DIR)
    ]
    for p in paths:
        print(f"  {p.name}")
        
    section("8. GENERATE FINDINGS")
    out_md = REPORTS_DIR / "eda_payment_amount_findings.md"
    generate_payment_amount_findings(monthly, by_target, sign_over, sign_tgt, change_over, change_tgt, corr, disp, anchors, out_md)
    print(f"  Saved: {out_md}")
    
    section("CHECKPOINT 2B2B1 COMPLETE")
    print(f"  Tables : {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}\n")

if __name__ == "__main__":
    main()

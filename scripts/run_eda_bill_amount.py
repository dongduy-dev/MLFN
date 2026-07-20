import sys
from pathlib import Path

from credit_default.data_loader import load_raw_dataset, compute_file_sha256
from credit_default.eda.bill_amount import (
    validate_chronological_schema,
    generate_schema_table,
    compute_monthly_summary,
    compute_by_target_summary,
    compute_sign_summary,
    compute_change_summary,
    compute_correlation_matrix,
    compute_display_range,
    run_quality_checks
)
from credit_default.eda.bill_amount_figures import (
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
from credit_default.eda.bill_amount_findings import generate_bill_amount_findings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables" / "eda" / "bill_amount"
FIGURES_DIR = REPORTS_DIR / "figures" / "eda" / "bill_amount"

def section(title: str) -> None:
    print(f"\n{'='*72}\n  {title}\n{'='*72}")

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    section("1. LOAD & VALIDATE SCHEMA")
    sha256 = compute_file_sha256()
    print(f"  Raw SHA-256: {sha256}")
    
    df = load_raw_dataset()
    validate_chronological_schema(df)
    
    schema = generate_schema_table(df)
    schema.to_csv(TABLES_DIR / "bill_amount_schema.csv", index=False)
    print("  Saved: bill_amount_schema.csv")
    print("  Chronological schema validated.")
    
    section("2. COMPUTE SUMMARIES")
    disp = compute_display_range(df)
    disp.to_csv(TABLES_DIR / "bill_amount_display_range.csv", index=False)
    print("  Saved: bill_amount_display_range.csv")
    
    monthly = compute_monthly_summary(df)
    monthly.to_csv(TABLES_DIR / "bill_amount_monthly_summary.csv", index=False)
    print("  Saved: bill_amount_monthly_summary.csv")
    
    by_target = compute_by_target_summary(df)
    by_target.to_csv(TABLES_DIR / "bill_amount_by_target_summary.csv", index=False)
    print("  Saved: bill_amount_by_target_summary.csv")
    
    sign_over, sign_tgt = compute_sign_summary(df)
    sign_over.to_csv(TABLES_DIR / "bill_amount_sign_summary.csv", index=False)
    sign_tgt.to_csv(TABLES_DIR / "bill_amount_sign_by_target.csv", index=False)
    print("  Saved: bill_amount_sign_summary.csv")
    print("  Saved: bill_amount_sign_by_target.csv")
    
    section("3. COMPUTE CHANGES")
    change_over, change_tgt = compute_change_summary(df)
    change_over.to_csv(TABLES_DIR / "bill_amount_change_summary.csv", index=False)
    change_tgt.to_csv(TABLES_DIR / "bill_amount_change_by_target.csv", index=False)
    print("  Saved: bill_amount_change_summary.csv")
    print("  Saved: bill_amount_change_by_target.csv")
    
    section("4. COMPUTE CORRELATIONS")
    corr = compute_correlation_matrix(df)
    corr.to_csv(TABLES_DIR / "bill_amount_correlation_matrix.csv", index=True)
    print("  Saved: bill_amount_correlation_matrix.csv")
    
    section("5. QUALITY CHECKS")
    qc = run_quality_checks(
        df, schema, monthly, by_target, sign_over, sign_tgt, change_over, change_tgt, corr, disp
    )
    qc.to_csv(TABLES_DIR / "bill_amount_quality_checks.csv", index=False)
    print("  Saved: bill_amount_quality_checks.csv")
    
    failed = qc[~qc["passed"]]
    if not failed.empty:
        print(f"\n  [WARNING] {len(failed)} quality check(s) FAILED:")
        print(failed.to_string(index=False))
        sys.exit(1)
    else:
        print(f"\n  Quality checks: {len(qc)}/{len(qc)} passed")
        
    section("6. GENERATE FIGURES")
    plot_monthly_median_iqr_overall(monthly, FIGURES_DIR)
    print("  1_median_iqr_trajectory.png")
    
    plot_monthly_median_iqr_by_target(by_target, FIGURES_DIR)
    print("  2_median_iqr_trajectory_by_target.png")
    
    plot_six_panel_distributions(df, disp, FIGURES_DIR)
    print("  3_distributions.png")
    
    plot_six_panel_distributions_by_target(df, disp, FIGURES_DIR)
    print("  4_distributions_by_target.png")
    
    plot_six_panel_boxplots(df, FIGURES_DIR)
    print("  5_boxplots_by_target.png")
    
    plot_sign_proportions(sign_over, FIGURES_DIR)
    print("  6_sign_proportions.png")
    
    plot_sign_default_rates(sign_over, FIGURES_DIR)
    print("  7_sign_default_rates.png")
    
    plot_adjacent_change_medians(change_over, change_tgt, FIGURES_DIR)
    print("  8_adjacent_change_medians.png")
    
    plot_adjacent_change_distributions(df, FIGURES_DIR)
    print("  9_adjacent_change_distributions.png")
    
    plot_correlation_heatmap(corr, FIGURES_DIR)
    print("  10_correlation_heatmap.png")
    
    plot_potential_extreme_counts(monthly, FIGURES_DIR)
    print("  11_potential_extremes.png")
    
    plot_outside_display_range(disp, FIGURES_DIR)
    print("  12_outside_display_range.png")
    
    section("7. GENERATE FINDINGS")
    out_md = REPORTS_DIR / "eda_bill_amount_findings.md"
    generate_bill_amount_findings(monthly, by_target, sign_over, sign_tgt, change_over, change_tgt, corr, disp, out_md)
    print(f"  Saved: {out_md}")
    
    section("CHECKPOINT 2B2A COMPLETE")
    print(f"  Tables : {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}\n")

if __name__ == "__main__":
    main()

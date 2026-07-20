import sys
from pathlib import Path

from credit_default.data_loader import load_raw_dataset, compute_file_sha256
from credit_default.eda.bill_payment_relationship import (
    validate_schema,
    generate_schema_table,
    compute_monthly_relationship_summary,
    compute_relationship_by_target,
    compute_positive_bill_ratio,
    compute_relationship_categories,
    generate_regression_anchors,
    run_quality_checks
)
from credit_default.eda.bill_payment_relationship_figures import (
    plot_monthly_medians,
    plot_six_panel_scatter,
    plot_monthly_correlations,
    plot_ratio_median_iqr,
    plot_category_proportions,
    plot_category_default_rates
)
from credit_default.eda.bill_payment_relationship_findings import generate_relationship_findings

def main():
    print("=" * 72)
    print("  1. LOAD & VALIDATE SCHEMA")
    print("=" * 72)
    
    sha256 = compute_file_sha256()
    print(f"  Raw SHA-256: {sha256}")
    
    df = load_raw_dataset()
    validate_schema(df)
    
    schema = generate_schema_table(df)
    
    tables_dir = Path("reports/tables/eda/bill_payment_relationship")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    schema_path = tables_dir / "1_relationship_schema.csv"
    schema.to_csv(schema_path, index=False)
    print("  Schema validated and saved.")

    print("\n" + "=" * 72)
    print("  2. COMPUTE RELATIONSHIP SUMMARIES")
    print("=" * 72)
    
    monthly = compute_monthly_relationship_summary(df)
    monthly.to_csv(tables_dir / "2_monthly_relationship_summary.csv", index=False)
    
    by_target = compute_relationship_by_target(df)
    by_target.to_csv(tables_dir / "3_relationship_by_target.csv", index=False)
    
    ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
    ratio_over.to_csv(tables_dir / "4_positive_bill_ratio_summary.csv", index=False)
    
    cats = compute_relationship_categories(df)
    cats.to_csv(tables_dir / "5_relationship_category_summary.csv", index=False)
    
    anchors = generate_regression_anchors(df, monthly, ratio_over, cats)
    
    print("  Summaries computed and saved.")

    print("\n" + "=" * 72)
    print("  3. QUALITY CHECKS")
    print("=" * 72)
    
    checks = run_quality_checks(df, schema, monthly, by_target, ratio_over, ratio_tgt, cats, anchors)
    checks.to_csv(tables_dir / "6_relationship_quality_checks.csv", index=False)
    
    passed_cnt = checks["passed"].sum()
    total_cnt = len(checks)
    print(f"  Quality checks: {passed_cnt}/{total_cnt} passed")
    
    if not checks["passed"].all():
        print("  ERROR: Quality checks failed!")
        print(checks[~checks["passed"]])
        sys.exit(1)

    print("\n" + "=" * 72)
    print("  4. GENERATE FIGURES")
    print("=" * 72)
    
    fig_dir = Path("reports/figures/eda/bill_payment_relationship")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plot_monthly_medians(monthly, fig_dir)
    plot_six_panel_scatter(df, fig_dir)
    plot_monthly_correlations(monthly, fig_dir)
    plot_ratio_median_iqr(ratio_over, ratio_tgt, fig_dir)
    plot_category_proportions(cats, fig_dir)
    plot_category_default_rates(cats, fig_dir)
    
    print(f"  Figures generated at: {fig_dir}")

    print("\n" + "=" * 72)
    print("  5. GENERATE FINDINGS")
    print("=" * 72)
    
    findings_path = Path("reports/eda_bill_payment_relationship_findings.md")
    generate_relationship_findings(monthly, ratio_over, ratio_tgt, cats, findings_path)
    
    print(f"  Findings saved to: {findings_path}")

    print("\n" + "=" * 72)
    print("  ACCELERATED PHASE 1 COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()

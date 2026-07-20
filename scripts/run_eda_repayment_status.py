"""
run_eda_repayment_status.py
===========================
Execute Checkpoint 2B1: temporal EDA for repayment statuses.
"""

import sys
from pathlib import Path

from credit_default.data_loader import load_raw_dataset, compute_file_sha256
from credit_default.eda.repayment_status import (
    validate_chronological_schema,
    generate_schema_table,
    compute_status_distribution_by_month,
    compute_distribution_conditioned_on_target,
    compute_month_to_month_transitions,
    compute_exact_sequence_patterns,
    run_temporal_quality_checks,
)
from credit_default.eda.repayment_status_figures import (
    plot_status_distribution_lines,
    plot_status_distribution_target,
    plot_default_rate_by_status_and_month,
    plot_distribution_heatmap,
    plot_default_rate_heatmap,
    plot_transition_heatmap,
    plot_top_sequence_patterns,
)
from credit_default.eda.repayment_status_findings import generate_repayment_findings_markdown
from credit_default.eda.static_features import POSITIVE_CLASS, NEGATIVE_CLASS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "eda" / "repayment_status"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eda" / "repayment_status"
EXPECTED_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


def section(title: str) -> None:
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    section("1. LOAD & VALIDATE SCHEMA")
    actual_sha256 = compute_file_sha256()
    print(f"  Raw SHA-256: {actual_sha256}")
    
    df = load_raw_dataset()
    validate_chronological_schema(df)
    
    schema_df = generate_schema_table(df)
    schema_df.to_csv(TABLES_DIR / "repayment_status_schema.csv", index=False)
    print("  Saved: repayment_status_schema.csv")
    print("  Chronological schema validated.")

    section("2. COMPUTE RAW DISTRIBUTIONS")
    dist_month = compute_status_distribution_by_month(df)
    dist_month.to_csv(TABLES_DIR / "status_distribution_by_month.csv", index=False)
    print("  Saved: status_distribution_by_month.csv")

    dist_target = compute_distribution_conditioned_on_target(df)
    dist_target.to_csv(TABLES_DIR / "status_distribution_by_month_and_target.csv", index=False)
    print("  Saved: status_distribution_by_month_and_target.csv")
    
    # We can write a specific subset table as requested: status_default_rates_by_month.csv
    # It is effectively covered by dist_month, but we'll extract relevant columns just in case
    dr_cols = ["month", "raw_status_value", "total_count", "default_count", "non_default_count", "default_rate", "observed_combination"]
    dist_month[dr_cols].to_csv(TABLES_DIR / "status_default_rates_by_month.csv", index=False)
    print("  Saved: status_default_rates_by_month.csv")

    section("3. COMPUTE TRANSITIONS")
    transitions = compute_month_to_month_transitions(df)
    transitions.to_csv(TABLES_DIR / "status_transition_summary.csv", index=False)
    print("  Saved: status_transition_summary.csv")

    section("4. COMPUTE EXACT SEQUENCE PATTERNS")
    patterns = compute_exact_sequence_patterns(df)
    patterns.to_csv(TABLES_DIR / "status_sequence_patterns.csv", index=False)
    print("  Saved: status_sequence_patterns.csv")
    
    section("5. QUALITY CHECKS")
    qc = run_temporal_quality_checks(df, dist_month, dist_target, transitions, patterns)
    qc.loc[len(qc)] = {"check_name": "raw_file_sha256_unchanged", "passed": actual_sha256 == EXPECTED_SHA256, "detail": "SHA matches."}
    
    qc.to_csv(TABLES_DIR / "repayment_status_quality_checks.csv", index=False)
    print("  Saved: repayment_status_quality_checks.csv")
    
    failed = qc[~qc["passed"]]
    if not failed.empty:
        print(f"\n  [WARNING] {len(failed)} quality check(s) FAILED:")
        print(failed.to_string(index=False))
        sys.exit(1)
    else:
        print(f"\n  Quality checks: {len(qc)}/{len(qc)} passed")

    section("6. GENERATE FIGURES")
    print("  1_status_dist_overall.png")
    plot_status_distribution_lines(dist_month, FIGURES_DIR)
    
    print("  2_status_dist_non_default.png")
    plot_status_distribution_target(dist_target, NEGATIVE_CLASS, "Status Distribution (Non-Default)", "2_status_dist_non_default.png", FIGURES_DIR)
    
    print("  3_status_dist_default.png")
    plot_status_distribution_target(dist_target, POSITIVE_CLASS, "Status Distribution (Default)", "3_status_dist_default.png", FIGURES_DIR)
    
    print("  4_default_rate_by_status.png")
    plot_default_rate_by_status_and_month(dist_month, FIGURES_DIR)
    
    print("  5_distribution_heatmap.png")
    plot_distribution_heatmap(dist_month, FIGURES_DIR)
    
    print("  6_default_rate_heatmap.png")
    plot_default_rate_heatmap(dist_month, FIGURES_DIR)
    
    print("  7_transition_heatmap_apr_may.png")
    # Apr->May pair index = 0
    plot_transition_heatmap(transitions, 0, "April to May Transitions", "7_transition_heatmap_apr_may.png", FIGURES_DIR)
    
    print("  8_transition_heatmap_aug_sep.png")
    # Aug->Sep pair index = 4
    plot_transition_heatmap(transitions, 4, "August to September Transitions", "8_transition_heatmap_aug_sep.png", FIGURES_DIR)
    
    print("  9_top10_patterns_total.png")
    plot_top_sequence_patterns(patterns, "total_count", "Top 10 Sequence Patterns (Total)", "9_top10_patterns_total.png", FIGURES_DIR)
    
    print("  10_top10_patterns_default.png")
    plot_top_sequence_patterns(patterns, "default_count", "Top 10 Sequence Patterns (Default Clients)", "10_top10_patterns_default.png", FIGURES_DIR)

    section("7. GENERATE FINDINGS")
    findings_path = PROJECT_ROOT / "reports" / "eda_repayment_status_findings.md"
    generate_repayment_findings_markdown(dist_month, dist_target, transitions, patterns, findings_path)
    print(f"  Saved: {findings_path}")
    
    section("CHECKPOINT 2B1 COMPLETE")
    print(f"  Tables : {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")

if __name__ == "__main__":
    main()

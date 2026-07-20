"""
run_eda_static.py
=================
Checkpoint 2A — Descriptive EDA for target and static customer features.

Features analyzed: default payment next month, LIMIT_BAL, AGE, SEX,
EDUCATION, MARRIAGE.

The PAY_x, BILL_AMTx, and PAY_AMTx groups are intentionally deferred to
Checkpoint 2B.

Run from the project root after editable installation:
    python -m scripts.run_eda_static

Scientific constraints honoured:
  - Descriptive EDA only.
  - No train/validation/test split.
  - No model construction or training.
  - No feature selection.
  - No scaling, encoding, imputation, or resampling.
  - No duplicate removal.
  - No modification of the raw dataset.
  - Causal language avoided.
"""

import io
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 on Windows to avoid cp1252 errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from credit_default.data_loader import (
    compute_file_sha256,
    get_raw_dataset_path,
    load_raw_dataset,
)
from credit_default.eda.static_features import (
    CATEGORICAL_COLS,
    DOCUMENTED_VALUES,
    NUMERIC_COLS,
    TARGET_COL,
    compute_categorical_default_rates,
    compute_categorical_distribution,
    compute_numeric_by_target,
    compute_numeric_summary,
    compute_target_summary,
    run_quality_checks,
)
from credit_default.eda.figures import (
    plot_categorical_counts,
    plot_categorical_default_rates,
    plot_numeric_boxplot_by_target,
    plot_numeric_distribution_by_target,
    plot_target_distribution,
)
from credit_default.eda.findings import generate_findings_markdown

# ---------------------------------------------------------------------------
# Expected raw-file SHA-256 (Checkpoint 1 verified)
# ---------------------------------------------------------------------------
EXPECTED_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
SMALL_N = 200

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "reports" / "tables" / "eda" / "static"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "eda" / "static"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main() -> None:
    print(SEP)
    print("  CHECKPOINT 2A — STATIC FEATURE EDA")
    print("  Credit Card Default Prediction (UCI)")
    print(SEP)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    section("LOADING DATASET")
    df = load_raw_dataset()
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 1. Target distribution
    # ------------------------------------------------------------------
    section("1. TARGET DISTRIBUTION")
    target_summary = compute_target_summary(df)
    print(target_summary.to_string(index=False))
    path = TABLES_DIR / "target_summary.csv"
    target_summary.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    fig_path = plot_target_distribution(target_summary, FIGURES_DIR)
    print(f"  Figure: {fig_path}")

    # ------------------------------------------------------------------
    # 2 & 3. Numeric features
    # ------------------------------------------------------------------
    section("2-3. NUMERIC FEATURES: LIMIT_BAL and AGE")

    all_numeric_overall = []
    all_numeric_by_target = []
    numeric_by_target_dfs: dict[str, pd.DataFrame] = {}

    for col in NUMERIC_COLS:
        print(f"\n  --- {col} ---")

        overall = compute_numeric_summary(df, col)
        by_target = compute_numeric_by_target(df, col)
        numeric_by_target_dfs[col] = by_target

        print("  Overall:")
        print(overall[["feature", "count", "mean", "std", "min", "q25", "median", "q75", "max", "iqr"]].to_string(index=False))
        print("  By target:")
        print(by_target[["feature", "target_label", "count", "mean", "std", "min", "median", "max", "iqr"]].to_string(index=False))
        print(f"  Extreme-value note: {overall['potential_extreme_note'].iloc[0]}")

        all_numeric_overall.append(overall)
        all_numeric_by_target.append(by_target)

        # Figures
        fp1 = plot_numeric_distribution_by_target(df, col, FIGURES_DIR)
        fp2 = plot_numeric_boxplot_by_target(df, col, FIGURES_DIR)
        print(f"  Figure (distribution): {fp1}")
        print(f"  Figure (boxplot):      {fp2}")

    overall_df = pd.concat(all_numeric_overall, ignore_index=True)
    by_target_df = pd.concat(all_numeric_by_target, ignore_index=True)

    p_overall = TABLES_DIR / "numeric_overall_summary.csv"
    p_bytarget = TABLES_DIR / "numeric_by_target_summary.csv"
    overall_df.to_csv(p_overall, index=False)
    by_target_df.to_csv(p_bytarget, index=False)
    print(f"\n  Saved: {p_overall}")
    print(f"  Saved: {p_bytarget}")

    # ------------------------------------------------------------------
    # 4–6. Categorical features
    # ------------------------------------------------------------------
    section("4-6. CATEGORICAL FEATURES: SEX, EDUCATION, MARRIAGE")

    all_cat_dist = []
    all_cat_rates = []
    cat_dist_dfs: dict[str, pd.DataFrame] = {}
    cat_rate_dfs: dict[str, pd.DataFrame] = {}

    for col in CATEGORICAL_COLS:
        print(f"\n  --- {col} ---")
        docs = DOCUMENTED_VALUES.get(col, set())

        cat_dist = compute_categorical_distribution(df, col, docs, SMALL_N)
        cat_rate = compute_categorical_default_rates(df, col, docs, SMALL_N)

        cat_dist_dfs[col] = cat_dist
        cat_rate_dfs[col] = cat_rate

        print("  Distribution:")
        print(cat_dist.to_string(index=False))
        print("  Default rates:")
        print(cat_rate[["raw_value", "total_count", "default_count", "default_rate", "documentation_status", "small_sample_warning"]].to_string(index=False))

        # Warn about small categories
        small = cat_rate[cat_rate["small_sample_warning"]]
        if not small.empty:
            for row in small.itertuples():
                print(f"  [WARNING] {col}={row.raw_value}: n={row.total_count} < {SMALL_N}; interpret cautiously.")

        all_cat_dist.append(cat_dist)
        all_cat_rates.append(cat_rate)

        fp_counts = plot_categorical_counts(cat_dist, col, FIGURES_DIR, SMALL_N)
        fp_rates = plot_categorical_default_rates(cat_rate, col, FIGURES_DIR, SMALL_N)
        print(f"  Figure (counts): {fp_counts}")
        print(f"  Figure (rates):  {fp_rates}")

    cat_dist_combined = pd.concat(all_cat_dist, ignore_index=True)
    cat_rate_combined = pd.concat(all_cat_rates, ignore_index=True)

    p_cat_dist = TABLES_DIR / "categorical_distribution.csv"
    p_cat_rate = TABLES_DIR / "categorical_default_rates.csv"
    cat_dist_combined.to_csv(p_cat_dist, index=False)
    cat_rate_combined.to_csv(p_cat_rate, index=False)
    print(f"\n  Saved: {p_cat_dist}")
    print(f"  Saved: {p_cat_rate}")

    # ------------------------------------------------------------------
    # 7. Quality checks
    # ------------------------------------------------------------------
    section("7. QUALITY CHECKS")
    raw_path = get_raw_dataset_path()
    qc = run_quality_checks(
        df,
        cat_dist_dfs=cat_dist_dfs,
        cat_rate_dfs=cat_rate_dfs,
        numeric_by_target_dfs=numeric_by_target_dfs,
        expected_sha256=EXPECTED_SHA256,
        raw_file_path=raw_path,
    )
    print(qc.to_string(index=False))
    failed = qc[~qc["passed"]]
    if failed.empty:
        print("\n  All quality checks passed.")
    else:
        print(f"\n  [WARNING] {len(failed)} quality check(s) FAILED:")
        print(failed.to_string(index=False))
        sys.exit(1)

    p_qc = TABLES_DIR / "static_eda_quality_checks.csv"
    qc.to_csv(p_qc, index=False)
    print(f"  Saved: {p_qc}")

    # ------------------------------------------------------------------
    # 8. Generate Findings Markdown
    # ------------------------------------------------------------------
    section("8. GENERATE FINDINGS DOCUMENT")
    findings_path = PROJECT_ROOT / "reports" / "eda_static_findings.md"
    generate_findings_markdown(
        target_summary=target_summary,
        numeric_overall=overall_df,
        numeric_by_target=by_target_df,
        cat_dist=cat_dist_combined,
        cat_rates=cat_rate_combined,
        out_path=findings_path
    )
    print(f"  Saved: {findings_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("CHECKPOINT 2A COMPLETE")
    print(f"  Tables : {TABLES_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  SHA-256: {compute_file_sha256(raw_path)}")
    print()
    print("  *** Checkpoint 2A covers only static features.")
    print("      PAY_x, BILL_AMTx, and PAY_AMTx are deferred to Checkpoint 2B.")
    print("      No preprocessing, splitting, or modelling was performed. ***")
    print()


if __name__ == "__main__":
    main()

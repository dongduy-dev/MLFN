"""
audit_dataset.py
================
Checkpoint 1 — Comprehensive raw-dataset audit.

Run from the project root:
    python -m scripts.audit_dataset

This script loads the raw UCI Credit Card Default dataset and produces a
thorough audit report.  It does NOT modify, transform, encode, impute,
split or model the data in any way.

All reported values come from executing against the actual dataset.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from credit_default.data_loader import (
    RAW_DATASET_FILENAME,
    compute_file_sha256,
    get_raw_dataset_path,
    load_raw_dataset,
)

# ---------------------------------------------------------------------------
# Output directory for machine-readable reports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: pretty section headers
# ---------------------------------------------------------------------------
SEPARATOR = "=" * 72


def section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Categorical columns of interest
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
TARGET_COL = "default payment next month"

# Documented valid values per official UCI documentation.
#
# Repayment status (PAY_x):
#   The UCI page explicitly defines:
#     -1  = pay duly
#     1-9 = payment delay in months
#   Values 0 and -2 are present in the raw dataset but are NOT explicitly
#   defined by the official UCI documentation.  They are reported as
#   undocumented but are NOT treated as invalid or removed.
DOCUMENTED_VALUES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    "PAY_STATUS": {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    TARGET_COL: {0, 1},
}

# Temporal column groups with month mapping (Sept = most recent)
# The dataset covers April 2005 – September 2005.
# Index 1 -> September 2005 (most recent), Index 6 -> April 2005 (oldest)
TEMPORAL_GROUPS = {
    "PAY_x (repayment status)": {
        "columns": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        "note": "PAY_0=Sep, PAY_2=Aug, PAY_3=Jul, PAY_4=Jun, PAY_5=May, PAY_6=Apr. "
                "PAY_1 is absent from the dataset.",
    },
    "BILL_AMTx (bill statement amount)": {
        "columns": [f"BILL_AMT{i}" for i in range(1, 7)],
        "note": "BILL_AMT1=Sep, BILL_AMT2=Aug, ..., BILL_AMT6=Apr.",
    },
    "PAY_AMTx (previous payment amount)": {
        "columns": [f"PAY_AMT{i}" for i in range(1, 7)],
        "note": "PAY_AMT1=Sep, PAY_AMT2=Aug, ..., PAY_AMT6=Apr.",
    },
}

MONTH_MAPPING = {
    1: "September 2005",
    2: "August 2005",
    3: "July 2005",
    4: "June 2005",
    5: "May 2005",
    6: "April 2005",
}


def main() -> None:
    """Run the complete dataset audit."""
    print(SEPARATOR)
    print("  CHECKPOINT 1 — RAW DATASET AUDIT")
    print("  Credit Card Default Prediction (UCI)")
    print(SEPARATOR)

    # ------------------------------------------------------------------
    # 1. File information & SHA-256
    # ------------------------------------------------------------------
    section("1. SOURCE FILE INFORMATION")
    raw_path = get_raw_dataset_path()
    file_size = raw_path.stat().st_size
    sha256 = compute_file_sha256()

    print(f"  File name     : {RAW_DATASET_FILENAME}")
    print(f"  Full path     : {raw_path}")
    print(f"  File size     : {file_size:,} bytes")
    print(f"  SHA-256 hash  : {sha256}")

    # ------------------------------------------------------------------
    # 2. Load the dataset
    # ------------------------------------------------------------------
    section("2. LOADING DATASET")
    df = load_raw_dataset()
    print(f"  DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # 3. Column list (ordered)
    # ------------------------------------------------------------------
    section("3. ORDERED COLUMN LIST")
    for i, col in enumerate(df.columns):
        print(f"  [{i:2d}] {col}")

    # ------------------------------------------------------------------
    # 4. Data types
    # ------------------------------------------------------------------
    section("4. DATA TYPES")
    dtype_info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
    })
    for _, row in dtype_info.iterrows():
        print(f"  {row['column']:40s}  {row['dtype']}")

    # ------------------------------------------------------------------
    # 5. First five rows
    # ------------------------------------------------------------------
    section("5. FIRST FIVE ROWS")
    print(df.head().to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Missing values per column
    # ------------------------------------------------------------------
    section("6. MISSING VALUES PER COLUMN")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    for col in df.columns:
        count = missing[col]
        print(f"  {col:40s}  {count}")
    print(f"\n  Total missing values: {total_missing}")

    # ------------------------------------------------------------------
    # 7. Infinite values
    # ------------------------------------------------------------------
    section("7. INFINITE VALUES CHECK")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    total_inf = 0
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        inf_counts[col] = n_inf
        total_inf += n_inf
        if n_inf > 0:
            print(f"  {col:40s}  {n_inf} infinite values")
    if total_inf == 0:
        print("  No infinite values found in any numeric column.")
    print(f"\n  Total infinite values: {total_inf}")

    # ------------------------------------------------------------------
    # 8. Duplicate rows
    # ------------------------------------------------------------------
    section("8. DUPLICATE ROW DETECTION")

    # 8a. Complete-row duplicates (including ID)
    dup_all = df.duplicated(keep=False).sum()
    dup_all_count = df.duplicated(keep="first").sum()
    print(f"  Complete-row duplicates (including ID):")
    print(f"    Rows involved in duplication : {dup_all}")
    print(f"    Duplicate rows (excl. first) : {dup_all_count}")

    # 8b. Duplicates after excluding ID
    cols_no_id = [c for c in df.columns if c != "ID"]
    dup_no_id = df.duplicated(subset=cols_no_id, keep=False).sum()
    dup_no_id_count = df.duplicated(subset=cols_no_id, keep="first").sum()
    print(f"\n  Duplicates after excluding ID column:")
    print(f"    Rows involved in duplication : {dup_no_id}")
    print(f"    Duplicate rows (excl. first) : {dup_no_id_count}")

    # ------------------------------------------------------------------
    # 9. Unique values per column
    # ------------------------------------------------------------------
    section("9. UNIQUE VALUE COUNTS PER COLUMN")
    nunique = df.nunique()
    for col in df.columns:
        print(f"  {col:40s}  {nunique[col]}")

    # ------------------------------------------------------------------
    # 10. ID uniqueness
    # ------------------------------------------------------------------
    section("10. ID COLUMN UNIQUENESS")
    id_unique = df["ID"].nunique() == len(df)
    print(f"  Total rows        : {len(df)}")
    print(f"  Unique ID values  : {df['ID'].nunique()}")
    print(f"  All IDs unique    : {id_unique}")

    # ------------------------------------------------------------------
    # 11. Categorical value distributions
    # ------------------------------------------------------------------
    section("11. CATEGORICAL COLUMN VALUES & FREQUENCIES")

    cat_records = []

    # SEX, EDUCATION, MARRIAGE
    for col in CATEGORICAL_COLS:
        print(f"\n  --- {col} ---")
        vc = df[col].value_counts().sort_index()
        for val, count in vc.items():
            pct = 100.0 * count / len(df)
            documented = val in DOCUMENTED_VALUES.get(col, set())
            flag = (
                ""
                if documented
                else "  [NOT EXPLICITLY DEFINED IN UCI DOCS]"
            )
            print(f"    {val:>6}  :  {count:>6} ({pct:5.2f}%){flag}")
            cat_records.append({
                "column": col,
                "value": val,
                "count": count,
                "percentage": round(pct, 4),
                "documented": documented,
            })

    # PAY status columns
    for col in PAY_STATUS_COLS:
        print(f"\n  --- {col} ---")
        vc = df[col].value_counts().sort_index()
        for val, count in vc.items():
            pct = 100.0 * count / len(df)
            documented = val in DOCUMENTED_VALUES.get("PAY_STATUS", set())
            flag = (
                ""
                if documented
                else "  [NOT EXPLICITLY DEFINED IN UCI DOCS]"
            )
            print(f"    {val:>6}  :  {count:>6} ({pct:5.2f}%){flag}")
            cat_records.append({
                "column": col,
                "value": val,
                "count": count,
                "percentage": round(pct, 4),
                "documented": documented,
            })

    # Target column
    print(f"\n  --- {TARGET_COL} ---")
    vc = df[TARGET_COL].value_counts().sort_index()
    for val, count in vc.items():
        pct = 100.0 * count / len(df)
        documented = val in DOCUMENTED_VALUES.get(TARGET_COL, set())
        flag = (
            ""
            if documented
            else "  [NOT EXPLICITLY DEFINED IN UCI DOCS]"
        )
        print(f"    {val:>6}  :  {count:>6} ({pct:5.2f}%){flag}")
        cat_records.append({
            "column": TARGET_COL,
            "value": val,
            "count": count,
            "percentage": round(pct, 4),
            "documented": documented,
        })

    # ------------------------------------------------------------------
    # 12. Target class distribution summary
    # ------------------------------------------------------------------
    section("12. TARGET CLASS DISTRIBUTION")
    target_vc = df[TARGET_COL].value_counts().sort_index()
    for val, count in target_vc.items():
        pct = 100.0 * count / len(df)
        label = "No default" if val == 0 else "Default"
        print(f"  {val} ({label:>10}) : {count:>6}  ({pct:5.2f}%)")
    print(f"\n  Class imbalance ratio (majority/minority): "
          f"{target_vc.max() / target_vc.min():.2f}:1")

    # ------------------------------------------------------------------
    # 13. Descriptive statistics for numeric columns
    # ------------------------------------------------------------------
    section("13. DESCRIPTIVE STATISTICS (NUMERIC COLUMNS)")
    desc = df.describe().T
    print(desc.to_string())

    # ------------------------------------------------------------------
    # 14. Undocumented / suspicious categorical values
    # ------------------------------------------------------------------
    section("14. VALUES PRESENT IN RAW DATA BUT NOT EXPLICITLY DEFINED IN UCI DOCS")
    print()
    print("  The following values appear in the dataset but are NOT explicitly")
    print("  defined by the official UCI documentation.  They are reported here")
    print("  for transparency.  They have NOT been removed, recoded, or treated")
    print("  as invalid.")

    undoc_records = []
    found_any = False

    for col in CATEGORICAL_COLS:
        actual = set(int(v) for v in df[col].unique())
        expected = DOCUMENTED_VALUES.get(col, set())
        undocumented = actual - expected
        if undocumented:
            found_any = True
            print(f"\n  {col}:")
            print(f"    Explicitly documented : {sorted(expected)}")
            print(f"    Actual in dataset     : {sorted(actual)}")
            print(f"    Not explicitly defined: {sorted(undocumented)}")
            for val in sorted(undocumented):
                count = int((df[col] == val).sum())
                print(f"      value={val} appears {count} times")
                undoc_records.append({
                    "column": col,
                    "value": val,
                    "count": count,
                    "note": "Present in raw dataset but not explicitly defined by official UCI documentation",
                })

    for col in PAY_STATUS_COLS:
        actual = set(int(v) for v in df[col].unique())
        expected = DOCUMENTED_VALUES.get("PAY_STATUS", set())
        undocumented = actual - expected
        if undocumented:
            found_any = True
            print(f"\n  {col}:")
            print(f"    Explicitly documented : {sorted(expected)}")
            print(f"    Actual in dataset     : {sorted(actual)}")
            print(f"    Not explicitly defined: {sorted(undocumented)}")
            for val in sorted(undocumented):
                count = int((df[col] == val).sum())
                print(f"      value={val} appears {count} times")
                undoc_records.append({
                    "column": col,
                    "value": val,
                    "count": count,
                    "note": "Present in raw dataset but not explicitly defined by official UCI documentation",
                })

    if not found_any:
        print("  No undocumented categorical values found.")

    # ------------------------------------------------------------------
    # 15. Temporal column groups & month mapping
    # ------------------------------------------------------------------
    section("15. TEMPORAL COLUMN GROUPS & MONTH MAPPING")

    for group_name, info in TEMPORAL_GROUPS.items():
        print(f"\n  {group_name}")
        print(f"    Columns: {info['columns']}")
        print(f"    Note   : {info['note']}")
        # Verify columns exist
        missing_cols = [c for c in info["columns"] if c not in df.columns]
        if missing_cols:
            print(f"    WARNING - MISSING COLUMNS: {missing_cols}")

    print(f"\n  Apparent chronological mapping (index -> month):")
    for idx, month in MONTH_MAPPING.items():
        print(f"    Index {idx} -> {month}")

    print(f"\n  NOTE: PAY_1 is absent from the dataset.")
    print(f"        PAY_0 corresponds to September 2005 (most recent month).")
    print(f"        The suffix numbering is NOT consistent across groups:")
    print(f"          PAY_x uses: 0, 2, 3, 4, 5, 6")
    print(f"          BILL_AMTx & PAY_AMTx use: 1, 2, 3, 4, 5, 6")

    # ------------------------------------------------------------------
    # 16. Save machine-readable reports
    # ------------------------------------------------------------------
    section("16. SAVING MACHINE-READABLE REPORTS")

    # a) Column info
    col_info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
        "missing_count": [int(df[c].isnull().sum()) for c in df.columns],
        "unique_count": [int(df[c].nunique()) for c in df.columns],
    })
    col_info_path = REPORTS_DIR / "column_info.csv"
    col_info.to_csv(col_info_path, index=False)
    print(f"  Saved: {col_info_path}")

    # b) Descriptive statistics
    desc_path = REPORTS_DIR / "descriptive_stats.csv"
    desc.to_csv(desc_path)
    print(f"  Saved: {desc_path}")

    # c) Target distribution
    target_df = pd.DataFrame({
        "value": target_vc.index,
        "count": target_vc.values,
        "percentage": [round(100.0 * c / len(df), 4) for c in target_vc.values],
    })
    target_path = REPORTS_DIR / "target_distribution.csv"
    target_df.to_csv(target_path, index=False)
    print(f"  Saved: {target_path}")

    # d) Categorical unique values and frequencies
    cat_df = pd.DataFrame(cat_records)
    cat_path = REPORTS_DIR / "categorical_values.csv"
    cat_df.to_csv(cat_path, index=False)
    print(f"  Saved: {cat_path}")

    # e) Undocumented values
    if undoc_records:
        undoc_df = pd.DataFrame(undoc_records)
        undoc_path = REPORTS_DIR / "undocumented_values.csv"
        undoc_df.to_csv(undoc_path, index=False)
        print(f"  Saved: {undoc_path}")

    # f) Duplicate summary
    dup_df = pd.DataFrame([
        {
            "scope": "all_columns_including_ID",
            "rows_involved": int(dup_all),
            "duplicate_rows_excl_first": int(dup_all_count),
        },
        {
            "scope": "all_columns_excluding_ID",
            "rows_involved": int(dup_no_id),
            "duplicate_rows_excl_first": int(dup_no_id_count),
        },
    ])
    dup_path = REPORTS_DIR / "duplicate_summary.csv"
    dup_df.to_csv(dup_path, index=False)
    print(f"  Saved: {dup_path}")

    # g) File integrity
    integrity_df = pd.DataFrame([{
        "filename": RAW_DATASET_FILENAME,
        "file_size_bytes": file_size,
        "sha256": sha256,
        "rows": df.shape[0],
        "columns": df.shape[1],
    }])
    integrity_path = REPORTS_DIR / "file_integrity.csv"
    integrity_df.to_csv(integrity_path, index=False)
    print(f"  Saved: {integrity_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    section("AUDIT COMPLETE")
    print(f"  Dataset           : {RAW_DATASET_FILENAME}")
    print(f"  Shape             : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  SHA-256           : {sha256}")
    print(f"  Missing values    : {total_missing}")
    print(f"  Infinite values   : {total_inf}")
    print(f"  ID uniqueness     : {id_unique}")
    print(f"  Dup rows (w/ ID)  : {dup_all_count}")
    print(f"  Dup rows (no ID)  : {dup_no_id_count}")
    print(f"  Report files      : {REPORTS_DIR}")
    print()
    print("  *** No preprocessing, splitting, EDA visualisation, or")
    print("      model training was performed in this audit. ***")
    print()


if __name__ == "__main__":
    main()

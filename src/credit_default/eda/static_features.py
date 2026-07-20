"""
static_features.py
==================
Checkpoint 2A — Descriptive EDA for the target variable and static customer
attributes: LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE.

This module is intentionally descriptive only.
  - No train/validation/test split.
  - No model construction or training.
  - No feature selection.
  - No scaling, encoding, imputation, or resampling.
  - No duplicate removal.
  - No modification of the raw dataset.
  - Raw category codes are preserved in all machine-readable outputs.
  - Undocumented category values are annotated, not merged.
  - Causal language is avoided; only associations are reported.

The PAY_x, BILL_AMTx, and PAY_AMTx groups are intentionally deferred to
Checkpoint 2B.

Public API
----------
compute_target_summary(df)
compute_numeric_summary(df, col)
compute_numeric_by_target(df, col)
compute_categorical_distribution(df, col, documented_values, small_n_threshold)
compute_categorical_default_rates(df, col, documented_values, small_n_threshold)
run_quality_checks(df, cat_dfs, numeric_dfs, expected_sha256)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "default payment next month"
POSITIVE_CLASS = 1   # 1 = default
NEGATIVE_CLASS = 0   # 0 = no default

NUMERIC_COLS = ["LIMIT_BAL", "AGE"]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

# Values explicitly defined by the official UCI documentation
DOCUMENTED_VALUES: dict[str, set[int]] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}

# Threshold below which a category is flagged as small sample
DEFAULT_SMALL_N = 200


# ---------------------------------------------------------------------------
# 1. Target summary
# ---------------------------------------------------------------------------
def compute_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising the target class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (must contain TARGET_COL).

    Returns
    -------
    pd.DataFrame with columns:
        class_value, class_label, count, percentage, imbalance_note
    """
    _require_columns(df, [TARGET_COL])
    n = len(df)
    vc = df[TARGET_COL].value_counts().sort_index()

    rows = []
    for val, count in vc.items():
        rows.append({
            "class_value": int(val),
            "class_label": "default" if val == POSITIVE_CLASS else "no_default",
            "count": int(count),
            "percentage": round(100.0 * count / n, 4),
        })
    out = pd.DataFrame(rows)

    # Attach imbalance ratio as a column on the majority row
    maj = vc.max()
    mn = vc.min()
    ratio = round(maj / mn, 4)
    out["imbalance_ratio_majority_to_minority"] = ratio
    out["positive_class"] = POSITIVE_CLASS
    return out


# ---------------------------------------------------------------------------
# 2. Numeric summary — overall
# ---------------------------------------------------------------------------
def compute_numeric_summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Return overall descriptive statistics for a numeric column.

    Returns a single-row DataFrame with columns:
        feature, count, mean, std, min, q25, median, q75, max, iqr,
        potential_extreme_note
    """
    _require_columns(df, [col])
    s = df[col]
    q25, median, q75 = s.quantile([0.25, 0.50, 0.75]).values
    iqr = q75 - q25
    # Flag values more than 3×IQR beyond the fences as potential extremes
    lower_fence = q25 - 3.0 * iqr
    upper_fence = q75 + 3.0 * iqr
    n_extreme = int(((s < lower_fence) | (s > upper_fence)).sum())
    note = (
        f"{n_extreme} values outside 3×IQR fences "
        f"[{lower_fence:.1f}, {upper_fence:.1f}]; "
        "not removed"
    )
    return pd.DataFrame([{
        "feature": col,
        "count": int(s.count()),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "q25": round(float(q25), 4),
        "median": round(float(median), 4),
        "q75": round(float(q75), 4),
        "max": round(float(s.max()), 4),
        "iqr": round(float(iqr), 4),
        "potential_extreme_note": note,
    }])


# ---------------------------------------------------------------------------
# 3. Numeric summary — by target group
# ---------------------------------------------------------------------------
def compute_numeric_by_target(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Return descriptive statistics for a numeric column, split by target class.

    Returns a two-row DataFrame (one row per class) with the same columns
    as compute_numeric_summary plus a 'target_value' and 'target_label' column.
    """
    _require_columns(df, [col, TARGET_COL])
    rows = []
    for tval in sorted(df[TARGET_COL].unique()):
        sub = df.loc[df[TARGET_COL] == tval, col]
        q25, median, q75 = sub.quantile([0.25, 0.50, 0.75]).values
        iqr = q75 - q25
        lower_fence = q25 - 3.0 * iqr
        upper_fence = q75 + 3.0 * iqr
        n_extreme = int(((sub < lower_fence) | (sub > upper_fence)).sum())
        note = (
            f"{n_extreme} values outside 3×IQR fences "
            f"[{lower_fence:.1f}, {upper_fence:.1f}]; "
            "not removed"
        )
        rows.append({
            "feature": col,
            "target_value": int(tval),
            "target_label": "default" if tval == POSITIVE_CLASS else "no_default",
            "count": int(sub.count()),
            "mean": round(float(sub.mean()), 4),
            "std": round(float(sub.std()), 4),
            "min": round(float(sub.min()), 4),
            "q25": round(float(q25), 4),
            "median": round(float(median), 4),
            "q75": round(float(q75), 4),
            "max": round(float(sub.max()), 4),
            "iqr": round(float(iqr), 4),
            "potential_extreme_note": note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Categorical distribution
# ---------------------------------------------------------------------------
def compute_categorical_distribution(
    df: pd.DataFrame,
    col: str,
    documented_values: set[int] | None = None,
    small_n_threshold: int = DEFAULT_SMALL_N,
) -> pd.DataFrame:
    """
    Return frequency distribution for a categorical column.

    All raw codes are preserved; undocumented ones are annotated.
    Categories with n < small_n_threshold get a warning flag.

    Returns a DataFrame with columns:
        feature, raw_value, count, percentage, documentation_status,
        small_sample_warning
    """
    _require_columns(df, [col])
    if documented_values is None:
        documented_values = DOCUMENTED_VALUES.get(col, set())
    n = len(df)
    vc = df[col].value_counts().sort_index()
    rows = []
    for val, count in vc.items():
        doc = int(val) in documented_values
        rows.append({
            "feature": col,
            "raw_value": int(val),
            "count": int(count),
            "percentage": round(100.0 * count / n, 4),
            "documentation_status": (
                "documented"
                if doc
                else "not_explicitly_defined_in_uci_docs"
            ),
            "small_sample_warning": count < small_n_threshold,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Categorical default rates
# ---------------------------------------------------------------------------
def compute_categorical_default_rates(
    df: pd.DataFrame,
    col: str,
    documented_values: set[int] | None = None,
    small_n_threshold: int = DEFAULT_SMALL_N,
) -> pd.DataFrame:
    """
    Return default counts and rates for every raw category code.

    Parameters match compute_categorical_distribution.

    Returns a DataFrame with columns:
        feature, raw_value, total_count, default_count, non_default_count,
        default_rate, population_percentage, documentation_status,
        small_sample_warning
    """
    _require_columns(df, [col, TARGET_COL])
    if documented_values is None:
        documented_values = DOCUMENTED_VALUES.get(col, set())
    n = len(df)
    rows = []
    for val in sorted(df[col].unique()):
        mask = df[col] == val
        total = int(mask.sum())
        defaults = int((mask & (df[TARGET_COL] == POSITIVE_CLASS)).sum())
        non_defaults = total - defaults
        rate = defaults / total if total > 0 else float("nan")
        doc = int(val) in documented_values
        rows.append({
            "feature": col,
            "raw_value": int(val),
            "total_count": total,
            "default_count": defaults,
            "non_default_count": non_defaults,
            "default_rate": round(rate, 6),
            "population_percentage": round(100.0 * total / n, 4),
            "documentation_status": (
                "documented"
                if doc
                else "not_explicitly_defined_in_uci_docs"
            ),
            "small_sample_warning": total < small_n_threshold,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Quality checks
# ---------------------------------------------------------------------------
def run_quality_checks(
    df: pd.DataFrame,
    cat_dist_dfs: dict[str, pd.DataFrame],
    cat_rate_dfs: dict[str, pd.DataFrame],
    numeric_by_target_dfs: dict[str, pd.DataFrame],
    expected_sha256: str,
    raw_file_path: Path,
) -> pd.DataFrame:
    """
    Run programmatic quality checks and return a tidy results DataFrame.

    Checks
    ------
    1. Target counts sum to dataset length.
    2. Category counts sum to dataset length for each categorical feature.
    3. default + non_default = total_count for every categorical row.
    4. All default_rates are in [0, 1].
    5. Numeric grouped counts sum correctly (sum over groups = dataset count).
    6. Raw file SHA-256 matches expected hash.
    7. Raw rows are not modified (re-load and compare shape & dtypes).

    Returns a DataFrame with columns: check_name, passed, detail.
    """
    from credit_default.data_loader import compute_file_sha256, load_raw_dataset

    results: list[dict[str, Any]] = []

    def _record(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check_name": name, "passed": passed, "detail": detail})

    n = len(df)

    # 1. Target counts
    target_total = int(df[TARGET_COL].count())
    _record(
        "target_counts_sum_to_dataset_size",
        target_total == n,
        f"target count={target_total}, dataset n={n}",
    )

    # 2. Category counts per feature
    for col, cdf in cat_dist_dfs.items():
        total = int(cdf["count"].sum())
        _record(
            f"cat_counts_sum_{col}",
            total == n,
            f"sum={total}, expected={n}",
        )

    # 3. default + non_default = total
    for col, rdf in cat_rate_dfs.items():
        ok = (rdf["default_count"] + rdf["non_default_count"] == rdf["total_count"]).all()
        _record(
            f"default_plus_nonfault_eq_total_{col}",
            bool(ok),
            "all rows ok" if ok else "mismatch detected",
        )

    # 4. Default rates in [0, 1]
    for col, rdf in cat_rate_dfs.items():
        rates = rdf["default_rate"].dropna()
        ok = bool((rates >= 0).all() and (rates <= 1).all())
        _record(
            f"default_rates_in_range_{col}",
            ok,
            f"min={rates.min():.4f}, max={rates.max():.4f}",
        )

    # 5. Numeric grouped counts
    for col, ndf in numeric_by_target_dfs.items():
        total = int(ndf["count"].sum())
        _record(
            f"numeric_grouped_counts_{col}",
            total == n,
            f"sum={total}, expected={n}",
        )

    # 6. SHA-256 check
    actual_hash = compute_file_sha256(raw_file_path)
    hash_ok = actual_hash == expected_sha256
    _record(
        "raw_file_sha256_unchanged",
        hash_ok,
        f"expected={expected_sha256[:16]}..., actual={actual_hash[:16]}...",
    )

    # 7. Raw row integrity (re-load and compare)
    df2 = load_raw_dataset()
    shape_ok = df2.shape == df.shape
    dtype_ok = list(df2.dtypes) == list(df.dtypes)
    _record(
        "raw_rows_not_modified_shape",
        shape_ok,
        f"shape={df2.shape}",
    )
    _record(
        "raw_rows_not_modified_dtypes",
        dtype_ok,
        "dtypes match" if dtype_ok else "dtype mismatch detected",
    )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

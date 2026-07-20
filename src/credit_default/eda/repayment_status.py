"""
repayment_status.py
===================
Checkpoint 2B1 — Temporal exploratory analysis of the six repayment-status columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_default.data_loader import load_raw_dataset
from credit_default.eda.static_features import (
    DOCUMENTED_VALUES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_COL,
)

# Chronological order of columns from April to September 2005
CHRONOLOGICAL_COLS = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
MONTH_NAMES = ["April", "May", "June", "July", "August", "September"]
MONTH_MAPPING = dict(zip(CHRONOLOGICAL_COLS, MONTH_NAMES))

DATASET_SIZE = 30_000

# The documented values for PAY_X
# -1 = pay duly, 1..9 = payment delay
# 0, -2 are undocumented
KNOWN_STATUS_CODES = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def validate_chronological_schema(df: pd.DataFrame) -> None:
    """Validate that the dataset matches the expected chronological schema."""
    if len(df) != DATASET_SIZE:
        raise ValueError(f"Expected dataset size {DATASET_SIZE}, got {len(df)}")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
        
    for col in CHRONOLOGICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required temporal column: {col}")
            
    if "PAY_1" in df.columns:
        raise ValueError("PAY_1 is present in the dataset, but expected to be absent.")
        
    target_vals = set(df[TARGET_COL].unique())
    if target_vals != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError(f"Expected binary target {{0, 1}}, got {target_vals}")


def _get_doc_status(val: int) -> str:
    """Return documentation status for a repayment status code."""
    if val in {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
        return "documented"
    return "not_explicitly_defined_in_uci_docs"

def generate_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the explicit chronological schema table."""
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_COLS, start=1):
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "raw_column": col,
            "expected_present": True,
            "actual_present": col in df.columns,
            "documentation_note": "Approved chronological mapping"
        })
    
    records.append({
        "chronological_index": 7,
        "month": "N/A",
        "raw_column": "PAY_1",
        "expected_present": False,
        "actual_present": "PAY_1" in df.columns,
        "documentation_note": "Expected to be absent"
    })
    
    return pd.DataFrame(records)


def compute_status_distribution_by_month(df: pd.DataFrame, small_n_threshold: int = 200) -> pd.DataFrame:
    """Calculate raw status distribution for each month. Include all known codes even if unobserved."""
    
    records = []
    
    for month_idx, col in enumerate(CHRONOLOGICAL_COLS):
        month_name = MONTH_MAPPING[col]
        # Get actual counts
        counts = df.groupby([col, TARGET_COL], dropna=False).size().unstack(fill_value=0)
        
        # Ensure all known codes exist in the index
        for code in KNOWN_STATUS_CODES:
            if code not in counts.index:
                counts.loc[code] = 0
                
        # Fill missing targets if any
        if POSITIVE_CLASS not in counts.columns:
            counts[POSITIVE_CLASS] = 0
        if NEGATIVE_CLASS not in counts.columns:
            counts[NEGATIVE_CLASS] = 0
            
        for code in sorted(counts.index):
            def_cnt = counts.loc[code, POSITIVE_CLASS]
            non_def_cnt = counts.loc[code, NEGATIVE_CLASS]
            total_cnt = def_cnt + non_def_cnt
            
            observed = bool(total_cnt > 0)
            pct = (total_cnt / DATASET_SIZE) * 100 if observed else np.nan
            def_rate = (def_cnt / total_cnt) if observed else np.nan
            
            records.append({
                "raw_column_name": col,
                "month": month_name,
                "chronological_month_index": month_idx,
                "raw_status_value": int(code),
                "total_count": int(total_cnt),
                "population_percentage": pct,
                "default_count": int(def_cnt),
                "non_default_count": int(non_def_cnt),
                "default_rate": def_rate,
                "observed_combination": observed,
                "documentation_status": _get_doc_status(int(code)),
                "small_sample_warning": total_cnt < small_n_threshold if observed else False
            })
            
    res = pd.DataFrame(records)
    # Sort deterministically
    return res.sort_values(by=["chronological_month_index", "raw_status_value"]).reset_index(drop=True)


def compute_distribution_conditioned_on_target(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage of each status code within the specific month and target class."""
    records = []
    
    # We want rows: month, target_class, status_code
    for month_idx, col in enumerate(CHRONOLOGICAL_COLS):
        month_name = MONTH_MAPPING[col]
        
        for tgt in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            sub = df[df[TARGET_COL] == tgt]
            tgt_total = len(sub)
            
            counts = sub[col].value_counts()
            
            for code in KNOWN_STATUS_CODES:
                cnt = counts.get(code, 0)
                observed = bool(cnt > 0)
                pct = (cnt / tgt_total) * 100 if observed else np.nan
                
                records.append({
                    "raw_column_name": col,
                    "month": month_name,
                    "chronological_month_index": month_idx,
                    "target_class": tgt,
                    "raw_status_value": int(code),
                    "status_count": int(cnt),
                    "percentage_within_target_class": pct,
                    "observed_combination": observed
                })
                
    res = pd.DataFrame(records)
    return res.sort_values(by=["chronological_month_index", "target_class", "raw_status_value"]).reset_index(drop=True)


def compute_month_to_month_transitions(df: pd.DataFrame, small_n_threshold: int = 200) -> pd.DataFrame:
    """Calculate raw transitions between adjacent months."""
    records = []
    
    # Adjacent pairs: (0, 1), (1, 2), etc.
    for i in range(len(CHRONOLOGICAL_COLS) - 1):
        src_col = CHRONOLOGICAL_COLS[i]
        dst_col = CHRONOLOGICAL_COLS[i+1]
        src_month = MONTH_MAPPING[src_col]
        dst_month = MONTH_MAPPING[dst_col]
        
        # Calculate source status totals for denominator
        src_totals = df[src_col].value_counts()
        
        counts = df.groupby([src_col, dst_col, TARGET_COL], dropna=False).size().unstack(fill_value=0)
        
        # Ensure all target columns exist
        if POSITIVE_CLASS not in counts.columns:
            counts[POSITIVE_CLASS] = 0
        if NEGATIVE_CLASS not in counts.columns:
            counts[NEGATIVE_CLASS] = 0
            
        # Re-index to ensure all KNOWN_STATUS_CODES combinations exist
        idx = pd.MultiIndex.from_product(
            [sorted(KNOWN_STATUS_CODES), sorted(KNOWN_STATUS_CODES)], 
            names=[src_col, dst_col]
        )
        counts = counts.reindex(idx, fill_value=0)
        
        for (src_code, dst_code), row in counts.iterrows():
            def_cnt = row[POSITIVE_CLASS]
            non_def_cnt = row[NEGATIVE_CLASS]
            total_cnt = def_cnt + non_def_cnt
            
            src_total = src_totals.get(src_code, 0)
            
            observed = bool(total_cnt > 0)
            pct = (total_cnt / src_total) * 100 if src_total > 0 else np.nan
            def_rate = (def_cnt / total_cnt) if observed else np.nan
            
            records.append({
                "source_month": src_month,
                "destination_month": dst_month,
                "chronological_pair_index": i,
                "source_raw_status": int(src_code),
                "destination_raw_status": int(dst_code),
                "source_status_total": int(src_total),
                "transition_count": int(total_cnt),
                "percentage_within_source_status": pct,
                "default_count": int(def_cnt),
                "non_default_count": int(non_def_cnt),
                "default_rate": def_rate,
                "observed_combination": observed,
                "small_sample_warning": total_cnt < small_n_threshold if observed else False
            })
            
    res = pd.DataFrame(records)
    return res.sort_values(
        by=["chronological_pair_index", "source_raw_status", "destination_raw_status"]
    ).reset_index(drop=True)


def compute_exact_sequence_patterns(df: pd.DataFrame, small_n_threshold: int = 200) -> pd.DataFrame:
    """Group complete 6-month trajectories into a summary."""
    # Build sequence pattern string for each row
    seq_series = df[CHRONOLOGICAL_COLS].astype(str).agg('|'.join, axis=1)
    
    tmp_df = pd.DataFrame({
        "sequence_pattern": seq_series,
        "target": df[TARGET_COL]
    })
    
    counts = tmp_df.groupby(["sequence_pattern", "target"]).size().unstack(fill_value=0)
    
    if POSITIVE_CLASS not in counts.columns:
        counts[POSITIVE_CLASS] = 0
    if NEGATIVE_CLASS not in counts.columns:
        counts[NEGATIVE_CLASS] = 0
        
    records = []
    for seq, row in counts.iterrows():
        def_cnt = row[POSITIVE_CLASS]
        non_def_cnt = row[NEGATIVE_CLASS]
        total_cnt = def_cnt + non_def_cnt
        
        observed = bool(total_cnt > 0)
        pct = (total_cnt / DATASET_SIZE) * 100 if observed else np.nan
        def_rate = (def_cnt / total_cnt) if observed else np.nan
        
        records.append({
            "sequence_pattern": str(seq),
            "total_count": int(total_cnt),
            "population_percentage": pct,
            "default_count": int(def_cnt),
            "non_default_count": int(non_def_cnt),
            "default_rate": def_rate,
            "observed_combination": observed,
            "small_sample_warning": total_cnt < small_n_threshold if observed else False
        })
        
    res = pd.DataFrame(records)
    # Sort deterministically: total_count DESC, sequence_pattern ASC
    return res.sort_values(
        by=["total_count", "sequence_pattern"], ascending=[False, True]
    ).reset_index(drop=True)


def run_temporal_quality_checks(
    df: pd.DataFrame,
    dist_month: pd.DataFrame,
    dist_target: pd.DataFrame,
    transitions: pd.DataFrame,
    patterns: pd.DataFrame
) -> pd.DataFrame:
    """Run required quality checks on the temporal analysis output."""
    checks = []
    
    def _add(name: str, passed: bool, detail: str):
        checks.append({"check_name": name, "passed": passed, "detail": detail})

    # 1. Dataset size and column exactness
    _add("dataset_size_30k", len(df) == 30000, f"n={len(df)}")
    
    # 2. Month totals
    for m in MONTH_NAMES:
        total = dist_month[dist_month["month"] == m]["total_count"].sum()
        _add(f"month_sum_30k_{m}", total == 30000, f"sum={total}")
        
    # 3. Target class percentage sums ~100
    for m in MONTH_NAMES:
        for t in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            sub = dist_target[(dist_target["month"] == m) & (dist_target["target_class"] == t)]
            pct_sum = sub["percentage_within_target_class"].sum()
            _add(f"target_pct_sum_{m}_c{t}", abs(pct_sum - 100.0) < 1e-3, f"sum={pct_sum:.2f}")

    # 4. Default + non_default = total for aggregations
    all_ok = (dist_month["default_count"] + dist_month["non_default_count"] == dist_month["total_count"]).all()
    _add("month_counts_add_up", all_ok, str(all_ok))
    
    all_ok = (transitions["default_count"] + transitions["non_default_count"] == transitions["transition_count"]).all()
    _add("transition_counts_add_up", all_ok, str(all_ok))
    
    all_ok = (patterns["default_count"] + patterns["non_default_count"] == patterns["total_count"]).all()
    _add("pattern_counts_add_up", all_ok, str(all_ok))

    # 5. Transition sums by pair = 30k
    for i in range(5):
        sub = transitions[transitions["chronological_pair_index"] == i]
        total = sub["transition_count"].sum()
        _add(f"transition_pair_{i}_sum_30k", total == 30000, f"sum={total}")
        
    # 6. Default rates in [0, 1]
    dr = dist_month["default_rate"].dropna()
    dr_ok = ((dr >= 0) & (dr <= 1)).all() if not dr.empty else True
    _add("month_default_rates_valid", dr_ok, "Rates in [0,1]")

    # 7. Pattern sums = 30k
    pat_total = patterns["total_count"].sum()
    _add("pattern_sum_30k", pat_total == 30000, f"sum={pat_total}")
    
    # 8. Sequences have 6 elements
    elem_lens = patterns["sequence_pattern"].str.split("|").str.len()
    _add("sequence_len_6", (elem_lens == 6).all(), "All patterns have 6 elements")
    
    # 9. 0 and -2 remain
    all_codes = set(dist_month["raw_status_value"].unique())
    _add("undocumented_codes_present", 0 in all_codes and -2 in all_codes, f"codes={all_codes}")

    return pd.DataFrame(checks)

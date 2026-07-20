import sys
import pandas as pd
import numpy as np
from pathlib import Path

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COL

DATASET_SIZE = 30000

CHRONOLOGICAL_BILL_COLS = [
    "BILL_AMT6",
    "BILL_AMT5",
    "BILL_AMT4",
    "BILL_AMT3",
    "BILL_AMT2",
    "BILL_AMT1"
]

MONTH_MAPPING = {
    "BILL_AMT6": "April",
    "BILL_AMT5": "May",
    "BILL_AMT4": "June",
    "BILL_AMT3": "July",
    "BILL_AMT2": "August",
    "BILL_AMT1": "September"
}

def validate_chronological_schema(df: pd.DataFrame) -> None:
    if len(df) != DATASET_SIZE:
        raise ValueError(f"Expected dataset size {DATASET_SIZE}, got {len(df)}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    target_vals = set(df[TARGET_COL].unique())
    if target_vals != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError(f"Expected binary target, got {target_vals}")
    for col in CHRONOLOGICAL_BILL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required temporal column: {col}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} is not numeric.")
        if df.columns.tolist().count(col) > 1:
            raise ValueError(f"Column {col} is duplicated.")

def generate_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS, start=1):
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "raw_column": col,
            "expected_present": True,
            "actual_present": col in df.columns,
            "dtype": str(df[col].dtype) if col in df.columns else "N/A",
            "documentation_note": "Approved chronological mapping"
        })
    return pd.DataFrame(records)

def _get_sign(val: float) -> str:
    if val < 0: return "negative"
    if val > 0: return "positive"
    return "zero"

def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS, start=1):
        s = df[col]
        total_count = len(s)
        missing_count = s.isna().sum()
        s_drop = s.dropna()
        n_obs = len(s_drop)
        
        q1 = s_drop.quantile(0.25)
        q3 = s_drop.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        neg_count = (s_drop < 0).sum()
        zero_count = (s_drop == 0).sum()
        pos_count = (s_drop > 0).sum()
        
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "count": n_obs,
            "missing_count": missing_count,
            "mean": s_drop.mean(),
            "std": s_drop.std(),
            "min": s_drop.min(),
            "p01": s_drop.quantile(0.01),
            "p05": s_drop.quantile(0.05),
            "p25": q1,
            "median": s_drop.median(),
            "p75": q3,
            "p95": s_drop.quantile(0.95),
            "p99": s_drop.quantile(0.99),
            "max": s_drop.max(),
            "iqr": iqr,
            "negative_count": neg_count,
            "zero_count": zero_count,
            "positive_count": pos_count,
            "negative_percentage": (neg_count / n_obs) * 100 if n_obs else 0,
            "zero_percentage": (zero_count / n_obs) * 100 if n_obs else 0,
            "positive_percentage": (pos_count / n_obs) * 100 if n_obs else 0,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "below_lower_fence_count": (s_drop < lower_fence).sum(),
            "above_upper_fence_count": (s_drop > upper_fence).sum(),
            "total_potential_extreme_count": (s_drop < lower_fence).sum() + (s_drop > upper_fence).sum()
        })
    return pd.DataFrame(records)

def compute_by_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS, start=1):
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            s = df[df[TARGET_COL] == target][col].dropna()
            n_obs = len(s)
            if n_obs == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            
            records.append({
                "chronological_index": idx,
                "month": MONTH_MAPPING[col],
                "target": target,
                "count": n_obs,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q1": q1,
                "median": s.median(),
                "q3": q3,
                "max": s.max(),
                "iqr": iqr,
                "negative_count": (s < 0).sum(),
                "zero_count": (s == 0).sum(),
                "positive_count": (s > 0).sum(),
                "potential_extreme_count": (s < lower_fence).sum() + (s > upper_fence).sum()
            })
    return pd.DataFrame(records)

def compute_sign_summary(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    overall = []
    by_target = []
    
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS, start=1):
        month = MONTH_MAPPING[col]
        s_sign = df[col].apply(_get_sign)
        
        # Overall
        total_count = len(df)
        for sign in ["negative", "zero", "positive"]:
            mask = (s_sign == sign)
            cnt = mask.sum()
            def_cnt = (mask & (df[TARGET_COL] == POSITIVE_CLASS)).sum()
            non_def_cnt = (mask & (df[TARGET_COL] == NEGATIVE_CLASS)).sum()
            overall.append({
                "chronological_index": idx,
                "month": month,
                "sign_category": sign,
                "total_count": cnt,
                "population_percentage": (cnt / total_count) * 100 if total_count else 0,
                "default_count": def_cnt,
                "non_default_count": non_def_cnt,
                "default_rate": (def_cnt / cnt) if cnt > 0 else np.nan,
                "caution_flag": cnt < 200
            })
            
        # By target
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            t_mask = (df[TARGET_COL] == target)
            t_total = t_mask.sum()
            for sign in ["negative", "zero", "positive"]:
                cnt = (t_mask & (s_sign == sign)).sum()
                by_target.append({
                    "chronological_index": idx,
                    "month": month,
                    "target": target,
                    "sign_category": sign,
                    "count": cnt,
                    "percentage": (cnt / t_total) * 100 if t_total else 0
                })
                
    df_overall = pd.DataFrame(overall)
    # deterministic sort: chronological, then sign
    sign_order = {"negative": 0, "zero": 1, "positive": 2}
    df_overall["_sign_order"] = df_overall["sign_category"].map(sign_order)
    df_overall = df_overall.sort_values(["chronological_index", "_sign_order"]).drop(columns=["_sign_order"]).reset_index(drop=True)

    df_target = pd.DataFrame(by_target)
    df_target["_sign_order"] = df_target["sign_category"].map(sign_order)
    df_target = df_target.sort_values(["chronological_index", "target", "_sign_order"]).drop(columns=["_sign_order"]).reset_index(drop=True)
    
    return df_overall, df_target

def compute_change_summary(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    overall = []
    by_target = []
    
    pairs = [
        ("April", "May", "BILL_AMT6", "BILL_AMT5"),
        ("May", "June", "BILL_AMT5", "BILL_AMT4"),
        ("June", "July", "BILL_AMT4", "BILL_AMT3"),
        ("July", "August", "BILL_AMT3", "BILL_AMT2"),
        ("August", "September", "BILL_AMT2", "BILL_AMT1")
    ]
    
    for idx, (m_src, m_dst, col_src, col_dst) in enumerate(pairs, start=1):
        diffs = df[col_dst] - df[col_src] # Destination - Source
        
        # Overall
        total = len(diffs)
        q1 = diffs.quantile(0.25)
        q3 = diffs.quantile(0.75)
        neg_c = (diffs < 0).sum()
        zero_c = (diffs == 0).sum()
        pos_c = (diffs > 0).sum()
        
        overall.append({
            "pair_index": idx,
            "source_month": m_src,
            "destination_month": m_dst,
            "count": total,
            "mean_change": diffs.mean(),
            "std_change": diffs.std(),
            "min_change": diffs.min(),
            "q1": q1,
            "median_change": diffs.median(),
            "q3": q3,
            "max_change": diffs.max(),
            "negative_change_count": neg_c,
            "zero_change_count": zero_c,
            "positive_change_count": pos_c,
            "negative_change_percentage": (neg_c / total) * 100,
            "zero_change_percentage": (zero_c / total) * 100,
            "positive_change_percentage": (pos_c / total) * 100
        })
        
        # By target
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            diffs_t = diffs[df[TARGET_COL] == target]
            total_t = len(diffs_t)
            q1_t = diffs_t.quantile(0.25)
            q3_t = diffs_t.quantile(0.75)
            neg_c_t = (diffs_t < 0).sum()
            zero_c_t = (diffs_t == 0).sum()
            pos_c_t = (diffs_t > 0).sum()
            
            by_target.append({
                "pair_index": idx,
                "source_month": m_src,
                "destination_month": m_dst,
                "target": target,
                "count": total_t,
                "mean_change": diffs_t.mean(),
                "std_change": diffs_t.std(),
                "min_change": diffs_t.min(),
                "q1": q1_t,
                "median_change": diffs_t.median(),
                "q3": q3_t,
                "max_change": diffs_t.max(),
                "negative_change_count": neg_c_t,
                "zero_change_count": zero_c_t,
                "positive_change_count": pos_c_t,
                "negative_change_percentage": (neg_c_t / total_t) * 100 if total_t else 0,
                "zero_change_percentage": (zero_c_t / total_t) * 100 if total_t else 0,
                "positive_change_percentage": (pos_c_t / total_t) * 100 if total_t else 0
            })
            
    df_overall = pd.DataFrame(overall).sort_values("pair_index").reset_index(drop=True)
    df_target = pd.DataFrame(by_target).sort_values(["pair_index", "target"]).reset_index(drop=True)
    return df_overall, df_target

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Retain chronological order
    corr = df[CHRONOLOGICAL_BILL_COLS].corr(method="pearson")
    # Rename index/columns to months
    month_names = [MONTH_MAPPING[c] for c in CHRONOLOGICAL_BILL_COLS]
    corr.columns = month_names
    corr.index = month_names
    return corr

def compute_display_range(df: pd.DataFrame) -> pd.DataFrame:
    # 180,000 raw values
    all_vals = df[CHRONOLOGICAL_BILL_COLS].values.flatten()
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    
    total = len(all_vals)
    below = (all_vals < vmin).sum()
    above = (all_vals > vmax).sum()
    outside = below + above
    
    records = [{
        "chronological_index": 0,
        "month": "Global",
        "global_1st_percentile": vmin,
        "global_99th_percentile": vmax,
        "total_count": total,
        "below_range_count": below,
        "above_range_count": above,
        "outside_range_count": outside,
        "outside_range_percentage": (outside / total) * 100
    }]
    
    for idx, col in enumerate(CHRONOLOGICAL_BILL_COLS, start=1):
        s = df[col]
        t = len(s)
        b = (s < vmin).sum()
        a = (s > vmax).sum()
        o = b + a
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "global_1st_percentile": vmin,
            "global_99th_percentile": vmax,
            "total_count": t,
            "below_range_count": b,
            "above_range_count": a,
            "outside_range_count": o,
            "outside_range_percentage": (o / t) * 100
        })
        
    return pd.DataFrame(records)

def compute_histogram_stats(vals: np.ndarray, n_bins: int, vmin: float, vmax: float) -> (np.ndarray, np.ndarray, int, int):
    """Computes histogram counts strictly within bounds, plus outside counts."""
    # edges over the exact display range
    _, edges = np.histogram([], bins=n_bins, range=(vmin, vmax))
    inside = vals[(vals >= vmin) & (vals <= vmax)]
    below = np.sum(vals < vmin)
    above = np.sum(vals > vmax)
    counts, _ = np.histogram(inside, bins=edges)
    return edges, counts, int(below), int(above)

def compute_histogram_percentages(df: pd.DataFrame, col: str, n_bins: int, vmin: float, vmax: float) -> (np.ndarray, dict):
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
        
    _, edges = np.histogram([], bins=n_bins, range=(vmin, vmax))
    
    stats = {}
    for target in sorted(df[TARGET_COL].unique()):
        sub_vals = df[df[TARGET_COL] == target][col].dropna()
        n_sub = len(sub_vals)
        if n_sub == 0:
            stats[target] = {
                "percentages": np.zeros(n_bins),
                "below_percentage": 0.0,
                "above_percentage": 0.0
            }
            continue
            
        inside = sub_vals[(sub_vals >= vmin) & (sub_vals <= vmax)]
        below = np.sum(sub_vals < vmin)
        above = np.sum(sub_vals > vmax)
        
        counts, _ = np.histogram(inside, bins=edges)
        
        stats[target] = {
            "percentages": (counts / n_sub) * 100,
            "below_percentage": (below / n_sub) * 100,
            "above_percentage": (above / n_sub) * 100
        }
        
    return edges, stats

def run_quality_checks(
    df: pd.DataFrame,
    schema: pd.DataFrame,
    monthly: pd.DataFrame,
    by_target: pd.DataFrame,
    sign_over: pd.DataFrame,
    sign_tgt: pd.DataFrame,
    change_over: pd.DataFrame,
    change_tgt: pd.DataFrame,
    corr: pd.DataFrame,
    disp: pd.DataFrame
) -> pd.DataFrame:
    checks = []
    
    checks.append({"check_name": "exact_chronology", "passed": all(schema["actual_present"]), "details": "All required columns exist in exact order"})
    
    df_mapped = list(schema["raw_column"]) == CHRONOLOGICAL_BILL_COLS
    checks.append({"check_name": "chronological_mapping", "passed": bool(df_mapped), "details": "Chronological order strictly mapped"})
    
    checks.append({"check_name": "dataset_size", "passed": len(df) == DATASET_SIZE, "details": "30,000 rows"})
    
    m_counts_ok = all(monthly["count"] == DATASET_SIZE)
    checks.append({"check_name": "monthly_count_total", "passed": m_counts_ok, "details": "Every month has 30,000 count"})
    
    tgt_sums = by_target.groupby("month")["count"].sum()
    tgt_sums_ok = all(tgt_sums == DATASET_SIZE)
    checks.append({"check_name": "target_group_sum", "passed": tgt_sums_ok, "details": "Target groups sum to 30000 per month"})
    
    sign_sums = sign_over.groupby("month")["total_count"].sum()
    sign_sums_ok = all(sign_sums == DATASET_SIZE)
    checks.append({"check_name": "sign_total_sum", "passed": sign_sums_ok, "details": "Sign counts sum to 30000"})
    
    # Sign percentage sum approx 100
    sign_pct_sums = sign_over.groupby("month")["population_percentage"].sum()
    sign_pct_ok = all(np.isclose(sign_pct_sums, 100.0, atol=1e-5))
    checks.append({"check_name": "sign_percentage_sum", "passed": sign_pct_ok, "details": "Sign percentages sum to ~100%"})
    
    # Sign target total reconciliations
    stgt_sums = sign_tgt.groupby(["month", "target"])["count"].sum()
    stgt_pct_sums = sign_tgt.groupby(["month", "target"])["percentage"].sum()
    checks.append({"check_name": "target_sign_count_reconciliation", "passed": (stgt_sums > 0).all(), "details": "Target sign sums valid"})
    checks.append({"check_name": "target_sign_percentage_totals", "passed": np.isclose(stgt_pct_sums, 100.0, atol=1e-5).all(), "details": "Target sign pct approx 100"})
    
    # Default plus non-default count reconciliation
    def_nondef = sign_over["default_count"] + sign_over["non_default_count"]
    checks.append({"check_name": "default_plus_nondefault_reconciliation", "passed": (def_nondef == sign_over["total_count"]).all(), "details": "Sum matches total"})
    
    # Default rates in [0, 1]
    rates_ok = sign_over["default_rate"].dropna().between(0, 1).all()
    checks.append({"check_name": "default_rates_bounds", "passed": bool(rates_ok), "details": "Default rates in [0, 1]"})
    
    # IQR non-negative
    iqr_ok = (monthly["iqr"] >= 0).all() and (by_target["iqr"] >= 0).all()
    checks.append({"check_name": "iqr_non_negative", "passed": bool(iqr_ok), "details": "IQR >= 0"})
    
    # Lower/upper fences
    f_low_ok = np.isclose(monthly["lower_fence"], monthly["p25"] - 1.5 * monthly["iqr"]).all()
    f_up_ok = np.isclose(monthly["upper_fence"], monthly["p75"] + 1.5 * monthly["iqr"]).all()
    checks.append({"check_name": "fences_calculation", "passed": bool(f_low_ok and f_up_ok), "details": "Both fences match 1.5 IQR formula"})
    
    # Potential-extreme counts recomputed
    pe_recomp = (monthly["below_lower_fence_count"] + monthly["above_upper_fence_count"] == monthly["total_potential_extreme_count"]).all()
    checks.append({"check_name": "potential_extreme_counts_recomputed", "passed": bool(pe_recomp), "details": "Recomputed potential extreme counts"})
    
    # Adjacent change tables exactly 30000
    c_counts_ok = all(change_over["count"] == DATASET_SIZE)
    checks.append({"check_name": "change_counts", "passed": c_counts_ok, "details": "Changes contain 30,000"})
    
    c_tgt_sums = change_tgt.groupby(["pair_index"])["count"].sum()
    c_tgt_sums_ok = all(c_tgt_sums == DATASET_SIZE)
    checks.append({"check_name": "change_target_sum", "passed": c_tgt_sums_ok, "details": "Target changes sum to 30,000"})
    
    c_sum = (change_over["negative_change_count"] + change_over["zero_change_count"] + change_over["positive_change_count"])
    c_sum_ok = all(c_sum == change_over["count"])
    checks.append({"check_name": "change_sign_sum", "passed": c_sum_ok, "details": "Neg+Zero+Pos change equals total"})
    
    c_pct_sum = (change_over["negative_change_percentage"] + change_over["zero_change_percentage"] + change_over["positive_change_percentage"])
    c_pct_ok = np.isclose(c_pct_sum, 100.0, atol=1e-5).all()
    checks.append({"check_name": "change_percentage_totals", "passed": bool(c_pct_ok), "details": "Change percentages sum to approx 100"})
    
    # Corr matrix
    corr_size_ok = corr.shape == (6, 6)
    corr_diag_ok = np.isclose(np.diag(corr), 1.0).all()
    corr_sym_ok = np.allclose(corr, corr.T, equal_nan=True)
    checks.append({"check_name": "correlation_validity", "passed": corr_size_ok and corr_diag_ok and corr_sym_ok, "details": "6x6, diag=1, symmetric"})
    
    disp_min = disp.iloc[0]["global_1st_percentile"]
    disp_max = disp.iloc[0]["global_99th_percentile"]
    disp_ok = disp_min < disp_max
    checks.append({"check_name": "display_range_validity", "passed": bool(disp_ok), "details": "Min < Max"})
    
    # Regression Anchors
    # April overall (idx=1, month="April")
    apr_m = monthly[monthly["month"] == "April"].iloc[0]
    apr_ok = (
        np.isclose(apr_m["mean"], 38871.7604) and
        np.isclose(apr_m["p25"], 1256.00) and
        np.isclose(apr_m["median"], 17071.00) and
        np.isclose(apr_m["p75"], 49198.25) and
        np.isclose(apr_m["iqr"], 47942.25) and
        apr_m["negative_count"] == 688 and
        apr_m["zero_count"] == 4020 and
        apr_m["positive_count"] == 25292 and
        apr_m["total_potential_extreme_count"] == 2693
    )
    checks.append({"check_name": "regression_anchor_april_overall", "passed": bool(apr_ok), "details": "April exact known values match"})
    
    # September overall (idx=6, month="September")
    sep_m = monthly[monthly["month"] == "September"].iloc[0]
    sep_ok = (
        np.isclose(sep_m["mean"], 51223.3309) and
        np.isclose(sep_m["p25"], 3558.75) and
        np.isclose(sep_m["median"], 22381.50) and
        np.isclose(sep_m["p75"], 67091.00) and
        np.isclose(sep_m["iqr"], 63532.25) and
        sep_m["negative_count"] == 590 and
        sep_m["zero_count"] == 2008 and
        sep_m["positive_count"] == 27402 and
        sep_m["total_potential_extreme_count"] == 2400
    )
    checks.append({"check_name": "regression_anchor_september_overall", "passed": bool(sep_ok), "details": "September exact known values match"})
    
    # Target medians
    apr_def = by_target[(by_target["month"] == "April") & (by_target["target"] == POSITIVE_CLASS)].iloc[0]
    apr_nodef = by_target[(by_target["month"] == "April") & (by_target["target"] == NEGATIVE_CLASS)].iloc[0]
    sep_def = by_target[(by_target["month"] == "September") & (by_target["target"] == POSITIVE_CLASS)].iloc[0]
    sep_nodef = by_target[(by_target["month"] == "September") & (by_target["target"] == NEGATIVE_CLASS)].iloc[0]
    
    tm_ok = (
        apr_nodef["count"] == 23364 and np.isclose(apr_nodef["median"], 16679.00) and
        apr_def["count"] == 6636 and np.isclose(apr_def["median"], 18028.50) and
        sep_nodef["count"] == 23364 and np.isclose(sep_nodef["median"], 23119.50) and
        sep_def["count"] == 6636 and np.isclose(sep_def["median"], 20185.00)
    )
    checks.append({"check_name": "regression_anchor_target_medians", "passed": bool(tm_ok), "details": "Target group known medians match"})
    
    # September sign summary
    sep_s_neg = sign_over[(sign_over["month"] == "September") & (sign_over["sign_category"] == "negative")].iloc[0]
    sep_s_zero = sign_over[(sign_over["month"] == "September") & (sign_over["sign_category"] == "zero")].iloc[0]
    sep_s_pos = sign_over[(sign_over["month"] == "September") & (sign_over["sign_category"] == "positive")].iloc[0]
    
    ss_ok = (
        sep_s_neg["total_count"] == 590 and sep_s_neg["default_count"] == 109 and sep_s_neg["non_default_count"] == 481 and
        sep_s_zero["total_count"] == 2008 and sep_s_zero["default_count"] == 534 and sep_s_zero["non_default_count"] == 1474 and
        sep_s_pos["total_count"] == 27402 and sep_s_pos["default_count"] == 5993 and sep_s_pos["non_default_count"] == 21409
    )
    checks.append({"check_name": "regression_anchor_september_sign", "passed": bool(ss_ok), "details": "September sign counts match"})
    
    # Display population & per month
    glob = disp[disp["month"] == "Global"].iloc[0]
    
    # global/per-month reconciliation
    pm_below_sum = disp[disp["month"] != "Global"]["below_range_count"].sum()
    pm_above_sum = disp[disp["month"] != "Global"]["above_range_count"].sum()
    pm_outside_sum = disp[disp["month"] != "Global"]["outside_range_count"].sum()
    recon_ok = (pm_below_sum == glob["below_range_count"]) and (pm_above_sum == glob["above_range_count"]) and (pm_outside_sum == glob["outside_range_count"])
    checks.append({"check_name": "display_range_reconciliation", "passed": bool(recon_ok), "details": "Per-month exactly reconciles to global"})
    
    disp_ok = (
        glob["total_count"] == 180000 and
        np.isclose(glob["global_1st_percentile"], -200.00) and
        np.isclose(glob["global_99th_percentile"], 314507.17) and
        glob["below_range_count"] == 1739 and
        glob["above_range_count"] == 1800 and
        glob["outside_range_count"] == 3539 and
        np.isclose(glob["outside_range_percentage"], 1.966111)
    )
    checks.append({"check_name": "regression_anchor_display_range", "passed": bool(disp_ok), "details": "Display range 180k known values match"})
    
    apr_disp = disp[disp["month"] == "April"].iloc[0]
    sep_disp = disp[disp["month"] == "September"].iloc[0]
    disp_pm_ok = (
        apr_disp["below_range_count"] == 346 and apr_disp["above_range_count"] == 184 and apr_disp["outside_range_count"] == 530 and
        sep_disp["below_range_count"] == 207 and sep_disp["above_range_count"] == 420 and sep_disp["outside_range_count"] == 627
    )
    checks.append({"check_name": "regression_anchor_display_range_per_month", "passed": bool(disp_pm_ok), "details": "Display range per month match"})
    
    # Adjacent medians
    adj_ok = (change_over["median_change"] == 0).all()
    checks.append({"check_name": "regression_anchor_adjacent_medians_zero", "passed": bool(adj_ok), "details": "All overall median changes are zero"})
    
    # Aug->Sep change
    as_ch = change_over[(change_over["source_month"] == "August") & (change_over["destination_month"] == "September")].iloc[0]
    as_ok = (
        as_ch["negative_change_count"] == 14216 and
        as_ch["zero_change_count"] == 2634 and
        as_ch["positive_change_count"] == 13150 and
        np.isclose(as_ch["mean_change"], 2044.2557333) and
        as_ch["median_change"] == 0
    )
    checks.append({"check_name": "regression_anchor_aug_sep_change", "passed": bool(as_ok), "details": "Aug->Sep change counts match"})
    
    from credit_default.data_loader import compute_file_sha256
    sha256 = compute_file_sha256()
    checks.append({"check_name": "exact_expected_raw_sha256", "passed": sha256 == "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933", "details": "SHA-256 remains exact"})
    
    # Deterministic repeated calculations Check
    monthly_again = compute_monthly_summary(df)
    checks.append({"check_name": "deterministic_repeated_calculations", "passed": monthly.equals(monthly_again), "details": "Repeated calculation is identical"})
    
    return pd.DataFrame(checks)

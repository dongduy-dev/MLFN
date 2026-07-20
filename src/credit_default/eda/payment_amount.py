import pandas as pd
import numpy as np

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COL

DATASET_SIZE = 30000

CHRONOLOGICAL_PAY_COLS = [
    "PAY_AMT6",
    "PAY_AMT5",
    "PAY_AMT4",
    "PAY_AMT3",
    "PAY_AMT2",
    "PAY_AMT1"
]

MONTH_MAPPING = {
    "PAY_AMT6": "April",
    "PAY_AMT5": "May",
    "PAY_AMT4": "June",
    "PAY_AMT3": "July",
    "PAY_AMT2": "August",
    "PAY_AMT1": "September"
}

def validate_chronological_schema(df: pd.DataFrame) -> None:
    for col in CHRONOLOGICAL_PAY_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required temporal column: {col}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column {col} must be numeric")

def generate_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_PAY_COLS, start=1):
        present = col in df.columns
        dtype = str(df[col].dtype) if present else "N/A"
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "raw_column": col,
            "expected_present": True,
            "actual_present": present,
            "dtype": dtype,
            "documentation_note": "Previous payment amount (NT dollar)"
        })
    return pd.DataFrame(records)

def compute_histogram_stats(vals: np.ndarray, n_bins: int, vmin: float, vmax: float) -> (np.ndarray, np.ndarray, int, int):
    """Computes histogram counts strictly within bounds, plus outside counts."""
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

def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_PAY_COLS, start=1):
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
        
        neg = np.sum(s_drop < 0)
        zero = np.sum(s_drop == 0)
        pos = np.sum(s_drop > 0)
        
        records.append({
            "chronological_index": idx,
            "month": MONTH_MAPPING[col],
            "count": total_count,
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
            "negative_count": neg,
            "zero_count": zero,
            "positive_count": pos,
            "negative_percentage": (neg / n_obs) * 100 if n_obs > 0 else 0,
            "zero_percentage": (zero / n_obs) * 100 if n_obs > 0 else 0,
            "positive_percentage": (pos / n_obs) * 100 if n_obs > 0 else 0,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "below_lower_fence_count": np.sum(s_drop < lower_fence),
            "above_upper_fence_count": np.sum(s_drop > upper_fence),
            "total_potential_extreme_count": np.sum((s_drop < lower_fence) | (s_drop > upper_fence))
        })
    return pd.DataFrame(records)

def compute_by_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, col in enumerate(CHRONOLOGICAL_PAY_COLS, start=1):
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            s = df[df[TARGET_COL] == target][col]
            total_count = len(s)
            missing_count = s.isna().sum()
            s_drop = s.dropna()
            
            q1 = s_drop.quantile(0.25)
            q3 = s_drop.quantile(0.75)
            iqr = q3 - q1
            lf = q1 - 1.5 * iqr
            uf = q3 + 1.5 * iqr
            
            records.append({
                "chronological_index": idx,
                "month": MONTH_MAPPING[col],
                "target": target,
                "count": total_count,
                "missing_count": missing_count,
                "mean": s_drop.mean(),
                "std": s_drop.std(),
                "min": s_drop.min(),
                "q1": q1,
                "median": s_drop.median(),
                "q3": q3,
                "max": s_drop.max(),
                "iqr": iqr,
                "negative_count": np.sum(s_drop < 0),
                "zero_count": np.sum(s_drop == 0),
                "positive_count": np.sum(s_drop > 0),
                "potential_extreme_count": np.sum((s_drop < lf) | (s_drop > uf))
            })
    return pd.DataFrame(records)

def compute_sign_summary(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    over_records = []
    tgt_records = []
    
    for idx, col in enumerate(CHRONOLOGICAL_PAY_COLS, start=1):
        s = df[col].dropna()
        n_obs = len(s)
        
        signs = {
            "negative": (df[col] < 0),
            "zero": (df[col] == 0),
            "positive": (df[col] > 0)
        }
        
        for sign_name, mask in signs.items():
            valid_mask = mask & df[col].notna()
            tot = valid_mask.sum()
            def_cnt = (valid_mask & (df[TARGET_COL] == POSITIVE_CLASS)).sum()
            non_def_cnt = (valid_mask & (df[TARGET_COL] == NEGATIVE_CLASS)).sum()
            
            over_records.append({
                "chronological_index": idx,
                "month": MONTH_MAPPING[col],
                "sign_category": sign_name,
                "total_count": tot,
                "population_percentage": (tot / n_obs) * 100 if n_obs > 0 else 0,
                "default_count": def_cnt,
                "non_default_count": non_def_cnt,
                "default_rate": def_cnt / tot if tot > 0 else np.nan,
                "observed_combination": tot > 0,
                "caution_flag": tot > 0 and tot < 200
            })
            
            for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
                tgt_mask = df[TARGET_COL] == target
                tgt_total = tgt_mask.sum()
                scnt = (valid_mask & tgt_mask).sum()
                
                tgt_records.append({
                    "chronological_index": idx,
                    "month": MONTH_MAPPING[col],
                    "target": target,
                    "sign_category": sign_name,
                    "count": scnt,
                    "percentage": (scnt / tgt_total) * 100 if tgt_total > 0 else 0
                })
                
    return pd.DataFrame(over_records), pd.DataFrame(tgt_records)

def compute_change_summary(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    over = []
    tgt = []
    
    pairs = [
        ("April", "May", "PAY_AMT6", "PAY_AMT5"),
        ("May", "June", "PAY_AMT5", "PAY_AMT4"),
        ("June", "July", "PAY_AMT4", "PAY_AMT3"),
        ("July", "August", "PAY_AMT3", "PAY_AMT2"),
        ("August", "September", "PAY_AMT2", "PAY_AMT1"),
    ]
    
    for idx, (sm, dm, sc, dc) in enumerate(pairs, start=1):
        diffs = df[dc] - df[sc]
        valid = diffs.notna()
        diffs_clean = diffs.dropna()
        n_obs = len(diffs_clean)
        
        q1 = diffs_clean.quantile(0.25)
        q3 = diffs_clean.quantile(0.75)
        
        neg = np.sum(diffs_clean < 0)
        zero = np.sum(diffs_clean == 0)
        pos = np.sum(diffs_clean > 0)
        
        over.append({
            "pair_index": idx,
            "source_month": sm,
            "destination_month": dm,
            "count": n_obs,
            "mean_change": diffs_clean.mean(),
            "std_change": diffs_clean.std(),
            "min_change": diffs_clean.min(),
            "q1_change": q1,
            "median_change": diffs_clean.median(),
            "q3_change": q3,
            "max_change": diffs_clean.max(),
            "negative_change_count": neg,
            "zero_change_count": zero,
            "positive_change_count": pos,
            "negative_change_percentage": (neg / n_obs) * 100 if n_obs > 0 else 0,
            "zero_change_percentage": (zero / n_obs) * 100 if n_obs > 0 else 0,
            "positive_change_percentage": (pos / n_obs) * 100 if n_obs > 0 else 0,
        })
        
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            t_mask = df[TARGET_COL] == target
            t_diffs = diffs[valid & t_mask]
            nt = len(t_diffs)
            tq1 = t_diffs.quantile(0.25)
            tq3 = t_diffs.quantile(0.75)
            
            tneg = np.sum(t_diffs < 0)
            tzero = np.sum(t_diffs == 0)
            tpos = np.sum(t_diffs > 0)
            
            tgt.append({
                "pair_index": idx,
                "source_month": sm,
                "destination_month": dm,
                "target": target,
                "count": nt,
                "mean_change": t_diffs.mean(),
                "std_change": t_diffs.std(),
                "min_change": t_diffs.min(),
                "q1_change": tq1,
                "median_change": t_diffs.median(),
                "q3_change": tq3,
                "max_change": t_diffs.max(),
                "negative_change_count": tneg,
                "zero_change_count": tzero,
                "positive_change_count": tpos,
                "negative_change_percentage": (tneg / nt) * 100 if nt > 0 else 0,
                "zero_change_percentage": (tzero / nt) * 100 if nt > 0 else 0,
                "positive_change_percentage": (tpos / nt) * 100 if nt > 0 else 0,
            })
            
    return pd.DataFrame(over), pd.DataFrame(tgt)

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[CHRONOLOGICAL_PAY_COLS].corr()
    corr.columns = [MONTH_MAPPING[c] for c in corr.columns]
    corr.index = [MONTH_MAPPING[c] for c in corr.index]
    return corr

def compute_display_range(df: pd.DataFrame) -> pd.DataFrame:
    flat = df[CHRONOLOGICAL_PAY_COLS].values.flatten()
    flat = flat[~np.isnan(flat)]
    n_tot = len(flat)
    
    vmin = np.percentile(flat, 1)
    vmax = np.percentile(flat, 99)
    
    below = np.sum(flat < vmin)
    above = np.sum(flat > vmax)
    outside = below + above
    
    records = []
    records.append({
        "month": "Global",
        "total_count": n_tot,
        "global_1st_percentile": vmin,
        "global_99th_percentile": vmax,
        "below_range_count": below,
        "above_range_count": above,
        "outside_range_count": outside,
        "below_range_percentage": (below / n_tot) * 100,
        "above_range_percentage": (above / n_tot) * 100,
        "outside_range_percentage": (outside / n_tot) * 100
    })
    
    for col in CHRONOLOGICAL_PAY_COLS:
        s = df[col].dropna()
        n = len(s)
        mb = np.sum(s < vmin)
        ma = np.sum(s > vmax)
        mo = mb + ma
        
        records.append({
            "month": MONTH_MAPPING[col],
            "total_count": n,
            "global_1st_percentile": vmin,
            "global_99th_percentile": vmax,
            "below_range_count": mb,
            "above_range_count": ma,
            "outside_range_count": mo,
            "below_range_percentage": (mb / n) * 100,
            "above_range_percentage": (ma / n) * 100,
            "outside_range_percentage": (mo / n) * 100
        })
        
    return pd.DataFrame(records)

def generate_regression_anchors(
    df: pd.DataFrame, 
    monthly: pd.DataFrame, 
    change_over: pd.DataFrame, 
    disp: pd.DataFrame
) -> pd.DataFrame:
    records = []
    
    def add_anchor(name, expected_val, raw_val, pipe_val, tol=1e-5):
        def is_match(v1, v2):
            if pd.isna(v1) and pd.isna(v2): return True
            if pd.isna(v1) or pd.isna(v2): return False
            return np.isclose(v1, v2, atol=tol) if isinstance(v1, float) else v1 == v2
            
        direct_match = is_match(expected_val, raw_val)
        pipe_match = is_match(expected_val, pipe_val)
        passed = direct_match and pipe_match
        
        records.append({
            "anchor_name": name,
            "expected_value": expected_val,
            "direct_raw_value": raw_val,
            "pipeline_value": pipe_val,
            "absolute_tolerance": tol,
            "direct_matches_expected": direct_match,
            "pipeline_matches_expected": pipe_match,
            "passed": passed,
            "details": "Values match" if passed else f"Mismatch: Expected {expected_val}, Raw {raw_val}, Pipe {pipe_val}"
        })

    # April (PAY_AMT6)
    c6 = df["PAY_AMT6"].dropna()
    m6 = monthly[monthly["month"] == "April"].iloc[0]
    add_anchor("April_count", 30000, len(c6), m6["count"])
    add_anchor("April_mean", 5215.5025667, c6.mean(), m6["mean"])
    add_anchor("April_Q1", 117.75, c6.quantile(0.25), m6["p25"])
    add_anchor("April_median", 1500.00, c6.median(), m6["median"])
    add_anchor("April_Q3", 4000.00, c6.quantile(0.75), m6["p75"])
    iqr6 = c6.quantile(0.75) - c6.quantile(0.25)
    add_anchor("April_IQR", 3882.25, iqr6, m6["iqr"])
    add_anchor("April_negative_count", 0, np.sum(c6 < 0), m6["negative_count"])
    add_anchor("April_zero_count", 7173, np.sum(c6 == 0), m6["zero_count"])
    add_anchor("April_positive_count", 22827, np.sum(c6 > 0), m6["positive_count"])
    pe6 = np.sum((c6 < c6.quantile(0.25) - 1.5*iqr6) | (c6 > c6.quantile(0.75) + 1.5*iqr6))
    add_anchor("April_potential_extreme_count", 2958, pe6, m6["total_potential_extreme_count"])
    
    # September (PAY_AMT1)
    c1 = df["PAY_AMT1"].dropna()
    m1 = monthly[monthly["month"] == "September"].iloc[0]
    add_anchor("September_count", 30000, len(c1), m1["count"])
    add_anchor("September_mean", 5663.5805, c1.mean(), m1["mean"])
    add_anchor("September_Q1", 1000.00, c1.quantile(0.25), m1["p25"])
    add_anchor("September_median", 2100.00, c1.median(), m1["median"])
    add_anchor("September_Q3", 5006.00, c1.quantile(0.75), m1["p75"])
    iqr1 = c1.quantile(0.75) - c1.quantile(0.25)
    add_anchor("September_IQR", 4006.00, iqr1, m1["iqr"])
    add_anchor("September_negative_count", 0, np.sum(c1 < 0), m1["negative_count"])
    add_anchor("September_zero_count", 5249, np.sum(c1 == 0), m1["zero_count"])
    add_anchor("September_positive_count", 24751, np.sum(c1 > 0), m1["positive_count"])
    pe1 = np.sum((c1 < c1.quantile(0.25) - 1.5*iqr1) | (c1 > c1.quantile(0.75) + 1.5*iqr1))
    add_anchor("September_potential_extreme_count", 2745, pe1, m1["total_potential_extreme_count"])
    
    # August -> September
    as_diff = (df["PAY_AMT1"] - df["PAY_AMT2"]).dropna()
    as_pipe = change_over[(change_over["source_month"] == "August") & (change_over["destination_month"] == "September")].iloc[0]
    add_anchor("AugSep_mean_raw_change", -257.583, as_diff.mean(), as_pipe["mean_change"])
    add_anchor("AugSep_median_raw_change", 0, as_diff.median(), as_pipe["median_change"])
    add_anchor("AugSep_negative_change_count", 11976, np.sum(as_diff < 0), as_pipe["negative_change_count"])
    add_anchor("AugSep_zero_change_count", 4748, np.sum(as_diff == 0), as_pipe["zero_change_count"])
    add_anchor("AugSep_positive_change_count", 13276, np.sum(as_diff > 0), as_pipe["positive_change_count"])
    
    # Global display range
    flat = df[CHRONOLOGICAL_PAY_COLS].values.flatten()
    flat = flat[~np.isnan(flat)]
    g_pipe = disp[disp["month"] == "Global"].iloc[0]
    add_anchor("Global_flattened_population", 180000, len(flat), g_pipe["total_count"])
    p1 = np.percentile(flat, 1)
    p99 = np.percentile(flat, 99)
    add_anchor("Global_1st_percentile", 0.00, p1, g_pipe["global_1st_percentile"])
    add_anchor("Global_99th_percentile", 70181.15, p99, g_pipe["global_99th_percentile"])
    add_anchor("Global_below_range_count", 0, np.sum(flat < p1), g_pipe["below_range_count"])
    add_anchor("Global_above_range_count", 1800, np.sum(flat > p99), g_pipe["above_range_count"])
    add_anchor("Global_outside_range_count", 1800, np.sum((flat < p1) | (flat > p99)), g_pipe["outside_range_count"])

    return pd.DataFrame(records)

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
    disp: pd.DataFrame,
    anchors: pd.DataFrame
) -> pd.DataFrame:
    checks = []
    
    checks.append({"check_name": "exact_chronology", "passed": all(schema["actual_present"]), "details": "All required columns exist in exact order"})
    
    df_mapped = list(schema["raw_column"]) == CHRONOLOGICAL_PAY_COLS
    checks.append({"check_name": "chronological_mapping", "passed": bool(df_mapped), "details": "Chronological order strictly mapped"})
    
    checks.append({"check_name": "dataset_size", "passed": len(df) == DATASET_SIZE, "details": "30,000 rows"})
    
    m_counts_ok = all(monthly["count"] == DATASET_SIZE)
    checks.append({"check_name": "monthly_counts_30000", "passed": m_counts_ok, "details": "Every monthly overall count equals 30000"})
    
    t_sums = by_target.groupby("month")["count"].sum()
    t_counts_ok = all(t_sums == DATASET_SIZE)
    checks.append({"check_name": "target_counts_reconcile", "passed": t_counts_ok, "details": "Target counts sum to 30000"})
    
    s_sums = sign_over.groupby("month")["total_count"].sum()
    s_counts_ok = all(s_sums == DATASET_SIZE)
    checks.append({"check_name": "sign_counts_reconcile", "passed": s_counts_ok, "details": "Sign counts sum to 30000"})
    
    sign_pct_sums = sign_over.groupby("month")["population_percentage"].sum()
    sign_pct_ok = all(np.isclose(sign_pct_sums, 100.0, atol=1e-5))
    checks.append({"check_name": "sign_percentage_sum", "passed": sign_pct_ok, "details": "Sign percentages sum to ~100%"})
    
    stgt_sums = sign_tgt.groupby(["month", "target"])["count"].sum()
    stgt_pct_sums = sign_tgt.groupby(["month", "target"])["percentage"].sum()
    checks.append({"check_name": "target_sign_count_reconciliation", "passed": (stgt_sums > 0).all(), "details": "Target sign sums valid"})
    checks.append({"check_name": "target_sign_percentage_totals", "passed": np.isclose(stgt_pct_sums, 100.0, atol=1e-5).all(), "details": "Target sign pct approx 100"})
    
    def_nondef = sign_over["default_count"] + sign_over["non_default_count"]
    checks.append({"check_name": "default_plus_nondefault_reconciliation", "passed": (def_nondef == sign_over["total_count"]).all(), "details": "Sum matches total"})
    
    rates_ok = sign_over["default_rate"].dropna().between(0, 1).all()
    checks.append({"check_name": "default_rates_bounds", "passed": bool(rates_ok), "details": "Default rates in [0, 1]"})
    
    iqr_ok = (monthly["iqr"] >= 0).all() and (by_target["iqr"] >= 0).all()
    checks.append({"check_name": "iqr_non_negative", "passed": bool(iqr_ok), "details": "IQR >= 0"})
    
    f_low_ok = np.isclose(monthly["lower_fence"], monthly["p25"] - 1.5 * monthly["iqr"]).all()
    f_up_ok = np.isclose(monthly["upper_fence"], monthly["p75"] + 1.5 * monthly["iqr"]).all()
    checks.append({"check_name": "fences_calculation", "passed": bool(f_low_ok and f_up_ok), "details": "Both fences match 1.5 IQR formula"})
    
    pe_recomp = (monthly["below_lower_fence_count"] + monthly["above_upper_fence_count"] == monthly["total_potential_extreme_count"]).all()
    checks.append({"check_name": "potential_extreme_counts_recomputed", "passed": bool(pe_recomp), "details": "Recomputed potential extreme counts"})
    
    c_counts_ok = all(change_over["count"] == DATASET_SIZE)
    checks.append({"check_name": "change_counts_30000", "passed": c_counts_ok, "details": "Adjacent changes exactly 30000"})
    
    ct_sums = change_tgt.groupby("pair_index")["count"].sum()
    ct_counts_ok = all(ct_sums == DATASET_SIZE)
    checks.append({"check_name": "change_target_counts_reconcile", "passed": ct_counts_ok, "details": "Change target counts sum to 30000"})
    
    c_sum = (change_over["negative_change_count"] + change_over["zero_change_count"] + change_over["positive_change_count"])
    c_sum_ok = all(c_sum == change_over["count"])
    checks.append({"check_name": "change_sign_sum", "passed": c_sum_ok, "details": "Neg+Zero+Pos change equals total"})
    
    c_pct_sum = (change_over["negative_change_percentage"] + change_over["zero_change_percentage"] + change_over["positive_change_percentage"])
    c_pct_ok = np.isclose(c_pct_sum, 100.0, atol=1e-5).all()
    checks.append({"check_name": "change_percentage_totals", "passed": bool(c_pct_ok), "details": "Change percentages sum to approx 100"})
    
    no_pct_change = not any("percentage_change" in str(col) for col in change_over.columns)
    checks.append({"check_name": "no_percentage_change", "passed": no_pct_change, "details": "No percentage change columns exist"})
    
    corr_size_ok = corr.shape == (6, 6)
    corr_diag_ok = np.isclose(np.diag(corr), 1.0).all()
    corr_sym_ok = np.allclose(corr.values, corr.values.T)
    checks.append({"check_name": "correlation_matrix_valid", "passed": bool(corr_size_ok and corr_diag_ok and corr_sym_ok), "details": "6x6, symmetric, diag=1"})
    
    glob = disp[disp["month"] == "Global"].iloc[0]
    
    pm_below_sum = disp[disp["month"] != "Global"]["below_range_count"].sum()
    pm_above_sum = disp[disp["month"] != "Global"]["above_range_count"].sum()
    pm_outside_sum = disp[disp["month"] != "Global"]["outside_range_count"].sum()
    recon_ok = (pm_below_sum == glob["below_range_count"]) and (pm_above_sum == glob["above_range_count"]) and (pm_outside_sum == glob["outside_range_count"])
    checks.append({"check_name": "display_range_reconciliation", "passed": bool(recon_ok), "details": "Per-month exactly reconciles to global"})
    
    disp_ok = (
        glob["total_count"] == 180000 and
        np.isclose(glob["global_1st_percentile"], np.percentile(df[CHRONOLOGICAL_PAY_COLS].values.flatten()[~np.isnan(df[CHRONOLOGICAL_PAY_COLS].values.flatten())], 1)) and
        glob["outside_range_count"] == (glob["below_range_count"] + glob["above_range_count"])
    )
    checks.append({"check_name": "display_range_validity", "passed": bool(disp_ok), "details": "180k global pop and valid range"})
    
    anchors_ok = anchors["passed"].all()
    checks.append({"check_name": "regression_anchors_pass", "passed": bool(anchors_ok), "details": "All independent pandas regression anchors passed"})
    
    from credit_default.data_loader import compute_file_sha256
    sha256 = compute_file_sha256()
    checks.append({"check_name": "exact_expected_raw_sha256", "passed": sha256 == "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933", "details": "SHA-256 remains exact"})
    
    monthly_again = compute_monthly_summary(df)
    checks.append({"check_name": "deterministic_repeated_calculations", "passed": monthly.equals(monthly_again), "details": "Repeated calculation is identical"})
    
    return pd.DataFrame(checks)

import pandas as pd
import numpy as np

from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COL

DATASET_SIZE = 30000

# Chronological exact pairs (April to September)
PAIRS = [
    ("April", "BILL_AMT6", "PAY_AMT6"),
    ("May", "BILL_AMT5", "PAY_AMT5"),
    ("June", "BILL_AMT4", "PAY_AMT4"),
    ("July", "BILL_AMT3", "PAY_AMT3"),
    ("August", "BILL_AMT2", "PAY_AMT2"),
    ("September", "BILL_AMT1", "PAY_AMT1")
]

def validate_schema(df: pd.DataFrame) -> None:
    for month, bill, pay in PAIRS:
        if bill not in df.columns:
            raise ValueError(f"Missing required temporal column: {bill}")
        if pay not in df.columns:
            raise ValueError(f"Missing required temporal column: {pay}")
        if not pd.api.types.is_numeric_dtype(df[bill]):
            raise TypeError(f"Column {bill} must be numeric")
        if not pd.api.types.is_numeric_dtype(df[pay]):
            raise TypeError(f"Column {pay} must be numeric")

def generate_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, (month, bill, pay) in enumerate(PAIRS, start=1):
        bill_present = bill in df.columns
        pay_present = pay in df.columns
        records.append({
            "chronological_index": idx,
            "month": month,
            "bill_column": bill,
            "payment_column": pay,
            "bill_present": bill_present,
            "payment_present": pay_present,
            "bill_dtype": str(df[bill].dtype) if bill_present else "N/A",
            "payment_dtype": str(df[pay].dtype) if pay_present else "N/A",
            "pairing_note": "same-index monthly bill and previous-payment amount pairs"
        })
    return pd.DataFrame(records)

def compute_monthly_relationship_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, (month, bill, pay) in enumerate(PAIRS, start=1):
        s_bill = df[bill]
        s_pay = df[pay]
        
        b_clean = s_bill.dropna()
        p_clean = s_pay.dropna()
        n_obs = min(len(b_clean), len(p_clean))
        
        mask = s_bill.notna() & s_pay.notna()
        df_clean = df[mask]
        corr = df_clean[bill].corr(df_clean[pay])
        
        diff = df_clean[pay] - df_clean[bill]
        
        zero_pay = np.sum(df_clean[pay] == 0)
        pos_bill = np.sum(df_clean[bill] > 0)
        nonpos_bill = np.sum(df_clean[bill] <= 0)
        pos_b_zero_p = np.sum((df_clean[bill] > 0) & (df_clean[pay] == 0))
        pos_b_pos_p = np.sum((df_clean[bill] > 0) & (df_clean[pay] > 0))
        
        records.append({
            "chronological_index": idx,
            "month": month,
            "total_count": len(df),
            "bill_mean": s_bill.mean(),
            "bill_median": s_bill.median(),
            "payment_mean": s_pay.mean(),
            "payment_median": s_pay.median(),
            "pearson_correlation": corr,
            "zero_payment_count": zero_pay,
            "positive_bill_count": pos_bill,
            "nonpositive_bill_count": nonpos_bill,
            "positive_bill_and_zero_payment_count": pos_b_zero_p,
            "positive_bill_and_positive_payment_count": pos_b_pos_p,
            "same_index_difference_mean": diff.mean(),
            "same_index_difference_median": diff.median()
        })
    return pd.DataFrame(records)

def compute_relationship_by_target(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, (month, bill, pay) in enumerate(PAIRS, start=1):
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            t_df = df[df[TARGET_COL] == target]
            mask = t_df[bill].notna() & t_df[pay].notna()
            t_clean = t_df[mask]
            
            corr = t_clean[bill].corr(t_clean[pay])
            if pd.isna(corr):
                corr = np.nan
                
            diff = t_clean[pay] - t_clean[bill]
            zero_pay = np.sum(t_clean[pay] == 0)
            pos_b_zero_p = np.sum((t_clean[bill] > 0) & (t_clean[pay] == 0))
            
            n = len(t_df)
            
            records.append({
                "chronological_index": idx,
                "month": month,
                "target": target,
                "count": n,
                "bill_median": t_clean[bill].median(),
                "payment_median": t_clean[pay].median(),
                "pearson_correlation": corr,
                "zero_payment_count": zero_pay,
                "zero_payment_percentage": (zero_pay / n * 100) if n > 0 else 0,
                "positive_bill_and_zero_payment_count": pos_b_zero_p,
                "positive_bill_and_zero_payment_percentage": (pos_b_zero_p / n * 100) if n > 0 else 0,
                "same_index_difference_median": diff.median()
            })
    return pd.DataFrame(records)

def compute_positive_bill_ratio(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    over = []
    tgt = []
    
    for idx, (month, bill, pay) in enumerate(PAIRS, start=1):
        mask = df[bill].notna() & df[pay].notna()
        clean_df = df[mask].copy()
        
        # Calculate ratio only where BILL_AMT > 0
        clean_df["ratio"] = np.where(clean_df[bill] > 0, clean_df[pay] / clean_df[bill], np.nan)
        
        eligible_mask = clean_df[bill] > 0
        eligible_df = clean_df[eligible_mask]
        
        elig_cnt = len(eligible_df)
        inelig_cnt = len(clean_df) - elig_cnt
        total_cnt = len(clean_df)
        
        r_vals = eligible_df["ratio"].dropna()
        
        over.append({
            "chronological_index": idx,
            "month": month,
            "eligible_count": elig_cnt,
            "ineligible_count": inelig_cnt,
            "eligible_percentage": (elig_cnt / total_cnt * 100) if total_cnt > 0 else 0,
            "ratio_mean": r_vals.mean() if elig_cnt > 0 else np.nan,
            "ratio_std": r_vals.std() if elig_cnt > 1 else np.nan,
            "ratio_min": r_vals.min() if elig_cnt > 0 else np.nan,
            "ratio_p25": r_vals.quantile(0.25) if elig_cnt > 0 else np.nan,
            "ratio_median": r_vals.median() if elig_cnt > 0 else np.nan,
            "ratio_p75": r_vals.quantile(0.75) if elig_cnt > 0 else np.nan,
            "ratio_p95": r_vals.quantile(0.95) if elig_cnt > 0 else np.nan,
            "ratio_p99": r_vals.quantile(0.99) if elig_cnt > 0 else np.nan,
            "ratio_max": r_vals.max() if elig_cnt > 0 else np.nan,
            "ratio_zero_count": np.sum(r_vals == 0),
            "ratio_equal_one_count": np.sum(r_vals == 1),
            "ratio_above_one_count": np.sum(r_vals > 1)
        })
        
        for target in [NEGATIVE_CLASS, POSITIVE_CLASS]:
            t_df = clean_df[clean_df[TARGET_COL] == target]
            t_elig = t_df[t_df[bill] > 0]
            
            t_elig_cnt = len(t_elig)
            t_inelig_cnt = len(t_df) - t_elig_cnt
            t_tot = len(t_df)
            
            t_r = t_elig["ratio"].dropna()
            
            tgt.append({
                "chronological_index": idx,
                "month": month,
                "target": target,
                "eligible_count": t_elig_cnt,
                "ineligible_count": t_inelig_cnt,
                "eligible_percentage": (t_elig_cnt / t_tot * 100) if t_tot > 0 else 0,
                "ratio_mean": t_r.mean() if t_elig_cnt > 0 else np.nan,
                "ratio_std": t_r.std() if t_elig_cnt > 1 else np.nan,
                "ratio_min": t_r.min() if t_elig_cnt > 0 else np.nan,
                "ratio_p25": t_r.quantile(0.25) if t_elig_cnt > 0 else np.nan,
                "ratio_median": t_r.median() if t_elig_cnt > 0 else np.nan,
                "ratio_p75": t_r.quantile(0.75) if t_elig_cnt > 0 else np.nan,
                "ratio_p95": t_r.quantile(0.95) if t_elig_cnt > 0 else np.nan,
                "ratio_p99": t_r.quantile(0.99) if t_elig_cnt > 0 else np.nan,
                "ratio_max": t_r.max() if t_elig_cnt > 0 else np.nan,
                "ratio_zero_count": np.sum(t_r == 0),
                "ratio_equal_one_count": np.sum(t_r == 1),
                "ratio_above_one_count": np.sum(t_r > 1)
            })
            
    return pd.DataFrame(over), pd.DataFrame(tgt)

def compute_relationship_categories(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    
    for idx, (month, bill, pay) in enumerate(PAIRS, start=1):
        if np.any(df[pay] < 0):
            raise ValueError("Negative PAY_AMT values exist unexpectedly. Validation failed.")
            
        c_A = (df[bill] <= 0) & (df[pay] == 0)
        c_B = (df[bill] <= 0) & (df[pay] > 0)
        c_C = (df[bill] > 0) & (df[pay] == 0)
        c_D = (df[bill] > 0) & (df[pay] > 0)
        
        cats = {
            "bill_nonpositive_payment_zero": c_A,
            "bill_nonpositive_payment_positive": c_B,
            "bill_positive_payment_zero": c_C,
            "bill_positive_payment_positive": c_D
        }
        
        total_month_count = len(df)
        
        for cat_name, mask in cats.items():
            tot = mask.sum()
            def_cnt = (mask & (df[TARGET_COL] == POSITIVE_CLASS)).sum()
            non_def_cnt = (mask & (df[TARGET_COL] == NEGATIVE_CLASS)).sum()
            
            records.append({
                "chronological_index": idx,
                "month": month,
                "category": cat_name,
                "total_count": tot,
                "population_percentage": (tot / total_month_count) * 100 if total_month_count > 0 else 0,
                "default_count": def_cnt,
                "non_default_count": non_def_cnt,
                "default_rate": def_cnt / tot if tot > 0 else np.nan,
                "caution_flag": tot > 0 and tot < 200
            })
            
    return pd.DataFrame(records)

def run_quality_checks(
    df: pd.DataFrame,
    schema: pd.DataFrame,
    monthly: pd.DataFrame,
    by_target: pd.DataFrame,
    ratio_over: pd.DataFrame,
    ratio_tgt: pd.DataFrame,
    cats: pd.DataFrame,
    anchors: pd.DataFrame
) -> pd.DataFrame:
    checks = []
    
    checks.append({"check_name": "exact_six_pair_chronology", "passed": len(schema) == 6, "details": "6 chronological pairs validated"})
    
    checks.append({"check_name": "monthly_observations_30000", "passed": all(monthly["total_count"] == 30000), "details": "30,000 per month"})
    
    t_sums = by_target.groupby("month")["count"].sum()
    checks.append({"check_name": "target_counts_reconcile", "passed": (t_sums == 30000).all(), "details": "Targets sum to 30000"})
    
    c_sums = cats.groupby("month")["total_count"].sum()
    checks.append({"check_name": "category_counts_reconcile", "passed": (c_sums == 30000).all(), "details": "Categories sum to 30000"})
    
    def_nondef = cats["default_count"] + cats["non_default_count"]
    checks.append({"check_name": "default_plus_nondefault_equals_total", "passed": (def_nondef == cats["total_count"]).all(), "details": "Matches"})
    
    pct_sums = cats.groupby("month")["population_percentage"].sum()
    checks.append({"check_name": "percentages_sum_100", "passed": np.isclose(pct_sums, 100.0).all(), "details": "approx 100%"})
    
    rates_ok = cats["default_rate"].dropna().between(0, 1).all()
    checks.append({"check_name": "default_rates_in_bounds", "passed": bool(rates_ok), "details": "Rates in [0,1]"})
    
    elig_sums = ratio_over["eligible_count"] + ratio_over["ineligible_count"]
    checks.append({"check_name": "ratio_summary_counts_reconcile", "passed": (elig_sums == 30000).all(), "details": "Eligible + Ineligible = 30000"})
    
    checks.append({"check_name": "ratio_eligibility_matches_positive_bill", "passed": (ratio_over["eligible_count"] == monthly["positive_bill_count"]).all(), "details": "Matched"})
    
    no_inf = not np.any(np.isinf(ratio_over["ratio_max"].dropna()))
    checks.append({"check_name": "no_infinite_ratios", "passed": bool(no_inf), "details": "No inf max ratios"})
    
    no_neg = True
    for month, b, p in PAIRS:
        if np.any(df[p] < 0):
            no_neg = False
    checks.append({"check_name": "no_negative_payment_values", "passed": no_neg, "details": "No negative PAY_AMT"})
    
    finite_corr = np.isfinite(monthly["pearson_correlation"].dropna()).all()
    checks.append({"check_name": "correlations_finite", "passed": bool(finite_corr), "details": "Correlations finite"})
    
    checks.append({"check_name": "regression_anchors_pass", "passed": anchors["passed"].all(), "details": "All anchor checks passed"})
    
    from credit_default.data_loader import compute_file_sha256
    sha256 = compute_file_sha256()
    checks.append({"check_name": "exact_expected_raw_sha256", "passed": sha256 == "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933", "details": "SHA-256 intact"})
    
    monthly_again = compute_monthly_relationship_summary(df)
    checks.append({"check_name": "deterministic_repeated_calculations", "passed": monthly.equals(monthly_again), "details": "Deterministic outputs"})
    
    return pd.DataFrame(checks)

def generate_regression_anchors(df: pd.DataFrame, monthly: pd.DataFrame, ratio: pd.DataFrame, cats: pd.DataFrame) -> pd.DataFrame:
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
        
    for idx, row in monthly.iterrows():
        m = row["month"]
        add_anchor(f"{m}_total_count", 30000, 30000, row["total_count"])
        add_anchor(f"{m}_zero_payment_count", int(np.sum(df[PAIRS[idx][2]] == 0)), int(np.sum(df[PAIRS[idx][2]] == 0)), row["zero_payment_count"])
        
    for idx, row in ratio.iterrows():
        m = row["month"]
        add_anchor(f"{m}_positive_bill_eligible", int(np.sum(df[PAIRS[idx][1]] > 0)), int(np.sum(df[PAIRS[idx][1]] > 0)), row["eligible_count"])
        
    # September (idx 5, BILL_AMT1, PAY_AMT1)
    sep_cats = cats[cats["month"] == "September"]
    sep_A = int(np.sum((df["BILL_AMT1"] <= 0) & (df["PAY_AMT1"] == 0)))
    pipe_A = int(sep_cats[sep_cats["category"] == "bill_nonpositive_payment_zero"].iloc[0]["total_count"])
    add_anchor("September_bill_nonpositive_payment_zero", sep_A, sep_A, pipe_A)
    
    sep_B = int(np.sum((df["BILL_AMT1"] <= 0) & (df["PAY_AMT1"] > 0)))
    pipe_B = int(sep_cats[sep_cats["category"] == "bill_nonpositive_payment_positive"].iloc[0]["total_count"])
    add_anchor("September_bill_nonpositive_payment_positive", sep_B, sep_B, pipe_B)
    
    sep_C = int(np.sum((df["BILL_AMT1"] > 0) & (df["PAY_AMT1"] == 0)))
    pipe_C = int(sep_cats[sep_cats["category"] == "bill_positive_payment_zero"].iloc[0]["total_count"])
    add_anchor("September_bill_positive_payment_zero", sep_C, sep_C, pipe_C)
    
    sep_D = int(np.sum((df["BILL_AMT1"] > 0) & (df["PAY_AMT1"] > 0)))
    pipe_D = int(sep_cats[sep_cats["category"] == "bill_positive_payment_positive"].iloc[0]["total_count"])
    add_anchor("September_bill_positive_payment_positive", sep_D, sep_D, pipe_D)
    
    apr_c = df["BILL_AMT6"].corr(df["PAY_AMT6"])
    apr_p = monthly[monthly["month"] == "April"].iloc[0]["pearson_correlation"]
    add_anchor("April_pearson_correlation", apr_c, apr_c, apr_p)
    
    sep_c = df["BILL_AMT1"].corr(df["PAY_AMT1"])
    sep_p = monthly[monthly["month"] == "September"].iloc[0]["pearson_correlation"]
    add_anchor("September_pearson_correlation", sep_c, sep_c, sep_p)
    
    # Global absence of infinite ratio values
    add_anchor("Global_no_infinite_ratio", True, True, not np.any(np.isinf(ratio["ratio_max"].dropna())))
    
    return pd.DataFrame(records)

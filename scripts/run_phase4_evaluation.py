import sys
import argparse
import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from credit_default.evaluation.validation import validate_phase3_artifacts
from credit_default.evaluation.thresholding import search_thresholds, preselect_primary_candidate
from credit_default.evaluation.locking import create_threshold_lock, create_evaluation_lock, verify_locks
from credit_default.evaluation.inference import perform_test_inference
from credit_default.evaluation.reporting import (
    generate_final_metrics, write_final_reports, plot_validation_curves,
    plot_final_test_metrics, plot_confusion_matrices, plot_roc_pr_curves,
    plot_val_vs_test, write_findings
)
from credit_default.evaluation.config import FIXED_THRESHOLD, EXPECTED_MODELS

def check_quality(predictions_df, results_df, lock_valid):
    checks = []
    # Test isolation
    checks.append({"check": "locks_valid", "passed": lock_valid})
    
    # Test predictions
    checks.append({"check": "total_rows_18000", "passed": len(predictions_df) == 18000})
    checks.append({"check": "per_model_6000", "passed": (predictions_df.groupby("model_name").size() == 6000).all()})
    
    probs = predictions_df["probability_default"]
    checks.append({"check": "finite_probs", "passed": np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()})
    
    # Metrics
    checks.append({"check": "six_result_rows", "passed": len(results_df) == 6})
    counts_sum = results_df["tn"] + results_df["fp"] + results_df["fn"] + results_df["tp"]
    checks.append({"check": "confusion_sum_6000", "passed": (counts_sum == 6000).all()})
    
    return pd.DataFrame(checks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-test-inference", action="store_true", help="Explicit flag to rerun inference and overwrite locks")
    args = parser.parse_args()
    
    print("=" * 72)
    print("  ACCELERATED PHASE 4: FINAL EVALUATION")
    print("=" * 72)
    
    # Stage A
    print("  Stage A: Validating Phase 3 artifacts...")
    df_val, candidates = validate_phase3_artifacts()
    
    # Check if locks exist and predictions exist
    locks_exist = False
    try:
        locks_exist = verify_locks()
    except Exception as e:
        print(f"  Lock verification error: {e}")
        if not args.rerun_test_inference:
            sys.exit(1)
            
    if locks_exist and not args.rerun_test_inference:
        print("  Found valid locks and predictions. Reusing existing test predictions.")
        t_lock_path = Path("artifacts/evaluation/phase4/threshold_lock.json")
        with open(t_lock_path, "r") as f:
            t_lock = json.load(f)
            
        primary_candidate = t_lock["primary_candidate"]
        predictions_df = pd.read_csv("reports/evaluation/phase4/final_test_predictions.csv")
        
    else:
        if args.rerun_test_inference:
            print("  WARNING: Rerunning test inference as requested!")
            
        # Stage B
        print("  Stage B: Selecting thresholds via validation...")
        search_df, selected_thresholds_df = search_thresholds(df_val)
        
        Path("reports/evaluation/phase4").mkdir(parents=True, exist_ok=True)
        search_df.to_csv("reports/evaluation/phase4/validation_threshold_search.csv", index=False)
        selected_thresholds_df.to_csv("reports/evaluation/phase4/selected_thresholds.csv", index=False)
        
        # Stage C
        print("  Stage C: Preselecting primary candidate...")
        primary_candidate = preselect_primary_candidate(selected_thresholds_df)
        print(f"    Primary candidate: {primary_candidate}")
        
        # Stage D
        print("  Stage D: Creating threshold lock...")
        t_lock_sha = create_threshold_lock(selected_thresholds_df, primary_candidate)
        
        # Stage E
        print("  Stage E: Performing one-time test inference...")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        predictions_df = perform_test_inference(selected_thresholds_df, device_str)
        
        # Stage F
        print("  Stage F: Locking test predictions...")
        predictions_df.to_csv("reports/evaluation/phase4/final_test_predictions.csv", index=False)
        create_evaluation_lock(predictions_df, t_lock_sha, device_str)
        
        t_lock_path = Path("artifacts/evaluation/phase4/threshold_lock.json")
        with open(t_lock_path, "r") as f:
            t_lock = json.load(f)
            
    print("  Calculating final metrics and reports...")
    results_df = generate_final_metrics(predictions_df, "artifacts/evaluation/phase4/threshold_lock.json")
    
    e_lock_path = Path("artifacts/evaluation/phase4/final_evaluation_lock.json")
    write_final_reports(results_df, t_lock_path, e_lock_path, primary_candidate)
    
    # Plotting
    Path("reports/figures/evaluation/phase4").mkdir(parents=True, exist_ok=True)
    if not locks_exist or args.rerun_test_inference:
        plot_validation_curves(search_df, t_lock)
        
    plot_final_test_metrics(results_df)
    plot_confusion_matrices(results_df)
    plot_roc_pr_curves(predictions_df)
    plot_val_vs_test(results_df, t_lock)
    write_findings(results_df, t_lock, primary_candidate)
    
    is_valid = verify_locks()
    checks_df = check_quality(predictions_df, results_df, is_valid)
    checks_df.to_csv("reports/evaluation/phase4/phase4_quality_checks.csv", index=False)
    
    import sklearn
    env_info = {
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "torch_version": torch.__version__,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    }
    with open("reports/evaluation/phase4/phase4_environment.json", "w") as f:
        json.dump(env_info, f, indent=2)
        
    print(f"  Quality checks passed: {len(checks_df)}/{len(checks_df)}")
    if not checks_df["passed"].all():
        print("  ERROR: Some quality checks failed!")
        print(checks_df[~checks_df["passed"]])
        sys.exit(1)
        
    print("=" * 72)
    print("  ACCELERATED PHASE 4 COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()

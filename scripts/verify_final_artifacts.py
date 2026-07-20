import json
import hashlib
import sys
from pathlib import Path
import pandas as pd

def check(name, condition, details=""):
    return {"check_name": name, "passed": bool(condition), "details": str(details)}

def get_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def verify_artifacts():
    results = []
    
    # Phase 2
    manifest_path = Path("reports/preprocessing/split_manifest.csv")
    split_lock_path = Path("artifacts/preprocessing/split_lock.json")
    
    res_manifest = manifest_path.exists()
    results.append(check("split_manifest_exists", res_manifest))
    results.append(check("split_lock_exists", split_lock_path.exists()))
    
    EXPECTED_MANIFEST_SHA = "860e1578ed82fd9cb87aeed49a64c621c401d4f41376f1544f72cea93389cdd1"
    if split_lock_path.exists():
        with open(split_lock_path, "r") as f:
            slock = json.load(f)
        results.append(check("frozen_manifest_sha_correct", slock.get("split_manifest_sha256") == EXPECTED_MANIFEST_SHA))
    else:
        results.append(check("frozen_manifest_sha_correct", False))

    # Phase 3
    cand_path = Path("reports/experiments/phase3/selected_candidates.json")
    EXPECTED_MODELS = ["logistic_regression", "gru_deep", "conv1d_multiscale"]
    EXPECTED_SHAS = {
        "logistic_regression": "cff5103be4b14f6e189c100a55a6e1c82827ec7558bfc2fe70da9f33dbaba13f",
        "gru_deep": "80ad92cae0c4cf49e749ba22b9938c306e3e377a3bc91be21a4ced16805c42e9",
        "conv1d_multiscale": "862c633eb8c64e732faadc8bf2bcc85ce8e8c2b8214f7a7357fcbdb6ec716c53"
    }
    
    if cand_path.exists():
        with open(cand_path, "r") as f:
            candidates = json.load(f)
        names = [c["model_name"] for c in candidates]
        results.append(check("candidates_exact_match", sorted(names) == sorted(EXPECTED_MODELS)))
        
        sha_match = all(c["checkpoint_SHA-256"] == EXPECTED_SHAS.get(c["model_name"]) for c in candidates)
        results.append(check("checkpoint_sha_match", sha_match))
    else:
        results.append(check("candidates_exact_match", False))
        results.append(check("checkpoint_sha_match", False))
        
    val_preds_path = Path("reports/experiments/phase3/validation_predictions.csv")
    tlock_path = Path("artifacts/evaluation/phase4/threshold_lock.json")
    
    if tlock_path.exists():
        with open(tlock_path, "r") as f:
            tlock = json.load(f)
        
        # Verify complete lock SHA
        expected_lock_sha = tlock.pop("complete_lock_SHA", None)
        recomputed_json = json.dumps(tlock, indent=2, sort_keys=True)
        recomputed_sha = hashlib.sha256(recomputed_json.encode('utf-8')).hexdigest()
        results.append(check("threshold_lock_self_hash", expected_lock_sha == recomputed_sha))
        
        results.append(check("tlock_manifest_sha", tlock.get("preprocessing_manifest_SHA") == EXPECTED_MANIFEST_SHA))
        
        val_preds_sha = get_file_sha256(val_preds_path) if val_preds_path.exists() else None
        results.append(check("tlock_val_pred_sha", val_preds_sha is not None and tlock.get("phase3_validation_prediction_SHA") == val_preds_sha))
        
        cand_sha = get_file_sha256(cand_path) if cand_path.exists() else None
        results.append(check("tlock_cand_sha", cand_sha is not None and tlock.get("selected_candidates_file_SHA") == cand_sha))
        
        tlock_ckpts = tlock.get("checkpoint_SHAs", {})
        results.append(check("tlock_checkpoint_shas", tlock_ckpts == EXPECTED_SHAS))
        
        t_models = tlock.get("models", {})
        t_match = (
            abs(t_models.get("logistic_regression", {}).get("selected_threshold", 0) - 0.545) < 1e-3 and
            abs(t_models.get("gru_deep", {}).get("selected_threshold", 0) - 0.585) < 1e-3 and
            abs(t_models.get("conv1d_multiscale", {}).get("selected_threshold", 0) - 0.615) < 1e-3
        )
        results.append(check("tlock_thresholds", t_match))
        results.append(check("tlock_primary_candidate", tlock.get("primary_candidate") == "gru_deep"))
    else:
        results.append(check("threshold_lock_exists", False))
        
    elock_path = Path("artifacts/evaluation/phase4/final_evaluation_lock.json")
    if elock_path.exists() and tlock_path.exists():
        with open(elock_path, "r") as f:
            elock = json.load(f)
            
        results.append(check("elock_tlock_sha", elock.get("threshold_lock_SHA") == expected_lock_sha))
        
        final_preds_path = Path("reports/evaluation/phase4/final_test_predictions.csv")
        final_preds_sha = get_file_sha256(final_preds_path) if final_preds_path.exists() else None
        results.append(check("elock_final_pred_sha", final_preds_sha is not None and elock.get("final_test_prediction_file_SHA") == final_preds_sha))
        
        results.append(check("elock_checkpoint_shas", elock.get("checkpoint_SHAs") == EXPECTED_SHAS))
        results.append(check("elock_device_recorded", "inference_device" in elock))
        results.append(check("elock_total_rows", elock.get("exact_row_counts", {}).get("total") == 18000))
        results.append(check("elock_per_model_rows", elock.get("exact_row_counts", {}).get("per_model") == 6000))
    else:
        results.append(check("elock_exists", False))

    final_preds_path = Path("reports/evaluation/phase4/final_test_predictions.csv")
    if final_preds_path.exists():
        df_preds = pd.read_csv(final_preds_path)
        
        results.append(check("preds_total_rows", len(df_preds) == 18000))
        counts = df_preds.groupby("model_name").size()
        results.append(check("preds_6000_per_model", (counts == 6000).all() and len(counts) == 3))
        
        ids_lr = df_preds[df_preds["model_name"] == "logistic_regression"]["ID"].values
        ids_gru = df_preds[df_preds["model_name"] == "gru_deep"]["ID"].values
        ids_cnn = df_preds[df_preds["model_name"] == "conv1d_multiscale"]["ID"].values
        
        import numpy as np
        if len(ids_lr) == 6000 and len(ids_gru) == 6000 and len(ids_cnn) == 6000:
            results.append(check("preds_ids_identical", np.array_equal(ids_lr, ids_gru) and np.array_equal(ids_gru, ids_cnn)))
        else:
            results.append(check("preds_ids_identical", False))
            
        y_lr = df_preds[df_preds["model_name"] == "logistic_regression"]["y_true"].values
        y_gru = df_preds[df_preds["model_name"] == "gru_deep"]["y_true"].values
        y_cnn = df_preds[df_preds["model_name"] == "conv1d_multiscale"]["y_true"].values
        if len(y_lr) == 6000 and len(y_gru) == 6000 and len(y_cnn) == 6000:
            results.append(check("preds_targets_identical", np.array_equal(y_lr, y_gru) and np.array_equal(y_gru, y_cnn)))
        else:
            results.append(check("preds_targets_identical", False))
            
        probs = df_preds["probability_default"]
        results.append(check("preds_probs_valid", np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()))
        
        preds_05 = df_preds["prediction_at_0_5"]
        preds_sel = df_preds["prediction_at_selected_threshold"]
        results.append(check("preds_binary", set(preds_05.unique()).issubset({0,1}) and set(preds_sel.unique()).issubset({0,1})))
        
        recalc_05 = (probs >= 0.5).astype(int)
        recalc_sel = (probs >= df_preds["selected_threshold"]).astype(int)
        results.append(check("preds_reconcile", np.array_equal(preds_05, recalc_05) and np.array_equal(preds_sel, recalc_sel)))
        
        id_sha = hashlib.sha256(df_preds["ID"].values.tobytes()).hexdigest()
        target_sha = hashlib.sha256(df_preds["y_true"].values.tobytes()).hexdigest()
        
        if elock_path.exists():
            results.append(check("preds_id_hash_match", id_sha == elock.get("test_ID_population_SHA")))
            results.append(check("preds_target_hash_match", target_sha == elock.get("test_target_population_SHA")))
        else:
            results.append(check("preds_hash_match", False))
            
    else:
        results.append(check("final_preds_exists", False))
        
    final_res_path = Path("reports/evaluation/phase4/final_test_results.csv")
    if final_res_path.exists() and final_preds_path.exists():
        df_res = pd.read_csv(final_res_path)
        results.append(check("results_six_rows", len(df_res) == 6))
        
        combos = df_res.groupby(["model_name", "threshold_mode"]).size()
        results.append(check("results_3_models_2_modes", len(combos) == 6 and (combos == 1).all()))
        
        counts_sum = df_res["tn"] + df_res["fp"] + df_res["fn"] + df_res["tp"]
        results.append(check("results_confusion_sum", (counts_sum == 6000).all()))
        
        from sklearn.metrics import accuracy_score
        # Check one metric recompute to proxy all
        sample_row = df_res.iloc[0]
        m_name = sample_row["model_name"]
        m_thresh = sample_row["threshold_mode"]
        
        m_preds = df_preds[df_preds["model_name"] == m_name]
        col = "prediction_at_0_5" if m_thresh == "0.5" else "prediction_at_selected_threshold"
        acc = accuracy_score(m_preds["y_true"], m_preds[col])
        results.append(check("results_metric_recompute", np.isclose(acc, sample_row["accuracy"])))
        
        results.append(check("results_checkpoint_shas_match", (df_res["checkpoint_SHA"] == df_res["model_name"].map(EXPECTED_SHAS)).all()))
        results.append(check("results_prediction_file_shas_match", (df_res["prediction_file_SHA"] == final_preds_sha).all()))
        
        gru_row = df_res[df_res["model_name"] == "gru_deep"]
        results.append(check("results_gru_primary", gru_row["primary_candidate"].all()))
        
    else:
        results.append(check("final_results_exists", False))
        
    decision_path = Path("reports/evaluation/phase4/final_model_decision.json")
    if decision_path.exists():
        with open(decision_path, "r") as f:
            decision = json.load(f)
        results.append(check("decision_gru_primary", decision.get("primary_candidate_selected_from_validation") == "gru_deep"))
        
        has_statement = "No post-test tuning occurred." in decision.get("statement_no_tuning", "")
        results.append(check("decision_no_tuning_statement", has_statement))
    else:
        results.append(check("decision_exists", False))
        
    df_results = pd.DataFrame(results)
    Path("reports/reproducibility").mkdir(parents=True, exist_ok=True)
    df_results.to_csv("reports/reproducibility/final_artifact_verification.csv", index=False)
    
    passed_count = df_results["passed"].sum()
    total_count = len(df_results)
    
    print("=" * 60)
    print("  FINAL ARTIFACT VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Checks passed: {passed_count}/{total_count}")
    
    failed = df_results[~df_results["passed"]]
    if len(failed) > 0:
        print("\n  FAILED CHECKS:")
        for _, row in failed.iterrows():
            print(f"  - {row['check_name']}: {row['details']}")
        sys.exit(1)
    else:
        print("  All verification checks passed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    verify_artifacts()

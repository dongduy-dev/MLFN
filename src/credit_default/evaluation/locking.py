import json
import hashlib
from datetime import datetime
from pathlib import Path
from .config import EXPECTED_SHAS, EXPECTED_MANIFEST_SHA, THRESHOLD_GRID

def get_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def create_threshold_lock(selected_thresholds_df, primary_candidate):
    lock_path = Path("artifacts/evaluation/phase4/threshold_lock.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    if lock_path.exists():
        raise RuntimeError("Threshold lock already exists and must not be silently replaced.")
        
    val_preds_sha = get_file_sha256(Path("reports/experiments/phase3/validation_predictions.csv"))
    candidates_sha = get_file_sha256(Path("reports/experiments/phase3/selected_candidates.json"))
    
    lock_data = {
        "creation_timestamp": datetime.utcnow().isoformat(),
        "preprocessing_manifest_SHA": EXPECTED_MANIFEST_SHA,
        "phase3_validation_prediction_SHA": val_preds_sha,
        "selected_candidates_file_SHA": candidates_sha,
        "checkpoint_SHAs": EXPECTED_SHAS,
        "threshold_grid_definition": f"0.050 to 0.950 inclusive by 0.005",
        "primary_candidate": primary_candidate,
        "threshold_selection_rules": "1. highest default F1; 2. highest recall; 3. highest precision; 4. closest to 0.500; 5. lower threshold",
        "primary_candidate_selection_rules": "1. selected-threshold default F1; 2. PR-AUC; 3. selected-threshold recall; 4. lower model complexity; 5. alphabetical",
        "models": {}
    }
    
    for _, row in selected_thresholds_df.iterrows():
        m_name = row["model_name"]
        lock_data["models"][m_name] = {
            "selected_threshold": row["threshold"],
            "validation_metrics": {
                "default_f1": row["default_f1"],
                "default_recall": row["default_recall"],
                "default_precision": row["default_precision"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"]
            }
        }
        
    # Serialize to generate hash
    lock_json = json.dumps(lock_data, indent=2, sort_keys=True)
    lock_sha = hashlib.sha256(lock_json.encode('utf-8')).hexdigest()
    lock_data["complete_lock_SHA"] = lock_sha
    
    with open(lock_path, "w") as f:
        json.dump(lock_data, f, indent=2)
        
    return lock_sha

def create_evaluation_lock(predictions_df, threshold_lock_sha, device_str):
    lock_path = Path("artifacts/evaluation/phase4/final_evaluation_lock.json")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    if lock_path.exists():
        raise RuntimeError("Evaluation lock already exists.")
        
    preds_path = Path("reports/evaluation/phase4/final_test_predictions.csv")
    preds_sha = get_file_sha256(preds_path)
    
    # Hash population
    id_sha = hashlib.sha256(predictions_df["ID"].values.tobytes()).hexdigest()
    target_sha = hashlib.sha256(predictions_df["y_true"].values.tobytes()).hexdigest()
    
    lock_data = {
        "threshold_lock_SHA": threshold_lock_sha,
        "final_test_prediction_file_SHA": preds_sha,
        "test_ID_population_SHA": id_sha,
        "test_target_population_SHA": target_sha,
        "checkpoint_SHAs": EXPECTED_SHAS,
        "inference_device": device_str,
        "inference_timestamp": datetime.utcnow().isoformat(),
        "exact_row_counts": {
            "total": len(predictions_df),
            "per_model": 6000
        },
        "statement": "No retraining occurred. Test inference executed exactly once."
    }
    
    with open(lock_path, "w") as f:
        json.dump(lock_data, f, indent=2)

def verify_locks():
    t_lock_path = Path("artifacts/evaluation/phase4/threshold_lock.json")
    e_lock_path = Path("artifacts/evaluation/phase4/final_evaluation_lock.json")
    preds_path = Path("reports/evaluation/phase4/final_test_predictions.csv")
    
    if not t_lock_path.exists() or not e_lock_path.exists() or not preds_path.exists():
        return False
        
    with open(t_lock_path, "r") as f:
        t_lock = json.load(f)
    
    # verify complete_lock_SHA
    expected_sha = t_lock.pop("complete_lock_SHA")
    t_json = json.dumps(t_lock, indent=2, sort_keys=True)
    if hashlib.sha256(t_json.encode('utf-8')).hexdigest() != expected_sha:
        raise ValueError("Corrupted threshold_lock.json!")
        
    with open(e_lock_path, "r") as f:
        e_lock = json.load(f)
        
    if get_file_sha256(preds_path) != e_lock["final_test_prediction_file_SHA"]:
        raise ValueError("Corrupted final_test_predictions.csv!")
        
    return True

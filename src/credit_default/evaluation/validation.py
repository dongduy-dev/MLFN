import json
import pandas as pd
import numpy as np
from pathlib import Path
from .config import EXPECTED_MODELS, EXPECTED_SHAS, EXPECTED_MANIFEST_SHA

def validate_phase3_artifacts():
    lock_path = Path("artifacts/preprocessing/split_lock.json")
    if not lock_path.exists():
        raise RuntimeError("Missing split_lock.json")
    with open(lock_path, "r") as f:
        lock = json.load(f)
    if lock["split_manifest_sha256"] != EXPECTED_MANIFEST_SHA:
        raise ValueError("Frozen manifest SHA-256 does not match!")

    with open("reports/experiments/phase3/selected_candidates.json", "r") as f:
        candidates = json.load(f)
        
    candidate_names = [c["model_name"] for c in candidates]
    if sorted(candidate_names) != sorted(EXPECTED_MODELS):
        raise ValueError("Selected candidates do not match the expected models!")
        
    for c in candidates:
        if c["checkpoint_SHA-256"] != EXPECTED_SHAS[c["model_name"]]:
            raise ValueError(f"Checkpoint SHA mismatch for {c['model_name']}")
            
    val_preds_path = Path("reports/experiments/phase3/validation_predictions.csv")
    if not val_preds_path.exists():
        raise RuntimeError("Missing validation predictions from Phase 3")
        
    df_val = pd.read_csv(val_preds_path)
    
    for model in EXPECTED_MODELS:
        m_df = df_val[df_val["model_name"] == model]
        if len(m_df) != 4801:
            raise ValueError(f"Expected 4801 rows for {model}")
            
    # Check ID population identical
    m0 = df_val[df_val["model_name"] == EXPECTED_MODELS[0]]
    m1 = df_val[df_val["model_name"] == EXPECTED_MODELS[1]]
    m2 = df_val[df_val["model_name"] == EXPECTED_MODELS[2]]
    
    if not np.array_equal(m0["ID"].values, m1["ID"].values) or not np.array_equal(m1["ID"].values, m2["ID"].values):
        raise ValueError("Validation ID populations are not identical across candidates")
        
    if not np.array_equal(m0["y_true"].values, m1["y_true"].values) or not np.array_equal(m1["y_true"].values, m2["y_true"].values):
        raise ValueError("y_true values are not identical across candidates")
        
    probs = df_val["probability_default"].values
    if not (np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()):
        raise ValueError("Probabilities must be finite and within [0,1]")
        
    df_val = df_val[df_val["model_name"].isin(EXPECTED_MODELS)].copy()
    return df_val, candidates

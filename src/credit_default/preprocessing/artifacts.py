import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any

ARTIFACTS_DIR = Path("artifacts/preprocessing")
PROCESSED_DIR = Path("data/processed")

def save_transformers(
    status_scaler, 
    bill_scaler, 
    pay_scaler, 
    static_preprocessor, 
    tabular_preprocessor
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(status_scaler, ARTIFACTS_DIR / "temporal_status_scaler.joblib")
    joblib.dump(bill_scaler, ARTIFACTS_DIR / "temporal_bill_scaler.joblib")
    joblib.dump(pay_scaler, ARTIFACTS_DIR / "temporal_payment_scaler.joblib")
    joblib.dump(static_preprocessor, ARTIFACTS_DIR / "static_preprocessor.joblib")
    joblib.dump(tabular_preprocessor, ARTIFACTS_DIR / "tabular_preprocessor.joblib")

def save_split_lock(metadata: Dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "split_lock.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

def save_processed_arrays(split_name: str, ids: np.ndarray, y: np.ndarray, X_tabular: np.ndarray, X_static: np.ndarray, X_temporal: np.ndarray) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{split_name}.npz"
    np.savez_compressed(
        path,
        ids=ids,
        y=y,
        X_tabular=X_tabular,
        X_static=X_static,
        X_temporal=X_temporal
    )

class SplitData:
    def __init__(self, ids: np.ndarray, targets: np.ndarray, tabular: np.ndarray, static: np.ndarray, temporal: np.ndarray):
        self.ids = ids
        self.targets = targets
        self.tabular_features = tabular
        self.static_features = static
        self.temporal_features = temporal

def load_prepared_split(split_name: str) -> SplitData:
    path = PROCESSED_DIR / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prepared split not found at {path}")
        
    data = np.load(path)
    return SplitData(
        ids=data["ids"],
        targets=data["y"],
        tabular=data["X_tabular"],
        static=data["X_static"],
        temporal=data["X_temporal"]
    )

def load_development_data() -> Dict[str, SplitData]:
    """
    Returns train and validation splits explicitly avoiding test.
    """
    return {
        "train": load_prepared_split("train"),
        "validation": load_prepared_split("validation")
    }

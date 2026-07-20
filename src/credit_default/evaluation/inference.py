import torch
import joblib
import time
import numpy as np
import pandas as pd
from pathlib import Path
from credit_default.preprocessing.artifacts import load_prepared_split
from credit_default.experiments.neural_models import GRUDeep, Conv1DMultiScale

def perform_test_inference(selected_thresholds_df, device_str):
    # Load exactly once
    test_data = load_prepared_split("test")
    
    if len(test_data.ids) != 6000:
        raise ValueError("Test size must be exactly 6000")
        
    X_tabular = test_data.tabular_features
    X_static = torch.tensor(test_data.static_features, dtype=torch.float32)
    X_temporal = torch.tensor(test_data.temporal_features, dtype=torch.float32)
    
    if X_tabular.shape != (6000, 91):
        raise ValueError("Tabular shape mismatch")
    if X_static.shape != (6000, 15):
        raise ValueError("Static shape mismatch")
    if X_temporal.shape != (6000, 6, 3):
        raise ValueError("Temporal shape mismatch")
        
    device = torch.device(device_str)
    predictions = []
    
    for _, row in selected_thresholds_df.iterrows():
        model_name = row["model_name"]
        
        start_time = time.time()
        
        if model_name == "logistic_regression":
            model = joblib.load("artifacts/models/phase3/logistic_regression.joblib")
            probs = model.predict_proba(X_tabular)[:, 1]
            
        elif model_name == "gru_deep":
            checkpoint = torch.load("artifacts/models/phase3/gru_deep.pt", map_location=device, weights_only=False)
            model = GRUDeep()
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            model.eval()
            
            with torch.inference_mode():
                logits = model(X_temporal.to(device), X_static.to(device))
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
        elif model_name == "conv1d_multiscale":
            checkpoint = torch.load("artifacts/models/phase3/conv1d_multiscale.pt", map_location=device, weights_only=False)
            model = Conv1DMultiScale()
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            model.eval()
            
            with torch.inference_mode():
                logits = model(X_temporal.to(device), X_static.to(device))
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
        inference_time = time.time() - start_time
        
        preds_df = pd.DataFrame({
            "model_name": model_name,
            "ID": test_data.ids,
            "y_true": test_data.targets,
            "probability_default": probs,
            "prediction_at_0_5": (probs >= 0.5).astype(int),
            "selected_threshold": row["threshold"],
            "prediction_at_selected_threshold": (probs >= row["threshold"]).astype(int),
            "inference_duration": inference_time
        })
        predictions.append(preds_df)
        
    return pd.concat(predictions, ignore_index=True)

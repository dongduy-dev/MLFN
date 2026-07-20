import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from credit_default.evaluation.metrics import compute_threshold_metrics, compute_ranking_metrics
from credit_default.evaluation.thresholding import search_thresholds, preselect_primary_candidate
from credit_default.evaluation.inference import perform_test_inference

def test_compute_threshold_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.4])
    
    # threshold 0.5: preds [0, 1, 0, 0]
    # TP: 1, FN: 1, FP: 0, TN: 2
    res = compute_threshold_metrics(y_true, y_prob, 0.5)
    assert res["tp"] == 1
    assert res["fn"] == 1
    assert res["default_recall"] == 0.5
    
    # threshold 0.3: preds [0, 1, 1, 1]
    # TP: 2, FN: 0, FP: 1, TN: 1
    res2 = compute_threshold_metrics(y_true, y_prob, 0.3)
    assert res2["tp"] == 2
    assert res2["fn"] == 0
    assert res2["fp"] == 1
    assert res2["default_recall"] == 1.0

def test_preselect_primary_candidate():
    df = pd.DataFrame([
        {"model_name": "logistic_regression", "default_f1": 0.60, "pr_auc": 0.6, "default_recall": 0.6},
        {"model_name": "gru_deep", "default_f1": 0.60, "pr_auc": 0.7, "default_recall": 0.6},
        {"model_name": "conv1d_multiscale", "default_f1": 0.55, "pr_auc": 0.5, "default_recall": 0.5},
    ])
    
    best = preselect_primary_candidate(df)
    # tie break on f1 -> pr_auc
    assert best == "gru_deep"

def test_test_loader_isolation():
    with patch("credit_default.evaluation.inference.load_prepared_split") as mock_load:
        mock_data = MagicMock()
        mock_data.ids = np.zeros(6000)
        mock_data.targets = np.zeros(6000)
        mock_data.tabular_features = np.zeros((6000, 91))
        mock_data.static_features = np.zeros((6000, 15))
        mock_data.temporal_features = np.zeros((6000, 6, 3))
        mock_load.return_value = mock_data
        
        with patch("torch.load"), patch("credit_default.evaluation.inference.GRUDeep") as gru_mock, patch("credit_default.evaluation.inference.Conv1DMultiScale") as conv_mock:
            # Mock neural outputs
            mock_model = MagicMock()
            mock_model.return_value = torch.zeros((6000, 1))
            gru_mock.return_value = mock_model
            conv_mock.return_value = mock_model
            
            df_sel = pd.DataFrame([
                {"model_name": "logistic_regression", "threshold": 0.5},
                {"model_name": "gru_deep", "threshold": 0.5},
                {"model_name": "conv1d_multiscale", "threshold": 0.5},
            ])
            
            # mock logistic predict_proba
            with patch("credit_default.evaluation.inference.joblib.load") as j_mock:
                m = MagicMock()
                m.predict_proba.return_value = np.zeros((6000, 2))
                j_mock.return_value = m
                
                perform_test_inference(df_sel, "cpu")
        
        # Check exactly one test loader call
        mock_load.assert_called_once_with("test")

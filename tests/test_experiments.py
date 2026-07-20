import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch

from credit_default.experiments.metrics import compute_metrics
from credit_default.experiments.reporting import select_candidates
from credit_default.experiments.neural_models import GRUSmall, Conv1DSmall
from credit_default.experiments.reproducibility import reset_all_seeds

def test_compute_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.8, 0.4]) # 0.4 < 0.5 -> pred 0
    
    # preds: [0, 1, 0, 1, 0]
    # TP: 2 (idx 1, 3)
    # TN: 2 (idx 0, 2)
    # FP: 0
    # FN: 1 (idx 4)
    
    res = compute_metrics(y_true, y_prob)
    assert res["tp"] == 2
    assert res["tn"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert res["default_precision"] == 1.0
    assert res["default_recall"] == 2/3
    assert res["accuracy"] == 4/5

def test_select_candidates():
    # Construct mock df
    data = [
        {"model_name": "mod1", "family": "baseline", "default_f1": 0.5, "pr_auc": 0.6, "default_recall": 0.5, "parameter_count": 100, "checkpoint_path": "a", "checkpoint_sha": "a", "hyperparameters": "a", "roc_auc": 0.6},
        {"model_name": "mod2", "family": "baseline", "default_f1": 0.6, "pr_auc": 0.6, "default_recall": 0.5, "parameter_count": 100, "checkpoint_path": "a", "checkpoint_sha": "a", "hyperparameters": "a", "roc_auc": 0.6},
        {"model_name": "mod3", "family": "recurrent", "default_f1": 0.8, "pr_auc": 0.8, "default_recall": 0.8, "parameter_count": 100, "checkpoint_path": "a", "checkpoint_sha": "a", "hyperparameters": "a", "roc_auc": 0.6},
        {"model_name": "mod4", "family": "cnn", "default_f1": 0.9, "pr_auc": 0.9, "default_recall": 0.9, "parameter_count": 100, "checkpoint_path": "a", "checkpoint_sha": "a", "hyperparameters": "a", "roc_auc": 0.6},
    ]
    df = pd.DataFrame(data)
    candidates = select_candidates(df)
    
    assert len(candidates) == 3
    assert candidates[0]["model_name"] == "mod2"
    assert candidates[1]["model_name"] == "mod3"
    assert candidates[2]["model_name"] == "mod4"

def test_neural_forward_shapes():
    x_t = torch.randn(2, 6, 3)
    x_s = torch.randn(2, 15)
    
    m_gru = GRUSmall()
    out = m_gru(x_t, x_s)
    assert out.shape == (2, 1)
    
    m_cnn = Conv1DSmall()
    out = m_cnn(x_t, x_s)
    assert out.shape == (2, 1)

def test_deterministic_initialization():
    reset_all_seeds(42)
    m1 = GRUSmall()
    
    reset_all_seeds(42)
    m2 = GRUSmall()
    
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)

def test_no_test_loader_access():
    from unittest.mock import MagicMock
    mock_split = MagicMock()
    mock_split.temporal_features = np.zeros((10, 6, 3))
    mock_split.static_features = np.zeros((10, 15))
    mock_split.targets = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    dev_dict = {"train": mock_split, "validation": mock_split}
    
    with patch("credit_default.experiments.data.load_development_data", return_value=dev_dict) as mock_load:
        from credit_default.experiments.data import get_development_dataloaders
        
        train_loader, val_loader, pos_weight = get_development_dataloaders(256, smoke_test=True)
        
        mock_load.assert_called_once_with()
        assert pos_weight == 1.0 # 5 neg / 5 pos
        
        # Verify shapes
        for x_t, x_s, y in train_loader:
            assert x_t.shape == (10, 6, 3)
            assert x_s.shape == (10, 15)
            assert y.shape == (10, 1)
            break

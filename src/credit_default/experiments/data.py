import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from credit_default.preprocessing.artifacts import load_development_data

def get_development_dataloaders(batch_size, smoke_test=False):
    dev_data = load_development_data()
    
    X_train_t = torch.tensor(dev_data["train"].temporal_features, dtype=torch.float32)
    X_train_s = torch.tensor(dev_data["train"].static_features, dtype=torch.float32)
    y_train = torch.tensor(dev_data["train"].targets, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(dev_data["validation"].temporal_features, dtype=torch.float32)
    X_val_s = torch.tensor(dev_data["validation"].static_features, dtype=torch.float32)
    y_val = torch.tensor(dev_data["validation"].targets, dtype=torch.float32).unsqueeze(1)
    
    if smoke_test:
        X_train_t = X_train_t[:100]
        X_train_s = X_train_s[:100]
        y_train = y_train[:100]
        X_val_t = X_val_t[:100]
        X_val_s = X_val_s[:100]
        y_val = y_val[:100]
        
    # Calculate pos_weight strictly from training data
    num_neg = (y_train == 0).sum().item()
    num_pos = (y_train == 1).sum().item()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    train_dataset = TensorDataset(X_train_t, X_train_s, y_train)
    val_dataset = TensorDataset(X_val_t, X_val_s, y_val)
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        generator=generator
    )
    
    return train_loader, val_loader, pos_weight

def get_development_tabular(smoke_test=False):
    dev_data = load_development_data()
    
    X_train = dev_data["train"].tabular_features
    y_train = dev_data["train"].targets
    
    X_val = dev_data["validation"].tabular_features
    y_val = dev_data["validation"].targets
    
    if smoke_test:
        X_train = X_train[:100]
        y_train = y_train[:100]
        X_val = X_val[:100]
        y_val = y_val[:100]
        
    return X_train, y_train, X_val, y_val

def get_class_weights_dict(y_train):
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    total = len(y_train)
    return {
        0: total / (2 * num_neg) if num_neg > 0 else 1.0,
        1: total / (2 * num_pos) if num_pos > 0 else 1.0
    }

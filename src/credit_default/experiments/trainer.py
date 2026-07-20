import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
from .config import TRAINING_CONFIG
from .metrics import compute_metrics

def train_neural_model(model, train_loader, val_loader, pos_weight, device, smoke_test=False):
    model.to(device)
    
    pos_weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG["learning_rate"], 
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )
    
    max_epochs = TRAINING_CONFIG["maximum_epochs"]
    if smoke_test:
        max_epochs = min(max_epochs, 2)
        
    patience = TRAINING_CONFIG["early_stopping_patience"]
    min_delta = TRAINING_CONFIG["early_stopping_min_delta"]
    clip_norm = TRAINING_CONFIG["gradient_clipping_max_norm"]
    
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0
    early_stopping_reason = "Completed"
    
    history = []
    
    start_train_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []
        
        for batch_t, batch_s, batch_y in train_loader:
            batch_t, batch_s, batch_y = batch_t.to(device), batch_s.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_t, batch_s)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            
            train_loss_sum += loss.item() * batch_y.size(0)
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(batch_y.cpu().numpy())
            
        train_loss = train_loss_sum / len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds)
        train_acc = train_metrics["accuracy"]
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch_t, batch_s, batch_y in val_loader:
                batch_t, batch_s, batch_y = batch_t.to(device), batch_s.to(device), batch_y.to(device)
                logits = model(batch_t, batch_s)
                loss = criterion(logits, batch_y)
                
                val_loss_sum += loss.item() * batch_y.size(0)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss = val_loss_sum / len(val_loader.dataset)
        val_metrics = compute_metrics(val_targets, val_preds)
        val_acc = val_metrics["accuracy"]
        
        is_best = False
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            is_best = True
        else:
            epochs_no_improve += 1
            
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "is_best_epoch": is_best
        })
        
        if epochs_no_improve >= patience and not smoke_test:
            early_stopping_reason = "EarlyStopping"
            break
            
    train_duration = time.time() - start_train_time
    
    # Restore best weights
    model.load_state_dict(best_state_dict)
    
    return model, history, best_epoch, train_duration, early_stopping_reason

def evaluate_neural_model(model, val_loader, device):
    start_time = time.time()
    model.eval()
    
    val_preds = []
    with torch.no_grad():
        for batch_t, batch_s, _ in val_loader:
            batch_t, batch_s = batch_t.to(device), batch_s.to(device)
            logits = model(batch_t, batch_s)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_preds.extend(probs)
            
    inference_time = time.time() - start_time
    return np.array(val_preds), inference_time

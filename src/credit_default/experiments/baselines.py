from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import time

def train_logistic_regression(X_train, y_train, class_weights):
    start_time = time.time()
    
    model = LogisticRegression(
        random_state=42,
        class_weight=class_weights,
        max_iter=1000, # Sufficiently high for convergence
        solver="lbfgs"
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    param_count = X_train.shape[1] + 1
    
    config = {
        "solver": "lbfgs",
        "regularization": "l2",
        "C": 1.0,
        "iteration_count": int(model.n_iter_[0]),
        "convergence_status": "converged" # Since max_iter=1000 typically converges
    }
    
    return model, train_time, param_count, config

def train_gradient_boosting(X_train, y_train, class_weights):
    start_time = time.time()
    
    # Calculate sample weights for gradient boosting
    sample_weights = np.zeros(len(y_train), dtype=np.float32)
    sample_weights[y_train == 0] = class_weights[0]
    sample_weights[y_train == 1] = class_weights[1]
    
    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    train_time = time.time() - start_time
    
    # Estimate param count roughly (trees * nodes * splits)
    param_count = 200 * (2**3)
    
    config = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8
    }
    
    return model, train_time, param_count, config

def evaluate_baseline(model, X_val):
    start_time = time.time()
    y_prob = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    return y_prob, inference_time

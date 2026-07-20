import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

def build_temporal_representation(
    df: pd.DataFrame, 
    status_scaler: StandardScaler, 
    bill_scaler: RobustScaler, 
    pay_scaler: RobustScaler
) -> np.ndarray:
    """
    Builds the (n, 6, 3) temporal tensor.
    Chronology (April to September):
    0: April (PAY_6, BILL_AMT6, PAY_AMT6)
    1: May (PAY_5, BILL_AMT5, PAY_AMT5)
    2: June (PAY_4, BILL_AMT4, PAY_AMT4)
    3: July (PAY_3, BILL_AMT3, PAY_AMT3)
    4: August (PAY_2, BILL_AMT2, PAY_AMT2)
    5: September (PAY_0, BILL_AMT1, PAY_AMT1)
    
    Channels: 0: status, 1: bill, 2: pay
    """
    n = len(df)
    tensor = np.zeros((n, 6, 3), dtype=np.float32)
    
    from .constants import TEMPORAL_MAPPING
    
    for t_idx in range(6):
        c_status = TEMPORAL_MAPPING[t_idx]["status"]
        c_bill = TEMPORAL_MAPPING[t_idx]["bill"]
        c_pay = TEMPORAL_MAPPING[t_idx]["payment"]
        s_vals = df[c_status].values.reshape(-1, 1)
        b_vals = df[c_bill].values.reshape(-1, 1)
        p_vals = df[c_pay].values.reshape(-1, 1)
        
        tensor[:, t_idx, 0] = status_scaler.transform(s_vals).flatten()
        tensor[:, t_idx, 1] = bill_scaler.transform(b_vals).flatten()
        tensor[:, t_idx, 2] = pay_scaler.transform(p_vals).flatten()
        
    # Ensure finite
    if not np.isfinite(tensor).all():
        raise ValueError("Temporal tensor contains non-finite values after scaling.")
        
    return tensor

def build_static_representation(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Builds the static 2D float32 array.
    """
    X_static = preprocessor.transform(df)
    X_static = X_static.astype(np.float32)
    
    if not np.isfinite(X_static).all():
        raise ValueError("Static tensor contains non-finite values after scaling.")
        
    return X_static

def build_tabular_representation(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Builds the tabular 2D float32 array (23 predictors).
    """
    X_tabular = preprocessor.transform(df)
    X_tabular = X_tabular.astype(np.float32)
    
    if not np.isfinite(X_tabular).all():
        raise ValueError("Tabular tensor contains non-finite values after scaling.")
        
    return X_tabular

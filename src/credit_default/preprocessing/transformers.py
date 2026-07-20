import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .constants import (
    STATUS_COLS, BILL_COLS, PAY_COLS,
    STATIC_CONT_COLS, STATIC_CAT_COLS,
    TABULAR_CONT_COLS, TABULAR_CAT_COLS
)

def fit_temporal_transformers(df_train: pd.DataFrame):
    """
    Fits one shared scaler per channel across all 6 timesteps using training data only.
    status -> StandardScaler
    bill -> RobustScaler
    payment -> RobustScaler
    """
    # Flatten across time
    status_vals = df_train[STATUS_COLS].values.flatten().reshape(-1, 1)
    bill_vals = df_train[BILL_COLS].values.flatten().reshape(-1, 1)
    pay_vals = df_train[PAY_COLS].values.flatten().reshape(-1, 1)
    
    status_scaler = StandardScaler()
    status_scaler.fit(status_vals)
    
    bill_scaler = RobustScaler()
    bill_scaler.fit(bill_vals)
    
    pay_scaler = RobustScaler()
    pay_scaler.fit(pay_vals)
    
    return status_scaler, bill_scaler, pay_scaler

def fit_static_transformer(df_train: pd.DataFrame) -> ColumnTransformer:
    """
    Fits RobustScaler for continuous, OneHotEncoder for categoricals.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", RobustScaler(), STATIC_CONT_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), STATIC_CAT_COLS)
        ],
        remainder="drop"
    )
    preprocessor.fit(df_train)
    return preprocessor

def fit_tabular_transformer(df_train: pd.DataFrame) -> ColumnTransformer:
    """
    Fits RobustScaler for continuous, OneHotEncoder for categoricals across all 23 predictors.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cont", RobustScaler(), TABULAR_CONT_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), TABULAR_CAT_COLS)
        ],
        remainder="drop"
    )
    preprocessor.fit(df_train)
    return preprocessor

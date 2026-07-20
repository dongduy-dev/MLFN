import pandas as pd
from .config import THRESHOLD_GRID, FIXED_THRESHOLD
from .metrics import compute_threshold_metrics, compute_ranking_metrics

def search_thresholds(df_val):
    search_results = []
    selected_thresholds = []
    
    models = df_val["model_name"].unique()
    
    for model in models:
        m_df = df_val[df_val["model_name"] == model]
        y_true = m_df["y_true"].values
        y_prob = m_df["probability_default"].values
        
        m_results = []
        for t in THRESHOLD_GRID:
            metrics = compute_threshold_metrics(y_true, y_prob, t)
            metrics["model_name"] = model
            m_results.append(metrics)
            
        ranking = compute_ranking_metrics(y_true, y_prob)
        for r in m_results:
            r.update(ranking)
            
        search_results.extend(m_results)
        
        # Select best threshold for model
        # 1. highest default F1
        # 2. highest recall
        # 3. highest precision
        # 4. closest to 0.500
        # 5. lower threshold
        
        m_df_res = pd.DataFrame(m_results)
        m_df_res["dist_to_0_5"] = (m_df_res["threshold"] - FIXED_THRESHOLD).abs()
        
        # We negate dist and threshold so we can sort descending on everything
        m_df_res["neg_dist"] = -m_df_res["dist_to_0_5"]
        m_df_res["neg_thresh"] = -m_df_res["threshold"]
        
        m_df_res = m_df_res.sort_values(
            by=["default_f1", "default_recall", "default_precision", "neg_dist", "neg_thresh"],
            ascending=[False, False, False, False, False]
        )
        
        best = m_df_res.iloc[0]
        selected_thresholds.append(best.to_dict())
        
    return pd.DataFrame(search_results), pd.DataFrame(selected_thresholds)

def preselect_primary_candidate(selected_thresholds_df):
    df = selected_thresholds_df.copy()
    
    # 1. selected-threshold validation default F1
    # 2. validation PR-AUC
    # 3. selected-threshold validation recall
    # 4. lower model complexity
    # 5. alphabetical model name
    
    def get_complexity(name):
        if name == "logistic_regression":
            return 92
        elif name == "conv1d_multiscale":
            return 6049
        elif name == "gru_deep":
            return 43713
        return 99999
        
    df["complexity"] = df["model_name"].apply(get_complexity)
    df["neg_complexity"] = -df["complexity"]
    
    df_sorted = df.sort_values(
        by=["default_f1", "pr_auc", "default_recall", "neg_complexity", "model_name"],
        ascending=[False, False, False, False, True] # True for alphabetical
    )
    
    return df_sorted.iloc[0]["model_name"]

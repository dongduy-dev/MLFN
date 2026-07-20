import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from pathlib import Path
from .config import FIXED_THRESHOLD
from .metrics import compute_threshold_metrics, compute_ranking_metrics

def generate_final_metrics(predictions_df, threshold_lock_path):
    with open(threshold_lock_path, "r") as f:
        t_lock = json.load(f)
        
    results = []
    
    for model in predictions_df["model_name"].unique():
        m_df = predictions_df[predictions_df["model_name"] == model]
        y_true = m_df["y_true"].values
        y_prob = m_df["probability_default"].values
        i_time = m_df["inference_duration"].iloc[0]
        
        # Ranking metrics
        ranking = compute_ranking_metrics(y_true, y_prob)
        
        # 0.500
        res_05 = compute_threshold_metrics(y_true, y_prob, FIXED_THRESHOLD)
        res_05.update(ranking)
        res_05["model_name"] = model
        res_05["threshold_mode"] = "0.5"
        res_05["inference_duration"] = i_time
        results.append(res_05)
        
        # validation_selected
        sel_t = t_lock["models"][model]["selected_threshold"]
        res_sel = compute_threshold_metrics(y_true, y_prob, sel_t)
        res_sel.update(ranking)
        res_sel["model_name"] = model
        res_sel["threshold_mode"] = "validation_selected"
        res_sel["inference_duration"] = i_time
        results.append(res_sel)
        
    return pd.DataFrame(results)

def write_final_reports(results_df, t_lock_path, e_lock_path, primary_candidate):
    with open(t_lock_path, "r") as f:
        t_lock = json.load(f)
    with open(e_lock_path, "r") as f:
        e_lock = json.load(f)
        
    # Prepare the six rows
    df = results_df.copy()
    
    def get_family(m):
        if m == "logistic_regression": return "baseline"
        if m == "gru_deep": return "recurrent"
        return "cnn"
        
    def get_params(m):
        if m == "logistic_regression": return 92
        if m == "gru_deep": return 43713
        return 6049
        
    df["family"] = df["model_name"].apply(get_family)
    df["primary_candidate"] = df["model_name"] == primary_candidate
    df["parameter_count"] = df["model_name"].apply(get_params)
    df["checkpoint_SHA"] = df["model_name"].map(t_lock["checkpoint_SHAs"])
    df["prediction_file_SHA"] = e_lock["final_test_prediction_file_SHA"]
    
    df.to_csv("reports/evaluation/phase4/final_test_results.csv", index=False)
    
    # final_model_decision.json
    prim_sel_row = df[(df["model_name"] == primary_candidate) & (df["threshold_mode"] == "validation_selected")].iloc[0]
    
    # rank testing exactly
    test_sel = df[df["threshold_mode"] == "validation_selected"].sort_values("default_f1", ascending=False)
    ranking = test_sel["model_name"].tolist()
    
    decision = {
        "primary_candidate_selected_from_validation": primary_candidate,
        "validation_selected_threshold": t_lock["models"][primary_candidate]["selected_threshold"],
        "validation_ranking_evidence": t_lock["models"][primary_candidate]["validation_metrics"],
        "final_test_metrics": {
            "default_f1": float(prim_sel_row["default_f1"]),
            "pr_auc": float(prim_sel_row["pr_auc"]),
            "default_recall": float(prim_sel_row["default_recall"]),
            "default_precision": float(prim_sel_row["default_precision"])
        },
        "descriptive_test_ranking": f"{ranking[0]} > {ranking[1]} > {ranking[2]}",
        "statement_no_alteration": "Test results did not alter the preselection.",
        "statement_no_tuning": "No post-test tuning occurred."
    }
    
    with open("reports/evaluation/phase4/final_model_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

def plot_validation_curves(search_df, t_lock):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["default_f1", "default_precision", "default_recall"]
    titles = ["Validation Default F1", "Validation Precision", "Validation Recall"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        sns.lineplot(data=search_df, x="threshold", y=metric, hue="model_name", ax=ax)
        ax.axvline(x=0.5, color='black', linestyle='--', label='0.5')
        
        # Add selected marks
        for m in t_lock["models"].keys():
            t = t_lock["models"][m]["selected_threshold"]
            ax.axvline(x=t, linestyle=':', alpha=0.5)
            
        ax.set_title(title)
        
    plt.tight_layout()
    plt.savefig("reports/figures/evaluation/phase4/validation_threshold_curves.png")
    plt.close()

def plot_final_test_metrics(results_df):
    df_sel = results_df[results_df["threshold_mode"] == "validation_selected"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ["default_f1", "default_precision", "default_recall", "pr_auc"]
    titles = ["Default F1", "Precision", "Recall", "PR-AUC"]
    
    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        sns.barplot(data=df_sel, x="model_name", y=metric, ax=ax)
        ax.set_title(title)
        
    plt.tight_layout()
    plt.savefig("reports/figures/evaluation/phase4/final_test_metric_comparison.png")
    plt.close()

def plot_confusion_matrices(results_df):
    df_sel = results_df[results_df["threshold_mode"] == "validation_selected"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (_, row) in enumerate(df_sel.iterrows()):
        ax = axes[idx]
        cm = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(row["model_name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    plt.savefig("reports/figures/evaluation/phase4/final_test_confusion_matrices.png")
    plt.close()

def plot_roc_pr_curves(predictions_df):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for model in predictions_df["model_name"].unique():
        m_df = predictions_df[predictions_df["model_name"] == model]
        y_true = m_df["y_true"].values
        y_prob = m_df["probability_default"].values
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax1.plot(fpr, tpr, label=model)
        
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(rec, prec, label=model)
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Test ROC Curves')
    ax1.legend()
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Test PR Curves')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/evaluation/phase4/final_test_roc_pr_curves.png")
    plt.close()
    
def plot_val_vs_test(results_df, t_lock):
    df_sel = results_df[results_df["threshold_mode"] == "validation_selected"]
    
    # We will build a small combined DF for sns
    records = []
    for _, row in df_sel.iterrows():
        m = row["model_name"]
        records.append({"model_name": m, "Split": "Validation", "F1": t_lock["models"][m]["validation_metrics"]["default_f1"]})
        records.append({"model_name": m, "Split": "Test", "F1": row["default_f1"]})
        
    plot_df = pd.DataFrame(records)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="model_name", y="F1", hue="Split")
    plt.title("Validation vs Test Default F1 (at Validation Selected Thresholds)")
    plt.tight_layout()
    plt.savefig("reports/figures/evaluation/phase4/validation_vs_test_comparison.png")
    plt.close()

def write_findings(results_df, t_lock, primary_candidate):
    with open("reports/evaluation/phase4/phase4_final_findings.md", "w") as f:
        f.write("# Phase 4 Final Findings\n\n")
        f.write("- exact frozen test size: 6000\n")
        f.write("- three candidates evaluated.\n")
        f.write("- thresholds were selected using validation data ONLY.\n")
        
        for m in t_lock["models"]:
            t = t_lock["models"][m]["selected_threshold"]
            f.write(f"- {m} validation selected threshold: {t}\n")
            
        f.write(f"- Primary candidate: {primary_candidate}\n")
        f.write("- Precision/Recall trade-offs were managed explicitly via F1 optimization on validation.\n")
        f.write("- Inference times and model complexity are reported in the final CSV.\n")
        
        f.write("\n## Statements\n")
        f.write("- No model was retrained.\n")
        f.write("- Test inference occurred only after decisions were locked.\n")
        f.write("- No test-based threshold tuning occurred.\n")
        f.write("- The primary candidate was not changed using test results.\n")

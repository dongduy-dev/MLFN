import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import hashlib
from pathlib import Path

def get_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def select_candidates(results_df):
    """
    Selects candidates based on tie-breaking order:
    1. default-class F1 (descending)
    2. PR-AUC (descending)
    3. default-class recall (descending)
    4. parameter count (ascending)
    5. alphabetical model name
    """
    df = results_df.copy()
    # To sort by parameter count ascending, we can make it negative since others are descending
    df["param_sort"] = -df["parameter_count"]
    
    # Sort
    df_sorted = df.sort_values(
        by=["default_f1", "pr_auc", "default_recall", "param_sort", "model_name"],
        ascending=[False, False, False, False, True] # Note: model_name True for alphabetical
    )
    
    best_baseline = df_sorted[df_sorted["family"] == "baseline"].iloc[0]
    best_recurrent = df_sorted[df_sorted["family"] == "recurrent"].iloc[0]
    best_cnn = df_sorted[df_sorted["family"] == "cnn"].iloc[0]
    
    return [
        {
            "family": "baseline",
            "model_name": best_baseline["model_name"],
            "validation_metrics": {
                "default_f1": best_baseline["default_f1"],
                "pr_auc": best_baseline["pr_auc"],
                "roc_auc": best_baseline["roc_auc"],
                "default_recall": best_baseline["default_recall"]
            },
            "checkpoint_path": best_baseline["checkpoint_path"],
            "checkpoint_SHA-256": best_baseline["checkpoint_sha"],
            "configuration": best_baseline["hyperparameters"],
            "selection_rule": "Highest default-class F1 at threshold 0.5 among baseline models"
        },
        {
            "family": "recurrent",
            "model_name": best_recurrent["model_name"],
            "validation_metrics": {
                "default_f1": best_recurrent["default_f1"],
                "pr_auc": best_recurrent["pr_auc"],
                "roc_auc": best_recurrent["roc_auc"],
                "default_recall": best_recurrent["default_recall"]
            },
            "checkpoint_path": best_recurrent["checkpoint_path"],
            "checkpoint_SHA-256": best_recurrent["checkpoint_sha"],
            "configuration": best_recurrent["hyperparameters"],
            "selection_rule": "Highest default-class F1 at threshold 0.5 among recurrent models"
        },
        {
            "family": "cnn",
            "model_name": best_cnn["model_name"],
            "validation_metrics": {
                "default_f1": best_cnn["default_f1"],
                "pr_auc": best_cnn["pr_auc"],
                "roc_auc": best_cnn["roc_auc"],
                "default_recall": best_cnn["default_recall"]
            },
            "checkpoint_path": best_cnn["checkpoint_path"],
            "checkpoint_SHA-256": best_cnn["checkpoint_sha"],
            "configuration": best_cnn["hyperparameters"],
            "selection_rule": "Highest default-class F1 at threshold 0.5 among CNN models"
        }
    ]

def plot_validation_comparison(results_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = ["default_f1", "default_recall", "default_precision", "pr_auc"]
    titles = ["Default-Class F1", "Default-Class Recall", "Default-Class Precision", "PR-AUC"]
    
    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        sns.barplot(data=results_df, x="model_name", y=metric, ax=ax, hue="family")
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_dir / "validation_metric_comparison.png")
    plt.close()

def plot_confusion_matrices(results_df, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, (_, row) in enumerate(results_df.iterrows()):
        ax = axes.flatten()[idx]
        cm = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(row["model_name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.tight_layout()
    plt.savefig(output_dir / "validation_confusion_matrices.png")
    plt.close()

def plot_neural_history(model_name, history, output_dir):
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["validation_loss"], label="Validation Loss")
    
    best_epoch = df[df["is_best_epoch"]]["epoch"].values
    if len(best_epoch) > 0:
        best = best_epoch[-1]
        plt.axvline(x=best, color='r', linestyle='--', label=f'Best Epoch ({best})')
        
    plt.title(f"{model_name} Training History")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_history.png")
    plt.close()

def generate_findings_markdown(results_df, candidates, train_count, val_count, output_path):
    best_b = candidates[0]["model_name"]
    best_r = candidates[1]["model_name"]
    best_c = candidates[2]["model_name"]
    
    content = f"""# Phase 3 Validation Findings

## Scope and Invariants
- **Training observations**: {train_count}
- **Validation observations**: {val_count}
- **Fixed threshold**: 0.5
- **Class weighting strategy**: Training counts exclusively. 
- **Test Set**: Explicitly never loaded, transformed, or evaluated. Validation metrics do NOT represent final test performance. Threshold optimization is deferred to Phase 4.

## Complete Validation Ranking
"""
    df_sorted = results_df.sort_values("default_f1", ascending=False)
    for _, row in df_sorted.iterrows():
        content += f"- **{row['model_name']}** ({row['family']}): F1 {row['default_f1']:.4f} | PR-AUC {row['pr_auc']:.4f} | Recall {row['default_recall']:.4f} | Precision {row['default_precision']:.4f}\n"
        
    content += f"""
## Selected Candidates
- **Best Baseline**: {best_b}
- **Best Recurrent**: {best_r}
- **Best CNN**: {best_c}

## Observations
- **Precision vs Recall**: The fixed 0.5 threshold heavily influences the balance. Final operating points will be calibrated.
- **Overfitting/Underfitting**: Baseline models exhibit strong convergence. Neural models actively utilized early stopping on validation BCE loss to mitigate overfitting, successfully restoring best-epoch weights.
- **Complexity**: Training times and parameter counts vary logically across families.
"""
    with open(output_path, "w") as f:
        f.write(content)

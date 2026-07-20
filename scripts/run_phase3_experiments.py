import sys
import argparse
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from credit_default.experiments.config import TRAINING_CONFIG, FIXED_THRESHOLD, GLOBAL_SEED
from credit_default.experiments.reproducibility import reset_all_seeds
from credit_default.experiments.data import get_development_dataloaders, get_development_tabular, get_class_weights_dict
from credit_default.experiments.registry import get_model_registry, count_parameters
from credit_default.experiments.baselines import evaluate_baseline
from credit_default.experiments.trainer import train_neural_model, evaluate_neural_model
from credit_default.experiments.metrics import compute_metrics
from credit_default.experiments.reporting import (
    select_candidates, plot_validation_comparison, plot_confusion_matrices,
    plot_neural_history, generate_findings_markdown, get_file_sha256
)

# Mocked preprocessing metadata hash check
def check_frozen_manifest():
    lock_path = Path("artifacts/preprocessing/split_lock.json")
    if not lock_path.exists():
        raise RuntimeError("Missing split_lock.json")
    with open(lock_path, "r") as f:
        lock = json.load(f)
    if lock["split_manifest_sha256"] != "860e1578ed82fd9cb87aeed49a64c621c401d4f41376f1544f72cea93389cdd1":
        raise ValueError("Frozen manifest SHA-256 does not match!")
    return lock["split_manifest_sha256"]

def run_quality_checks(results_df, predictions_df, history_df, candidates, smoke_test):
    checks = []
    
    # Check 1: exactly eight expected models (if not smoke testing specific models)
    checks.append({"check": "eight_models", "passed": len(results_df) == 8 if not smoke_test else True})
    
    # Check 2: exactly 4801 predictions per model
    preds_counts = predictions_df.groupby("model_name").size()
    checks.append({"check": "exact_prediction_counts", "passed": (preds_counts == 4801).all() if not smoke_test else True})
    
    # Check 3: predictions binary, probabilities in [0,1]
    probs = predictions_df["probability_default"]
    preds = predictions_df["prediction_at_0_5"]
    checks.append({"check": "valid_probabilities", "passed": (probs >= 0).all() and (probs <= 1).all() and np.isfinite(probs).all()})
    checks.append({"check": "binary_predictions", "passed": set(preds.unique()).issubset({0, 1})})
    
    # Check 4: Exactly one candidate per family
    fams = [c["family"] for c in candidates]
    checks.append({"check": "candidate_families", "passed": sorted(fams) == ["baseline", "cnn", "recurrent"]})
    
    # Check 5: metrics independently recompute
    # Just take one model and check F1
    if not results_df.empty:
        m = results_df.iloc[0]["model_name"]
        m_preds = predictions_df[predictions_df["model_name"] == m]
        f1_recalc = compute_metrics(m_preds["y_true"], m_preds["probability_default"])["default_f1"]
        f1_orig = results_df.iloc[0]["default_f1"]
        checks.append({"check": "metrics_recomputable", "passed": np.isclose(f1_recalc, f1_orig)})
        
    return pd.DataFrame(checks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke test mode")
    args = parser.parse_args()
    
    print("=" * 72)
    print("  ACCELERATED PHASE 3: MODEL EXPERIMENTS")
    print("=" * 72)
    
    manifest_sha = check_frozen_manifest()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load data
    train_loader, val_loader, pos_weight = get_development_dataloaders(TRAINING_CONFIG["batch_size"], args.smoke_test)
    X_train, y_train, X_val, y_val = get_development_tabular(args.smoke_test)
    class_weights = get_class_weights_dict(y_train)
    
    train_count = len(y_train)
    val_count = len(y_val)
    print(f"  Train: {train_count} | Validation: {val_count}")
    
    registry = get_model_registry()
    if args.models:
        allowed = args.models.split(",")
        registry = [m for m in registry if m["name"] in allowed]
        
    if args.smoke_test:
        artifacts_dir = Path("artifacts/models/phase3_smoke")
        reports_dir = Path("reports/experiments/phase3_smoke")
        figures_dir = Path("reports/figures/experiments/phase3_smoke")
    else:
        artifacts_dir = Path("artifacts/models/phase3")
        reports_dir = Path("reports/experiments/phase3")
        figures_dir = Path("reports/figures/experiments/phase3")
        
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    histories = []
    all_predictions = []
    
    # To attach actual validation IDs, we need to extract them from validation.npz 
    # but we can get them from load_development_data easily:
    from credit_default.preprocessing.artifacts import load_development_data
    dev_data = load_development_data()
    val_ids = dev_data["validation"].ids
    if args.smoke_test:
        val_ids = val_ids[:100]
        
    for model_info in registry:
        name = model_info["name"]
        family = model_info["family"]
        print(f"  Training {name} ({family})...")
        
        reset_all_seeds()
        
        if family == "baseline":
            model, t_time, p_count, config = model_info["train_fn"](X_train, y_train, class_weights)
            y_prob, i_time = evaluate_baseline(model, X_val)
            best_epoch = None
            
            checkpoint_path = artifacts_dir / f"{name}.joblib"
            joblib.dump(model, checkpoint_path)
            
        else:
            model = model_info["model_class"]()
            p_count = count_parameters(model)
            model, history, best_epoch, t_time, _ = train_neural_model(
                model, train_loader, val_loader, pos_weight, device, args.smoke_test
            )
            y_prob, i_time = evaluate_neural_model(model, val_loader, device)
            
            for h in history:
                h["model_name"] = name
            histories.extend(history)
            
            plot_neural_history(name, history, figures_dir)
            
            checkpoint_path = artifacts_dir / f"{name}.pt"
            torch.save({
                "model_name": name,
                "configuration": TRAINING_CONFIG,
                "state_dict": model.state_dict(),
                "input_dimensions": {"temporal": (6, 3), "static": 15},
                "seed": GLOBAL_SEED,
                "best_epoch": best_epoch,
                "manifest_sha": manifest_sha,
                "class_weights": class_weights
            }, checkpoint_path)
            
            config = TRAINING_CONFIG
            
        metrics = compute_metrics(y_val, y_prob)
        
        preds_df = pd.DataFrame({
            "model_name": name,
            "ID": val_ids,
            "y_true": y_val,
            "probability_default": y_prob,
            "prediction_at_0_5": (y_prob >= FIXED_THRESHOLD).astype(int)
        })
        all_predictions.append(preds_df)
        
        chk_sha = get_file_sha256(checkpoint_path)
        
        row = {
            "model_name": name,
            "family": family,
            "representation": "tabular" if family == "baseline" else "temporal+static",
            "parameter_count": p_count,
            "hyperparameters": json.dumps(config),
            "duration": t_time,
            "inference_duration": i_time,
            "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha": chk_sha
        }
        row.update(metrics)
        results.append(row)
        
    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    history_df = pd.DataFrame(histories)
    
    candidates = select_candidates(results_df)
    
    checks_df = run_quality_checks(results_df, predictions_df, history_df, candidates, args.smoke_test)
    if not checks_df["passed"].all():
        print("  ERROR: Quality checks failed!")
        print(checks_df[~checks_df["passed"]])
        sys.exit(1)
        
    print(f"  Quality checks passed: {len(checks_df)}/{len(checks_df)}")
    
    # Save reports
    results_df.to_csv(reports_dir / "experiment_results.csv", index=False)
    predictions_df.to_csv(reports_dir / "validation_predictions.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(reports_dir / "neural_training_history.csv", index=False)
        
    with open(reports_dir / "selected_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
        
    with open(reports_dir / "model_registry.json", "w") as f:
        json.dump([{"name": m["name"], "family": m["family"]} for m in registry], f, indent=2)
        
    checks_df.to_csv(reports_dir / "phase3_quality_checks.csv", index=False)
    
    plot_validation_comparison(results_df, figures_dir)
    plot_confusion_matrices(results_df, figures_dir)
    generate_findings_markdown(results_df, candidates, train_count, val_count, reports_dir / "phase3_validation_findings.md")
    
    import sklearn
    env_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    }
    with open(reports_dir / "phase3_environment.json", "w") as f:
        json.dump(env_info, f, indent=2)
        
    print("=" * 72)
    print("  ACCELERATED PHASE 3 COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()

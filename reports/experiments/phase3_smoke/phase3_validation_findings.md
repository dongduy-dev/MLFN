# Phase 3 Validation Findings

## Scope and Invariants
- **Training observations**: 100
- **Validation observations**: 100
- **Fixed threshold**: 0.5
- **Class weighting strategy**: Training counts exclusively. 
- **Test Set**: Explicitly never loaded, transformed, or evaluated. Validation metrics do NOT represent final test performance. Threshold optimization is deferred to Phase 4.

## Complete Validation Ranking
- **lstm_deep** (recurrent): F1 0.4510 | PR-AUC 0.3026 | Recall 0.9583 | Precision 0.2949
- **gru_deep** (recurrent): F1 0.4103 | PR-AUC 0.3168 | Recall 1.0000 | Precision 0.2581
- **gru_small** (recurrent): F1 0.4082 | PR-AUC 0.3543 | Recall 0.8333 | Precision 0.2703
- **conv1d_deep** (cnn): F1 0.4000 | PR-AUC 0.3424 | Recall 0.9583 | Precision 0.2527
- **logistic_regression** (baseline): F1 0.3846 | PR-AUC 0.3562 | Recall 0.4167 | Precision 0.3571
- **gradient_boosting** (baseline): F1 0.3556 | PR-AUC 0.4124 | Recall 0.3333 | Precision 0.3810
- **conv1d_small** (cnn): F1 0.2182 | PR-AUC 0.2421 | Recall 0.2500 | Precision 0.1935
- **conv1d_multiscale** (cnn): F1 0.0000 | PR-AUC 0.2814 | Recall 0.0000 | Precision 0.0000

## Selected Candidates
- **Best Baseline**: logistic_regression
- **Best Recurrent**: lstm_deep
- **Best CNN**: conv1d_deep

## Observations
- **Precision vs Recall**: The fixed 0.5 threshold heavily influences the balance. Final operating points will be calibrated.
- **Overfitting/Underfitting**: Baseline models exhibit strong convergence. Neural models actively utilized early stopping on validation BCE loss to mitigate overfitting, successfully restoring best-epoch weights.
- **Complexity**: Training times and parameter counts vary logically across families.

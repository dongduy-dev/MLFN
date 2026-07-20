# Phase 3 Validation Findings

## Scope and Invariants
- **Training observations**: 19199
- **Validation observations**: 4801
- **Fixed threshold**: 0.5
- **Class weighting strategy**: Training counts exclusively. 
- **Test Set**: Explicitly never loaded, transformed, or evaluated. Validation metrics do NOT represent final test performance. Threshold optimization is deferred to Phase 4.

## Complete Validation Ranking
- **logistic_regression** (baseline): F1 0.5452 | PR-AUC 0.5585 | Recall 0.5791 | Precision 0.5151
- **gru_deep** (recurrent): F1 0.5415 | PR-AUC 0.5722 | Recall 0.6544 | Precision 0.4618
- **conv1d_multiscale** (cnn): F1 0.5404 | PR-AUC 0.5547 | Recall 0.6356 | Precision 0.4701
- **gradient_boosting** (baseline): F1 0.5398 | PR-AUC 0.5742 | Recall 0.6252 | Precision 0.4750
- **conv1d_small** (cnn): F1 0.5385 | PR-AUC 0.5465 | Recall 0.6356 | Precision 0.4671
- **conv1d_deep** (cnn): F1 0.5384 | PR-AUC 0.5647 | Recall 0.6497 | Precision 0.4597
- **gru_small** (recurrent): F1 0.5377 | PR-AUC 0.5676 | Recall 0.6375 | Precision 0.4650
- **lstm_deep** (recurrent): F1 0.5325 | PR-AUC 0.5665 | Recall 0.6403 | Precision 0.4558

## Selected Candidates
- **Best Baseline**: logistic_regression
- **Best Recurrent**: gru_deep
- **Best CNN**: conv1d_multiscale

## Observations
- **Precision vs Recall**: The fixed 0.5 threshold heavily influences the balance. Final operating points will be calibrated.
- **Overfitting/Underfitting**: Baseline models exhibit strong convergence. Neural models actively utilized early stopping on validation BCE loss to mitigate overfitting, successfully restoring best-epoch weights.
- **Complexity**: Training times and parameter counts vary logically across families.

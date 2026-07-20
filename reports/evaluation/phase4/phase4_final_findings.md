# Phase 4 Final Findings

- exact frozen test size: 6000
- three candidates evaluated.
- thresholds were selected using validation data ONLY.
- logistic_regression validation selected threshold: 0.5449999999999998
- gru_deep validation selected threshold: 0.5849999999999997
- conv1d_multiscale validation selected threshold: 0.6149999999999998
- Primary candidate: gru_deep
- Precision/Recall trade-offs were managed explicitly via F1 optimization on validation.
- Inference times and model complexity are reported in the final CSV.

## Statements
- No model was retrained.
- Test inference occurred only after decisions were locked.
- No test-based threshold tuning occurred.
- The primary candidate was not changed using test results.

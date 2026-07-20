# Credit Card Default Prediction

## Project Overview
This is an experimental project focused on predicting credit card defaults using the UCI Default of Credit Card Clients dataset. The dataset contains 30,000 observations with a binary prediction target and consists of tabular features along with a six-month ordered history of repayment statuses, bill amounts, and payment amounts.

## Scientific Workflow
1. Immutable raw-data audit.
2. Static and temporal EDA.
3. Predictor-group-aware train/validation/test split.
4. Training-only preprocessing.
5. Traditional and neural development experiments.
6. Validation-only candidate and threshold selection.
7. One-time frozen-test evaluation.

## Dataset
- **Source**: UCI Machine Learning Repository
- **Local raw-data path**: `data/raw/default of credit card clients.xls`
- **Raw SHA-256**: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`
- The raw data is strictly read-only and is not modified.
- `ID` is excluded from model features.
- `default payment next month` (target) is excluded from predictor grouping and feature matrices.

## Frozen Splits
| Split | Rows | Non-default | Default |
| --- | --- | --- | --- |
| Train | 19,199 | 14,952 | 4,247 |
| Validation | 4,801 | 3,739 | 1,062 |
| Test | 6,000 | 4,673 | 1,327 |

Identical predictor vectors are grouped and never cross partitions, preventing information leakage. The test set is explicitly isolated and excluded from model selection and threshold selection.

## Representations

**Tabular:**
- Shape: 91 features.
- Used by traditional baselines: Logistic Regression and Gradient Boosting.

**Static:**
- Shape: 15 features.

**Temporal:**
- Shape: 6 timesteps × 3 channels.
- Timeline: April 2005 through September 2005.
- Channels: repayment status, bill amount, payment amount.

Neural models (GRU, LSTM, Conv1D) use temporal + static inputs via a multi-branch fusion architecture.

## Models Evaluated

**Baselines:**
- Logistic Regression
- Gradient Boosting

**Recurrent:**
- GRU Small
- GRU Deep
- LSTM Deep

**CNN:**
- Conv1D Small
- Conv1D Deep
- Conv1D Multiscale

## Development Results
Based strictly on validation metrics, the following family winners were selected during Accelerated Phase 3:
- **Baseline**: `logistic_regression`
- **Recurrent**: `gru_deep`
- **CNN**: `conv1d_multiscale`

*Note: These are validation results, not final test results.*

## Final Frozen-Test Results
The final results evaluated precisely once on the test split at the validation-selected thresholds are as follows:

| Model | Threshold | F1 | Precision | Recall | PR-AUC | ROC-AUC | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.545 | 0.5349 | 0.5304 | 0.5396 | 0.5285 | 0.7684 | 0.7925 |
| gru_deep | 0.585 | 0.5386 | 0.5362 | 0.5411 | 0.5435 | 0.7804 | 0.7950 |
| conv1d_multiscale | 0.615 | 0.5318 | 0.5258 | 0.5381 | 0.5247 | 0.7766 | 0.7905 |

- `gru_deep` was selected as the primary candidate before test access.
- Test results did not change that selection.
- No post-test tuning occurred.
- Threshold improvements over 0.5 were small.

## Reproducibility
Setup the environment:
```bash
python -m pip install -e ".[dev]"
```

Run the pipeline:
```bash
python -m scripts.audit_dataset
python -m scripts.run_eda_static
python -m scripts.run_eda_repayment_status
python -m scripts.run_eda_bill_amount
python -m scripts.run_eda_payment_amount
python -m scripts.run_eda_bill_payment_relationship
python -m scripts.prepare_model_data
python -m scripts.run_phase3_experiments
python -m scripts.run_phase4_evaluation
python -m scripts.verify_final_artifacts
python -m pytest tests/ -v
```

**WARNING:**
- Running Phase 3 retrains models.
- Running Phase 4 normally must reuse the locked test predictions.
- **Do not** use `--rerun-test-inference` after the final evaluation has been frozen.

## Project Structure
- `src/credit_default/eda/`
- `src/credit_default/preprocessing/`
- `src/credit_default/experiments/`
- `src/credit_default/evaluation/`
- `scripts/`
- `tests/`
- `reports/`
- `artifacts/`

## Key Artifacts
- **Split manifest**: `reports/preprocessing/split_manifest.csv`
- **Split lock**: `artifacts/preprocessing/split_lock.json`
- **Preprocessing metadata**: `reports/preprocessing/preprocessing_metadata.json`
- **Phase 3 results**: `reports/experiments/phase3/experiment_results.csv`
- **Selected candidates**: `reports/experiments/phase3/selected_candidates.json`
- **Threshold lock**: `artifacts/evaluation/phase4/threshold_lock.json`
- **Final evaluation lock**: `artifacts/evaluation/phase4/final_evaluation_lock.json`
- **Final test results**: `reports/evaluation/phase4/final_test_results.csv`
- **Final model decision**: `reports/evaluation/phase4/final_model_decision.json`

## Limitations
- This model is based on one public dataset.
- The temporal observation window is strictly limited to a six-month historical period.
- There is a large class imbalance in defaults.
- There is no external behavioral, socio-economic, or macroeconomic data.
- Moderate final PR-AUC shows predictive limitations using this data alone.
- Neural models only slightly outperform or closely match the linear baseline.
- Results should not be interpreted causally.

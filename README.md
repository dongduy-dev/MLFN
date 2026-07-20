# Credit Card Default Prediction

**Course**: Machine Learning Final Project  
**Dataset**: [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
**Current Checkpoint**: **1.1 — Project Foundation & Raw Dataset Audit (corrected)**

---

## Project Purpose

Predict whether a credit card client will default on their next payment,
using the UCI "Default of Credit Card Clients" dataset (30,000 records,
23 features, 1 binary target).

Future checkpoints will explore traditional ML baselines (Logistic Regression,
Random Forest, XGBoost) and deep-learning architectures (GRU, LSTM, Conv1D).

> **No modelling, preprocessing, data splitting, or EDA visualisation
> has been performed yet. This checkpoint covers project setup and raw-data
> auditing only.**

---

## Dataset Source

| Property | Value |
|----------|-------|
| Name | Default of Credit Card Clients |
| Provider | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients |
| Records | 30,000 |
| Features | 23 + 1 ID + 1 Target |
| Format | Excel 97-2003 (`.xls`) |
| File | `data/raw/default of credit card clients.xls` |

---

## Project Structure

```
MLFN/
├── data/
│   └── raw/                              # immutable raw dataset
│       └── default of credit card clients.xls
├── reports/
│   └── tables/                           # machine-readable audit outputs (CSV)
├── scripts/
│   ├── __init__.py
│   └── audit_dataset.py                  # CP1 dataset audit
├── src/
│   └── credit_default/                   # main project package
│       ├── __init__.py
│       └── data_loader.py               # reusable data loading module
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py              # data loader & schema tests (15 tests)
├── .gitignore
├── pyproject.toml                        # editable install configuration
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv

# 2. Activate the environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Install the project package in editable mode
python -m pip install -e .
```

---

## Commands

### Run the dataset audit (Checkpoint 1)

```bash
python -m scripts.audit_dataset
```

This will print a detailed audit report to the console and save
machine-readable CSV files under `reports/tables/`.

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Checkpoint Status

| Checkpoint | Description | Status |
|-----------|-------------|--------|
| 1 | Project Foundation & Raw Dataset Audit | ✅ Complete |
| 1.1 | Correction: documentation accuracy, package layout | ✅ Complete |
| 2 | EDA & Visualisation | ⬜ Not started |
| 3 | Preprocessing & Feature Engineering | ⬜ Not started |
| 4 | Traditional ML Baselines | ⬜ Not started |
| 5 | Deep Learning (GRU / LSTM / Conv1D) | ⬜ Not started |
| 6 | Evaluation & Final Report | ⬜ Not started |

---

## Known Data Issues for Future Checkpoints

### Duplicate feature vectors (Checkpoint 3)

The raw dataset contains 35 rows (70 rows in 35 pairs) that have identical
feature values when the ID column is excluded.  Since each pair has distinct
IDs, they are preserved in the raw data.

**Checkpoint 3 must investigate** whether these identical feature vectors
(excluding both ID and the target column) should be kept, merged, or handled
specially.  In particular, identical feature groups must **not be allowed to
cross train, validation, and test boundaries**, as this would create data
leakage.

### Undocumented categorical values

Values `0` and `-2` in repayment status columns (PAY_0, PAY_2–PAY_6), as
well as EDUCATION values `0`, `5`, `6` and MARRIAGE value `0`, are present
in the raw dataset but are **not explicitly defined** by the official UCI
documentation.  They have not been removed or recoded; any future
preprocessing must first decide how to handle them.

---

## License

This project is for educational purposes.  
The dataset is provided by the UCI Machine Learning Repository under their terms of use.

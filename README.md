# Credit Card Default Prediction

**Course**: Machine Learning Final Project  
**Dataset**: [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
**Current Checkpoint**: **2B1 — Temporal EDA (Repayment Status)**

---

## Project Purpose

Predict whether a credit card client will default on their next payment,
using the UCI "Default of Credit Card Clients" dataset (30,000 records,
23 features, 1 binary target).

Future checkpoints will explore traditional ML baselines (Logistic Regression,
Random Forest, XGBoost) and deep-learning architectures (GRU, LSTM, Conv1D).

> **No preprocessing, data splitting, feature selection, or model training
> has been performed. Checkpoint 2B1 is descriptive temporal EDA only.**

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
│   ├── eda_static_findings.md            # CP2A human-readable findings
│   ├── eda_repayment_status_findings.md  # CP2B1 human-readable findings
│   ├── figures/
│   │   ├── eda/static/                  # CP2A PNG figures
│   │   └── eda/repayment_status/        # CP2B1 PNG figures (10 files)
│   └── tables/
│       ├── eda/static/                  # CP2A CSV tables
│       └── eda/repayment_status/        # CP2B1 CSV tables (7 files)
├── scripts/
│   ├── __init__.py
│   ├── audit_dataset.py                 # CP1 dataset audit
│   ├── run_eda_static.py               # CP2A static feature EDA
│   └── run_eda_repayment_status.py     # CP2B1 temporal EDA
├── src/
│   └── credit_default/                  # main project package
│       ├── __init__.py
│       ├── data_loader.py              # reusable data loading module
│       └── eda/                        # EDA sub-package
│           ├── __init__.py
│           ├── figures.py              # CP2A figure generation
│           ├── findings.py             # CP2A findings generation
│           ├── static_features.py     # CP2A EDA calculations
│           ├── repayment_status.py         # CP2B1 calculations
│           ├── repayment_status_figures.py # CP2B1 figures
│           └── repayment_status_findings.py# CP2B1 findings
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py             # CP1 tests
│   ├── test_eda_static.py             # CP2A tests
│   └── test_eda_repayment_status.py   # CP2B1 tests
├── .gitignore
├── pyproject.toml
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

### Run static feature EDA (Checkpoint 2A)

```bash
python -m scripts.run_eda_static
```

### Run temporal EDA (Checkpoint 2B1)

```bash
python -m scripts.run_eda_repayment_status
```

Outputs:
- Tables: `reports/tables/eda/repayment_status/*.csv` (7 files)
- Figures: `reports/figures/eda/repayment_status/*.png` (10 files)
- Findings: `reports/eda_repayment_status_findings.md`

### Run temporal EDA (Checkpoint 2B2A - Bill Amounts)

```bash
python -m scripts.run_eda_bill_amount
```

Outputs:
- Tables: `reports/tables/eda/bill_amount/*.csv` (10 files)
- Figures: `reports/figures/eda/bill_amount/*.png` (12 files)
- Findings: `reports/eda_bill_amount_findings.md`

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
| 2A | Descriptive EDA — Target & Static Features | ✅ Complete |
| 2A.2 | Correction: static EDA wording and tests | ✅ Complete |
| 2B1 | Descriptive EDA — Temporal Features (PAY_x) | ✅ Complete |
| 2B2A | Descriptive EDA — Temporal Features (BILL_AMTx) | ✅ Complete |
| 2B2B | Descriptive EDA — Temporal Features (PAY_AMTx) | ⬜ Not started |
| 3 | Preprocessing & Feature Engineering | ⬜ Not started |
| 4 | Traditional ML Baselines | ⬜ Not started |
| 5 | Deep Learning (GRU / LSTM / Conv1D) | ⬜ Not started |
| 6 | Evaluation & Final Report | ⬜ Not started |

**Note**: Checkpoint 2B2B (payment amounts), preprocessing, splitting, and modelling have not started.

---

## Known Data Issues for Future Checkpoints

### Duplicate feature vectors (Checkpoint 3)

Checkpoint 1 identified 35 duplicate rows when ID was excluded from the
comparison. Because the target was still included, this does not yet establish
the number of predictor-only duplicate groups. Before splitting, Checkpoint 3
will group rows using all predictive features while excluding both ID and the
target, identify groups with consistent or conflicting targets, and prevent
identical predictor groups from crossing dataset boundaries.

### Undocumented categorical values

EDUCATION values `0`, `5`, `6` and MARRIAGE value `0` are present in the raw
dataset but are **not explicitly defined** by the official UCI documentation.
They have not been removed or recoded; any future preprocessing must decide
how to handle them.

The treatment of raw codes 0 and -2 during preprocessing remains unresolved.
Future checkpoints should compare defensible representations while preserving
the raw values and documenting any transformation.

---

## License

This project is for educational purposes.  
The dataset is provided by the UCI Machine Learning Repository under their terms of use.

# Checkpoint 2A — Static Feature EDA Findings
**Scope**: Descriptive analysis of `default payment next month`, `LIMIT_BAL`,
`AGE`, `SEX`, `EDUCATION`, and `MARRIAGE`.  
**Dataset**: Complete raw dataset — 30,000 clients, no modifications.  
**Temporal features deferred**: PAY_x, BILL_AMTx, and PAY_AMTx columns are
intentionally excluded from this checkpoint and will be analyzed in Checkpoint 2B.  
**No model evaluation has occurred.** These findings describe observed associations
in the raw data only.

All numerical values below are taken directly from the generated CSV tables
in `reports/tables/eda/static/`. Those CSVs are the authoritative
machine-readable source.

---

## 1. Target Distribution

| Class value | Class label | Count | Percentage |
|:-----------:|:-----------:|------:|----------:|
| 0 | No default | 23,364 | 77.88% |
| 1 | **Default** | 6,636 | 22.12% |

- **Positive class**: value `1` (default next month).
- **Imbalance ratio** (majority : minority): **3.52 : 1**.
- The dataset is moderately imbalanced. Any future classifier evaluation should
  account for this imbalance; metrics such as accuracy alone will be misleading.
  (*Note: no resampling is applied in this checkpoint.*)

---

## LIMIT_BAL

### Overall statistics
| Statistic | Value |
|-----------|------:|
| Count | 30,000 |
| Mean | 167,484.3227 |
| Std | 129,747.6616 |
| Min | 10,000.0000 |
| Q25 | 50,000.0000 |
| Median | 140,000.0000 |
| Q75 | 240,000.0000 |
| Max | 1,000,000.0000 |
| IQR | 190,000.0000 |

### By target group
| Group | Count | Mean | Median | Std |
|-------|------:|-----:|-------:|----:|
| No default | 23,364 | 178,099.7261 | 150,000.0000 | 131,628.3597 |
| Default | 6,636 | 130,109.6564 | 90,000.0000 | 115,378.5406 |

- Clients who **defaulted** had a **lower mean credit limit** (130,110) compared
  to those who did not (178,100). The medians differ by 60,000 (90,000 vs 150,000).
- Both groups show right-skewed distributions. The 3×IQR fence analysis found
  **3 potential extreme values** in the No Default group and **4** in the
  Default group; all are retained as-is.
- This is a descriptive association only; lower credit limits do not necessarily
  cause default.

---

## AGE

### Overall statistics
| Statistic | Value |
|-----------|------:|
| Count | 30,000 |
| Mean | 35.4855 |
| Std | 9.2179 |
| Min | 21.0000 |
| Q25 | 28.0000 |
| Median | 34.0000 |
| Q75 | 41.0000 |
| Max | 79.0000 |
| IQR | 13.0000 |

### By target group
| Group | Count | Mean | Median | Std |
|-------|------:|-----:|-------:|----:|
| No default | 23,364 | 35.4173 | 34.0000 | 9.0774 |
| Default | 6,636 | 35.7257 | 34.0000 | 9.6934 |

- The age distributions of defaulters and non-defaulters are very similar.
  Mean difference is approximately **0.31 years** (35.73 vs 35.42), with
  identical medians (34) for both groups.
- No extreme values were flagged by the 3×IQR check.
- The dataset's recorded age range is 21–79 years.

---

## 4. SEX

| Raw code | Count | % of total | Default count | Default rate | N | Note |
|:--------:|------:|----------:|:-------------:|:------------:|--:|:-----|
| 1 | 11,888 | 39.63% | 2,873 | **24.2%** | 11,888 |  |
| 2 | 18,112 | 60.37% | 3,763 | **20.8%** | 18,112 |  |

- Both categories contain many observations, so their observed rates are less
  sensitive to individual records than those of the flagged small categories.
- Male clients show a slightly higher observed default rate than female
  clients. This is a descriptive association; no causal interpretation
  is made.
- Both codes are explicitly documented in the UCI description.

---

## 5. EDUCATION

| Raw code | Count | % of total | Default count | Default rate | N | Note |
|:--------:|------:|----------:|:-------------:|:------------:|--:|:-----|
| 0 | 14 | 0.05% | 0 | **0.0%** | 14 | **Not explicitly defined in UCI docs; n < 200** |
| 1 | 10,585 | 35.28% | 2,036 | **19.2%** | 10,585 |  |
| 2 | 14,030 | 46.77% | 3,330 | **23.7%** | 14,030 |  |
| 3 | 4,917 | 16.39% | 1,237 | **25.2%** | 4,917 |  |
| 4 | 123 | 0.41% | 7 | **5.7%** | 123 | **n < 200** |
| 5 | 280 | 0.93% | 18 | **6.4%** | 280 | **Not explicitly defined in UCI docs** |
| 6 | 51 | 0.17% | 8 | **15.7%** | 51 | **Not explicitly defined in UCI docs; n < 200** |

**Undocumented categories**: values are present in the
raw data but are not explicitly defined by the official UCI documentation. They
are retained as-is; no recoding has been performed.

Categories with fewer than 200 observations are flagged under the project's
caution threshold. Their rates should be interpreted carefully because they
are more sensitive to individual observations.

Among the large documented categories, there is a visible gradient. However,
this pattern should not be interpreted causally, and feature utility should not
be assessed before modelling.

---

## 6. MARRIAGE

| Raw code | Count | % of total | Default count | Default rate | N | Note |
|:--------:|------:|----------:|:-------------:|:------------:|--:|:-----|
| 0 | 54 | 0.18% | 5 | **9.3%** | 54 | **Not explicitly defined in UCI docs; n < 200** |
| 1 | 13,659 | 45.53% | 3,206 | **23.5%** | 13,659 |  |
| 2 | 15,964 | 53.21% | 3,341 | **20.9%** | 15,964 |  |
| 3 | 323 | 1.08% | 84 | **26.0%** | 323 |  |

**Undocumented category**: values are not explicitly defined in the
UCI documentation. They are retained and annotated.

Categories with fewer than 200 observations are flagged under the project's
caution threshold. Their rates should be interpreted carefully because they
are more sensitive to individual observations.

These are associations only.

---

## Summary of Undocumented and Small-Sample Categories

| Feature | Value | Count | Note |
|---------|------:|------:|------|
| EDUCATION | 0 | 14 | Not explicitly defined in UCI docs; small sample |
| EDUCATION | 4 | 123 | small sample |
| EDUCATION | 5 | 280 | Not explicitly defined in UCI docs |
| EDUCATION | 6 | 51 | Not explicitly defined in UCI docs; small sample |
| MARRIAGE | 0 | 54 | Not explicitly defined in UCI docs; small sample |

None of these values have been removed or merged. Their handling must be
explicitly decided in Checkpoint 3 (preprocessing).

---

## Methodological Notes

- This EDA uses the **complete raw dataset** (n=30,000) for descriptive
  understanding. No train/validation/test split has been applied.
- No model evaluation has occurred.
- Temporal features (PAY_x, BILL_AMTx, PAY_AMTx) are **intentionally deferred**
  to Checkpoint 2B.
- Associations between features and the target are reported, not causal
  relationships.
- Feature importance or utility for prediction has **not** been assessed here.
  That requires model-based evaluation in later checkpoints.
- No preprocessing, encoding, scaling, imputation, or resampling was performed.

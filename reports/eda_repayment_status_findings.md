# Checkpoint 2B1 — Repayment Status Temporal EDA Findings

**Scope**: Descriptive analysis of repayment statuses over 6 months.
**Features Deferred**: `BILL_AMTx` and `PAY_AMTx` are deferred to Checkpoint 2B2.

## Chronological Mapping
The analysis follows this exact chronological mapping based on dataset documentation:
- **PAY_6**: April
- **PAY_5**: May
- **PAY_4**: June
- **PAY_3**: July
- **PAY_2**: August
- **PAY_0**: September

**Note on PAY_1**: The dataset schema skips `PAY_1`, transitioning directly from `PAY_2` to `PAY_0`.
This is a known quirk of the UCI dataset structure and has been preserved programmatically.

**Chronological sequence**: `PAY_6|PAY_5|PAY_4|PAY_3|PAY_2|PAY_0`
**Months**: April → May → June → July → August → September

## Undocumented Categories
Raw code 0 is the most frequent observed repayment-status value. It is present
in the raw dataset but is not explicitly defined by the official UCI
documentation.

Raw code -2 (observed 24,415 times in total) is also frequently present
but not explicitly defined in the UCI documentation. These codes have been retained
without recoding or assumptions of linear progression.

## Target Class Distributions
There are observable associations between status distributions and the target variable:
- **No Default (September)**: Most common status is `0` (55.0%).
- **Default (September)**: Most common status is `0` (28.5%).

## Selected Default Rates
The following examples highlight the association between raw status codes and default rate:

- **September, Code 0**: 1,888 defaults / 14,737 total = **12.8%**
- **September, Code 2**: 1,844 defaults / 2,667 total = **69.1%**
- **September, Code 3**: 244 defaults / 322 total = **75.8%**

## Sequence Patterns
There are **1,106 unique 6-month sequence patterns** observed in the dataset. The top 10 most common patterns account for **17,202 customers** (57.34% of all clients).

### Most Common Patterns
- **Overall**: `0|0|0|0|0|0` (Total: 9,821, Defaults: 1,026, Rate: 10.4%)
- **Among Defaults**: `0|0|0|0|0|0` (Total: 9,821, Defaults: 1,026, Rate: 10.4%)
- **Among Non-Defaults**: `0|0|0|0|0|0` (Total: 9,821, Non-Defaults: 8,795)

## Methodological Caveats & Unresolved Concerns
- **Caution on Small Samples**: Categories, transitions, or patterns with small sample counts (n < 200) should be interpreted cautiously due to their sensitivity to individual records.
- **Unresolved Codes 0 and -2**: The treatment of raw codes 0 and -2 during preprocessing remains unresolved. Future checkpoints should compare defensible representations while preserving the raw values and documenting any transformation.
- **Association $\neq$ Causation**: Changes over time and their correlation with the target are observational. No causal inferences are made.
- **Model Expectations**: While this temporal structure supports experimenting with sequence models (e.g., GRU, LSTM, CNN), this descriptive analysis does not guarantee they will outperform tabular baselines.
- **Deferred Work**: `BILL_AMT` and `PAY_AMT` analysis is deferred to Checkpoint 2B2.

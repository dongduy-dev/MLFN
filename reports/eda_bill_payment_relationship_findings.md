# Accelerated Phase 1 — Bill/Payment Relationship EDA

## Scope and Constraints
This phase analyzes same-index monthly bill and previous-payment amount pairs (e.g., BILL_AMT6 paired with PAY_AMT6 for April).
**Important Caveats:**
- This is a purely numerical same-index comparison.
- No causal or accounting-settlement interpretation is assumed. We do not claim that PAY_AMTx necessarily settles BILL_AMTx.
- All EDA is strictly descriptive. No train/validation/test splitting, no preprocessing for modelling, and no model training has begun.

## Chronological Same-Index Pairs
- **April**: BILL_AMT6 with PAY_AMT6
- **May**: BILL_AMT5 with PAY_AMT5
- **June**: BILL_AMT4 with PAY_AMT4
- **July**: BILL_AMT3 with PAY_AMT3
- **August**: BILL_AMT2 with PAY_AMT2
- **September**: BILL_AMT1 with PAY_AMT1

## Monthly Pearson Correlations
- **April**: 0.1155
- **May**: 0.1416
- **June**: 0.1302
- **July**: 0.1300
- **August**: 0.1009
- **September**: 0.1403

## Eligible Positive-Bill Ratios
> **Note:** The ratio (PAY_AMT / BILL_AMT) is calculated *only* where BILL_AMT > 0. Ratios strictly exclude nonpositive bills to avoid infinite or undefined values. Because very small positive bills can create extreme ratios, the IQR and percentiles provide a more robust summary than the mean.

### April
- **Positive-Bill Eligibility:** 25,292 of 30,000 (84.31%)
- **Overall Ratio:** Median = 0.0461 (IQR: 0.2283)
- **By Target:** Target 0 Median = 0.0498 (IQR: 0.3128) | Target 1 Median = 0.0404 (IQR: 0.0739)

### May
- **Positive-Bill Eligibility:** 25,839 of 30,000 (86.13%)
- **Overall Ratio:** Median = 0.0455 (IQR: 0.1906)
- **By Target:** Target 0 Median = 0.0485 (IQR: 0.2649) | Target 1 Median = 0.0404 (IQR: 0.0699)

### June
- **Positive-Bill Eligibility:** 26,130 of 30,000 (87.10%)
- **Overall Ratio:** Median = 0.0421 (IQR: 0.1554)
- **By Target:** Target 0 Median = 0.0438 (IQR: 0.2126) | Target 1 Median = 0.0386 (IQR: 0.0726)

### July
- **Positive-Bill Eligibility:** 26,475 of 30,000 (88.25%)
- **Overall Ratio:** Median = 0.0485 (IQR: 0.1847)
- **By Target:** Target 0 Median = 0.0515 (IQR: 0.2592) | Target 1 Median = 0.0406 (IQR: 0.0834)

### August
- **Positive-Bill Eligibility:** 26,825 of 30,000 (89.42%)
- **Overall Ratio:** Median = 0.0600 (IQR: 0.2267)
- **By Target:** Target 0 Median = 0.0645 (IQR: 0.3011) | Target 1 Median = 0.0495 (IQR: 0.1090)

### September
- **Positive-Bill Eligibility:** 27,402 of 30,000 (91.34%)
- **Overall Ratio:** Median = 0.0611 (IQR: 0.2365)
- **By Target:** Target 0 Median = 0.0658 (IQR: 0.2985) | Target 1 Median = 0.0503 (IQR: 0.1286)

## Relationship Categories and Default Rates
Observations are assigned to mutually exclusive raw numerical categories. The sum of categories across each month is exactly 30,000.

### April
- **bill_positive_payment_positive**: n = 21,823 (Default rate: 20.13%)
- **bill_positive_payment_zero**: n = 3,469 (Default rate: 33.32%)
- **bill_nonpositive_payment_positive**: n = 1,004 (Default rate: 16.33%)
- **bill_nonpositive_payment_zero**: n = 3,704 (Default rate: 24.89%)
- **Total Zero-Payment Occurrences**: 7,173 of 30,000

### May
- **bill_positive_payment_positive**: n = 22,293 (Default rate: 20.13%)
- **bill_positive_payment_zero**: n = 3,546 (Default rate: 32.54%)
- **bill_nonpositive_payment_positive**: n = 1,004 (Default rate: 17.83%)
- **bill_nonpositive_payment_zero**: n = 3,157 (Default rate: 25.85%)
- **Total Zero-Payment Occurrences**: 6,703 of 30,000

### June
- **bill_positive_payment_positive**: n = 22,518 (Default rate: 19.87%)
- **bill_positive_payment_zero**: n = 3,612 (Default rate: 34.97%)
- **bill_nonpositive_payment_positive**: n = 1,074 (Default rate: 15.74%)
- **bill_nonpositive_payment_zero**: n = 2,796 (Default rate: 26.11%)
- **Total Zero-Payment Occurrences**: 6,408 of 30,000

### July
- **bill_positive_payment_positive**: n = 23,079 (Default rate: 19.72%)
- **bill_positive_payment_zero**: n = 3,396 (Default rate: 37.25%)
- **bill_nonpositive_payment_positive**: n = 953 (Default rate: 16.05%)
- **bill_nonpositive_payment_zero**: n = 2,572 (Default rate: 25.89%)
- **Total Zero-Payment Occurrences**: 5,968 of 30,000

### August
- **bill_positive_payment_positive**: n = 23,619 (Default rate: 19.74%)
- **bill_positive_payment_zero**: n = 3,206 (Default rate: 38.33%)
- **bill_nonpositive_payment_positive**: n = 985 (Default rate: 17.66%)
- **bill_nonpositive_payment_zero**: n = 2,190 (Default rate: 26.03%)
- **Total Zero-Payment Occurrences**: 5,396 of 30,000

### September
- **bill_positive_payment_positive**: n = 23,907 (Default rate: 19.25%)
- **bill_positive_payment_zero**: n = 3,495 (Default rate: 39.77%)
- **bill_nonpositive_payment_positive**: n = 844 (Default rate: 17.30%)
- **bill_nonpositive_payment_zero**: n = 1,754 (Default rate: 28.34%)
- **Total Zero-Payment Occurrences**: 5,249 of 30,000

## Conclusion
Descriptive Exploratory Data Analysis (EDA) is now entirely complete for static features, repayment statuses, bill amounts, previous-payment amounts, and same-index relationships.
Leakage-safe dataset splitting and preprocessing will be conducted in subsequent checkpoints. **No splitting, preprocessing, or modelling has occurred.**

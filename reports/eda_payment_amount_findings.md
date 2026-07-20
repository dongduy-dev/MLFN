# Checkpoint 2B2B1 — Temporal EDA Findings (Payment Amounts)

**Scope**: Descriptive analysis of the six previous-payment amount columns (`PAY_AMT1` to `PAY_AMT6`) over 6 months.
**Features Deferred**: `BILL_AMT`/`PAY_AMT` relationship analysis is deferred to Checkpoint 2B2B2.

## Chronological Mapping
The analysis follows this exact chronological mapping based on dataset documentation:
- **PAY_AMT6**: April
- **PAY_AMT5**: May
- **PAY_AMT4**: June
- **PAY_AMT3**: July
- **PAY_AMT2**: August
- **PAY_AMT1**: September

**Chronological sequence**: `PAY_AMT6` → `PAY_AMT5` → `PAY_AMT4` → `PAY_AMT3` → `PAY_AMT2` → `PAY_AMT1`
**Months**: April → May → June → July → August → September

## Overall Monthly Medians and Spread
Payment amounts exhibit right skewness, as evidenced by means consistently exceeding medians. All calculations maintain the 30,000 raw rows without filtering or outlier removal.

- **April**: Mean = 5,215.5026, Q1 = 117.75, Median = 1,500.00, Q3 = 4,000.00, IQR = 3,882.25
- **May**: Mean = 4,799.3876, Q1 = 252.50, Median = 1,500.00, Q3 = 4,031.50, IQR = 3,779.00
- **June**: Mean = 4,826.0769, Q1 = 296.00, Median = 1,500.00, Q3 = 4,013.25, IQR = 3,717.25
- **July**: Mean = 5,225.6815, Q1 = 390.00, Median = 1,800.00, Q3 = 4,505.00, IQR = 4,115.00
- **August**: Mean = 5,921.1635, Q1 = 833.00, Median = 2,009.00, Q3 = 5,000.00, IQR = 4,167.00
- **September**: Mean = 5,663.5805, Q1 = 1,000.00, Median = 2,100.00, Q3 = 5,006.00, IQR = 4,006.00

## Target Group Comparisons
- **April**: Target 0 (n=23,364) Median = 1,706.00 | Target 1 (n=6,636) Median = 1,000.00
- **May**: Target 0 (n=23,364) Median = 1,765.00 | Target 1 (n=6,636) Median = 1,000.00
- **June**: Target 0 (n=23,364) Median = 1,734.00 | Target 1 (n=6,636) Median = 1,000.00
- **July**: Target 0 (n=23,364) Median = 2,000.00 | Target 1 (n=6,636) Median = 1,222.00
- **August**: Target 0 (n=23,364) Median = 2,247.50 | Target 1 (n=6,636) Median = 1,533.50
- **September**: Target 0 (n=23,364) Median = 2,459.50 | Target 1 (n=6,636) Median = 1,636.00

## Sign Category Findings
No negative PAY_AMT values were observed in any month. Zero and positive values are reported descriptively without assigning business interpretations to zero payments.

**April**:
- Negative: 0
- Zero: 7,173 (Default rate: 29.0%)
- Positive: 22,827 (Default rate: 20.0%)

**May**:
- Negative: 0
- Zero: 6,703 (Default rate: 29.4%)
- Positive: 23,297 (Default rate: 20.0%)

**June**:
- Negative: 0
- Zero: 6,408 (Default rate: 31.1%)
- Positive: 23,592 (Default rate: 19.7%)

**July**:
- Negative: 0
- Zero: 5,968 (Default rate: 32.4%)
- Positive: 24,032 (Default rate: 19.6%)

**August**:
- Negative: 0
- Zero: 5,396 (Default rate: 33.3%)
- Positive: 24,604 (Default rate: 19.7%)

**September**:
- Negative: 0
- Zero: 5,249 (Default rate: 35.9%)
- Positive: 24,751 (Default rate: 19.2%)

## Potential Extremes
Values falling outside $Q1 - 1.5 \times IQR$ or $Q3 + 1.5 \times IQR$ are flagged as potential extreme under the 1.5 × IQR rule:
- **April**: 2,958 potential extremes
- **May**: 2,945 potential extremes
- **June**: 2,994 potential extremes
- **July**: 2,598 potential extremes
- **August**: 2,714 potential extremes
- **September**: 2,745 potential extremes
These observations are not dropped.

## Adjacent-Month Raw Changes
Changes are computed as $Destination - Source$.
- **April → May**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **May → June**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **June → July**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **July → August**: Overall Median = 71.00 | Target 0 = 100.00 | Target 1 = 0.00
- **August → September**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00

This is a descriptive property of the observed change distributions and does not by itself establish a business explanation.

## Correlation matrix
Cross-month correlations between payment amounts:

- **April–May**: 0.1549
- **August–September**: 0.2856
- **April–September**: 0.1857

## Display-Only Range Disclosure
Some distribution figures limit the displayed range to the global 1st (0.00) and 99th (70,181.15) percentiles for visual legibility. There are 1,800 of 180,000 observations outside this range (approximately 1.0000%).

- **April**: 358 outside range (1.1933%)
- **May**: 275 outside range (0.9167%)
- **June**: 273 outside range (0.9100%)
- **July**: 294 outside range (0.9800%)
- **August**: 327 outside range (1.0900%)
- **September**: 273 outside range (0.9100%)

Values outside the range are not moved into boundary bins and remain included in all calculations.

## Methodological Caveats
- **Association vs Causation**: Association does not imply causation.
- **Modelling Implications**: Payment history provides ordered information suitable for sequence-model experiments but does not prove sequence models will outperform tabular baselines.
- **Validation Anchors**: 31/31 anchors passed independent validation tests.

# Checkpoint 2B2A — Temporal EDA Findings (Bill Amounts)

**Scope**: Descriptive analysis of the six bill-statement amount columns (`BILL_AMT1` to `BILL_AMT6`) over 6 months.
**Features Deferred**: `PAY_AMTx` analysis and bill/payment relationships are deferred to Checkpoint 2B2B.

## Chronological Mapping
The analysis follows this exact chronological mapping based on dataset documentation:
- **BILL_AMT6**: April
- **BILL_AMT5**: May
- **BILL_AMT4**: June
- **BILL_AMT3**: July
- **BILL_AMT2**: August
- **BILL_AMT1**: September

**Chronological sequence**: `BILL_AMT6` → `BILL_AMT5` → `BILL_AMT4` → `BILL_AMT3` → `BILL_AMT2` → `BILL_AMT1`
**Months**: April → May → June → July → August → September

## Overall Monthly Medians and Spread
Bill amounts exhibit right skewness, as evidenced by means consistently exceeding medians. All calculations maintain the 30,000 raw rows without filtering or outlier removal.

- **April**: Mean = 38,871.7604, Q1 = 1,256.00, Median = 17,071.00, Q3 = 49,198.25, IQR = 47,942.25
- **May**: Mean = 40,311.4010, Q1 = 1,763.00, Median = 18,104.50, Q3 = 50,190.50, IQR = 48,427.50
- **June**: Mean = 43,262.9490, Q1 = 2,326.75, Median = 19,052.00, Q3 = 54,506.00, IQR = 52,179.25
- **July**: Mean = 47,013.1548, Q1 = 2,666.25, Median = 20,088.50, Q3 = 60,164.75, IQR = 57,498.50
- **August**: Mean = 49,179.0752, Q1 = 2,984.75, Median = 21,200.00, Q3 = 64,006.25, IQR = 61,021.50
- **September**: Mean = 51,223.3309, Q1 = 3,558.75, Median = 22,381.50, Q3 = 67,091.00, IQR = 63,532.25

## Target Group Comparisons
- **April**: Target 0 (n=23,364) Median = 16,679.00 | Target 1 (n=6,636) Median = 18,028.50
- **May**: Target 0 (n=23,364) Median = 17,998.00 | Target 1 (n=6,636) Median = 18,478.50
- **June**: Target 0 (n=23,364) Median = 19,000.00 | Target 1 (n=6,636) Median = 19,119.50
- **July**: Target 0 (n=23,364) Median = 20,202.50 | Target 1 (n=6,636) Median = 19,834.50
- **August**: Target 0 (n=23,364) Median = 21,660.50 | Target 1 (n=6,636) Median = 20,300.50
- **September**: Target 0 (n=23,364) Median = 23,119.50 | Target 1 (n=6,636) Median = 20,185.00

## Sign Category Findings
Negative and zero values exist as raw entries and are recorded descriptively without inferring business definitions.

**April**:
- Negative: 688 (Default rate: 19.3%)
- Zero: 4,020 (Default rate: 23.7%)
- Positive: 25,292 (Default rate: 21.9%)

**May**:
- Negative: 655 (Default rate: 19.8%)
- Zero: 3,506 (Default rate: 24.7%)
- Positive: 25,839 (Default rate: 21.8%)

**June**:
- Negative: 675 (Default rate: 17.5%)
- Zero: 3,195 (Default rate: 24.4%)
- Positive: 26,130 (Default rate: 22.0%)

**July**:
- Negative: 655 (Default rate: 19.7%)
- Zero: 2,870 (Default rate: 24.0%)
- Positive: 26,475 (Default rate: 22.0%)

**August**:
- Negative: 669 (Default rate: 19.0%)
- Zero: 2,506 (Default rate: 24.6%)
- Positive: 26,825 (Default rate: 22.0%)

**September**:
- Negative: 590 (Default rate: 18.5%)
- Zero: 2,008 (Default rate: 26.6%)
- Positive: 27,402 (Default rate: 21.9%)

## Potential Extremes
Values falling outside $Q1 - 1.5 \times IQR$ or $Q3 + 1.5 \times IQR$ are flagged as potential extreme under the 1.5 × IQR rule:
- **April**: 2,693 potential extremes
- **May**: 2,725 potential extremes
- **June**: 2,622 potential extremes
- **July**: 2,469 potential extremes
- **August**: 2,395 potential extremes
- **September**: 2,400 potential extremes
These observations are not dropped.

## Adjacent-Month Raw Changes
Changes are computed as $Destination - Source$.
- **April → May**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **May → June**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **June → July**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **July → August**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00
- **August → September**: Overall Median = 0.00 | Target 0 = 0.00 | Target 1 = 0.00

Every adjacent-month overall median raw change is zero. This is a descriptive property of the observed change distributions and does not by itself establish a business explanation.

## Correlation matrix
Adjacent months typically exhibit strong positive correlations, decaying gradually over longer temporal distances.

- **April–May**: 0.9462
- **August–September**: 0.9515
- **April–September**: 0.8027

## Display-Only Range Disclosure
Some distribution figures clip the axes between the global 1st (-200.00) and 99th (314,507.17) percentiles for visual legibility. There are 3,539 of 180,000 observations outside this range (approximately 1.9661%).

- **April**: 530 outside range (1.7667%)
- **May**: 527 outside range (1.7567%)
- **June**: 579 outside range (1.9300%)
- **July**: 627 outside range (2.0900%)
- **August**: 649 outside range (2.1633%)
- **September**: 627 outside range (2.0900%)

Values outside the range are not moved into boundary bins and remain included in all calculations.

## Methodological Caveats
- **Association $\neq$ Causation**: Changes over time and correlations are observational.
- **Model Expectations**: Bill history provides ordered monthly information suitable for later sequence-model experiments but does not prove sequence models will outperform tabular baselines.

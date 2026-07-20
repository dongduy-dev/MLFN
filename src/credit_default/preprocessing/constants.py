TEMPORAL_MAPPING = {
    0: {"month": "April", "status": "PAY_6", "bill": "BILL_AMT6", "payment": "PAY_AMT6"},
    1: {"month": "May", "status": "PAY_5", "bill": "BILL_AMT5", "payment": "PAY_AMT5"},
    2: {"month": "June", "status": "PAY_4", "bill": "BILL_AMT4", "payment": "PAY_AMT4"},
    3: {"month": "July", "status": "PAY_3", "bill": "BILL_AMT3", "payment": "PAY_AMT3"},
    4: {"month": "August", "status": "PAY_2", "bill": "BILL_AMT2", "payment": "PAY_AMT2"},
    5: {"month": "September", "status": "PAY_0", "bill": "BILL_AMT1", "payment": "PAY_AMT1"}
}

def get_temporal_columns():
    """Returns status, bill, and payment column lists strictly in chronological order."""
    status_cols = [TEMPORAL_MAPPING[i]["status"] for i in range(6)]
    bill_cols = [TEMPORAL_MAPPING[i]["bill"] for i in range(6)]
    pay_cols = [TEMPORAL_MAPPING[i]["payment"] for i in range(6)]
    return status_cols, bill_cols, pay_cols

STATUS_COLS, BILL_COLS, PAY_COLS = get_temporal_columns()

STATIC_CONT_COLS = ["LIMIT_BAL", "AGE"]
STATIC_CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

TABULAR_CONT_COLS = ["LIMIT_BAL", "AGE"] + BILL_COLS + PAY_COLS
TABULAR_CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"] + STATUS_COLS

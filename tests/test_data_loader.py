"""
test_data_loader.py
===================
Tests for the data-loading module and key schema assumptions.

Run from the project root:
    python -m pytest tests/ -v
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from credit_default.data_loader import (
    RAW_DATASET_PATH,
    compute_file_sha256,
    get_raw_dataset_path,
    load_raw_dataset,
)


# ---------------------------------------------------------------------------
# Expected constants
# ---------------------------------------------------------------------------
EXPECTED_ROWS = 30_000
EXPECTED_COLS = 25

EXPECTED_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default payment next month",
]

TARGET_COL = "default payment next month"

# Known SHA-256 of the raw dataset file
EXPECTED_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"


# ---------------------------------------------------------------------------
# Fixture: load dataset once for all tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Load the raw dataset once and share across tests in this module."""
    return load_raw_dataset()


# ---------------------------------------------------------------------------
# Test: successful loading
# ---------------------------------------------------------------------------
class TestDataLoading:
    """Tests for basic data loading functionality."""

    def test_load_returns_dataframe(self, dataset: pd.DataFrame) -> None:
        """load_raw_dataset() must return a pandas DataFrame."""
        assert isinstance(dataset, pd.DataFrame)

    def test_expected_shape(self, dataset: pd.DataFrame) -> None:
        """Dataset must have exactly 30,000 rows and 25 columns."""
        assert dataset.shape == (EXPECTED_ROWS, EXPECTED_COLS), (
            f"Expected ({EXPECTED_ROWS}, {EXPECTED_COLS}), got {dataset.shape}"
        )

    def test_expected_columns(self, dataset: pd.DataFrame) -> None:
        """All expected column names must be present in the correct order."""
        assert list(dataset.columns) == EXPECTED_COLUMNS

    def test_all_dtypes_numeric(self, dataset: pd.DataFrame) -> None:
        """Every column should be a numeric dtype (int or float)."""
        for col in dataset.columns:
            assert pd.api.types.is_numeric_dtype(dataset[col]), (
                f"Column '{col}' has non-numeric dtype: {dataset[col].dtype}"
            )


# ---------------------------------------------------------------------------
# Test: schema assumptions
# ---------------------------------------------------------------------------
class TestSchemaAssumptions:
    """Tests for important schema properties."""

    def test_pay_0_present(self, dataset: pd.DataFrame) -> None:
        """PAY_0 must be present in the dataset."""
        assert "PAY_0" in dataset.columns

    def test_pay_1_absent(self, dataset: pd.DataFrame) -> None:
        """PAY_1 must NOT be present (known dataset quirk)."""
        assert "PAY_1" not in dataset.columns

    def test_target_column_present(self, dataset: pd.DataFrame) -> None:
        """The target column 'default payment next month' must exist."""
        assert TARGET_COL in dataset.columns

    def test_target_values_binary(self, dataset: pd.DataFrame) -> None:
        """Target column should contain only 0 and 1."""
        unique_values = set(dataset[TARGET_COL].unique())
        assert unique_values == {0, 1}, (
            f"Expected target values {{0, 1}}, got {unique_values}"
        )

    def test_id_uniqueness(self, dataset: pd.DataFrame) -> None:
        """Every ID value must be unique."""
        assert dataset["ID"].is_unique, (
            f"ID column has {dataset['ID'].duplicated().sum()} duplicates"
        )


# ---------------------------------------------------------------------------
# Test: file-not-found error
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for error handling in the data loader."""

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Loading from a non-existent path must raise FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.xls"
        with patch(
            "credit_default.data_loader.get_raw_dataset_path",
            return_value=fake_path,
        ):
            with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
                load_raw_dataset()


# ---------------------------------------------------------------------------
# Test: immutability — SHA-256 of raw file
# ---------------------------------------------------------------------------
class TestFileIntegrity:
    """Tests for raw file integrity."""

    def test_sha256_exact_match(self) -> None:
        """Raw dataset SHA-256 must match the known expected hash."""
        actual = compute_file_sha256()
        assert actual == EXPECTED_SHA256, (
            f"SHA-256 mismatch!\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            f"  Actual:   {actual}"
        )

    def test_sha256_is_consistent(self) -> None:
        """SHA-256 hash must be deterministic (same file, same hash)."""
        hash1 = compute_file_sha256()
        hash2 = compute_file_sha256()
        assert hash1 == hash2, "SHA-256 hash changed between two reads!"

    def test_sha256_is_hex_string(self) -> None:
        """SHA-256 output must be a valid 64-character hex string."""
        h = compute_file_sha256()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_raw_file_not_mutated(self, dataset: pd.DataFrame) -> None:
        """
        After loading the dataset, the raw file's SHA-256 must still match
        the hash computed before loading.
        """
        hash_before = compute_file_sha256()
        # Force a fresh load
        _ = load_raw_dataset()
        hash_after = compute_file_sha256()
        assert hash_before == hash_after, (
            "Raw file was mutated during loading!"
        )

    def test_sha256_missing_file_raises_error(self, tmp_path: Path) -> None:
        """compute_file_sha256 on a missing file must raise FileNotFoundError."""
        fake = tmp_path / "nope.xls"
        with pytest.raises(FileNotFoundError):
            compute_file_sha256(fake)

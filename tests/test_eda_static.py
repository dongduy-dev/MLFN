"""
test_eda_static.py
==================
Tests for the Checkpoint 2A static-feature EDA module.

Run from the project root:
    python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_default.data_loader import compute_file_sha256, load_raw_dataset
from credit_default.eda.static_features import (
    CATEGORICAL_COLS,
    DOCUMENTED_VALUES,
    NUMERIC_COLS,
    POSITIVE_CLASS,
    TARGET_COL,
    compute_categorical_default_rates,
    compute_categorical_distribution,
    compute_numeric_by_target,
    compute_numeric_summary,
    compute_target_summary,
    run_quality_checks,
    _require_columns,
)

# ---------------------------------------------------------------------------
# Expected constants
# ---------------------------------------------------------------------------
EXPECTED_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
DATASET_SIZE = 30_000


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Load raw dataset once per module."""
    return load_raw_dataset()


# ---------------------------------------------------------------------------
# Minimal synthetic dataset for fast unit tests
# ---------------------------------------------------------------------------
@pytest.fixture
def small_df() -> pd.DataFrame:
    """Small deterministic DataFrame for unit-level tests."""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 5, 6],
        TARGET_COL: [0, 0, 0, 1, 1, 1],
        "LIMIT_BAL": [10_000, 50_000, 100_000, 20_000, 60_000, 200_000],
        "AGE": [25, 30, 40, 22, 35, 50],
        "SEX": [1, 2, 1, 2, 1, 2],
        "EDUCATION": [1, 2, 3, 1, 2, 4],
        "MARRIAGE": [1, 2, 3, 1, 2, 3],
    })


# ---------------------------------------------------------------------------
# Tests: target summary
# ---------------------------------------------------------------------------
class TestTargetSummary:

    def test_returns_dataframe(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        assert isinstance(result, pd.DataFrame)

    def test_two_classes(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        assert len(result) == 2

    def test_counts_sum_to_dataset_size(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        assert result["count"].sum() == DATASET_SIZE

    def test_percentages_sum_to_100(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        assert abs(result["percentage"].sum() - 100.0) < 0.01

    def test_positive_class_identified(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        assert (result["positive_class"] == POSITIVE_CLASS).all()

    def test_imbalance_ratio_positive(self, df: pd.DataFrame) -> None:
        result = compute_target_summary(df)
        ratio = result["imbalance_ratio_majority_to_minority"].iloc[0]
        assert ratio > 1.0

    def test_synthetic_correctness(self, small_df: pd.DataFrame) -> None:
        result = compute_target_summary(small_df)
        assert result["count"].sum() == 6
        default_row = result[result["class_label"] == "default"]
        assert default_row["count"].iloc[0] == 3
        assert abs(default_row["percentage"].iloc[0] - 50.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: categorical distribution
# ---------------------------------------------------------------------------
class TestCategoricalDistribution:

    @pytest.mark.parametrize("col", CATEGORICAL_COLS)
    def test_counts_sum_to_dataset_size(self, df: pd.DataFrame, col: str) -> None:
        result = compute_categorical_distribution(df, col)
        assert result["count"].sum() == DATASET_SIZE

    @pytest.mark.parametrize("col", CATEGORICAL_COLS)
    def test_percentages_sum_to_100(self, df: pd.DataFrame, col: str) -> None:
        result = compute_categorical_distribution(df, col)
        assert abs(result["percentage"].sum() - 100.0) < 0.01

    def test_undocumented_education_preserved(self, df: pd.DataFrame) -> None:
        result = compute_categorical_distribution(df, "EDUCATION")
        undoc = result[result["documentation_status"] == "not_explicitly_defined_in_uci_docs"]
        assert set(undoc["raw_value"].tolist()) == {0, 5, 6}, (
            f"Expected undocumented EDUCATION values {{0,5,6}}, got {set(undoc['raw_value'])}"
        )

    def test_undocumented_marriage_preserved(self, df: pd.DataFrame) -> None:
        result = compute_categorical_distribution(df, "MARRIAGE")
        undoc = result[result["documentation_status"] == "not_explicitly_defined_in_uci_docs"]
        assert 0 in undoc["raw_value"].tolist()

    def test_raw_codes_not_merged(self, df: pd.DataFrame) -> None:
        """All unique raw values for each column must appear as separate rows."""
        for col in CATEGORICAL_COLS:
            result = compute_categorical_distribution(df, col)
            expected = set(int(v) for v in df[col].unique())
            actual = set(result["raw_value"].tolist())
            assert expected == actual, f"{col}: expected {expected}, got {actual}"

    def test_small_sample_flag_education(self, df: pd.DataFrame) -> None:
        """EDUCATION codes 0 (n=14) should be flagged as small sample."""
        result = compute_categorical_distribution(df, "EDUCATION", small_n_threshold=200)
        flagged = result[result["small_sample_warning"]]
        assert 0 in flagged["raw_value"].tolist()

    def test_missing_column_raises(self, small_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Required column"):
            compute_categorical_distribution(small_df, "NONEXISTENT_COL")


# ---------------------------------------------------------------------------
# Tests: categorical default rates
# ---------------------------------------------------------------------------
class TestCategoricalDefaultRates:

    @pytest.mark.parametrize("col", CATEGORICAL_COLS)
    def test_default_plus_nonfault_equals_total(self, df: pd.DataFrame, col: str) -> None:
        result = compute_categorical_default_rates(df, col)
        assert (result["default_count"] + result["non_default_count"] == result["total_count"]).all()

    @pytest.mark.parametrize("col", CATEGORICAL_COLS)
    def test_total_counts_sum_to_dataset_size(self, df: pd.DataFrame, col: str) -> None:
        result = compute_categorical_default_rates(df, col)
        assert result["total_count"].sum() == DATASET_SIZE

    @pytest.mark.parametrize("col", CATEGORICAL_COLS)
    def test_default_rates_in_unit_interval(self, df: pd.DataFrame, col: str) -> None:
        result = compute_categorical_default_rates(df, col)
        rates = result["default_rate"].dropna()
        assert (rates >= 0).all() and (rates <= 1).all()

    def test_synthetic_rate_correctness(self, small_df: pd.DataFrame) -> None:
        result = compute_categorical_default_rates(small_df, "SEX")
        # small_df:
        #   SEX=1 at rows 0,2,4 (IDs 1,3,5): targets [0, 0, 1] -> 1 default of 3
        #   SEX=2 at rows 1,3,5 (IDs 2,4,6): targets [0, 1, 1] -> 2 defaults of 3
        sex1 = result[result["raw_value"] == 1].iloc[0]
        assert sex1["default_count"] == 1
        assert sex1["total_count"] == 3
        assert abs(sex1["default_rate"] - 1 / 3) < 1e-5

        sex2 = result[result["raw_value"] == 2].iloc[0]
        assert sex2["default_count"] == 2
        assert sex2["total_count"] == 3
        assert abs(sex2["default_rate"] - 2 / 3) < 1e-5

    def test_required_columns_present(self, df: pd.DataFrame) -> None:
        result = compute_categorical_default_rates(df, "SEX")
        required = {"feature", "raw_value", "total_count", "default_count",
                    "non_default_count", "default_rate", "population_percentage",
                    "documentation_status"}
        assert required.issubset(set(result.columns))

    def test_missing_column_raises(self, small_df: pd.DataFrame) -> None:
        bad_df = small_df.drop(columns=[TARGET_COL])
        with pytest.raises(ValueError, match="Required column"):
            compute_categorical_default_rates(bad_df, "SEX")


# ---------------------------------------------------------------------------
# Tests: numeric summary
# ---------------------------------------------------------------------------
class TestNumericSummary:

    @pytest.mark.parametrize("col", NUMERIC_COLS)
    def test_count_correct(self, df: pd.DataFrame, col: str) -> None:
        result = compute_numeric_summary(df, col)
        assert result["count"].iloc[0] == DATASET_SIZE

    @pytest.mark.parametrize("col", NUMERIC_COLS)
    def test_by_target_group_counts_sum(self, df: pd.DataFrame, col: str) -> None:
        result = compute_numeric_by_target(df, col)
        assert result["count"].sum() == DATASET_SIZE

    @pytest.mark.parametrize("col", NUMERIC_COLS)
    def test_by_target_has_two_groups(self, df: pd.DataFrame, col: str) -> None:
        result = compute_numeric_by_target(df, col)
        assert len(result) == 2

    def test_limit_bal_min_max(self, df: pd.DataFrame) -> None:
        result = compute_numeric_summary(df, "LIMIT_BAL")
        assert result["min"].iloc[0] == 10_000.0
        assert result["max"].iloc[0] == 1_000_000.0

    def test_age_min_max(self, df: pd.DataFrame) -> None:
        result = compute_numeric_summary(df, "AGE")
        assert result["min"].iloc[0] == 21.0
        assert result["max"].iloc[0] == 79.0

    def test_missing_column_raises(self, df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Required column"):
            compute_numeric_summary(df, "NONEXISTENT")


# ---------------------------------------------------------------------------
# Tests: raw file integrity
# ---------------------------------------------------------------------------
class TestRawFileIntegrity:

    def test_sha256_unchanged_after_eda(self, df: pd.DataFrame) -> None:
        """Loading data for EDA must not alter the raw file."""
        actual = compute_file_sha256()
        assert actual == EXPECTED_SHA256, (
            f"SHA-256 mismatch after EDA!\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            f"  Actual:   {actual}"
        )


# ---------------------------------------------------------------------------
# Tests: output directory (use tmp_path)
# ---------------------------------------------------------------------------
class TestOutputGeneration:

    def test_target_summary_saved_to_tmp(self, df: pd.DataFrame, tmp_path: Path) -> None:
        summary = compute_target_summary(df)
        out = tmp_path / "target_summary.csv"
        summary.to_csv(out, index=False)
        loaded = pd.read_csv(out)
        assert loaded["count"].sum() == DATASET_SIZE

    def test_cat_default_rates_saved_to_tmp(self, df: pd.DataFrame, tmp_path: Path) -> None:
        for col in CATEGORICAL_COLS:
            result = compute_categorical_default_rates(df, col)
            out = tmp_path / f"{col}_default_rates.csv"
            result.to_csv(out, index=False)
            loaded = pd.read_csv(out)
            assert loaded["total_count"].sum() == DATASET_SIZE

# ---------------------------------------------------------------------------
# Tests: Figures
# ---------------------------------------------------------------------------
class TestFigures:
    
    def test_plot_target_distribution(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.figures import plot_target_distribution
        import matplotlib.pyplot as plt
        summary = compute_target_summary(df)
        path = plot_target_distribution(summary, tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0
        assert not plt.fignum_exists(1)  # Ensure closed

    def test_plot_numeric_distribution(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.figures import plot_numeric_distribution_by_target
        import matplotlib.pyplot as plt
        path = plot_numeric_distribution_by_target(df, "LIMIT_BAL", tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0
        assert not plt.get_fignums()

    def test_plot_categorical_default_rates(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.figures import plot_categorical_default_rates
        import matplotlib.pyplot as plt
        cat_rate = compute_categorical_default_rates(df, "SEX")
        path = plot_categorical_default_rates(cat_rate, "SEX", tmp_path)
        assert path.exists()
        assert path.stat().st_size > 0
        assert not plt.get_fignums()

    def test_compute_class_histogram_percentages(self, df: pd.DataFrame) -> None:
        from credit_default.eda.figures import compute_class_histogram_percentages
        bins, pct_neg, pct_pos = compute_class_histogram_percentages(df, "LIMIT_BAL", n_bins=20)
        assert len(bins) == 21
        assert len(pct_neg) == 20
        assert len(pct_pos) == 20
        # If there are data points, they should sum to 100%
        if pct_neg.sum() > 0:
            assert abs(pct_neg.sum() - 100.0) < 1e-5
        if pct_pos.sum() > 0:
            assert abs(pct_pos.sum() - 100.0) < 1e-5
            
        with pytest.raises(ValueError):
            compute_class_histogram_percentages(df, "LIMIT_BAL", n_bins=-1)
            
        with pytest.raises(ValueError):
            compute_class_histogram_percentages(df, "NONEXISTENT", n_bins=20)

    def test_compute_categorical_axis_limits(self) -> None:
        from credit_default.eda.figures import compute_categorical_axis_limits
        lower, upper = compute_categorical_axis_limits(0.5)
        assert lower == 0.0
        assert upper > 0.0
        assert upper <= 100.0

# ---------------------------------------------------------------------------
# Tests: Findings and Determinism
# ---------------------------------------------------------------------------
class TestFindingsAndDeterminism:

    def test_repeated_calculations_equal(self, df: pd.DataFrame) -> None:
        calc1 = compute_numeric_summary(df, "LIMIT_BAL")
        calc2 = compute_numeric_summary(df, "LIMIT_BAL")
        pd.testing.assert_frame_equal(calc1, calc2)

    def test_generate_findings(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.findings import generate_findings_markdown
        target_summary = compute_target_summary(df)
        
        numeric_overalls = []
        numeric_by_targets = []
        for col in NUMERIC_COLS:
            numeric_overalls.append(compute_numeric_summary(df, col))
            numeric_by_targets.append(compute_numeric_by_target(df, col))
        numeric_overall = pd.concat(numeric_overalls, ignore_index=True)
        numeric_by_target = pd.concat(numeric_by_targets, ignore_index=True)
        
        cat_dists = []
        cat_rates = []
        for col in CATEGORICAL_COLS:
            docs = DOCUMENTED_VALUES.get(col, set())
            cat_dists.append(compute_categorical_distribution(df, col, docs))
            cat_rates.append(compute_categorical_default_rates(df, col, docs))
        cat_dist = pd.concat(cat_dists, ignore_index=True)
        cat_rates_df = pd.concat(cat_rates, ignore_index=True)

        out_path = tmp_path / "findings.md"
        generate_findings_markdown(
            target_summary, numeric_overall, numeric_by_target, cat_dist, cat_rates_df, out_path
        )
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "Checkpoint 2A" in content
        
        # Verify LIMIT_BAL correctness
        no_def = numeric_by_target[numeric_by_target["target_value"] == 0].iloc[0]
        def_row = numeric_by_target[numeric_by_target["target_value"] == 1].iloc[0]
        
        assert f"{no_def['mean']:,.4f}" in content
        assert f"{def_row['mean']:,.4f}" in content
        assert f"{no_def['median']:,.4f}" in content
        assert f"{def_row['median']:,.4f}" in content

        # Check for Checkpoint 2A.1 findings values (approx/exact matches based on real data)
        assert "178,099.7261" in content
        assert "130,109.6564" in content
        assert "150,000" in content
        assert "90,000" in content
        assert "3 potential extreme values" in content
        assert "4" in content

        assert "rates are stable" not in content
        assert "Not in UCI docs" not in content

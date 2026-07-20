"""
test_eda_repayment_status.py
============================
Tests for Checkpoint 2B1 temporal EDA on repayment statuses.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_default.data_loader import load_raw_dataset
from credit_default.eda.static_features import TARGET_COL
from credit_default.eda.repayment_status import (
    validate_chronological_schema,
    compute_status_distribution_by_month,
    compute_distribution_conditioned_on_target,
    compute_month_to_month_transitions,
    compute_exact_sequence_patterns,
    CHRONOLOGICAL_COLS,
    MONTH_NAMES,
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Load raw dataset once per module."""
    return load_raw_dataset()


class TestSchemaValidation:
    
    def test_chronological_columns_exist(self, df: pd.DataFrame) -> None:
        for col in CHRONOLOGICAL_COLS:
            assert col in df.columns
            
    def test_pay_1_is_absent(self, df: pd.DataFrame) -> None:
        assert "PAY_1" not in df.columns
        
    def test_unexpected_pay_1_raises_error(self, df: pd.DataFrame) -> None:
        bad_df = df.copy()
        bad_df["PAY_1"] = 0
        with pytest.raises(ValueError, match="PAY_1 is present in the dataset, but expected to be absent."):
            validate_chronological_schema(bad_df)
            
    def test_missing_column_raises_error(self, df: pd.DataFrame) -> None:
        bad_df = df.drop(columns=["PAY_6"])
        with pytest.raises(ValueError, match="Missing required temporal column: PAY_6"):
            validate_chronological_schema(bad_df)
            
    def test_invalid_target_values_raises_error(self, df: pd.DataFrame) -> None:
        bad_df = df.copy()
        bad_df.loc[0, TARGET_COL] = 99
        with pytest.raises(ValueError, match="Expected binary target"):
            validate_chronological_schema(bad_df)


    def test_schema_table_generation(self, df: pd.DataFrame) -> None:
        from credit_default.eda.repayment_status import generate_schema_table
        schema_df = generate_schema_table(df)
        
        # Must have exactly 7 rows (6 chronological cols + PAY_1)
        assert len(schema_df) == 7
        
        # Check chronological mapping
        mapping = [
            (1, "April", "PAY_6", True),
            (2, "May", "PAY_5", True),
            (3, "June", "PAY_4", True),
            (4, "July", "PAY_3", True),
            (5, "August", "PAY_2", True),
            (6, "September", "PAY_0", True),
            (7, "N/A", "PAY_1", False),
        ]
        
        for i, (idx, month, col, exp_pres) in enumerate(mapping):
            row = schema_df.iloc[i]
            assert row["chronological_index"] == idx
            assert row["month"] == month
            assert row["raw_column"] == col
            assert row["expected_present"] == exp_pres
            
            # Check absence of PAY_1 is correctly captured
            if col == "PAY_1":
                assert row["actual_present"] == False
            else:
                assert row["actual_present"] == True


class TestRawDistributions:

    def test_monthly_count_totals(self, df: pd.DataFrame) -> None:
        dist_month = compute_status_distribution_by_month(df)
        for m in MONTH_NAMES:
            total = dist_month[dist_month["month"] == m]["total_count"].sum()
            assert total == len(df)
            
    def test_target_conditioned_percentage_totals(self, df: pd.DataFrame) -> None:
        dist_target = compute_distribution_conditioned_on_target(df)
        for m in MONTH_NAMES:
            for t in [0, 1]:
                sub = dist_target[(dist_target["month"] == m) & (dist_target["target_class"] == t)]
                pct_sum = sub["percentage_within_target_class"].dropna().sum()
                # Should sum to 100% since we only dropna on missing combos
                assert abs(pct_sum - 100.0) < 1e-3
                
    def test_zero_count_combinations_have_nan_rates(self, df: pd.DataFrame) -> None:
        dist_month = compute_status_distribution_by_month(df)
        zero_counts = dist_month[dist_month["total_count"] == 0]
        assert len(zero_counts) > 0, "Expected some unobserved combinations"
        for _, row in zero_counts.iterrows():
            assert not row["observed_combination"]
            assert np.isnan(row["default_rate"])
            assert np.isnan(row["population_percentage"])
            
    def test_observed_zero_rates_remain_numeric_zero(self, df: pd.DataFrame) -> None:
        dist_month = compute_status_distribution_by_month(df)
        observed = dist_month[dist_month["observed_combination"]]
        # Let's see if any have 0 defaults
        zero_defaults = observed[observed["default_count"] == 0]
        for _, row in zero_defaults.iterrows():
            assert row["default_rate"] == 0.0
            
    def test_raw_0_and_m2_preservation(self, df: pd.DataFrame) -> None:
        dist_month = compute_status_distribution_by_month(df)
        assert 0 in dist_month["raw_status_value"].values
        assert -2 in dist_month["raw_status_value"].values


class TestTransitions:

    def test_transition_count_totals(self, df: pd.DataFrame) -> None:
        trans = compute_month_to_month_transitions(df)
        for i in range(5):
            sub = trans[trans["chronological_pair_index"] == i]
            assert sub["transition_count"].sum() == len(df)
            
    def test_transition_percentage_uses_source_status_total(self, df: pd.DataFrame) -> None:
        trans = compute_month_to_month_transitions(df)
        # Take a row with some counts
        row = trans[trans["transition_count"] > 0].iloc[0]
        expected_pct = (row["transition_count"] / row["source_status_total"]) * 100
        assert abs(row["percentage_within_source_status"] - expected_pct) < 1e-5


class TestSequencePatterns:

    def test_pattern_counts_sum_to_dataset_size(self, df: pd.DataFrame) -> None:
        patterns = compute_exact_sequence_patterns(df)
        assert patterns["total_count"].sum() == len(df)
        
    def test_six_pipe_separated_values(self, df: pd.DataFrame) -> None:
        patterns = compute_exact_sequence_patterns(df)
        elem_lens = patterns["sequence_pattern"].str.split("|").str.len()
        assert (elem_lens == 6).all()
        
    def test_pattern_ordering_is_deterministic(self, df: pd.DataFrame) -> None:
        pat1 = compute_exact_sequence_patterns(df)
        pat2 = compute_exact_sequence_patterns(df)
        pd.testing.assert_frame_equal(pat1, pat2)
        
        # Check sort order: total_count DESC, sequence_pattern ASC
        counts = pat1["total_count"].values
        # Verify descending counts
        assert (np.diff(counts) <= 0).all()


class TestFigures:

    def test_figure_generation(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.repayment_status_figures import (
            plot_status_distribution_lines,
            plot_default_rate_by_status_and_month,
            plot_distribution_heatmap,
            plot_top_sequence_patterns,
        )
        import matplotlib.pyplot as plt
        
        dist_month = compute_status_distribution_by_month(df)
        patterns = compute_exact_sequence_patterns(df)
        
        f1 = plot_status_distribution_lines(dist_month, tmp_path)
        assert f1.exists() and f1.stat().st_size > 0
        
        f2 = plot_default_rate_by_status_and_month(dist_month, tmp_path)
        assert f2.exists() and f2.stat().st_size > 0
        
        f3 = plot_distribution_heatmap(dist_month, tmp_path)
        assert f3.exists() and f3.stat().st_size > 0
        
        f4 = plot_top_sequence_patterns(patterns, "total_count", "Title", "top10.png", tmp_path)
        assert f4.exists() and f4.stat().st_size > 0
        
        assert not plt.get_fignums()
        
    def test_status_styles_deterministic_and_unique(self) -> None:
        from credit_default.eda.repayment_status_figures import get_status_style
        from credit_default.eda.repayment_status import KNOWN_STATUS_CODES
        
        styles = [get_status_style(c) for c in KNOWN_STATUS_CODES]
        
        # Ensure repeatable
        styles2 = [get_status_style(c) for c in KNOWN_STATUS_CODES]
        assert styles == styles2
        
        # Ensure 12 unique styles
        unique_styles = set(tuple(s.items()) for s in styles)
        assert len(unique_styles) == 12


class TestFindings:

    def test_findings_contains_calculated_values(self, df: pd.DataFrame, tmp_path: Path) -> None:
        from credit_default.eda.repayment_status_findings import generate_repayment_findings_markdown
        
        dist_month = compute_status_distribution_by_month(df)
        dist_target = compute_distribution_conditioned_on_target(df)
        transitions = compute_month_to_month_transitions(df)
        patterns = compute_exact_sequence_patterns(df)
        
        out_md = tmp_path / "findings.md"
        generate_repayment_findings_markdown(dist_month, dist_target, transitions, patterns, out_md)
        
        content = out_md.read_text(encoding="utf-8")
        assert "PAY_1" in content
        assert "PAY_6" in content
        assert "-2" in content
        
        # Exact statistics based on 2B1.2
        assert "1,106" in content
        assert "17,202" in content
        assert "57.34%" in content
        assert "9,821" in content
        assert "1,026" in content
        assert "8,795" in content
        
        # Must not contain incorrect past values
        assert "8,312" not in content
        assert "8,301" not in content
        assert "772" not in content

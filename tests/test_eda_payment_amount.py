import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

from credit_default.data_loader import load_raw_dataset, compute_file_sha256
from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS, TARGET_COL
from credit_default.eda.payment_amount import (
    validate_chronological_schema,
    generate_schema_table,
    compute_monthly_summary,
    compute_by_target_summary,
    compute_sign_summary,
    compute_change_summary,
    compute_correlation_matrix,
    compute_display_range,
    compute_histogram_percentages,
    generate_regression_anchors,
    run_quality_checks,
    CHRONOLOGICAL_PAY_COLS
)
from credit_default.eda.payment_amount_figures import (
    plot_monthly_median_iqr_overall,
    plot_monthly_median_iqr_by_target,
    plot_six_panel_distributions,
    plot_six_panel_distributions_by_target,
    plot_six_panel_boxplots,
    plot_sign_proportions,
    plot_sign_default_rates,
    plot_adjacent_change_medians,
    plot_adjacent_change_distributions,
    plot_correlation_heatmap,
    plot_potential_extreme_counts,
    plot_outside_display_range
)

@pytest.fixture(scope="module")
def df():
    return load_raw_dataset()

class TestSchemaValidation:
    def test_chronological_columns_exist_in_order(self, df):
        validate_chronological_schema(df)
        assert CHRONOLOGICAL_PAY_COLS == ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]
        
    def test_required_column_numeric_dtype(self, df):
        for col in CHRONOLOGICAL_PAY_COLS:
            assert pd.api.types.is_numeric_dtype(df[col])
            
    def test_missing_column_raises_error(self, df):
        bad_df = df.drop(columns=["PAY_AMT3"])
        with pytest.raises(ValueError, match="Missing required temporal column: PAY_AMT3"):
            validate_chronological_schema(bad_df)

class TestCalculations:
    def test_monthly_count_reconciliation_and_no_row_removal(self, df):
        assert len(df) == 30000
        monthly = compute_monthly_summary(df)
        assert len(monthly) == 6
        assert (monthly["count"] == 30000).all()
        
    def test_quantile_iqr_and_extreme_fence_calculations(self):
        fake = pd.DataFrame({
            TARGET_COL: [0]*10,
            "PAY_AMT6": [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
            "PAY_AMT5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "PAY_AMT4": [1]*10,
            "PAY_AMT3": [1]*10,
            "PAY_AMT2": [1]*10,
            "PAY_AMT1": [1]*10
        })
        m = compute_monthly_summary(fake)
        b6 = m[m["month"] == "April"].iloc[0]
        q1, q3 = b6["p25"], b6["p75"]
        assert b6["iqr"] == q3 - q1
        assert b6["lower_fence"] == q1 - 1.5 * (q3 - q1)
        assert b6["upper_fence"] == q3 + 1.5 * (q3 - q1)
        assert b6["above_upper_fence_count"] >= 1
        
    def test_sign_classification(self, df):
        sign_over, sign_tgt = compute_sign_summary(df)
        for m in sign_over["month"].unique():
            sub = sign_over[sign_over["month"] == m]
            assert sub["total_count"].sum() == 30000
            assert np.isclose(sub["population_percentage"].sum(), 100.0)
            
    def test_default_rate_calculations(self, df):
        sign_over, sign_tgt = compute_sign_summary(df)
        for _, row in sign_over.iterrows():
            if row["total_count"] > 0:
                assert np.isclose(row["default_rate"], row["default_count"] / row["total_count"])
                
    def test_negative_values_unobserved(self, df):
        sign_over, _ = compute_sign_summary(df)
        neg_rows = sign_over[sign_over["sign_category"] == "negative"]
        assert len(neg_rows) == 6
        assert (neg_rows["observed_combination"] == False).all()
        assert neg_rows["default_rate"].isna().all()
                
    def test_adjacent_month_change_direction(self, df):
        chg_over, chg_tgt = compute_change_summary(df)
        assert len(chg_over) == 5
        r1 = chg_over[chg_over["pair_index"] == 1].iloc[0]
        diffs = df["PAY_AMT5"] - df["PAY_AMT6"]
        assert np.isclose(r1["mean_change"], diffs.mean())
        assert r1["count"] == 30000
        assert "percentage_change" not in r1.index
        
    def test_change_count_reconciliation(self, df):
        chg_over, chg_tgt = compute_change_summary(df)
        assert (chg_over["count"] == 30000).all()
        tgt_sums = chg_tgt.groupby("pair_index")["count"].sum()
        assert (tgt_sums == 30000).all()
        
    def test_correlation_matrix_symmetry_and_diagonal(self, df):
        corr = compute_correlation_matrix(df)
        assert corr.shape == (6, 6)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr.values, corr.values.T)
        
    def test_display_range_from_180k(self, df):
        disp = compute_display_range(df)
        glob = disp[disp["month"] == "Global"].iloc[0]
        assert glob["total_count"] == 180000
        sub = disp[disp["month"] != "Global"]
        assert sub["total_count"].sum() == 180000
        assert sub["outside_range_count"].sum() == glob["outside_range_count"]

    def test_independent_regression_anchors(self, df):
        monthly = compute_monthly_summary(df)
        change_over, change_tgt = compute_change_summary(df)
        disp = compute_display_range(df)
        anchors = generate_regression_anchors(df, monthly, change_over, disp)
        assert anchors["passed"].all()

class TestFiguresAndFindings:
    def test_histogram_shared_bins_and_class_percentages(self, df):
        edges, pct = compute_histogram_percentages(df, "PAY_AMT1", 40, vmin=0, vmax=100)
        assert len(edges) == 41
        assert np.isclose(pct[NEGATIVE_CLASS]["percentages"].sum(), 100.0 - pct[NEGATIVE_CLASS]["below_percentage"] - pct[NEGATIVE_CLASS]["above_percentage"])
        assert np.isclose(pct[POSITIVE_CLASS]["percentages"].sum(), 100.0 - pct[POSITIVE_CLASS]["below_percentage"] - pct[POSITIVE_CLASS]["above_percentage"])
        
    def test_histogram_stats_no_clipping(self):
        from credit_default.eda.payment_amount import compute_histogram_stats
        vals = np.array([-10, 5, 50, 120])
        edges, counts, below, above = compute_histogram_stats(vals, n_bins=10, vmin=0, vmax=100)
        assert below == 1
        assert above == 1
        assert counts.sum() == 2
        assert edges[0] == 0
        assert edges[-1] == 100
        
    def test_figures_generation_and_closing(self, df, tmp_path):
        disp = compute_display_range(df)
        monthly = compute_monthly_summary(df)
        by_target = compute_by_target_summary(df)
        sign_over, sign_tgt = compute_sign_summary(df)
        chg_over, chg_tgt = compute_change_summary(df)
        corr = compute_correlation_matrix(df)
        
        paths = [
            plot_monthly_median_iqr_overall(monthly, tmp_path),
            plot_monthly_median_iqr_by_target(by_target, tmp_path),
            plot_six_panel_distributions(df, disp, tmp_path),
            plot_six_panel_distributions_by_target(df, disp, tmp_path),
            plot_six_panel_boxplots(df, tmp_path),
            plot_sign_proportions(sign_over, tmp_path),
            plot_sign_default_rates(sign_over, tmp_path),
            plot_adjacent_change_medians(chg_over, chg_tgt, tmp_path),
            plot_adjacent_change_distributions(df, tmp_path),
            plot_correlation_heatmap(corr, tmp_path),
            plot_potential_extreme_counts(monthly, tmp_path),
            plot_outside_display_range(disp, tmp_path)
        ]
        
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
            
        assert not plt.get_fignums()
        
    def test_default_rate_axes_starting_at_zero(self, df, monkeypatch, tmp_path):
        sign_over, _ = compute_sign_summary(df)
        captured = []
        original = plt.Figure.savefig
        def mock_savefig(self, *args, **kwargs):
            captured.append(self)
            original(self, *args, **kwargs)
        monkeypatch.setattr(plt.Figure, "savefig", mock_savefig)
        
        plot_sign_default_rates(sign_over, tmp_path)
        assert len(captured) == 1
        ax = captured[0].axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin <= 0.0

    def test_matrix_annotated_in_change_medians(self, df, monkeypatch, tmp_path):
        change_over, change_tgt = compute_change_summary(df)
        captured = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *a, **k: captured.append(self))
        plot_adjacent_change_medians(change_over, change_tgt, tmp_path)
        
        ax = captured[0].axes[0]
        assert len(ax.tables) > 0
        tab = ax.tables[0]
        text_content = [cell.get_text().get_text() for (row, col), cell in tab.get_celld().items()]
        assert "Overall" in text_content
        assert "No Default" in text_content
        assert "Default" in text_content

    def test_sign_proportions_palette(self, df, monkeypatch, tmp_path):
        sign_over, _ = compute_sign_summary(df)
        captured = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *a, **k: captured.append(self))
        plot_sign_proportions(sign_over, tmp_path)
        
        ax = captured[0].axes[0]
        colors = [p.get_facecolor() for p in ax.patches]
        colors_hex = [plt.matplotlib.colors.to_hex(c) for c in colors]
        assert "#ff7f0e" in colors_hex
        assert "#7f7f7f" in colors_hex
        assert "#1f77b4" in colors_hex

    def test_findings_derived_from_tables(self, df, tmp_path):
        from credit_default.eda.payment_amount_findings import generate_payment_amount_findings
        monthly = compute_monthly_summary(df)
        by_target = compute_by_target_summary(df)
        sign_over, sign_tgt = compute_sign_summary(df)
        change_over, change_tgt = compute_change_summary(df)
        corr = compute_correlation_matrix(df)
        disp = compute_display_range(df)
        anchors = generate_regression_anchors(df, monthly, change_over, disp)
        
        out = tmp_path / "findings.md"
        generate_payment_amount_findings(monthly, by_target, sign_over, sign_tgt, change_over, change_tgt, corr, disp, anchors, out)
        
        content = out.read_text(encoding="utf-8")
        apr = monthly[monthly["month"] == "April"].iloc[0]
        assert f"{apr['median']:,.2f}" in content
        sep_zero = sign_over[(sign_over["month"] == "September") & (sign_over["sign_category"] == "zero")].iloc[0]
        assert f"{int(sep_zero['total_count']):,}" in content
        
    def test_figure_7_no_negative_trajectory(self, df, monkeypatch, tmp_path):
        sign_over, _ = compute_sign_summary(df)
        captured = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *a, **k: captured.append(self))
        plot_sign_default_rates(sign_over, tmp_path)
        
        assert len(captured) == 1
        fig = captured[0]
        ax = fig.axes[0]
        
        # Check legends
        handles, labels = ax.get_legend_handles_labels()
        assert "Negative" not in labels
        assert "Zero" in labels
        assert "Positive" in labels
        
        # Check lines
        lines = ax.get_lines()
        # Should be 2 lines (Zero, Positive)
        assert len(lines) == 2
        
        # Check text note
        texts = [t.get_text() for t in fig.texts]
        assert any("No negative PAY_AMT observations were present in the raw dataset." in t for t in texts)
        
        assert not plt.get_fignums()

class TestQualityCheckRunner:
    def test_quality_check_failure_nonzero(self, df, tmp_path):
        monthly = compute_monthly_summary(df)
        monthly.loc[0, "count"] = 29999
        schema = generate_schema_table(df)
        by_target = compute_by_target_summary(df)
        sign_over, sign_tgt = compute_sign_summary(df)
        change_over, change_tgt = compute_change_summary(df)
        corr = compute_correlation_matrix(df)
        disp = compute_display_range(df)
        anchors = generate_regression_anchors(df, monthly, change_over, disp)
        
        qc = run_quality_checks(df, schema, monthly, by_target, sign_over, sign_tgt, change_over, change_tgt, corr, disp, anchors)
        assert not qc["passed"].all()

    def test_raw_hash_unchanged(self):
        h = compute_file_sha256()
        assert h == "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"

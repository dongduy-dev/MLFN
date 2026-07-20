import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from credit_default.data_loader import load_raw_dataset
from credit_default.eda.static_features import NEGATIVE_CLASS, POSITIVE_CLASS
from credit_default.eda.bill_payment_relationship import (
    validate_schema,
    generate_schema_table,
    compute_monthly_relationship_summary,
    compute_relationship_by_target,
    compute_positive_bill_ratio,
    compute_relationship_categories,
    generate_regression_anchors,
    run_quality_checks,
    PAIRS
)
from credit_default.eda.bill_payment_relationship_figures import (
    plot_monthly_medians,
    plot_six_panel_scatter,
    plot_monthly_correlations,
    plot_ratio_median_iqr,
    plot_category_proportions,
    plot_category_default_rates
)

@pytest.fixture(scope="module")
def df():
    return load_raw_dataset()

class TestCalculations:
    def test_exact_chronology(self, df):
        validate_schema(df)
        schema = generate_schema_table(df)
        assert len(schema) == 6
        assert schema.iloc[0]["month"] == "April"
        assert schema.iloc[5]["month"] == "September"
        
    def test_neutral_same_index_difference_direction(self, df):
        monthly = compute_monthly_relationship_summary(df)
        assert "same_index_difference_mean" in monthly.columns
        apr = monthly[monthly["month"] == "April"].iloc[0]
        diff = df["PAY_AMT6"] - df["BILL_AMT6"]
        assert np.isclose(apr["same_index_difference_mean"], diff.mean())
        
    def test_ratio_eligibility_and_nan_behavior(self, df):
        ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
        apr = ratio_over[ratio_over["month"] == "April"].iloc[0]
        pos_bill_cnt = np.sum(df["BILL_AMT6"] > 0)
        nonpos_bill_cnt = len(df) - pos_bill_cnt
        assert apr["eligible_count"] == pos_bill_cnt
        assert apr["ineligible_count"] == nonpos_bill_cnt
        
        # Test NaN on mock
        mock_data = {
            "default payment next month": [0, 0, 1, 1],
            "BILL_AMT6": [100, 0, -100, 200],
            "PAY_AMT6": [50, 50, 50, 0]
        }
        for _, b, p in PAIRS:
            if b != "BILL_AMT6": mock_data[b] = [1, 1, 1, 1]
            if p != "PAY_AMT6": mock_data[p] = [1, 1, 1, 1]
            
        mock = pd.DataFrame(mock_data)
        ro, rt = compute_positive_bill_ratio(mock)
        r = ro[ro["month"] == "April"].iloc[0]
        assert r["eligible_count"] == 2
        assert r["ineligible_count"] == 2
        assert np.isclose(r["ratio_max"], 0.5)
        
    def test_no_infinite_ratios(self, df):
        ratio_over, _ = compute_positive_bill_ratio(df)
        assert not np.any(np.isinf(ratio_over["ratio_max"].dropna()))
        
    def test_category_exhaustiveness(self, df):
        cats = compute_relationship_categories(df)
        for m in cats["month"].unique():
            sub = cats[cats["month"] == m]
            assert sub["total_count"].sum() == 30000
            
    def test_target_reconciliation(self, df):
        by_target = compute_relationship_by_target(df)
        for m in by_target["month"].unique():
            sub = by_target[by_target["month"] == m]
            assert sub["count"].sum() == 30000
            
    def test_regression_anchors(self, df):
        monthly = compute_monthly_relationship_summary(df)
        ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
        cats = compute_relationship_categories(df)
        anchors = generate_regression_anchors(df, monthly, ratio_over, cats)
        assert anchors["passed"].all()
        
    def test_deterministic_output(self, df):
        m1 = compute_monthly_relationship_summary(df)
        m2 = compute_monthly_relationship_summary(df)
        assert m1.equals(m2)

class TestFiguresAndFindings:
    def test_six_figures_generated_and_closed(self, df, tmp_path):
        monthly = compute_monthly_relationship_summary(df)
        ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
        cats = compute_relationship_categories(df)
        
        paths = [
            plot_monthly_medians(monthly, tmp_path),
            plot_six_panel_scatter(df, tmp_path),
            plot_monthly_correlations(monthly, tmp_path),
            plot_ratio_median_iqr(ratio_over, ratio_tgt, tmp_path),
            plot_category_proportions(cats, tmp_path),
            plot_category_default_rates(cats, tmp_path)
        ]
        
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0
            
        assert not plt.get_fignums()
        
    def test_zero_based_default_rate_axes(self, df, monkeypatch, tmp_path):
        cats = compute_relationship_categories(df)
        captured = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *a, **k: captured.append(self))
        plot_category_default_rates(cats, tmp_path)
        
        ax = captured[0].axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin <= 0.0
        
    def test_caution_labels_only_when_applicable(self, df, monkeypatch, tmp_path):
        cats = compute_relationship_categories(df)
        captured = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, *a, **k: captured.append(self))
        plot_category_default_rates(cats, tmp_path)
        
        fig = captured[0]
        texts = [t.get_text() for t in fig.texts if "n < 200" in t.get_text()]
        needs_note = (cats["caution_flag"] == True).any()
        if needs_note:
            assert len(texts) > 0
        else:
            assert len(texts) == 0

    def test_findings_derived_from_tables(self, df, tmp_path):
        from credit_default.eda.bill_payment_relationship_findings import generate_relationship_findings
        monthly = compute_monthly_relationship_summary(df)
        ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
        cats = compute_relationship_categories(df)
        
        out = tmp_path / "findings.md"
        generate_relationship_findings(monthly, ratio_over, ratio_tgt, cats, out)
        
        content = out.read_text(encoding="utf-8")
        apr = monthly[monthly["month"] == "April"].iloc[0]
        assert f"{apr['pearson_correlation']:.4f}" in content
        
        apr_r = ratio_over[ratio_over["month"] == "April"].iloc[0]
        assert f"{int(apr_r['eligible_count']):,}" in content

class TestQualityCheckRunner:
    def test_runner_failing_when_quality_check_fails(self, df):
        schema = generate_schema_table(df)
        monthly = compute_monthly_relationship_summary(df)
        monthly.loc[0, "total_count"] = 29999
        by_target = compute_relationship_by_target(df)
        ratio_over, ratio_tgt = compute_positive_bill_ratio(df)
        cats = compute_relationship_categories(df)
        anchors = generate_regression_anchors(df, monthly, ratio_over, cats)
        
        checks = run_quality_checks(df, schema, monthly, by_target, ratio_over, ratio_tgt, cats, anchors)
        assert not checks["passed"].all()

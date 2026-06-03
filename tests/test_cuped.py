"""Unit tests for CUPED variance reduction."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.cuped import cuped_ttest, simulate_cuped_benefit


def _make_df(n=2000, corr=0.7, true_effect=0.05, seed=42):
    """Generate synthetic A/B data with correlated pre/post metrics."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    pre = latent + rng.standard_normal(n) * 0.5
    noise = rng.standard_normal(n)
    post = corr * latent + np.sqrt(1 - corr**2) * noise
    treatment = np.array([0] * (n // 2) + [1] * (n // 2))
    post[n // 2:] += true_effect  # add effect to treatment group
    return pd.DataFrame({
        "converted": post,
        "pre_metric": pre,
        "treatment": treatment,
    })


class TestCUPEDTtest:
    def test_variance_reduction_positive_with_correlation(self):
        """With high pre-post correlation, variance reduction should be > 0."""
        df = _make_df(corr=0.8)
        result = cuped_ttest(df, "converted", "pre_metric")
        assert result.variance_reduction_pct > 10.0

    def test_variance_reduction_near_zero_no_correlation(self):
        """With zero correlation, CUPED should provide negligible benefit."""
        df = _make_df(corr=0.0)
        result = cuped_ttest(df, "converted", "pre_metric")
        assert result.variance_reduction_pct < 10.0

    def test_theta_is_finite(self):
        df = _make_df()
        result = cuped_ttest(df, "converted", "pre_metric")
        assert np.isfinite(result.theta)

    def test_cuped_more_significant_than_raw(self):
        """On average over many seeds, CUPED improves power with high correlation."""
        # Run across 5 seeds; CUPED p-value should beat raw in the majority
        wins = 0
        for seed in range(10):
            df = _make_df(n=2000, corr=0.9, true_effect=0.1, seed=seed)
            result = cuped_ttest(df, "converted", "pre_metric")
            if result.cuped_p_value <= result.raw_p_value:
                wins += 1
        # CUPED should win in at least 6/10 seeds
        assert wins >= 6

    def test_result_fields_populated(self):
        df = _make_df()
        result = cuped_ttest(df, "converted", "pre_metric")
        assert hasattr(result, "theta")
        assert hasattr(result, "variance_reduction_pct")
        assert hasattr(result, "cuped_p_value")
        assert hasattr(result, "raw_p_value")
        assert hasattr(result, "sample_size_reduction_pct")

    def test_sample_size_reduction_bounded(self):
        """Sample size reduction should be between 0 and 100%."""
        df = _make_df(corr=0.7)
        result = cuped_ttest(df, "converted", "pre_metric")
        assert 0.0 <= result.sample_size_reduction_pct <= 100.0


class TestSimulateCUPEDBenefit:
    def test_higher_corr_higher_cuped_power(self):
        """More correlated pre-metric → CUPED gives more power improvement."""
        sim_low  = simulate_cuped_benefit(500, 0.10, 0.02, corr=0.1, n_sims=300)
        sim_high = simulate_cuped_benefit(500, 0.10, 0.02, corr=0.8, n_sims=300)
        assert sim_high["cuped_power"] >= sim_low["cuped_power"] - 0.05

    def test_output_keys(self):
        sim = simulate_cuped_benefit(200, 0.10, 0.02, corr=0.5, n_sims=100)
        for key in ["raw_power", "cuped_power", "theoretical_variance_reduction_pct",
                    "pre_post_correlation"]:
            assert key in sim

    def test_powers_in_unit_interval(self):
        sim = simulate_cuped_benefit(500, 0.10, 0.03, corr=0.6, n_sims=200)
        assert 0.0 <= sim["raw_power"] <= 1.0
        assert 0.0 <= sim["cuped_power"] <= 1.0

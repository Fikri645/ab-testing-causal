"""Unit tests for frequentist A/B testing functions."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.frequentist import (
    two_proportion_ztest, two_sample_ttest, chi_square_test,
    compute_power, required_sample_size, fdr_correction,
)


class TestTwoProportionZtest:
    def test_significant_difference(self):
        """Large difference between groups should be significant."""
        result = two_proportion_ztest(n_a=5000, conv_a=500,
                                      n_b=5000, conv_b=600, alpha=0.05)
        assert result.significant
        assert result.p_value < 0.05

    def test_no_difference(self):
        """Identical conversion rates should not be significant."""
        result = two_proportion_ztest(n_a=1000, conv_a=100,
                                      n_b=1000, conv_b=100, alpha=0.05)
        assert not result.significant
        assert result.observed_diff == pytest.approx(0.0, abs=1e-9)

    def test_ci_contains_true_effect(self):
        """95% CI should cover the true difference for a large sample."""
        true_diff = 0.05
        result = two_proportion_ztest(n_a=50000, conv_a=5000,
                                      n_b=50000, conv_b=7500, alpha=0.05)
        # True CVR: 0.10 vs 0.15, diff = 0.05
        assert result.ci_lower < 0.05 < result.ci_upper

    def test_positive_lift_when_b_better(self):
        result = two_proportion_ztest(n_a=1000, conv_a=100,
                                      n_b=1000, conv_b=150)
        assert result.relative_lift > 0
        assert result.observed_diff > 0

    def test_z_statistic_sign(self):
        """Positive z when B > A, negative when B < A."""
        r_pos = two_proportion_ztest(1000, 100, 1000, 150)
        r_neg = two_proportion_ztest(1000, 150, 1000, 100)
        assert r_pos.statistic > 0
        assert r_neg.statistic < 0

    def test_effect_name(self):
        result = two_proportion_ztest(1000, 100, 1000, 120)
        assert result.effect_name == "Cohen's h"

    def test_alpha_respected(self):
        """With α=0.01 instead of 0.05, same data may not be significant."""
        r05 = two_proportion_ztest(1000, 100, 1000, 115, alpha=0.05)
        r01 = two_proportion_ztest(1000, 100, 1000, 115, alpha=0.01)
        # Strict alpha means harder to reject
        assert r05.alpha == 0.05
        assert r01.alpha == 0.01


class TestTwoSampleTtest:
    def test_significant_revenue_difference(self):
        result = two_sample_ttest(
            mean_a=50.0, std_a=20.0, n_a=5000,
            mean_b=55.0, std_b=22.0, n_b=5000,
        )
        assert result.significant
        assert result.observed_diff == pytest.approx(5.0, abs=0.01)

    def test_no_difference_t(self):
        result = two_sample_ttest(
            mean_a=50.0, std_a=20.0, n_a=1000,
            mean_b=50.0, std_b=20.0, n_b=1000,
        )
        assert not result.significant
        assert result.observed_diff == pytest.approx(0.0, abs=0.01)

    def test_effect_name(self):
        result = two_sample_ttest(50, 10, 1000, 55, 10, 1000)
        assert result.effect_name == "Cohen's d"

    def test_ci_crosses_zero_when_not_significant(self):
        result = two_sample_ttest(50, 20, 100, 51, 20, 100)
        # Small n, small diff → CI should cross zero
        assert result.ci_lower < 0 < result.ci_upper


class TestPowerAnalysis:
    def test_power_increases_with_n(self):
        p1 = compute_power(500, 0.10, 0.02)
        p2 = compute_power(2000, 0.10, 0.02)
        assert p2 > p1

    def test_power_reaches_target(self):
        n = required_sample_size(0.10, 0.02, alpha=0.05, power=0.80)
        actual_power = compute_power(n, 0.10, 0.02)
        assert actual_power >= 0.78  # allow small numerical tolerance

    def test_required_n_positive(self):
        n = required_sample_size(0.05, 0.01)
        assert n > 0

    def test_smaller_mde_needs_more_samples(self):
        n1 = required_sample_size(0.10, 0.05)  # larger MDE
        n2 = required_sample_size(0.10, 0.01)  # smaller MDE
        assert n2 > n1

    def test_higher_alpha_needs_fewer_samples(self):
        n05 = required_sample_size(0.10, 0.02, alpha=0.05)
        n10 = required_sample_size(0.10, 0.02, alpha=0.10)
        assert n05 > n10

    def test_power_bounds(self):
        p = compute_power(1000, 0.10, 0.02)
        assert 0.0 <= p <= 1.0


class TestFDRCorrection:
    def test_bonferroni_is_conservative(self):
        """BH FDR should reject more tests than Bonferroni at same level."""
        pvals = [0.001, 0.01, 0.04, 0.08, 0.20]
        adj, sig = fdr_correction(pvals, alpha=0.05)
        # At least the very small p-values should be significant
        assert sig[0] is True

    def test_no_rejections_when_all_large(self):
        pvals = [0.5, 0.6, 0.7, 0.8]
        _, sig = fdr_correction(pvals, alpha=0.05)
        assert all(not s for s in sig)

    def test_adjusted_p_monotone(self):
        """Sorted p-values should produce monotone-non-decreasing adjusted p."""
        pvals = sorted([0.001, 0.005, 0.02, 0.10, 0.30])
        adj, _ = fdr_correction(pvals)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-12

    def test_output_length_matches(self):
        pvals = [0.01, 0.05, 0.10]
        adj, sig = fdr_correction(pvals)
        assert len(adj) == len(pvals)
        assert len(sig) == len(pvals)

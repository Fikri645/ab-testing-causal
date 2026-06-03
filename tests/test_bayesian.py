"""Unit tests for Bayesian A/B testing functions."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.bayesian import bayesian_proportion_test, bayesian_mean_test


class TestBayesianProportionTest:
    def test_high_probability_when_b_clearly_better(self):
        """When B has many more conversions, P(B>A) should be very high."""
        result = bayesian_proportion_test(
            n_a=5000, conv_a=500,    # 10% CVR
            n_b=5000, conv_b=750,    # 15% CVR
        )
        assert result.prob_b_beats_a > 0.99

    def test_near_50pct_when_equal(self):
        """Equal groups → P(B>A) ≈ 0.5."""
        result = bayesian_proportion_test(
            n_a=10000, conv_a=1000,
            n_b=10000, conv_b=1000,
        )
        assert 0.45 <= result.prob_b_beats_a <= 0.55

    def test_credible_interval_coverage(self):
        """95% CI should be wide enough to include zero when groups are equal."""
        result = bayesian_proportion_test(
            n_a=500, conv_a=50,
            n_b=500, conv_b=50,
        )
        assert result.ci_lower < 0 < result.ci_upper

    def test_expected_loss_trade_off(self):
        """
        When B is much better, expected loss of choosing A should be large,
        and loss of choosing B should be near zero.
        """
        result = bayesian_proportion_test(
            n_a=5000, conv_a=500,
            n_b=5000, conv_b=750,
        )
        assert result.expected_loss_choosing_a > result.expected_loss_choosing_b
        assert result.expected_loss_choosing_b < 0.001

    def test_model_label(self):
        result = bayesian_proportion_test(100, 10, 100, 12)
        assert result.model == "Beta-Binomial"

    def test_posterior_mean_close_to_mle(self):
        """With large n, posterior mean should be close to sample proportion."""
        n, conv = 10000, 1000
        result = bayesian_proportion_test(n, conv, n, conv)
        assert abs(result.posterior_mean_a - conv / n) < 0.005

    def test_probability_in_unit_interval(self):
        result = bayesian_proportion_test(200, 20, 200, 30)
        assert 0.0 <= result.prob_b_beats_a <= 1.0

    def test_n_samples_stored(self):
        result = bayesian_proportion_test(100, 10, 100, 12, n_samples=50_000)
        assert result.n_samples == 50_000


class TestBayesianMeanTest:
    def test_high_probability_b_better(self):
        result = bayesian_mean_test(
            mean_a=50.0, std_a=10.0, n_a=2000,
            mean_b=55.0, std_b=10.0, n_b=2000,
        )
        assert result.prob_b_beats_a > 0.99

    def test_near_50_when_equal(self):
        result = bayesian_mean_test(
            mean_a=50.0, std_a=10.0, n_a=1000,
            mean_b=50.0, std_b=10.0, n_b=1000,
        )
        assert 0.40 <= result.prob_b_beats_a <= 0.60

    def test_model_label(self):
        result = bayesian_mean_test(50, 10, 100, 55, 10, 100)
        assert result.model == "Normal-Normal"

    def test_ci_sign_when_b_better(self):
        result = bayesian_mean_test(
            mean_a=50.0, std_a=5.0, n_a=5000,
            mean_b=55.0, std_b=5.0, n_b=5000,
        )
        # Both bounds of CI should be positive (B is clearly better)
        assert result.ci_lower > 0

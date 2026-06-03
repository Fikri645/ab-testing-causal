"""Unit tests for sequential testing (mSPRT)."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.sequential import sequential_test, confidence_sequence, simulate_peeking


class TestSequentialTest:
    def test_detects_large_effect(self):
        """A large effect should be detected well before the end of the stream."""
        rng = np.random.default_rng(0)
        # Large true effect
        diffs = rng.normal(1.0, 1.0, 300).tolist()
        result = sequential_test(diffs, alpha=0.05)
        assert result.final_rejected
        assert result.rejected_at is not None
        assert result.rejected_at < 300

    def test_valid_under_null(self):
        """
        Under H0 over many simulations, the false-positive rate should be ≤ α.
        (We use a generous tolerance because it's a probabilistic test.)
        """
        alpha = 0.05
        rng = np.random.default_rng(99)
        n_sims = 500
        rejections = 0
        for _ in range(n_sims):
            diffs = rng.normal(0.0, 1.0, 200).tolist()
            r = sequential_test(diffs, alpha=alpha)
            if r.final_rejected:
                rejections += 1
        fpr = rejections / n_sims
        # Should be ≤ α with some tolerance (not strictly, due to finite sims)
        assert fpr <= alpha * 3  # generous tolerance for unit tests

    def test_threshold_is_inverse_alpha(self):
        diffs = [0.1] * 10
        result = sequential_test(diffs, alpha=0.05)
        assert result.threshold == pytest.approx(1.0 / 0.05)

    def test_e_values_length(self):
        diffs = list(range(50))
        result = sequential_test(diffs)
        assert len(result.e_values) == 50

    def test_e_values_positive(self):
        rng = np.random.default_rng(7)
        diffs = rng.standard_normal(100).tolist()
        result = sequential_test(diffs)
        assert all(ev >= 0 for ev in result.e_values)


class TestConfidenceSequence:
    def test_output_lengths(self):
        obs = list(range(1, 101))
        means, lowers, uppers = confidence_sequence(obs)
        assert len(means) == 100
        assert len(lowers) == 100
        assert len(uppers) == 100

    def test_lower_le_mean_le_upper(self):
        rng = np.random.default_rng(42)
        obs = rng.normal(5.0, 2.0, 200).tolist()
        means, lowers, uppers = confidence_sequence(obs)
        for m, lo, up in zip(means, lowers, uppers):
            assert lo <= m <= up

    def test_interval_shrinks_over_time(self):
        """Confidence sequence should tighten as n grows (on average)."""
        rng = np.random.default_rng(42)
        obs = rng.normal(0.0, 1.0, 500).tolist()
        _, lowers, uppers = confidence_sequence(obs)
        widths = [u - l for u, l in zip(uppers, lowers)]
        # Width at n=500 should be narrower than at n=10
        assert widths[499] < widths[9]


class TestPeekingSimulation:
    def test_traditional_inflated(self):
        """Traditional testing with peeking should inflate the FPR above α."""
        sim = simulate_peeking(n_total=500, n_experiments=500, alpha=0.05, seed=123)
        assert sim.traditional_fpr > 0.05

    def test_msprt_controlled(self):
        """mSPRT should not inflate beyond α (with some tolerance)."""
        sim = simulate_peeking(n_total=500, n_experiments=500, alpha=0.05, seed=123)
        # Allow 2× tolerance for finite simulation noise
        assert sim.msprt_fpr <= 0.05 * 2.5

    def test_peek_schedule_stored(self):
        sim = simulate_peeking(n_total=100, n_experiments=50,
                               peek_fractions=[0.5, 1.0])
        assert 50 in sim.peek_schedule
        assert 100 in sim.peek_schedule

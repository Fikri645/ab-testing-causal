"""
Sequential / Always-Valid Testing.

Reference: Johari, Pekelis, Walsh (2015) "Always Valid Inference:
Bringing Sequential Analysis to A/B Testing." arXiv:1512.04922.

Traditional A/B testing requires a fixed sample size decided *before* the
experiment. If you "peek" at the data and stop early when p < α, the actual
false-positive rate inflates to well above α.

The mixture Sequential Probability Ratio Test (mSPRT) solves this by
maintaining a martingale M_t whose expected value under H0 is 1. We reject
when M_t ≥ 1/α — valid at *any* stopping time, with E[false positives] ≤ α.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SequentialResult:
    n_obs: int
    e_values: List[float]           # running mSPRT e-value
    rejected_at: Optional[int]      # first time M_t ≥ 1/α  (None if never)
    final_rejected: bool
    alpha: float
    threshold: float                # = 1 / alpha


@dataclass
class PeekingSimulation:
    """Results of simulating α-inflation from peeking."""
    n_experiments: int
    peek_schedule: List[int]        # sample sizes where we peek
    traditional_fpr: float          # false positive rate with peeking
    msprt_fpr: float                # false positive rate with mSPRT (≈ α)
    alpha: float


# ── Core mSPRT computation ─────────────────────────────────────────────────────

def _msprt_e_value(
    x_bar: float,        # sample mean of X_i (each X_i = treatment - control obs)
    n: int,              # number of paired observations seen so far
    sigma: float,        # estimated SD of the per-observation differences
    rho: float,          # mixing prior SD (hyperparameter; use rho=sigma for diffuse prior)
) -> float:
    """
    mSPRT e-value for the normal model at sample size n.

    M_n = sqrt(sigma^2 / (sigma^2 + n*rho^2))
           * exp(n^2 * x_bar^2 * rho^2 / (2*sigma^2*(sigma^2 + n*rho^2)))

    Under H0 (true mean = 0):  E[M_n] = 1  for all n.
    Under H1 (true mean ≠ 0):  M_n → ∞  (detection guaranteed).
    """
    sigma2 = sigma ** 2
    rho2   = rho   ** 2
    denom  = sigma2 + n * rho2

    factor1 = np.sqrt(sigma2 / denom)
    s       = n * x_bar          # sum of differences
    factor2 = np.exp(s ** 2 * rho2 / (2 * sigma2 * denom))

    return float(factor1 * factor2)


def sequential_test(
    diffs: List[float],   # per-observation treatment - control differences
    alpha: float = 0.05,
    rho_scale: float = 1.0,   # ρ = rho_scale * sigma
) -> SequentialResult:
    """
    Run mSPRT on a stream of paired differences.

    Parameters
    ----------
    diffs      : list of (X_treatment_i - X_control_i) values
    alpha      : significance level
    rho_scale  : mixing prior width relative to sigma
    """
    obs      = np.array(diffs, dtype=float)
    sigma    = float(obs.std()) if obs.std() > 0 else 1.0
    rho      = rho_scale * sigma
    threshold = 1.0 / alpha

    e_values    = []
    rejected_at = None

    for n in range(1, len(obs) + 1):
        x_bar = obs[:n].mean()
        ev = _msprt_e_value(x_bar, n, sigma, rho)
        e_values.append(round(ev, 4))

        if rejected_at is None and ev >= threshold:
            rejected_at = n

    return SequentialResult(
        n_obs=len(obs),
        e_values=e_values,
        rejected_at=rejected_at,
        final_rejected=rejected_at is not None,
        alpha=alpha,
        threshold=threshold,
    )


# ── Confidence sequence ────────────────────────────────────────────────────────

def confidence_sequence(
    obs: List[float],
    alpha: float = 0.05,
    sigma: float = None,
) -> tuple[List[float], List[float], List[float]]:
    """
    Anytime-valid confidence sequence for the mean.

    The width at sample size n is proportional to sqrt(log(log(n)) / n),
    wider than fixed CIs at small n (honest about uncertainty) but valid
    for ALL stopping rules simultaneously.

    Returns (means, lower_bounds, upper_bounds).
    """
    obs = np.array(obs, dtype=float)
    if sigma is None:
        sigma = float(obs.std()) if obs.std() > 0 else 1.0

    means, lowers, uppers = [], [], []
    for n in range(1, len(obs) + 1):
        xbar = float(obs[:n].mean())
        # Howard et al. (2021) – stitched normal mixture CS
        # Half-width: sigma * sqrt(2*(n+v)/(n^2*v) * log(sqrt((n+v)/v) / alpha))
        # with v = 1 (intrinsic time = 1)
        v = 1.0
        inner = max((n + v) / (n ** 2 * v) * np.log(np.sqrt((n + v) / v) / alpha), 0)
        hw = sigma * np.sqrt(2 * inner)
        means.append(round(xbar, 6))
        lowers.append(round(xbar - hw, 6))
        uppers.append(round(xbar + hw, 6))

    return means, lowers, uppers


# ── Peeking simulation ────────────────────────────────────────────────────────

def simulate_peeking(
    n_total: int = 1000,
    n_experiments: int = 2000,
    peek_fractions: List[float] = None,
    alpha: float = 0.05,
    sigma: float = 1.0,
    seed: int = 42,
) -> PeekingSimulation:
    """
    Simulate the α-inflation caused by peeking under the null hypothesis.

    Under H0 (no treatment effect), both methods are run:
      1. Traditional: reject if ANY peek gives p < α
      2. mSPRT:       reject when M_t ≥ 1/α at any peek

    Expected result:
      - Traditional FPR ≈ 2× to 3× the nominal α (due to multiple comparisons)
      - mSPRT FPR ≈ α (always-valid)
    """
    if peek_fractions is None:
        peek_fractions = [0.25, 0.50, 0.75, 1.00]

    rng = np.random.default_rng(seed)
    peek_schedule = [int(f * n_total) for f in peek_fractions]
    threshold = 1.0 / alpha

    traditional_rejects = 0
    msprt_rejects = 0

    for _ in range(n_experiments):
        # Generate n_total i.i.d. observations under H0 (true mean = 0)
        obs = rng.normal(0.0, sigma, n_total)
        sigma_hat = float(obs.std()) if obs.std() > 0 else 1.0
        rho = sigma_hat

        trad_rejected = False
        msprt_rejected = False

        for peek_n in peek_schedule:
            if peek_n == 0:
                continue
            window = obs[:peek_n]

            # Traditional: two-sided z-test
            if not trad_rejected:
                z = window.mean() / (sigma_hat / np.sqrt(peek_n))
                p = 2 * (1 - _norm_cdf(abs(z)))
                if p < alpha:
                    trad_rejected = True

            # mSPRT
            if not msprt_rejected:
                ev = _msprt_e_value(window.mean(), peek_n, sigma_hat, rho)
                if ev >= threshold:
                    msprt_rejected = True

        if trad_rejected:
            traditional_rejects += 1
        if msprt_rejected:
            msprt_rejects += 1

    return PeekingSimulation(
        n_experiments=n_experiments,
        peek_schedule=peek_schedule,
        traditional_fpr=round(traditional_rejects / n_experiments, 4),
        msprt_fpr=round(msprt_rejects / n_experiments, 4),
        alpha=alpha,
    )


def simulate_detection_speed(
    true_effect: float,
    n_max: int = 500,
    n_experiments: int = 1000,
    sigma: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Simulate how quickly mSPRT detects a true effect vs. a fixed-n test.

    Returns distribution of stopping times for mSPRT (in number of observations).
    """
    rng = np.random.default_rng(seed)
    threshold = 1.0 / alpha
    stopping_times = []
    never_detected = 0

    for _ in range(n_experiments):
        obs = rng.normal(true_effect, sigma, n_max)
        sigma_hat = sigma  # known in this simulation
        rho = sigma_hat
        detected = False

        for n in range(1, n_max + 1):
            ev = _msprt_e_value(obs[:n].mean(), n, sigma_hat, rho)
            if ev >= threshold:
                stopping_times.append(n)
                detected = True
                break

        if not detected:
            never_detected += 1
            stopping_times.append(n_max)

    arr = np.array(stopping_times)
    return {
        "true_effect": true_effect,
        "alpha": alpha,
        "n_max": n_max,
        "n_experiments": n_experiments,
        "power": round(1 - never_detected / n_experiments, 4),
        "median_stopping_time": int(np.median(arr)),
        "p25_stopping_time": int(np.percentile(arr, 25)),
        "p75_stopping_time": int(np.percentile(arr, 75)),
        "stopping_times": arr.tolist(),
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (avoid scipy import in tight loop)."""
    from math import erfc, sqrt
    return 0.5 * erfc(-x / sqrt(2))

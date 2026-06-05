"""
Bayesian A/B testing using conjugate priors.

Conversion rates  → Beta-Binomial model (exact posterior)
Continuous metrics → Normal-Normal model (posterior on mean difference)

Key outputs:
  - P(B > A)        : probability treatment is better (decision metric)
  - Expected loss   : expected cost of the wrong decision (risk metric)
  - Credible interval: Bayesian analogue of confidence interval
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class BayesianResult:
    model: str
    prob_b_beats_a: float           # P(θ_B > θ_A | data)
    expected_loss_choosing_b: float # E[max(θ_A - θ_B, 0)]  — loss if B is wrong choice
    expected_loss_choosing_a: float # E[max(θ_B - θ_A, 0)]  — loss if A is wrong choice
    ci_lower: float                 # 95 % credible interval for (θ_B - θ_A)
    ci_upper: float
    posterior_mean_a: float
    posterior_mean_b: float
    posterior_std_a: float
    posterior_std_b: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


# ── Beta-Binomial (conversion rates) ─────────────────────────────────────────

def bayesian_proportion_test(
    n_a: int, conv_a: int,
    n_b: int, conv_b: int,
    prior_alpha: float = 1.0,   # Beta(α, β) prior — default Uniform
    prior_beta: float  = 1.0,
    n_samples: int = 100_000,
    seed: int = 42,
) -> BayesianResult:
    """
    Bayesian A/B test for binary conversion rates.

    Prior:     θ ~ Beta(α₀, β₀)
    Likelihood: X | θ ~ Binomial(n, θ)
    Posterior: θ | X ~ Beta(α₀ + conv, β₀ + n - conv)

    The Beta-Binomial is a conjugate model — no MCMC required.
    """
    rng = np.random.default_rng(seed)

    # Posterior parameters
    a_a = prior_alpha + conv_a
    b_a = prior_beta  + (n_a - conv_a)
    a_b = prior_alpha + conv_b
    b_b = prior_beta  + (n_b - conv_b)

    # Monte-Carlo samples from posteriors
    samples_a = rng.beta(a_a, b_a, n_samples)
    samples_b = rng.beta(a_b, b_b, n_samples)
    lift = samples_b - samples_a

    prob_b_beats_a = float(np.mean(lift > 0))

    # Expected loss (decision-theoretic risk)
    # loss_B = cost of deploying B when A was better
    # loss_A = cost of keeping A when B was better
    expected_loss_b = float(np.mean(np.maximum(-lift, 0)))
    expected_loss_a = float(np.mean(np.maximum(lift,  0)))

    ci = (float(np.percentile(lift, 2.5)), float(np.percentile(lift, 97.5)))

    # Posterior summary statistics (Beta moments)
    mean_a = a_a / (a_a + b_a)
    mean_b = a_b / (a_b + b_b)
    std_a  = np.sqrt(a_a * b_a / ((a_a + b_a) ** 2 * (a_a + b_a + 1)))
    std_b  = np.sqrt(a_b * b_b / ((a_b + b_b) ** 2 * (a_b + b_b + 1)))

    return BayesianResult(
        model="Beta-Binomial",
        prob_b_beats_a=round(prob_b_beats_a, 5),
        expected_loss_choosing_b=round(expected_loss_b, 6),
        expected_loss_choosing_a=round(expected_loss_a, 6),
        ci_lower=round(ci[0], 6),
        ci_upper=round(ci[1], 6),
        posterior_mean_a=round(float(mean_a), 6),
        posterior_mean_b=round(float(mean_b), 6),
        posterior_std_a=round(float(std_a), 6),
        posterior_std_b=round(float(std_b), 6),
        n_samples=n_samples,
    )


# ── Normal-Normal (continuous metrics, e.g. revenue) ─────────────────────────

def bayesian_mean_test(
    mean_a: float, std_a: float, n_a: int,
    mean_b: float, std_b: float, n_b: int,
    prior_mean: float = 0.0,
    prior_std: float = 1000.0,    # diffuse prior
    n_samples: int = 100_000,
    seed: int = 42,
) -> BayesianResult:
    """
    Bayesian test for difference in means with known-variance Normal model.

    Prior:       μ ~ N(μ₀, σ₀²)
    Likelihood:  X̄ | μ ~ N(μ, σ²/n)  (known σ approximated by sample std)
    Posterior:   μ | X̄ ~ N(μₙ, σₙ²)  (standard conjugate update)
    """
    rng = np.random.default_rng(seed)

    def normal_posterior(obs_mean, obs_std, n):
        prior_var = prior_std ** 2
        obs_var   = obs_std ** 2 / n
        post_var  = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        post_mean = post_var * (prior_mean / prior_var + obs_mean / obs_var)
        return post_mean, np.sqrt(post_var)

    pm_a, ps_a = normal_posterior(mean_a, std_a, n_a)
    pm_b, ps_b = normal_posterior(mean_b, std_b, n_b)

    samples_a = rng.normal(pm_a, ps_a, n_samples)
    samples_b = rng.normal(pm_b, ps_b, n_samples)
    lift = samples_b - samples_a

    prob_b_beats_a = float(np.mean(lift > 0))
    expected_loss_b = float(np.mean(np.maximum(-lift, 0)))
    expected_loss_a = float(np.mean(np.maximum(lift,  0)))
    ci = (float(np.percentile(lift, 2.5)), float(np.percentile(lift, 97.5)))

    return BayesianResult(
        model="Normal-Normal",
        prob_b_beats_a=round(prob_b_beats_a, 5),
        expected_loss_choosing_b=round(expected_loss_b, 6),
        expected_loss_choosing_a=round(expected_loss_a, 6),
        ci_lower=round(ci[0], 4),
        ci_upper=round(ci[1], 4),
        posterior_mean_a=round(pm_a, 4),
        posterior_mean_b=round(pm_b, 4),
        posterior_std_a=round(ps_a, 6),
        posterior_std_b=round(ps_b, 6),
        n_samples=n_samples,
    )

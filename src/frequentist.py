"""
Frequentist A/B testing: z-test, t-test, power analysis, FDR correction.

All functions return typed dataclasses for easy serialisation.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple

from scipy import stats


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    ci_lower: float          # lower bound of CI / credible interval for diff
    ci_upper: float
    observed_diff: float     # point estimate of (B – A)
    relative_lift: float     # (B – A) / A  [%]
    effect_size: float
    effect_name: str
    significant: bool
    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Two-proportion Z-test ─────────────────────────────────────────────────────

def two_proportion_ztest(
    n_a: int, conv_a: int,
    n_b: int, conv_b: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> TestResult:
    """
    Z-test for difference in conversion rates.

    Uses a pooled standard error under H0 (standard frequentist approach) and
    unpooled SE for the confidence interval (correct coverage semantics).
    """
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    # Pooled SE for the test statistic
    se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_test == 0:
        se_test = 1e-12

    z = (p_b - p_a) / se_test
    p_value = 2 * (1 - stats.norm.cdf(abs(z))) if two_tailed else (1 - stats.norm.cdf(z))

    # Unpooled SE for the CI
    se_ci = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    ci = (diff - z_crit * se_ci, diff + z_crit * se_ci)

    # Cohen's h effect size
    cohen_h = 2 * np.arcsin(np.sqrt(p_b)) - 2 * np.arcsin(np.sqrt(p_a))

    rel_lift = diff / p_a * 100 if p_a > 0 else 0.0

    return TestResult(
        test_name="Two-proportion Z-test",
        statistic=round(z, 4),
        p_value=round(p_value, 6),
        ci_lower=round(ci[0], 6),
        ci_upper=round(ci[1], 6),
        observed_diff=round(diff, 6),
        relative_lift=round(rel_lift, 2),
        effect_size=round(cohen_h, 4),
        effect_name="Cohen's h",
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


# ── Two-sample t-test ─────────────────────────────────────────────────────────

def two_sample_ttest(
    mean_a: float, std_a: float, n_a: int,
    mean_b: float, std_b: float, n_b: int,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TestResult:
    """
    Welch's t-test for difference in means (e.g., revenue per user).
    equal_var=False uses Welch's approximation; equal_var=True uses Student's.
    """
    t, p_value = stats.ttest_ind_from_stats(
        mean_a, std_a, n_a, mean_b, std_b, n_b, equal_var=equal_var
    )

    # Cohen's d (pooled SD denominator)
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

    # CI for the difference (Welch approximation)
    diff = mean_b - mean_a
    se = np.sqrt(std_a ** 2 / n_a + std_b ** 2 / n_b)
    # Welch–Satterthwaite df
    df_num = (std_a ** 2 / n_a + std_b ** 2 / n_b) ** 2
    df_den = (std_a ** 2 / n_a) ** 2 / (n_a - 1) + (std_b ** 2 / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else n_a + n_b - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - t_crit * se, diff + t_crit * se)

    rel_lift = diff / mean_a * 100 if mean_a != 0 else 0.0

    return TestResult(
        test_name="Welch's t-test",
        statistic=round(float(t), 4),
        p_value=round(float(p_value), 6),
        ci_lower=round(ci[0], 4),
        ci_upper=round(ci[1], 4),
        observed_diff=round(diff, 4),
        relative_lift=round(rel_lift, 2),
        effect_size=round(cohens_d, 4),
        effect_name="Cohen's d",
        significant=bool(p_value < alpha),
        alpha=alpha,
    )


# ── Power analysis ────────────────────────────────────────────────────────────

def compute_power(
    n_per_group: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> float:
    """
    Statistical power for a two-proportion z-test.

    Power = P(reject H0 | H1 is true).
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    se = np.sqrt(2 * p_avg * (1 - p_avg) / n_per_group)
    if se == 0:
        return 0.0

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    delta = abs(p2 - p1)
    z = delta / se - z_alpha
    return float(stats.norm.cdf(z))


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """
    Minimum sample size per group for a two-proportion z-test.

    Uses the exact formula rather than binary search for speed.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    z_beta  = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2

    return int(np.ceil(numerator / denominator))


def power_curve(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    n_max_multiplier: float = 3.0,
) -> Tuple[List[int], List[float]]:
    """
    Compute power vs sample size for plotting.

    Returns (sample_sizes, powers).
    """
    n_req = required_sample_size(baseline_rate, mde, alpha, power=0.80)
    n_max = max(int(n_req * n_max_multiplier), 500)
    ns = list(range(50, n_max, max(1, n_max // 200)))
    powers = [compute_power(n, baseline_rate, mde, alpha) for n in ns]
    return ns, powers


# ── Multiple testing correction ───────────────────────────────────────────────

def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Benjamini–Hochberg FDR correction for multiple simultaneous tests.

    Returns (adjusted_p_values, significance_flags).
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    # BH step-up procedure
    adjusted = sorted_p * n / np.arange(1, n + 1)
    # Enforce monotonicity (right to left minimum)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.empty(n)
    result[order] = adjusted

    return list(result), [bool(v < alpha) for v in result]


# ── Chi-square test of independence ──────────────────────────────────────────

def chi_square_test(
    n_a: int, conv_a: int,
    n_b: int, conv_b: int,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-square test of independence for a 2×2 contingency table.
    Equivalent to the z-test for proportions (z² = χ²) but more familiar
    to some practitioners.
    """
    table = np.array([
        [conv_a, n_a - conv_a],
        [conv_b, n_b - conv_b],
    ])
    chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)

    p_a = conv_a / n_a
    p_b = conv_b / n_b
    diff = p_b - p_a

    # Cramér's V effect size
    n_total = n_a + n_b
    cramers_v = np.sqrt(chi2 / n_total)

    return TestResult(
        test_name="Chi-square test",
        statistic=round(chi2, 4),
        p_value=round(p_value, 6),
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        observed_diff=round(diff, 6),
        relative_lift=round(diff / p_a * 100 if p_a > 0 else 0.0, 2),
        effect_size=round(cramers_v, 4),
        effect_name="Cramér's V",
        significant=bool(p_value < alpha),
        alpha=alpha,
    )

"""
CUPED – Controlled-experiment Using Pre-Existing Data.

Reference: Deng, Xu, Kohavi, Walker (2013) "Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data."
Microsoft Research. KDD 2013.

Key idea
--------
The post-experiment metric Y is correlated with a pre-experiment covariate X
(e.g., last month's conversion rate). We subtract the part of Y that is
"predictable" from X, leaving a lower-variance residual. This reduces the
required sample size by (1 – ρ²), where ρ = Corr(Y, X).

Usage
-----
    from src.cuped import CUPEDResult, cuped_ttest
    result = cuped_ttest(df, metric_col="converted", pre_metric_col="pre_conv")
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from scipy import stats


@dataclass
class CUPEDResult:
    theta: float                  # regression coefficient of X on Y
    corr_pre_post: float          # Pearson ρ between pre and post metric
    variance_reduction_pct: float # % reduction in metric variance

    # Raw (unadjusted) test
    raw_mean_control: float
    raw_mean_treatment: float
    raw_t_stat: float
    raw_p_value: float
    raw_significant: bool

    # CUPED-adjusted test
    cuped_mean_control: float
    cuped_mean_treatment: float
    cuped_t_stat: float
    cuped_p_value: float
    cuped_significant: bool

    # Effective sample size saving
    sample_size_reduction_pct: float   # how much smaller an experiment could be

    alpha: float

    def to_dict(self) -> dict:
        return asdict(self)


def _cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
    x_global_mean: float,
) -> tuple[np.ndarray, float]:
    """
    Return (y_adjusted, theta) where:
        y_adj = y - theta * (x - E[X])
        theta = Cov(Y, X) / Var(X)
    """
    cov_matrix = np.cov(y, x)
    theta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
    y_adj = y - theta * (x - x_global_mean)
    return y_adj, theta


def cuped_ttest(
    df: pd.DataFrame,
    metric_col: str,
    pre_metric_col: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05,
) -> CUPEDResult:
    """
    Run a t-test on both raw and CUPED-adjusted metrics.

    Parameters
    ----------
    df            : DataFrame with columns [metric_col, pre_metric_col, treatment_col]
    metric_col    : post-experiment outcome (e.g., "revenue", "converted")
    pre_metric_col: pre-experiment covariate (same metric from before the experiment)
    treatment_col : 0 = control, 1 = treatment
    alpha         : significance level
    """
    ctrl = df[df[treatment_col] == 0]
    trt  = df[df[treatment_col] == 1]

    y_ctrl = ctrl[metric_col].values.astype(float)
    y_trt  = trt[metric_col].values.astype(float)
    x_ctrl = ctrl[pre_metric_col].values.astype(float)
    x_trt  = trt[pre_metric_col].values.astype(float)

    # Theta is estimated on the full dataset (pooled) — avoids bias
    y_all = df[metric_col].values.astype(float)
    x_all = df[pre_metric_col].values.astype(float)
    x_global_mean = x_all.mean()

    _, theta = _cuped_adjust(y_all, x_all, x_global_mean)

    # Apply same theta to each group (uses global X̄ to keep E[Y_adj] = E[Y])
    y_ctrl_adj = y_ctrl - theta * (x_ctrl - x_global_mean)
    y_trt_adj  = y_trt  - theta * (x_trt  - x_global_mean)

    # Variance reduction
    var_raw   = np.var(y_all)
    var_cuped = np.var(y_all - theta * (x_all - x_global_mean))
    var_reduction = (1.0 - var_cuped / var_raw) * 100.0 if var_raw > 0 else 0.0

    # Correlation of pre and post metric
    corr = float(np.corrcoef(y_all, x_all)[0, 1])

    # Raw t-test
    t_raw, p_raw = stats.ttest_ind(y_ctrl, y_trt)

    # CUPED t-test
    t_cup, p_cup = stats.ttest_ind(y_ctrl_adj, y_trt_adj)

    # Sample size reduction (CUPED needs (1-ρ²) fraction of original n)
    ss_reduction = corr ** 2 * 100.0

    return CUPEDResult(
        theta=round(float(theta), 6),
        corr_pre_post=round(corr, 4),
        variance_reduction_pct=round(var_reduction, 2),
        raw_mean_control=round(float(y_ctrl.mean()), 6),
        raw_mean_treatment=round(float(y_trt.mean()), 6),
        raw_t_stat=round(float(t_raw), 4),
        raw_p_value=round(float(p_raw), 6),
        raw_significant=bool(p_raw < alpha),
        cuped_mean_control=round(float(y_ctrl_adj.mean()), 6),
        cuped_mean_treatment=round(float(y_trt_adj.mean()), 6),
        cuped_t_stat=round(float(t_cup), 4),
        cuped_p_value=round(float(p_cup), 6),
        cuped_significant=bool(p_cup < alpha),
        sample_size_reduction_pct=round(ss_reduction, 2),
        alpha=alpha,
    )


def simulate_cuped_benefit(
    n_per_group: int,
    baseline_rate: float,
    true_effect: float,
    corr: float,          # correlation between pre and post metric
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Simulate A/B experiments at a given pre-post correlation to show
    how CUPED variance reduction improves power.

    Returns power estimates for raw and CUPED-adjusted tests.
    """
    rng = np.random.default_rng(seed)
    raw_sig   = 0
    cuped_sig = 0

    for _ in range(n_sims):
        # Generate pre-experiment metric (user latent quality)
        latent = rng.standard_normal(n_per_group * 2)
        pre    = latent + rng.standard_normal(n_per_group * 2)

        # Post-experiment metric correlated with pre
        noise  = rng.standard_normal(n_per_group * 2)
        post_base = corr * latent + np.sqrt(1 - corr ** 2) * noise

        ctrl_post = post_base[:n_per_group]
        trt_post  = post_base[n_per_group:] + true_effect  # add treatment effect

        ctrl_pre = pre[:n_per_group]
        trt_pre  = pre[n_per_group:]

        # Raw test
        _, p_raw = stats.ttest_ind(ctrl_post, trt_post)

        # CUPED adjustment
        y_all = np.concatenate([ctrl_post, trt_post])
        x_all = np.concatenate([ctrl_pre, trt_pre])
        x_mean = x_all.mean()
        cov = np.cov(y_all, x_all)
        theta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

        ctrl_adj = ctrl_post - theta * (ctrl_pre - x_mean)
        trt_adj  = trt_post  - theta * (trt_pre  - x_mean)
        _, p_cup = stats.ttest_ind(ctrl_adj, trt_adj)

        if p_raw < alpha:
            raw_sig += 1
        if p_cup < alpha:
            cuped_sig += 1

    return {
        "n_per_group": n_per_group,
        "true_effect": true_effect,
        "pre_post_correlation": corr,
        "theoretical_variance_reduction_pct": round(corr ** 2 * 100, 2),
        "raw_power": round(raw_sig / n_sims, 4),
        "cuped_power": round(cuped_sig / n_sims, 4),
        "power_gain_pct": round((cuped_sig - raw_sig) / max(raw_sig, 1) * 100, 1),
    }

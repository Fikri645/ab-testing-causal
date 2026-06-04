"""
A/B Testing & Causal Inference Simulator — Gradio App

4-tab interactive dashboard:
  Tab 1: Power Analysis         — sample size calculator + power curve
  Tab 2: A/B Test Analyzer      — Frequentist vs Bayesian vs CUPED
  Tab 3: Sequential Testing     — peeking problem + mSPRT solution (pre-computed)
  Tab 4: Uplift Modeling (HTE)  — heterogeneous treatment effects from Hillstrom

Heavy computations (HTE training) are pre-computed and loaded from JSON.
Light computations (z-test, Bayesian) run on-the-fly with scipy only.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frequentist import (
    two_proportion_ztest, compute_power, required_sample_size,
)
from src.bayesian import bayesian_proportion_test

# ── Load pre-computed results ─────────────────────────────────────────────────

def _load(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _load_hte(path: Path):
    """
    Load pre-processed HTE JSON (histograms already computed).
    Falls back to the full hte_results.json and converts on-the-fly if needed.
    """
    data = _load(path)
    if data is None:
        return None
    # If the file still has raw CATE arrays (legacy), convert them now
    for outcome in ["conversion", "spend"]:
        if outcome not in data:
            continue
        for mk in ["causalforest", "x_learner", "t_learner"]:
            col = f"{mk}_cates"
            if col in data[outcome]:
                arr = np.array(data[outcome][col], dtype=float)
                counts, edges = np.histogram(arr, bins=60)
                data[outcome][f"{mk}_hist"] = {
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "pct_positive": float((arr > 0).mean() * 100),
                }
                del data[outcome][col]
    return data

ANALYSIS = _load(ROOT / "data" / "processed" / "analysis_results.json")
SEQ_SIM  = _load(ROOT / "data" / "processed" / "sequential_sim.json")
# Use pre-processed lightweight file (23 KB vs 5.4 MB for hte_results.json)
_hte_app_path = ROOT / "data" / "processed" / "hte_app.json"
_hte_full_path = ROOT / "data" / "processed" / "hte_results.json"
HTE_DATA = _load_hte(_hte_app_path if _hte_app_path.exists() else _hte_full_path)

# ── Plotting helpers ──────────────────────────────────────────────────────────

DARK_BG    = "#0f172a"
PANEL_BG   = "#1e293b"
PURPLE     = "#7c3aed"
PURPLE_L   = "#a78bfa"
BLUE       = "#60a5fa"
GREEN      = "#10b981"
RED        = "#f87171"
TEXT_WHITE = "#f1f5f9"
GRID_COLOR = "#334155"

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_WHITE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT_WHITE, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_WHITE, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_WHITE, fontsize=10)

def _new_fig(ncols=1, figsize=None, nrows=1):
    if figsize is None:
        figsize = (9 * ncols, 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    plt.subplots_adjust(wspace=0.35, hspace=0.4)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Power Analysis
# ══════════════════════════════════════════════════════════════════════════════

def power_analysis(baseline_cvr: float, mde_pct: float, alpha: float, power_target: float):
    plt.close("all")  # prevent memory leak
    mde = mde_pct / 100.0
    new_cvr = baseline_cvr + mde
    if new_cvr >= 1.0:
        return None, "MDE too large for baseline CVR — treatment rate would exceed 100%."

    n_req = required_sample_size(baseline_cvr, mde, alpha=alpha, power=power_target)
    actual_power = compute_power(n_req, baseline_cvr, mde, alpha=alpha)

    # ── Power curve ──
    n_max = max(int(n_req * 3), 500)
    ns = list(range(30, n_max, max(1, n_max // 300)))
    powers = [compute_power(n, baseline_cvr, mde, alpha=alpha) * 100 for n in ns]

    fig, ax = _new_fig(figsize=(9, 5))
    _style_ax(ax, "Power Curve — Sample Size vs Detected Power",
              "Sample size per group", "Statistical power (%)")
    ax.plot(ns, powers, color=PURPLE, linewidth=2.5, label="Power")
    ax.fill_between(ns, powers, alpha=0.15, color=PURPLE)
    ax.axhline(power_target * 100, color=PURPLE_L, linestyle="--", alpha=0.7,
               label=f"Target: {power_target*100:.0f}%")
    ax.axvline(n_req, color=GREEN, linestyle="--", linewidth=1.8,
               label=f"Required n = {n_req:,}")
    ax.scatter([n_req], [actual_power * 100], color=GREEN, s=90, zorder=6)
    ax.set_ylim(0, 108)
    ax.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)
    plt.tight_layout()

    total_n = n_req * 2
    relative_lift = mde / baseline_cvr * 100

    summary = f"""
### Required sample size: **{n_req:,} per group** ({total_n:,} total)

| Parameter | Value |
|:---|:---|
| Baseline conversion rate | {baseline_cvr*100:.2f}% |
| Treatment conversion rate | {new_cvr*100:.2f}% |
| Minimum Detectable Effect | +{mde_pct:.2f}pp ({relative_lift:.1f}% relative) |
| Significance level (α) | {alpha*100:.0f}% |
| Target power | {power_target*100:.0f}% |
| Actual power at n = {n_req:,} | {actual_power*100:.1f}% |

**How to read:** This curve shows how statistical power grows as you collect more data.
At n = **{n_req:,}** per group you have a **{actual_power*100:.0f}% chance** of detecting
the {mde_pct:.1f}pp lift — if it truly exists — and only a {alpha*100:.0f}% chance of a false positive.

> **Practical tip:** Running fewer than {n_req//2:,} per group means you're more likely to *miss*
> a real effect than to find it. Running more than {n_req*2:,} per group rarely helps.
"""
    return fig, summary


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: A/B Test Analyzer (Frequentist + Bayesian + CUPED demo)
# ══════════════════════════════════════════════════════════════════════════════

def ab_test_analyze(n_a: int, conv_a: int, n_b: int, conv_b: int,
                    alpha: float, corr_cuped: float):
    plt.close("all")  # prevent matplotlib memory leak

    # Safety checks — guard against zero/negative inputs
    n_a = max(int(n_a), 1)
    n_b = max(int(n_b), 1)
    conv_a = max(0, min(int(conv_a), n_a))
    conv_b = max(0, min(int(conv_b), n_b))
    p_a = conv_a / n_a
    p_b = conv_b / n_b

    # ── Frequentist ──
    freq = two_proportion_ztest(n_a, conv_a, n_b, conv_b, alpha=alpha)

    # ── Bayesian ──
    bayes = bayesian_proportion_test(n_a, conv_a, n_b, conv_b, n_samples=50_000)

    # ── Figure 1: CVR bar chart + posterior ──
    fig1, (ax1, ax2) = _new_fig(ncols=2, figsize=(12, 5))

    # Bar chart
    _style_ax(ax1, "Conversion Rate Comparison", "", "Conversion rate (%)")
    bar_colors = [BLUE, PURPLE_L]
    bars = ax1.bar(["Control (A)", "Treatment (B)"], [p_a * 100, p_b * 100],
                   color=bar_colors, alpha=0.85, width=0.45)
    for bar, n, c in zip(bars, [n_a, n_b], [conv_a, conv_b]):
        p = c / n
        se = (p * (1 - p) / n) ** 0.5
        ax1.errorbar(bar.get_x() + bar.get_width() / 2, p * 100,
                     yerr=1.96 * se * 100, fmt="none",
                     color=TEXT_WHITE, capsize=5, linewidth=1.8)
        ax1.text(bar.get_x() + bar.get_width() / 2, p * 100 + 0.05,
                 f"{p*100:.2f}%", ha="center", va="bottom",
                 color=TEXT_WHITE, fontsize=10, fontweight="bold")

    sig_color = GREEN if freq.significant else RED
    sig_label = ("SIGNIFICANT" if freq.significant else "NOT SIGNIFICANT") + f"  (p={freq.p_value:.4f})"
    ax1.set_title(sig_label, color=sig_color, fontsize=11, fontweight="bold")

    # Posterior distributions
    _style_ax(ax2, f"Posterior Distributions  |  P(B > A) = {bayes.prob_b_beats_a:.1%}",
              "Conversion rate", "Posterior density")
    margin = 0.06
    lo = max(0, min(p_a, p_b) - margin)
    hi = min(1, max(p_a, p_b) + margin)
    x = np.linspace(lo, hi, 400)
    a_a = 1 + conv_a; b_a = 1 + n_a - conv_a
    a_b = 1 + conv_b; b_b = 1 + n_b - conv_b
    pdf_a = sp_stats.beta.pdf(x, a_a, b_a)
    pdf_b = sp_stats.beta.pdf(x, a_b, b_b)
    ax2.fill_between(x, pdf_a, alpha=0.35, color=BLUE)
    ax2.fill_between(x, pdf_b, alpha=0.35, color=PURPLE_L)
    ax2.plot(x, pdf_a, color=BLUE, linewidth=2.2, label="Control (A)")
    ax2.plot(x, pdf_b, color=PURPLE_L, linewidth=2.2, label="Treatment (B)")
    ax2.axvline(p_a, color=BLUE, linestyle=":", alpha=0.7)
    ax2.axvline(p_b, color=PURPLE_L, linestyle=":", alpha=0.7)
    ax2.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)
    plt.tight_layout()

    # ── Figure 2: CUPED theoretical power chart (instant — no simulation) ──
    # Theoretical basis: CUPED reduces variance by (1 - ρ²), equivalent to
    # having n_effective = n / (1 - ρ²) samples. Power improves accordingly.
    corrs = np.linspace(0, 0.95, 40)
    mde = abs(p_b - p_a) if abs(p_b - p_a) > 0.001 else 0.01
    raw_power_val = compute_power(n_a, p_a, mde)

    # Theoretical CUPED power: effective n scales as 1/(1-rho²)
    cuped_powers_theory = [
        compute_power(max(int(n_a / max(1 - c**2, 0.01)), 10), p_a, mde) * 100
        for c in corrs
    ]
    raw_power_line = [raw_power_val * 100] * len(corrs)

    # Variance reduction percentage
    var_reduction_at_current = corr_cuped ** 2 * 100
    n_savings_at_current = int(n_a * corr_cuped**2)

    fig2, (ax3, ax4) = _new_fig(ncols=2, figsize=(12, 5))

    # Left: Power gain curve
    _style_ax(ax3, "CUPED Theoretical Power Gain",
              "Pre-post metric correlation (ρ)", "Statistical power (%)")
    ax3.plot(corrs, raw_power_line, color=BLUE, linewidth=2, linestyle="--",
             label="Without CUPED")
    ax3.plot(corrs, cuped_powers_theory, color=GREEN, linewidth=2.5,
             label="With CUPED (theoretical)")
    ax3.fill_between(corrs, raw_power_line, cuped_powers_theory,
                     alpha=0.18, color=GREEN, label="Power gain")
    ax3.axvline(corr_cuped, color=PURPLE_L, linestyle=":", linewidth=1.8,
                label=f"Current ρ = {corr_cuped:.2f}")
    ax3.set_ylim(0, 108)
    ax3.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)

    # Right: Sample size savings
    _style_ax(ax4, "Sample Size Savings from CUPED",
              "Pre-post metric correlation (ρ)", "Sample size reduction (%)")
    savings_pct = [c**2 * 100 for c in corrs]
    ax4.plot(corrs, savings_pct, color=GREEN, linewidth=2.5)
    ax4.fill_between(corrs, 0, savings_pct, alpha=0.18, color=GREEN)
    ax4.axvline(corr_cuped, color=PURPLE_L, linestyle=":", linewidth=1.8,
                label=f"ρ={corr_cuped:.2f} → save {var_reduction_at_current:.0f}%")
    ax4.axhline(var_reduction_at_current, color=PURPLE_L, linestyle=":", alpha=0.6)
    ax4.set_ylim(0, 105)
    ax4.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)
    plt.tight_layout()

    # ── Markdown summary ──
    sig_icon  = "✅" if freq.significant else "❌"
    bay_icon  = ("🟢 Deploy B" if bayes.prob_b_beats_a > 0.95
                 else ("🟡 Lean B" if bayes.prob_b_beats_a > 0.5 else "🔴 Keep A"))
    cuped_power_at_rho = compute_power(
        max(int(n_a / max(1 - corr_cuped**2, 0.01)), 10), p_a, mde
    ) * 100

    results_md = f"""
## Results Summary

| | **Frequentist** | **Bayesian (Beta-Binomial)** |
|:---|:---|:---|
| Test | Two-proportion Z-test | Conjugate posterior update |
| Statistic | z = {freq.statistic:+.3f} | P(B > A) = {bayes.prob_b_beats_a:.1%} |
| p-value / Expected loss | p = {freq.p_value:.5f} | Loss(deploy B) = {bayes.expected_loss_choosing_b:.5f} |
| 95% Interval for lift | [{freq.ci_lower:+.4f}, {freq.ci_upper:+.4f}] | [{bayes.ci_lower:+.4f}, {bayes.ci_upper:+.4f}] |
| Effect size | {freq.effect_name} = {freq.effect_size:.3f} | — |
| Decision (α = {alpha}) | {sig_icon} {"Reject H₀" if freq.significant else "Fail to reject H₀"} | {bay_icon} |

**Observed lift:** {conv_a:,}/{n_a:,} = {p_a:.2%} → {conv_b:,}/{n_b:,} = {p_b:.2%}
(**{freq.observed_diff:+.4f}pp absolute**, {freq.relative_lift:+.1f}% relative)

---

### CUPED Variance Reduction (at ρ = {corr_cuped:.2f}) — Theoretical
| Metric | Without CUPED | With CUPED |
|:---|:---|:---|
| Power | {raw_power_val*100:.1f}% | {cuped_power_at_rho:.1f}% |
| Effective sample size | {n_a:,} | {min(int(n_a / max(1-corr_cuped**2, 0.01)), n_a*10):,} equivalent |
| Sample size savings | — | ~{var_reduction_at_current:.0f}% ({n_savings_at_current:,} fewer users) |

> **Why CUPED?** CUPED uses a *pre-experiment* metric (e.g., last month's purchases) to
> remove user-level noise, reducing outcome variance by ρ². At ρ = {corr_cuped:.2f},
> you could achieve the same power with **~{var_reduction_at_current:.0f}% fewer users**.
> _(Theoretical: Deng et al. 2013, Microsoft KDD)_
"""
    return fig1, fig2, results_md


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Sequential Testing
# ══════════════════════════════════════════════════════════════════════════════

def sequential_testing():
    plt.close("all")  # prevent memory leak
    if SEQ_SIM is None:
        return None, "Pre-computed sequential simulation not found.\nRun: python scripts/run_analysis.py"

    peek = SEQ_SIM["peeking_simulation"]
    alpha     = peek["alpha"]
    trad_fpr  = peek["traditional_fpr"]
    msprt_fpr = peek["msprt_fpr"]
    n_exp     = peek["n_experiments"]

    speed_s = SEQ_SIM.get("detection_speed_small_effect", {})
    speed_m = SEQ_SIM.get("detection_speed_medium_effect", {})
    hist_s  = SEQ_SIM.get("stopping_times_small_hist", {})
    hist_m  = SEQ_SIM.get("stopping_times_medium_hist", {})

    fig, axes = _new_fig(ncols=2, figsize=(13, 5.5))
    ax1, ax2 = axes

    # ── Left: FPR comparison ──
    _style_ax(ax1, f"False Positive Rate Under H₀\n({n_exp:,} simulations, peeking at 25/50/75/100%)",
              "", "False positive rate (%)")
    methods = ["Traditional\n(with peeking)", "mSPRT\n(always valid)"]
    fprs    = [trad_fpr * 100, msprt_fpr * 100]
    colors  = [RED, GREEN]
    bars = ax1.bar(methods, fprs, color=colors, alpha=0.85, width=0.42)
    ax1.axhline(alpha * 100, color=TEXT_WHITE, linestyle="--", alpha=0.5,
                linewidth=1.5, label=f"Nominal α = {alpha*100:.0f}%")
    for bar, val in zip(bars, fprs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                 f"{val:.1f}%", ha="center", color=TEXT_WHITE,
                 fontweight="bold", fontsize=14)
    ax1.set_ylim(0, max(fprs) * 1.4)
    ax1.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)

    # ── Right: Stopping time histograms ──
    _style_ax(ax2, "mSPRT Stops Earlier for Larger Effects\n(Under H₁: True effect exists)",
              "Stopping time (n observations)", "Count of experiments")
    if hist_s and hist_m:
        es = hist_s["bin_edges"]
        cs = [(es[i] + es[i+1]) / 2 for i in range(len(es) - 1)]
        w = (es[1] - es[0]) * 0.45
        ax2.bar(cs, hist_s["counts"], width=w, color=BLUE, alpha=0.65,
                label=f"Small effect (d=0.2)  power={speed_s.get('power',0):.0%}")

        em = hist_m["bin_edges"]
        cm = [(em[i] + em[i+1]) / 2 for i in range(len(em) - 1)]
        wm = (em[1] - em[0]) * 0.45
        ax2.bar(cm, hist_m["counts"], width=wm, color=PURPLE_L, alpha=0.65,
                label=f"Medium effect (d=0.5)  power={speed_m.get('power',0):.0%}")

    ax2.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)
    plt.tight_layout(pad=2.5)

    summary = f"""
## Sequential Testing (mSPRT — Always Valid Inference)

**Reference:** Johari, Pekelis & Walsh (2015) — *"Always Valid Inference"* (arXiv:1512.04922)

---

### The Peeking Problem
Traditional A/B testing requires you to decide your sample size *before* the experiment.
If you check significance midway and stop early (a common practice!), you inflate the false
positive rate far above the nominal α.

| Method | False Positive Rate | vs Nominal α = 5% |
|:---|:---|:---|
| Traditional + 4 peeks | **{trad_fpr*100:.1f}%** | {trad_fpr/alpha:.1f}× inflation |
| mSPRT (any time) | **{msprt_fpr*100:.1f}%** | ✅ Controlled |

Simulation: {n_exp:,} experiments, true null, 4 peeks at 25/50/75/100% of n=1,000.

---

### How mSPRT Works
Instead of a p-value, mSPRT maintains a martingale **M_t** (e-value):

```
M_t = √(σ²/(σ²+t·ρ²)) · exp( (tX̄)² · ρ² / (2σ²(σ²+t·ρ²)) )
```

- Under H₀: E[M_t] = 1 for *all* t
- Under H₁: M_t → ∞  (detection is guaranteed)
- **Reject H₀ when M_t ≥ 1/α** — valid at any stopping time

---

### Detection Speed Comparison (right chart)

| Effect size | Power (mSPRT) | Median stop time | Fixed-n would need |
|:---|:---|:---|:---|
| Small (Cohen's d = 0.2) | {speed_s.get('power',0):.0%} | {speed_s.get('median_stopping_time','?')} obs | 500 obs |
| Medium (Cohen's d = 0.5) | {speed_m.get('power',0):.0%} | {speed_m.get('median_stopping_time','?')} obs | 500 obs |

mSPRT detects medium effects at **median {speed_m.get('median_stopping_time','?')} observations** — potentially
much earlier than the fixed-n design, without inflating false positives.
"""
    return fig, summary


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Uplift Modeling (HTE)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_MAP = {
    "CausalForest (DML) — Recommended": "causalforest",
    "X-Learner": "x_learner",
    "T-Learner": "t_learner",
}

def plot_hte(model_label: str, outcome_label: str):
    plt.close("all")  # prevent memory leak
    if HTE_DATA is None:
        return None, "Pre-computed HTE results not found.\nRun: python scripts/run_hte.py"

    outcome_key = "conversion" if "Conversion" in outcome_label else "spend"
    data = HTE_DATA.get(outcome_key, {})
    mk   = _MODEL_MAP.get(model_label, "causalforest")
    hist = data.get(f"{mk}_hist", {})
    segs = data.get("segment_summaries", {}).get(mk, [])
    overall = data.get("overall_ate", {}).get(mk, {})

    fig, (ax1, ax2) = _new_fig(ncols=2, figsize=(13, 5.5))

    # ── CATE distribution ──
    _style_ax(ax1, f"CATE Distribution — {model_label}  [{outcome_label}]",
              "Individual treatment effect (CATE)", "Count")
    if hist:
        edges  = hist["edges"]
        counts = hist["counts"]
        ctrs   = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
        widths = [(edges[i+1] - edges[i]) * 0.9 for i in range(len(edges) - 1)]
        bar_colors = [GREEN if c > 0 else RED for c in ctrs]
        ax1.bar(ctrs, counts, width=widths, color=bar_colors, alpha=0.75)
        ate = hist.get("mean", 0.0)
        ax1.axvline(0,   color=TEXT_WHITE, linestyle=":", alpha=0.4, linewidth=1.5)
        ax1.axvline(ate, color=GREEN,      linestyle="-",  linewidth=2.0,
                    label=f"ATE = {ate:.5f}")
        pct_pos = hist.get("pct_positive", 0)
        ax1.set_title(f"CATE Distribution  |  {pct_pos:.1f}% users benefit",
                      color=TEXT_WHITE, fontsize=11, fontweight="bold")
        ax1.legend(facecolor=PANEL_BG, edgecolor=PURPLE_L, labelcolor=TEXT_WHITE, fontsize=9)

    # ── Segment bar chart ──
    _style_ax(ax2, f"Average CATE by Segment — {model_label}", "Mean CATE", "")
    if segs:
        segs_sorted = sorted(segs, key=lambda x: x["cate_mean"], reverse=True)
        labels = [s["segment"] for s in segs_sorted]
        values = [s["cate_mean"] for s in segs_sorted]
        seg_colors = [GREEN if v > 0 else RED for v in values]
        bars = ax2.barh(labels, values, color=seg_colors, alpha=0.8)
        ax2.axvline(0, color=TEXT_WHITE, alpha=0.3, linewidth=1)
        # Add value labels
        for bar, val in zip(bars, values):
            x_pos = val + max(abs(v) for v in values) * 0.02
            ax2.text(x_pos if val >= 0 else val - max(abs(v) for v in values) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:+.5f}", va="center", color=TEXT_WHITE, fontsize=8)
    plt.tight_layout(pad=2.5)

    # ── Summary markdown ──
    top3 = sorted(segs, key=lambda x: x["cate_mean"], reverse=True)[:3]
    bot3 = sorted(segs, key=lambda x: x["cate_mean"])[:3]
    ate_val  = overall.get("ate_mean", 0)
    pct_pos  = overall.get("pct_positive", 0)
    naive    = data.get("naive_ate", 0)

    def _seg_rows(slist):
        return "\n".join(f"| {s['segment']} | {s['cate_mean']:+.5f} | {s['n']:,} |"
                         for s in slist)

    summary = f"""
## Uplift Modeling / HTE Results — {model_label} [{outcome_label}]

**Dataset:** Hillstrom Email Marketing — 64,000 customers, 3-arm RCT (2008)
**Treatment:** Any e-mail (Men's or Women's) vs. No e-mail (control)

---

### Average Treatment Effect

| Estimator | ATE | % Users Benefiting |
|:---|:---|:---|
| Naive (difference in means) | {naive:+.5f} | — |
| {model_label} | **{ate_val:+.5f}** | **{pct_pos:.1f}%** |

---

### Top 3 Segments to Target (highest CATE)
| Segment | Est. CATE | n |
|:---|:---|:---|
{_seg_rows(top3)}

### Lowest Responding Segments
| Segment | Est. CATE | n |
|:---|:---|:---|
{_seg_rows(bot3)}

---

### How to Use This

1. **Ranking:** Sort users by their estimated CATE — target those above a threshold.
2. **Budget constraint:** With a fixed email budget, send only to the top-N% by CATE.
3. **A/B validation:** Run a follow-up experiment *only* on the high-CATE segment to verify.

> **Key insight:** The average treatment effect ({naive:+.5f}) can mask huge heterogeneity.
> Some segments may show 2–3× the average response — targeting them delivers
> the same conversion uplift at a fraction of the marketing cost.

---
*Three estimators compared (select via dropdown):*
- **CausalForest DML** — doubly-robust SOTA; orthogonalizes Y and T residuals before fitting causal forest
- **X-Learner** — cross-fitted CATE; better for imbalanced treatment/control sizes
- **T-Learner** — separate outcome models per arm; simple baseline
"""
    return fig, summary


# ══════════════════════════════════════════════════════════════════════════════
# Gradio Layout
# ══════════════════════════════════════════════════════════════════════════════

_DESC = """
## A/B Testing & Causal Inference Simulator

A four-tab interactive dashboard demonstrating **state-of-the-art experimentation methods**
used at companies like Netflix, Spotify, Microsoft, and Airbnb.

| Tab | Method | What it demonstrates |
|:---|:---|:---|
| 1 Power Analysis | Z-test power formula | Sample size planning |
| 2 A/B Test Analyzer | Frequentist · Bayesian · CUPED | Multi-method comparison |
| 3 Sequential Testing | mSPRT (Always-Valid Inference) | Safe continuous monitoring |
| 4 Uplift Modeling | CausalForest · X-Learner · T-Learner | Heterogeneous treatment effects |

**Dataset (Tabs 3 & 4):** [Hillstrom E-mail Analytics Challenge](https://www.minethatdata.com/)
— 64,000 customers, 3-arm RCT, 2008.
"""

with gr.Blocks(title="A/B Testing & Causal Inference Simulator") as demo:
    gr.Markdown(_DESC)

    # ── TAB 1: Power Analysis ─────────────────────────────────────────────────
    with gr.Tab("1. Power Analysis"):
        gr.Markdown("""
### Sample Size & Power Calculator
Compute the required experiment size before running your A/B test.
A well-powered experiment is the foundation of valid inference.
""")
        with gr.Row():
            with gr.Column(scale=1):
                t1_baseline = gr.Slider(0.01, 0.50, value=0.10, step=0.01,
                                        label="Baseline conversion rate (Control CVR)")
                t1_mde = gr.Slider(0.10, 10.0, value=2.0, step=0.1,
                                   label="Minimum Detectable Effect (pp, absolute)")
                t1_alpha = gr.Dropdown([0.01, 0.05, 0.10], value=0.05,
                                       label="Significance level (α)")
                t1_power = gr.Slider(0.70, 0.95, value=0.80, step=0.05,
                                     label="Target power (1 - β)")
                t1_btn = gr.Button("Calculate", variant="primary")
            with gr.Column(scale=2):
                t1_plot = gr.Plot()
        t1_md = gr.Markdown()

        t1_btn.click(
            fn=power_analysis,
            inputs=[t1_baseline, t1_mde, t1_alpha, t1_power],
            outputs=[t1_plot, t1_md],
        )
        demo.load(
            fn=power_analysis,
            inputs=[t1_baseline, t1_mde, t1_alpha, t1_power],
            outputs=[t1_plot, t1_md],
        )

    # ── TAB 2: A/B Test Analyzer ──────────────────────────────────────────────
    with gr.Tab("2. A/B Test Analyzer"):
        gr.Markdown("""
### Multi-Method A/B Test Analysis
Enter observed results from any A/B test to compare **Frequentist**, **Bayesian**,
and **CUPED** methods side-by-side.
""")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Group A (Control)")
                t2_na   = gr.Number(value=21306, label="Users in Control (n_A)", precision=0)
                t2_ca   = gr.Number(value=121,   label="Conversions in Control", precision=0)
                gr.Markdown("#### Group B (Treatment)")
                t2_nb   = gr.Number(value=42694, label="Users in Treatment (n_B)", precision=0)
                t2_cb   = gr.Number(value=457,   label="Conversions in Treatment", precision=0)
                t2_alpha = gr.Dropdown([0.01, 0.05, 0.10], value=0.05,
                                       label="Significance level (α)")
                t2_corr = gr.Slider(0.0, 0.95, value=0.5, step=0.05,
                                    label="Pre-post metric correlation (for CUPED demo, ρ)")
                t2_btn  = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                t2_plot1 = gr.Plot(label="Conversion rates & Posteriors")
        t2_plot2 = gr.Plot(label="CUPED Power Gain vs Correlation")
        t2_md    = gr.Markdown()

        t2_btn.click(
            fn=ab_test_analyze,
            inputs=[t2_na, t2_ca, t2_nb, t2_cb, t2_alpha, t2_corr],
            outputs=[t2_plot1, t2_plot2, t2_md],
        )
        demo.load(
            fn=ab_test_analyze,
            inputs=[t2_na, t2_ca, t2_nb, t2_cb, t2_alpha, t2_corr],
            outputs=[t2_plot1, t2_plot2, t2_md],
        )

    # ── TAB 3: Sequential Testing ─────────────────────────────────────────────
    with gr.Tab("3. Sequential Testing"):
        gr.Markdown("""
### mSPRT — Always Valid Inference

Traditional A/B testing breaks if you peek at results mid-experiment.
The **mixture Sequential Probability Ratio Test** (mSPRT) lets you monitor continuously
without inflating the false positive rate. Pre-computed on 3,000 simulated experiments.
""")
        t3_plot = gr.Plot()
        t3_md   = gr.Markdown()
        demo.load(fn=sequential_testing, inputs=[], outputs=[t3_plot, t3_md])

    # ── TAB 4: Uplift Modeling ────────────────────────────────────────────────
    with gr.Tab("4. Uplift Modeling (HTE)"):
        gr.Markdown("""
### Heterogeneous Treatment Effect (HTE) Estimation

Not all users respond equally to a treatment. **Uplift modeling** estimates each user's
individual treatment effect (CATE) using three ML-based causal estimators from
[Microsoft EconML](https://github.com/py-why/EconML).

**Data:** Hillstrom E-mail Marketing — 64,000 customers, randomized 3-arm experiment.
**Treatment:** Any marketing email vs. no email.
""")
        with gr.Row():
            t4_model = gr.Dropdown(
                list(_MODEL_MAP.keys()),
                value="CausalForest (DML) — Recommended",
                label="Causal estimator",
            )
            t4_outcome = gr.Dropdown(
                ["Conversion (binary)", "Spend (continuous)"],
                value="Conversion (binary)",
                label="Outcome variable",
            )
            t4_btn = gr.Button("Show Results", variant="primary")
        t4_plot = gr.Plot()
        t4_md   = gr.Markdown()

        t4_btn.click(
            fn=plot_hte,
            inputs=[t4_model, t4_outcome],
            outputs=[t4_plot, t4_md],
        )
        demo.load(
            fn=plot_hte,
            inputs=[t4_model, t4_outcome],
            outputs=[t4_plot, t4_md],
        )

    gr.Markdown("""
---
Built by [Muhammad Fikri Wahidin](https://github.com/Fikri645) ·
[GitHub](https://github.com/Fikri645/ab-testing-causal) ·
Methods: CUPED (Microsoft 2013) · mSPRT (Johari et al. 2015) ·
CausalForestDML (Athey & Wager 2019, via EconML)
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="blue", neutral_hue="slate"),
    )

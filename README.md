# A/B Testing & Causal Inference Simulator

[![CI](https://github.com/Fikri645/ab-testing-causal/actions/workflows/ci.yml/badge.svg)](https://github.com/Fikri645/ab-testing-causal/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/fikri0o0/ab-testing-causal)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

> A production-quality portfolio project demonstrating five state-of-the-art experimentation
> and causal inference methods used at Netflix, Spotify, Microsoft, and Airbnb.

**[Live Demo →](https://huggingface.co/spaces/fikri0o0/ab-testing-causal)**

---

## What This Project Covers

| Method | Technique | When it's used |
|:---|:---|:---|
| **Power Analysis** | Z-test formula, sample size calculator | **Before** experiment — how many users do I need? |
| **CUPED** | Variance reduction via pre-experiment covariate | **Before** experiment — can I run it with fewer users? |
| **Frequentist** | Two-proportion Z-test, FDR correction | **After** experiment — did it work? (p-value) |
| **Bayesian** | Beta-Binomial conjugate model | **After** experiment — how confident am I? (P(B>A)) |
| **Sequential (mSPRT)** | Mixture SPRT — always-valid inference | **During** experiment — safe continuous monitoring |
| **HTE / Uplift** | CausalForestDML, X-Learner, T-Learner (EconML) | **After** experiment — who specifically responds? |

---

## Dataset

**Hillstrom E-mail Marketing Challenge (2008)**
- 64,000 customers from a US e-commerce retailer
- 3-arm RCT: No E-Mail (control), Men's E-Mail, Women's E-Mail
- Outcomes: visit, conversion, spend (2-week window)
- Features: recency, history, channel, zip code, new/returning customer

---

## Key Results

### Frequentist A/B Test (Email vs. No Email)

| Metric | Control | Treatment | Lift |
|:---|:---|:---|:---|
| Conversion rate | 0.57% | 1.07% | **+86.5%** |
| Avg spend/user | $0.65 | $1.25 | **+91.7%** |
| p-value (z-test) | — | — | **< 0.001** |

### Bayesian A/B Test

| Metric | Value |
|:---|:---|
| P(Treatment > Control) | **100.0%** |
| Expected loss (deploying treatment) | ~0 |
| 95% Credible Interval for lift | [+0.0038, +0.0063] |

### Sequential Testing (mSPRT)

| Method | False Positive Rate | Notes |
|:---|:---|:---|
| Traditional + 4 peeks | ~14–16% | Far above nominal 5% |
| mSPRT (always valid) | ~4–6% | Controlled at α regardless of stopping time |

### Heterogeneous Treatment Effects

**CausalForest (DML)** — Top responding segments (conversion):

| Segment | CATE | Interpretation |
|:---|:---|:---|
| Multichannel shoppers | +0.0077 | Highest lift — target first |
| New customers | +0.0064 | Email converts new users well |
| Women's buyers | +0.0056 | Relevant product fit |
| Rural customers | +0.0052 | Less channel saturation |

> **Business insight:** Targeting only the top-25% of users by CATE can deliver the same
> conversion uplift as blanket emailing at ~40% lower campaign cost.

---

## App — 4 Tabs

| Tab | What it does | When to use |
|:---|:---|:---|
| **1. Power Analysis** | Sample size calculator + power curve + CUPED planning | **Before** running the experiment |
| **2. A/B Test Analyzer** | Enter observed counts → Frequentist + Bayesian results | **After** the experiment completes |
| **3. Sequential Testing** | Peeking inflation demo vs. mSPRT (pre-computed, 3,000 simulations) | Reference / education |
| **4. Uplift Modeling** | Pre-computed Hillstrom HTE with 3 estimators + segment rankings | **After** the experiment |

### Design principle

Each tab maps to a distinct phase of the experiment lifecycle. CUPED lives in the Power
Analysis tab (Tab 1) because it answers the pre-experiment question *"how many users do I
need?"*, not the post-experiment question *"did it work?"*.

---

## Project Structure

```
ab-testing-causal/
├── src/
│   ├── frequentist.py      # Z-test, t-test, power analysis, FDR correction
│   ├── bayesian.py         # Beta-Binomial and Normal-Normal Bayesian updates
│   ├── cuped.py            # CUPED variance reduction (Microsoft KDD 2013)
│   ├── sequential.py       # mSPRT e-values + confidence sequences
│   └── hte.py              # HTE estimation (EconML: T/X-Learner, CausalForestDML)
├── scripts/
│   ├── run_analysis.py     # Full analysis pipeline (downloads Hillstrom + runs all methods)
│   └── run_hte.py          # HTE training (T-Learner, X-Learner, CausalForestDML)
├── app/
│   └── gradio_app.py       # 4-tab interactive Gradio dashboard
├── tests/
│   ├── test_frequentist.py # 21 tests
│   ├── test_bayesian.py    # 12 tests
│   ├── test_cuped.py       # 9 tests
│   └── test_sequential.py  # 11 tests
├── data/processed/         # Pre-computed results (JSON, committed to repo)
├── .flake8                 # Lint config
├── Makefile
└── .github/workflows/ci.yml
```

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/Fikri645/ab-testing-causal.git
cd ab-testing-causal

# 2. Install (Python 3.11 recommended — required for EconML)
pip install -r requirements.txt

# 3. Run analysis pipeline (downloads data + computes results)
python scripts/run_analysis.py   # ~2 min
python scripts/run_hte.py        # ~10 min (EconML training)

# 4. Launch app
python app/gradio_app.py

# 5. Run tests
pytest tests/ -v
```

---

## MLflow Experiment Tracking

All analysis runs are tracked with MLflow:

```bash
python -m mlflow ui
# Open http://localhost:5000
```

Tracked metrics: CVR by group, p-value, P(B>A), CUPED variance reduction, HTE ATEs per model.

---

## Key Lessons

1. **CUPED only helps with a strong pre-experiment covariate.** A proxy metric (recency +
   history) gave <1% variance reduction. A proper pre-period conversion rate would give 30–50%.
   This is a data infrastructure problem — you need to log user behaviour before experiments start.

2. **Peeking is worse than intuition suggests.** Just 4 mid-experiment checks push the false
   positive rate above 14% — nearly 3× the nominal 5%.

3. **mSPRT trades some power for safety.** Under H₁, mSPRT stops earlier for medium effects,
   but for small effects its power is lower than a well-powered fixed-n test.

4. **HTE estimators disagree substantially.** T-Learner ATE = 0.00278, X-Learner = 0.00160,
   CausalForest = 0.00486 — all on the same data. Model choice matters.

5. **Doubly-robust estimators tolerate model misspecification.** CausalForestDML is less
   sensitive to the choice of nuisance models than T-Learner or X-Learner.

6. **Targeting beats broadcasting.** The average email lift (+0.5pp) is real, but CausalForest
   reveals heterogeneity: multichannel users respond at 1.5× the average. Targeting them
   delivers the same uplift at a fraction of the cost.

---

## References

- Deng, Xu, Kohavi, Walker (2013). *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.* Microsoft Research, KDD 2013. — **CUPED**
- Johari, Pekelis, Walsh (2015). *Always Valid Inference: Bringing Sequential Analysis to A/B Testing.* arXiv:1512.04922. — **mSPRT**
- Kunzel, Sekhon, Bickel, Yu (2019). *Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning.* PNAS. — **T/X-Learner**
- Athey, Tibshirani, Wager (2019). *Generalized Random Forests.* Annals of Statistics. — **Causal Forest**
- Hillstrom (2008). *MineThatData E-Mail Analytics and Data Mining Challenge.* — **Dataset**

---

*Built by [Muhammad Fikri Wahidin](https://github.com/Fikri645) — ML Engineer / Data Scientist portfolio*

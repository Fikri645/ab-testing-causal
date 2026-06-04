"""
Heterogeneous Treatment Effect (HTE) estimation using EconML.

We estimate the Conditional Average Treatment Effect (CATE):
    τ(x) = E[Y(1) - Y(0) | X = x]

Three estimators are compared:
  1. T-Learner  – separate outcome models per arm; simple but high variance
  2. X-Learner  – cross-fitted residuals; better for imbalanced arms
  3. CausalForestDML – doubly-robust DML + causal forest; SOTA for HTE

Reference for CausalForestDML: Athey, Tibshirani, Wager (2019)
"Generalized Random Forests." Annals of Statistics.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


FEATURE_COLS = ["recency", "history", "mens", "womens", "newbie",
                "zip_urban", "zip_suburban", "zip_rural",
                "channel_web", "channel_phone", "channel_multichannel"]


def prepare_hillstrom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the Hillstrom dataset for HTE estimation.

    Binary treatment: 0 = No E-Mail, 1 = any e-mail (Men's or Women's)
    Outcome: conversion (binary), spend (continuous)
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Binary treatment
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)

    # One-hot encode zip_code and channel
    df["zip_urban"]        = (df["zip_code"] == "Urban").astype(int)
    df["zip_suburban"]     = (df["zip_code"] == "Surban").astype(int)
    df["zip_rural"]        = (df["zip_code"] == "Rural").astype(int)
    df["channel_web"]      = (df["channel"] == "Web").astype(int)
    df["channel_phone"]    = (df["channel"] == "Phone").astype(int)
    df["channel_multichannel"] = (df["channel"] == "Multichannel").astype(int)

    # Rename outcome columns for clarity
    df.rename(columns={"conversion": "conversion", "spend": "spend"}, inplace=True)

    # Log-transform history to reduce skew
    df["history"] = np.log1p(df["history"])

    return df


def run_hte_analysis(
    df: pd.DataFrame,
    outcome_col: str = "conversion",
    n_estimators: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run all three HTE estimators and return a summary dict.

    Returns CATE estimates per user plus segment-level aggregations.
    """
    try:
        from econml.metalearners import TLearner, XLearner
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(f"Install econml and scikit-learn: {e}")

    df_clean = prepare_hillstrom(df)

    available_feats = [c for c in FEATURE_COLS if c in df_clean.columns]
    X = df_clean[available_feats].values.astype(float)
    T = df_clean["treatment"].values.astype(float)
    Y = df_clean[outcome_col].values.astype(float)

    results: dict[str, np.ndarray] = {}

    # ── T-Learner ─────────────────────────────────────────────────────────────
    # Each learner gets its OWN model instance — sharing causes cross-contamination.
    def _make_base():
        if outcome_col == "conversion":
            return GradientBoostingClassifier(n_estimators=n_estimators,
                                              max_depth=4, random_state=seed)
        return GradientBoostingRegressor(n_estimators=n_estimators,
                                         max_depth=4, random_state=seed)

    t_learner = TLearner(models=_make_base())
    t_learner.fit(Y, T, X=X)
    results["T-Learner"] = t_learner.effect(X).flatten()

    # ── X-Learner ─────────────────────────────────────────────────────────────
    x_learner = XLearner(models=_make_base())
    x_learner.fit(Y, T, X=X)
    results["X-Learner"] = x_learner.effect(X).flatten()

    # ── CausalForestDML (SOTA) ────────────────────────────────────────────────
    # DML residualises both Y and T using regressors:
    # - model_y: fits E[Y|X]  (conditional outcome, treated as continuous)
    # - model_t: fits E[T|X]  (propensity score, treated as continuous ∈ [0,1])
    # Using regressors for both avoids DML's classifier-rejection check.
    model_y_cf = GradientBoostingRegressor(n_estimators=100, random_state=seed)
    model_t_cf = GradientBoostingRegressor(n_estimators=100, random_state=seed,
                                            loss='squared_error')

    cf = CausalForestDML(
        model_y=model_y_cf,
        model_t=model_t_cf,
        n_estimators=n_estimators,
        random_state=seed,
        verbose=0,
    )
    cf.fit(Y, T, X=X)
    cate_cf = cf.effect(X)
    cate_lb, cate_ub = cf.effect_interval(X)
    results["CausalForest"] = cate_cf.flatten()
    results["CausalForest_lb"] = cate_lb.flatten()
    results["CausalForest_ub"] = cate_ub.flatten()

    # ── Build output DataFrame ────────────────────────────────────────────────
    cate_df = df_clean[available_feats + ["treatment", outcome_col]].copy()
    cate_df["segment_email"] = df["segment"] if "segment" in df.columns else df_clean["treatment"].map({0: "No E-Mail", 1: "E-Mail"})
    for name, vals in results.items():
        cate_df[f"cate_{name.lower().replace('-', '_')}"] = vals

    # ── Segment-level CATE summaries ──────────────────────────────────────────
    segment_cols = {
        "zip_urban": "Urban",
        "zip_suburban": "Suburban",
        "zip_rural": "Rural",
        "channel_web": "Web Channel",
        "channel_phone": "Phone Channel",
        "channel_multichannel": "Multichannel",
        "newbie": "New Customer",
        "mens": "Men's Buyer",
        "womens": "Women's Buyer",
    }

    seg_summaries = {}
    for col, label in segment_cols.items():
        if col not in cate_df.columns:
            continue
        for val, grp_label in [(1, label), (0, f"Not {label}")]:
            mask = cate_df[col] == val
            if mask.sum() < 5:
                continue
            for model_key in ["t_learner", "x_learner", "causalforest"]:
                col_name = f"cate_{model_key}"
                if col_name not in cate_df.columns:
                    continue
                seg_summaries.setdefault(model_key, []).append({
                    "segment": grp_label,
                    "n": int(mask.sum()),
                    "cate_mean": round(float(cate_df.loc[mask, col_name].mean()), 5),
                    "cate_std": round(float(cate_df.loc[mask, col_name].std()), 5),
                    "outcome_mean": round(float(cate_df.loc[mask, outcome_col].mean()), 4),
                })

    # ── Overall ATE ───────────────────────────────────────────────────────────
    overall = {}
    for model_key in ["t_learner", "x_learner", "causalforest"]:
        col_name = f"cate_{model_key}"
        if col_name not in cate_df.columns:
            continue
        overall[model_key] = {
            "ate_mean": round(float(cate_df[col_name].mean()), 5),
            "ate_std": round(float(cate_df[col_name].std()), 5),
            "pct_positive": round(float((cate_df[col_name] > 0).mean() * 100), 2),
        }

    # ── Raw experiment ATE (naive difference in means) ───────────────────────
    ctrl_outcome = cate_df.loc[cate_df["treatment"] == 0, outcome_col].mean()
    trt_outcome  = cate_df.loc[cate_df["treatment"] == 1, outcome_col].mean()
    naive_ate    = trt_outcome - ctrl_outcome

    return {
        "outcome": outcome_col,
        "n_samples": len(cate_df),
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "naive_ate": round(float(naive_ate), 5),
        "overall_ate": overall,
        "segment_summaries": seg_summaries,
        # Top 10 and bottom 10 users by CausalForest CATE
        "top10_causalforest": cate_df.nlargest(10, "cate_causalforest")[
            available_feats + ["cate_causalforest"]
        ].round(4).to_dict("records"),
        "bottom10_causalforest": cate_df.nsmallest(10, "cate_causalforest")[
            available_feats + ["cate_causalforest"]
        ].round(4).to_dict("records"),
        # Distribution for histogram
        "causalforest_cates": cate_df["cate_causalforest"].round(5).tolist(),
        "x_learner_cates": cate_df["cate_x_learner"].round(5).tolist(),
        "t_learner_cates": cate_df["cate_t_learner"].round(5).tolist(),
    }


def save_hte_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_hte_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

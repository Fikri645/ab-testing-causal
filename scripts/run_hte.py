"""
HTE (Heterogeneous Treatment Effect) analysis on the Hillstrom dataset.

Trains three estimators:
  - T-Learner      (simple baseline)
  - X-Learner      (better for imbalanced arms)
  - CausalForestDML (SOTA doubly-robust estimator)

Saves CATE estimates and segment summaries to data/processed/hte_results.json

Usage:
    C:/Users/fikri/AppData/Local/Programs/Python/Python311/python.exe scripts/run_hte.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import mlflow

from src.config import (
    DATA_PROCESSED,
    HILLSTROM_RAW, HTE_RESULTS,
    SEED, HTE_N_ESTIMATORS,
)
from src.hte import run_hte_analysis, save_hte_results

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def main():
    if not HILLSTROM_RAW.exists():
        print("[hte] Hillstrom dataset not found. Run scripts/run_analysis.py first.")
        sys.exit(1)

    df = pd.read_csv(HILLSTROM_RAW)
    df.columns = df.columns.str.lower().str.strip()
    print(f"[hte] Loaded {len(df):,} rows")
    print(f"[hte] Segment distribution:\n{df['segment'].value_counts()}\n")

    mlflow.set_experiment("ab_testing_hte")

    # ── Conversion outcome ────────────────────────────────────────────────────
    print("[hte] Estimating HTE on conversion … (this takes 2–5 min)")
    with mlflow.start_run(run_name="hte_conversion"):
        results_conv = run_hte_analysis(
            df, outcome_col="conversion",
            n_estimators=HTE_N_ESTIMATORS, seed=SEED,
        )
        mlflow.log_metric("naive_ate_conversion", results_conv["naive_ate"])
        mlflow.log_metric("n_samples", results_conv["n_samples"])
        for model, stats in results_conv["overall_ate"].items():
            mlflow.log_metric(f"ate_{model}", stats["ate_mean"])
            mlflow.log_metric(f"pct_positive_{model}", stats["pct_positive"])

        print(f"  Naive ATE (conversion): {results_conv['naive_ate']:.4f}")
        for model, stats in results_conv["overall_ate"].items():
            print(f"  {model:20s}: ATE = {stats['ate_mean']:.5f}, "
                  f"Positive responders: {stats['pct_positive']:.1f}%")

    # ── Spend outcome ─────────────────────────────────────────────────────────
    print("[hte] Estimating HTE on spend … (this takes 2–5 min)")
    with mlflow.start_run(run_name="hte_spend"):
        results_spend = run_hte_analysis(
            df, outcome_col="spend",
            n_estimators=HTE_N_ESTIMATORS, seed=SEED,
        )
        mlflow.log_metric("naive_ate_spend", results_spend["naive_ate"])
        for model, stats in results_spend["overall_ate"].items():
            mlflow.log_metric(f"ate_spend_{model}", stats["ate_mean"])

        print(f"  Naive ATE (spend):      {results_spend['naive_ate']:.4f}")

    # ── Save combined results ─────────────────────────────────────────────────
    combined = {
        "conversion": results_conv,
        "spend": results_spend,
    }
    save_hte_results(combined, HTE_RESULTS)
    print(f"\n✅ HTE results saved → {HTE_RESULTS}")

    # Pretty-print top segments (by CausalForest CATE, conversion)
    segs = results_conv.get("segment_summaries", {}).get("causalforest", [])
    if segs:
        sorted_segs = sorted(segs, key=lambda x: x["cate_mean"], reverse=True)
        print("\n── Top responding segments (CausalForest, conversion) ──")
        for s in sorted_segs[:5]:
            print(f"  {s['segment']:30s}  CATE = {s['cate_mean']:+.4f}  (n={s['n']:,})")
        print("\n── Least responding segments ──")
        for s in sorted_segs[-5:]:
            print(f"  {s['segment']:30s}  CATE = {s['cate_mean']:+.4f}  (n={s['n']:,})")


if __name__ == "__main__":
    main()

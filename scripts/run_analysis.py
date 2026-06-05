"""
Main analysis pipeline for the Hillstrom e-mail campaign dataset.

Runs:
  1. Data download & preprocessing
  2. Frequentist A/B test (z-test + chi-square)
  3. Bayesian A/B test (Beta-Binomial)
  4. CUPED variance reduction demo
  5. Sequential testing simulation (peeking + mSPRT)
  6. Saves results to data/processed/analysis_results.json

Usage:
    python scripts/run_analysis.py
"""
import sys, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import mlflow

from src.config import (
    DATA_RAW, DATA_PROCESSED,
    HILLSTROM_URL, HILLSTROM_RAW,
    ANALYSIS_RESULTS, SEQUENTIAL_SIM,
    ALPHA, SEED,
)
from src.frequentist import (
    two_proportion_ztest, chi_square_test,
    two_sample_ttest, required_sample_size, power_curve,
)
from src.bayesian import bayesian_proportion_test
from src.cuped import cuped_ttest, simulate_cuped_benefit
from src.sequential import simulate_peeking, simulate_detection_speed

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ── 1. Download Hillstrom dataset ─────────────────────────────────────────────

def download_hillstrom() -> pd.DataFrame:
    if HILLSTROM_RAW.exists():
        print(f"[data] Using cached {HILLSTROM_RAW}")
    else:
        print("[data] Downloading Hillstrom dataset …")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(HILLSTROM_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
            HILLSTROM_RAW.write_bytes(content)
            print(f"[data] Saved to {HILLSTROM_RAW} ({len(content):,} bytes)")
        except Exception as e:
            print(f"[data] Primary URL failed: {e}")
            # Try fallback
            FALLBACK = "https://raw.githubusercontent.com/humboldt-ai/causal/main/data/hillstrom.csv"
            try:
                req2 = urllib.request.Request(FALLBACK, headers=headers)
                with urllib.request.urlopen(req2, timeout=30) as resp:
                    content = resp.read()
                HILLSTROM_RAW.write_bytes(content)
                print(f"[data] Saved via fallback ({len(content):,} bytes)")
            except Exception as e2:
                raise RuntimeError(f"Could not download Hillstrom dataset: {e2}")

    df = pd.read_csv(HILLSTROM_RAW)
    df.columns = df.columns.str.lower().str.strip()
    return df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (full_df, binary_df) where binary_df collapses the three segments
    into: control (No E-Mail) vs. treatment (any e-mail).
    """
    df_binary = df.copy()
    df_binary["treatment"] = (df_binary["segment"] != "No E-Mail").astype(int)

    # Synthetic pre-experiment metric: use recency + history as a proxy
    # (In a real experiment these would be pre-period outcome metrics)
    df_binary["pre_metric"] = (
        0.4 * (1 / (df_binary["recency"] + 1))
        + 0.6 * np.log1p(df_binary["history"]) / np.log1p(df_binary["history"].max())
    )

    return df, df_binary


# ── 3. Frequentist A/B test ───────────────────────────────────────────────────

def run_frequentist(df: pd.DataFrame) -> dict:
    ctrl = df[df["treatment"] == 0]
    trt  = df[df["treatment"] == 1]

    n_ctrl, n_trt = len(ctrl), len(trt)
    conv_ctrl = int(ctrl["conversion"].sum())
    conv_trt  = int(trt["conversion"].sum())

    spend_ctrl_mean = float(ctrl["spend"].mean())
    spend_ctrl_std  = float(ctrl["spend"].std())
    spend_trt_mean  = float(trt["spend"].mean())
    spend_trt_std   = float(trt["spend"].std())

    z_result  = two_proportion_ztest(n_ctrl, conv_ctrl, n_trt, conv_trt, alpha=ALPHA)
    chi_result = chi_square_test(n_ctrl, conv_ctrl, n_trt, conv_trt, alpha=ALPHA)
    t_result  = two_sample_ttest(
        spend_ctrl_mean, spend_ctrl_std, n_ctrl,
        spend_trt_mean,  spend_trt_std,  n_trt,
        alpha=ALPHA,
    )

    # Power analysis
    baseline_cvr = conv_ctrl / n_ctrl
    ns, powers = power_curve(baseline_cvr, mde=0.01, alpha=ALPHA)
    n_req = required_sample_size(baseline_cvr, mde=0.01, alpha=ALPHA, power=0.80)

    return {
        "n_control": n_ctrl, "n_treatment": n_trt,
        "conv_control": conv_ctrl, "conv_treatment": conv_trt,
        "cvr_control": round(conv_ctrl / n_ctrl, 5),
        "cvr_treatment": round(conv_trt / n_trt, 5),
        "spend_mean_control": round(spend_ctrl_mean, 4),
        "spend_mean_treatment": round(spend_trt_mean, 4),
        "ztest": z_result.to_dict(),
        "chitest": chi_result.to_dict(),
        "ttest_spend": t_result.to_dict(),
        "power_curve": {"sample_sizes": ns, "powers": powers},
        "required_n_per_group": n_req,
    }


# ── 4. Bayesian A/B test ──────────────────────────────────────────────────────

def run_bayesian(freq_results: dict) -> dict:
    n_c  = freq_results["n_control"]
    n_t  = freq_results["n_treatment"]
    c_c  = freq_results["conv_control"]
    c_t  = freq_results["conv_treatment"]

    prop_result = bayesian_proportion_test(n_c, c_c, n_t, c_t, n_samples=100_000, seed=SEED)

    return {"conversion": prop_result.to_dict()}


# ── 5. CUPED ──────────────────────────────────────────────────────────────────

def run_cuped(df: pd.DataFrame) -> dict:
    cuped_result = cuped_ttest(df, metric_col="conversion",
                               pre_metric_col="pre_metric",
                               treatment_col="treatment")

    # Sweep across correlation strengths for visualization
    sim_results = []
    for corr in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        sim = simulate_cuped_benefit(
            n_per_group=1000,
            baseline_rate=0.10,
            true_effect=0.02,
            corr=corr,
            n_sims=1000,
            seed=SEED,
        )
        sim_results.append(sim)

    return {
        "hillstrom_cuped": cuped_result.to_dict(),
        "power_vs_correlation": sim_results,
    }


# ── 6. Sequential testing simulation ─────────────────────────────────────────

def run_sequential() -> dict:
    print("[sequential] Simulating peeking inflation …")
    peeking_sim = simulate_peeking(
        n_total=1000, n_experiments=3000,
        peek_fractions=[0.25, 0.50, 0.75, 1.00],
        alpha=ALPHA, sigma=1.0, seed=SEED,
    )

    print("[sequential] Simulating detection speed under H1 …")
    speed_small = simulate_detection_speed(
        true_effect=0.2, n_max=500, n_experiments=1000,
        sigma=1.0, alpha=ALPHA, seed=SEED,
    )
    speed_medium = simulate_detection_speed(
        true_effect=0.5, n_max=500, n_experiments=1000,
        sigma=1.0, alpha=ALPHA, seed=SEED,
    )

    return {
        "peeking_simulation": {
            "n_experiments": peeking_sim.n_experiments,
            "peek_schedule": peeking_sim.peek_schedule,
            "traditional_fpr": peeking_sim.traditional_fpr,
            "msprt_fpr": peeking_sim.msprt_fpr,
            "alpha": peeking_sim.alpha,
        },
        "detection_speed_small_effect": {
            k: v for k, v in speed_small.items() if k != "stopping_times"
        },
        "detection_speed_medium_effect": {
            k: v for k, v in speed_medium.items() if k != "stopping_times"
        },
        # Store histogram bins only (full list is large)
        "stopping_times_small_hist": _to_histogram(speed_small["stopping_times"], bins=30),
        "stopping_times_medium_hist": _to_histogram(speed_medium["stopping_times"], bins=30),
    }


def _to_histogram(values: list, bins: int = 30) -> dict:
    import numpy as np
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mlflow.set_experiment("ab_testing_analysis")

    with mlflow.start_run(run_name="hillstrom_full_analysis"):
        # --- Data ---
        df = download_hillstrom()
        print(f"[data] Loaded {len(df):,} rows, columns: {df.columns.tolist()}")
        df, df_binary = preprocess(df)

        # --- Frequentist ---
        print("[freq] Running frequentist tests …")
        freq = run_frequentist(df_binary)
        mlflow.log_metric("cvr_control",   freq["cvr_control"])
        mlflow.log_metric("cvr_treatment", freq["cvr_treatment"])
        mlflow.log_metric("ztest_pvalue",  freq["ztest"]["p_value"])
        mlflow.log_metric("ztest_significant", int(freq["ztest"]["significant"]))

        # --- Bayesian ---
        print("[bayes] Running Bayesian tests …")
        bayes = run_bayesian(freq)
        mlflow.log_metric("prob_treatment_better",
                           bayes["conversion"]["prob_b_beats_a"])

        # --- CUPED ---
        print("[cuped] Running CUPED …")
        cuped = run_cuped(df_binary)
        mlflow.log_metric("cuped_variance_reduction_pct",
                           cuped["hillstrom_cuped"]["variance_reduction_pct"])
        mlflow.log_metric("cuped_pvalue",
                           cuped["hillstrom_cuped"]["cuped_p_value"])

        # --- Combine and save ---
        results = {
            "frequentist": freq,
            "bayesian": bayes,
            "cuped": cuped,
        }
        with open(ANALYSIS_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(ANALYSIS_RESULTS))
        print(f"[done] Analysis results -> {ANALYSIS_RESULTS}")

    # Sequential testing (separate run, no Hillstrom dependency)
    with mlflow.start_run(run_name="sequential_testing_simulation"):
        print("[seq] Running sequential testing simulation …")
        seq = run_sequential()
        with open(SEQUENTIAL_SIM, "w") as f:
            json.dump(seq, f, indent=2)
        mlflow.log_metric("traditional_fpr",
                           seq["peeking_simulation"]["traditional_fpr"])
        mlflow.log_metric("msprt_fpr",
                           seq["peeking_simulation"]["msprt_fpr"])
        mlflow.log_artifact(str(SEQUENTIAL_SIM))
        print(f"[done] Sequential sim results -> {SEQUENTIAL_SIM}")

    print("\n✅ All analyses complete.")
    print(f"   Frequentist: z={freq['ztest']['statistic']}, p={freq['ztest']['p_value']}")
    print(f"   Bayesian: P(treatment better) = {bayes['conversion']['prob_b_beats_a']:.4f}")
    print(f"   CUPED variance reduction: {cuped['hillstrom_cuped']['variance_reduction_pct']:.1f}%")
    print(f"   Peeking FPR: traditional={seq['peeking_simulation']['traditional_fpr']:.3f}, "
          f"mSPRT={seq['peeking_simulation']['msprt_fpr']:.3f}")


if __name__ == "__main__":
    main()

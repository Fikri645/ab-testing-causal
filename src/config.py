"""
Project-wide configuration for A/B Testing & Causal Inference.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
NOTEBOOKS = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"

# ── Hillstrom dataset ────────────────────────────────────────────────────────
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_FALLBACK_URL = (
    "https://raw.githubusercontent.com/EmanueleCannizzaro/sklift/"
    "master/sklift/datasets/hillstrom.csv"
)
HILLSTROM_RAW = DATA_RAW / "hillstrom.csv"
HILLSTROM_PROCESSED = DATA_PROCESSED / "hillstrom_processed.csv"

# ── Processed outputs ────────────────────────────────────────────────────────
ANALYSIS_RESULTS = DATA_PROCESSED / "analysis_results.json"
HTE_RESULTS      = DATA_PROCESSED / "hte_results.json"
SEQUENTIAL_SIM   = DATA_PROCESSED / "sequential_sim.json"

# ── Frequentist defaults ─────────────────────────────────────────────────────
ALPHA  = 0.05   # significance level
POWER  = 0.80   # target statistical power
MDE    = 0.02   # minimum detectable effect (absolute, for conversions)

# ── Bayesian defaults ────────────────────────────────────────────────────────
PRIOR_ALPHA = 1.0   # Beta(1,1) = Uniform prior
PRIOR_BETA  = 1.0
N_SAMPLES   = 100_000  # Monte-Carlo draws for posterior

# ── Sequential testing ───────────────────────────────────────────────────────
MSPRT_RHO_SCALE = 1.0   # ρ = σ * RHO_SCALE  (mixing prior std)

# ── HTE / Uplift ─────────────────────────────────────────────────────────────
HTE_SEED = 42
HTE_N_ESTIMATORS = 200
TREATMENT_COL = "treatment"
OUTCOME_CONVERSION = "conversion"
OUTCOME_SPEND = "spend"
FEATURE_COLS = ["recency", "history", "mens", "womens", "newbie",
                "zip_code", "channel"]

# ── Random seeds ─────────────────────────────────────────────────────────────
SEED = 42

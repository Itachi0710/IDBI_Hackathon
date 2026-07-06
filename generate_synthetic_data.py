"""
Synthetic MSME alternate-data generator.

WHY THIS DESIGN:
Real credit data is never a clean readout of "is this business healthy".
It's a NOISY PROXY for it -- two businesses with the same true health can
report different GST turnover, and two businesses with different true
health can look similar on paper. If we generate features that are a
*perfect* function of health, our later ML model will look artificially
good and won't teach you (or the judges) anything about how real credit
scoring actually behaves.

So the trick here: every MSME gets a hidden `true_health` score (0-100)
that we NEVER show the scoring model. Every observable feature (GST,
UPI/AA, EPFO) is generated as: `rho * health_signal + noise`, where `rho`
(rho, the correlation strength) controls how tightly that feature tracks
the truth. High-rho features are strong signals; low-rho features are
weak/noisy signals -- exactly what you'd expect from real alternate data.
"""

import numpy as np
import pandas as pd

N_MSMES = 350
SEED = 42
rng = np.random.default_rng(SEED)

# Each archetype represents a realistic MSME segment. `mu`/`sigma` define
# a normal distribution over the hidden true_health score for that segment.
ARCHETYPES = {
    "stable_growth": dict(p=0.25, mu=75, sigma=8),
    "seasonal": dict(p=0.20, mu=60, sigma=12),
    "declining": dict(p=0.15, mu=35, sigma=10),
    "credit_invisible_viable": dict(
        p=0.25, mu=68, sigma=10
    ),  # the segment we most want to catch
    "high_risk_ntc": dict(p=0.15, mu=30, sigma=15),
}


def correlated_feature(
    health_0_100: np.ndarray, rho: float, lo: float, hi: float, invert: bool = False
) -> np.ndarray:
    """
    Generate one feature column correlated with the latent health score.

    rho=1.0 -> feature almost perfectly reveals health (rare in real data)
    rho=0.0 -> feature is pure noise, uninformative
    invert=True -> higher health means LOWER feature value (e.g. filing delay,
                   bounce rate -- healthier businesses have less of these)

    This is a VECTORIZED operation: it computes all 350 values in one shot
    using numpy array math, instead of looping row-by-row in Python. For
    350 rows the difference is invisible, but the same loop over 1M rows
    would take ~50-100x longer in pure Python than in numpy. Always reach
    for array ops over `for` loops when generating or transforming tabular
    data -- it's one of the highest-leverage habits in data engineering.
    """
    z = (health_0_100 - 50) / 25  # standardize health to ~[-2, 2]
    if invert:
        z = -z
    noise = rng.normal(0, 1, size=len(health_0_100))
    signal = rho * z + np.sqrt(max(1 - rho**2, 0)) * noise
    pct = 1 / (1 + np.exp(-signal))  # sigmoid squashes signal into (0, 1)
    return lo + pct * (hi - lo)


def generate_dataset(n: int = N_MSMES) -> pd.DataFrame:
    names = list(ARCHETYPES.keys())
    probs = [ARCHETYPES[k]["p"] for k in names]
    archetype = rng.choice(names, size=n, p=probs)

    mu = np.array([ARCHETYPES[a]["mu"] for a in archetype])
    sigma = np.array([ARCHETYPES[a]["sigma"] for a in archetype])
    true_health = np.clip(rng.normal(mu, sigma), 0, 100)

    df = pd.DataFrame(
        {
            "msme_id": [f"MSME{i:05d}" for i in range(n)],
            "archetype": archetype,  # ground truth - NOT a model input
            "true_health": true_health.round(1),  # ground truth - NOT a model input
        }
    )

    # --- GST signals ---
    df["gst_turnover_lakhs_avg"] = correlated_feature(
        true_health, rho=0.60, lo=2, hi=500
    ).round(1)
    df["gst_turnover_growth_yoy_pct"] = correlated_feature(
        true_health, rho=0.50, lo=-30, hi=60
    ).round(1)
    df["gst_filing_delay_days_avg"] = correlated_feature(
        true_health, rho=0.50, lo=0, hi=45, invert=True
    ).round(1)
    df["gst_filing_consistency_pct"] = correlated_feature(
        true_health, rho=0.60, lo=40, hi=100
    ).round(1)

    # --- UPI / Account Aggregator (banking) signals ---
    df["upi_aa_avg_monthly_inflow_lakhs"] = correlated_feature(
        true_health, rho=0.55, lo=1, hi=300
    ).round(1)
    outflow_ratio = correlated_feature(
        true_health, rho=0.50, lo=0.5, hi=1.15, invert=True
    )
    df["upi_aa_avg_monthly_outflow_lakhs"] = (
        df["upi_aa_avg_monthly_inflow_lakhs"] * outflow_ratio
    ).round(1)
    df["cash_flow_volatility_cv"] = correlated_feature(
        true_health, rho=0.40, lo=0.05, hi=0.90, invert=True
    ).round(2)
    df["bounce_rate_pct"] = correlated_feature(
        true_health, rho=0.60, lo=0, hi=25, invert=True
    ).round(1)
    df["avg_bank_balance_runway_days"] = correlated_feature(
        true_health, rho=0.50, lo=3, hi=120
    ).round(0)

    # --- EPFO signals ---
    df["epfo_employee_count"] = (
        correlated_feature(true_health, rho=0.40, lo=1, hi=200).round(0).astype(int)
    )
    df["epfo_employee_growth_yoy_pct"] = correlated_feature(
        true_health, rho=0.45, lo=-40, hi=50
    ).round(1)
    df["epfo_pf_compliance_score"] = correlated_feature(
        true_health, rho=0.55, lo=20, hi=100
    ).round(1)
    df["epfo_avg_wage_growth_yoy_pct"] = correlated_feature(
        true_health, rho=0.35, lo=-10, hi=25
    ).round(1)

    # --- Label: did this MSME default within 12 months? ---
    # Logistic function of true_health -> probability, then a Bernoulli draw.
    # This keeps the label PROBABILISTIC rather than a hard threshold, so
    # even healthy businesses occasionally default (real-world noise) and
    # some risky ones survive -- avoiding an unrealistically "easy" dataset.
    default_prob = 1 / (1 + np.exp((true_health - 45) / 10))
    df["defaulted_12m"] = rng.binomial(1, default_prob)

    return df


if __name__ == "__main__":
    data = generate_dataset()

    print(f"Generated {len(data)} synthetic MSME profiles")
    print(f"Default rate: {data['defaulted_12m'].mean():.1%}")
    print(
        "\nDefault rate by archetype (sanity check -- should roughly match mu ordering):"
    )
    print(
        data.groupby("archetype")["defaulted_12m"]
        .mean()
        .sort_values(ascending=False)
        .round(3)
    )

    out_path = "/Users/a.shrivastava/Desktop/HAckatoh/IDBI_Hackathon/msme_health_card/msme_synthetic_data.csv"
    data.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(
        "\nNOTE: 'archetype' and 'true_health' are ground-truth columns for our "
        "own validation only. Drop them before training/scoring -- a real bank "
        "would never have access to a business's 'true health', only the "
        "observable GST/UPI/EPFO signals."
    )

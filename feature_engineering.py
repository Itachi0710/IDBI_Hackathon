"""
Feature engineering: turn raw GST / UPI-AA / EPFO columns into 5 normalized
(0-100) dimension scores.

DESIGN NOTE: we normalize against FIXED, domain-informed bounds (e.g. "0-45
days filing delay"), not against min/max of our current sample. If we
normalized against the sample's own min/max, the meaning of "80/100" would
silently shift every time the underlying population changed (e.g. if next
quarter's applicants are all worse, the same business would suddenly score
higher). Fixed bounds make the score STABLE and COMPARABLE across time --
a property real credit scores must have. This is a common beginner mistake
in scoring systems worth remembering.
"""

import os

import numpy as np
import pandas as pd


def normalize(series: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Min-max normalize a column to 0-100 against FIXED bounds, clipping outliers."""
    clipped = series.clip(lo, hi)
    pct = (clipped - lo) / (hi - lo)
    if invert:
        pct = 1 - pct
    return (pct * 100).round(1)


# (column, lo, hi, invert) -- invert=True means LOWER raw value is BETTER
DIMENSION_CONFIG = {
    "revenue_stability_score": {
        "weights": {"gst_turnover_lakhs_avg": 0.4, "gst_turnover_growth_yoy_pct": 0.6},
        "bounds": {
            "gst_turnover_lakhs_avg": (2, 500, False),
            "gst_turnover_growth_yoy_pct": (-30, 60, False),
        },
    },
    "cash_flow_health_score": {
        "weights": {"net_margin_pct": 0.45, "cash_flow_volatility_cv": 0.30, "bounce_rate_pct": 0.25},
        "bounds": {
            "net_margin_pct": (-20, 40, False),
            "cash_flow_volatility_cv": (0.05, 0.90, True),
            "bounce_rate_pct": (0, 25, True),
        },
    },
    "compliance_score": {
        "weights": {"gst_filing_delay_days_avg": 0.35, "gst_filing_consistency_pct": 0.35,
                     "epfo_pf_compliance_score": 0.30},
        "bounds": {
            "gst_filing_delay_days_avg": (0, 45, True),
            "gst_filing_consistency_pct": (40, 100, False),
            "epfo_pf_compliance_score": (20, 100, False),
        },
    },
    "growth_momentum_score": {
        "weights": {"epfo_employee_growth_yoy_pct": 0.55, "epfo_avg_wage_growth_yoy_pct": 0.45},
        "bounds": {
            "epfo_employee_growth_yoy_pct": (-40, 50, False),
            "epfo_avg_wage_growth_yoy_pct": (-10, 25, False),
        },
    },
    "repayment_capacity_score": {
        "weights": {"avg_bank_balance_runway_days": 0.5, "net_margin_pct": 0.5},
        "bounds": {
            "avg_bank_balance_runway_days": (3, 120, False),
            "net_margin_pct": (-20, 40, False),
        },
    },
}

# Weights for blending the 5 dimensions into one composite rule-based score.
COMPOSITE_WEIGHTS = {
    "revenue_stability_score": 0.20,
    "cash_flow_health_score": 0.25,
    "compliance_score": 0.20,
    "growth_momentum_score": 0.15,
    "repayment_capacity_score": 0.20,
}


def add_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # CREDIT SCORING SAFETY: guard the denominator before dividing.
    # If upi_aa_avg_monthly_inflow_lakhs is 0 (possible for a real applicant with
    # no recorded inflow), the division produces inf/NaN, which silently
    # propagates through every downstream dimension score and into the composite
    # score.  NaN comparisons (e.g. ``NaN >= 70``) are ALWAYS False in
    # Python/pandas, so a broken record would be silently bucketed as "High risk"
    # with no error or warning -- indistinguishable from a genuinely bad
    # business.  In a credit-scoring context this is dangerous: it can deny
    # credit to an applicant whose data was simply missing/zeroed, with no audit
    # trail.  We replace 0 with a tiny epsilon so the result is an
    # extreme-but-finite number that survives clipping in normalize() rather
    # than contaminating the pipeline with NaN.
    _inflow = df["upi_aa_avg_monthly_inflow_lakhs"].replace(0, 1e-9)
    df["net_margin_pct"] = (
        (df["upi_aa_avg_monthly_inflow_lakhs"] - df["upi_aa_avg_monthly_outflow_lakhs"])
        / _inflow * 100
    ).round(1)

    for dim_name, cfg in DIMENSION_CONFIG.items():
        normalized_cols = []
        for col, (lo, hi, invert) in cfg["bounds"].items():
            norm_col = f"_norm_{col}"
            df[norm_col] = normalize(df[col], lo, hi, invert)
            normalized_cols.append((norm_col, cfg["weights"][col]))

        df[dim_name] = sum(df[c] * w for c, w in normalized_cols).round(1)

    # Drop the temporary _norm_ helper columns, keep the final dimension scores
    df = df.drop(columns=[c for c in df.columns if c.startswith("_norm_")])

    df["composite_rule_score"] = sum(
        df[dim] * w for dim, w in COMPOSITE_WEIGHTS.items()
    ).round(1)

    return df


if __name__ == "__main__":
    _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msme_synthetic_data.csv")
    raw = pd.read_csv(_csv)
    enriched = add_dimension_scores(raw)
    dims = list(DIMENSION_CONFIG.keys()) + ["composite_rule_score"]
    print(enriched[["msme_id", "archetype", "true_health"] + dims].head(8).to_string(index=False))
    enriched.to_csv(_csv, index=False)
    print("\nSaved enriched dataset with dimension scores.")
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
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split


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
    # If upi_aa_avg_monthly_inflow_lakhs is 0 (possible for a real applicant),
    # the division produces inf/NaN which silently propagates through all
    # dimension scores. NaN comparisons (e.g. NaN >= 70) are ALWAYS False, so a
    # broken record would be silently bucketed as "High risk" with no audit
    # trail -- dangerous in a credit-scoring context. Floor to epsilon instead.
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


# ---------------------------------------------------------------------------
# ML scoring layer
# ---------------------------------------------------------------------------

# Raw alternate-data input columns used to train the XGBoost model.
# These are the observable GST / UPI-AA / EPFO fields -- NOT the derived
# dimension scores, and NOT the ground-truth columns (archetype, true_health).
RAW_FEATURES = [
    "gst_turnover_lakhs_avg",
    "gst_turnover_growth_yoy_pct",
    "gst_filing_delay_days_avg",
    "gst_filing_consistency_pct",
    "upi_aa_avg_monthly_inflow_lakhs",
    "upi_aa_avg_monthly_outflow_lakhs",
    "cash_flow_volatility_cv",
    "bounce_rate_pct",
    "avg_bank_balance_runway_days",
    "epfo_employee_count",
    "epfo_employee_growth_yoy_pct",
    "epfo_pf_compliance_score",
    "epfo_avg_wage_growth_yoy_pct",
]

_DIMENSION_SCORES = list(DIMENSION_CONFIG.keys())


def train_model(df: pd.DataFrame):
    """Train an XGBoost binary classifier to predict 12-month default.

    Returns (model, test_indices) -- we hold out 25% as a test set so the
    /portfolio/scores endpoint can show out-of-sample predictions only.
    """
    X = df[RAW_FEATURES]
    y = df["defaulted_12m"]

    X_train, _X_test, y_train, _y_test, _idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(_X_test, _y_test)], verbose=False)
    return model, idx_test


def build_explainer(model, X: pd.DataFrame):
    """Build a SHAP TreeExplainer and compute shap values for X.

    Returns (explainer, shap_values) where shap_values has shape
    (n_samples, n_features).  Positive shap value = pushes toward default.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


# ---------------------------------------------------------------------------
# Named constants -- change these in ONE place instead of hunting magic numbers.
# ---------------------------------------------------------------------------

# Risk-tier cutoffs applied to the final blended health score (0-100).
RISK_TIER_LOW_CUTOFF = 65       # >= this => "Low risk"
RISK_TIER_MEDIUM_CUTOFF = 45    # >= this (and < LOW) => "Medium risk"

# Rule-based vs ML blend in the final health score.
# 50/50 is deliberately transparent for a demo; in production you would tune
# this ratio once you have labelled ground-truth outcomes.
BLEND_WEIGHT_RULE = 0.5
BLEND_WEIGHT_ML   = 0.5


def _risk_tier(score: float) -> str:
    if score >= RISK_TIER_LOW_CUTOFF:
        return "Low risk"
    if score >= RISK_TIER_MEDIUM_CUTOFF:
        return "Medium risk"
    return "High risk"


def generate_health_card(row: pd.Series, model, explainer) -> dict:
    """Generate a complete health-card dict for a single enriched MSME row."""
    # Validate required columns are present before touching pandas -- a missing
    # column would otherwise surface as an opaque KeyError deep inside pandas.
    required = RAW_FEATURES + ["composite_rule_score"]
    missing = [c for c in required if c not in row.index]
    if missing:
        raise ValueError(
            f"generate_health_card() is missing required column(s): {missing}. "
            "Ensure add_dimension_scores() has been called on the dataframe first."
        )
    X = pd.DataFrame([row[RAW_FEATURES]])
    default_prob = float(model.predict_proba(X)[0, 1])
    ml_health_score = round((1 - default_prob) * 100, 1)
    ml_default_pct = round(default_prob * 100, 1)

    composite = round(float(row["composite_rule_score"]), 1)
    final_score = round(BLEND_WEIGHT_RULE * composite + BLEND_WEIGHT_ML * ml_health_score, 1)

    dim_scores = {d: round(float(row[d]), 1) for d in _DIMENSION_SCORES}

    # SHAP: positive value = pushes toward default = risk; negative = strength
    sv = explainer.shap_values(X)[0]  # shape (n_features,)
    contributions = sorted(zip(RAW_FEATURES, sv), key=lambda x: x[1])

    top_strengths = [
        f"{f.replace('_', ' ').title()}: {abs(v):.2f} positive impact"
        for f, v in contributions[:3]
        if v < 0
    ]
    top_risks = [
        f"{f.replace('_', ' ').title()}: {abs(v):.2f} risk contribution"
        for f, v in contributions[-3:]
        if v > 0
    ]

    return {
        "msme_id": str(row.get("msme_id", "N/A")),
        "final_health_score": final_score,
        "risk_tier": _risk_tier(final_score),
        "composite_rule_score": composite,
        "rule_based_score": composite,
        "ml_health_score": ml_health_score,
        "ml_default_probability": ml_default_pct,
        "dimension_scores": dim_scores,
        "top_strengths": top_strengths,
        "top_risks": top_risks,
    }


def score_portfolio(df: pd.DataFrame, model) -> pd.DataFrame:
    """Score all MSMEs in a dataframe, adding ml/final score columns."""
    # Validate required columns up-front to give a clear error instead of a
    # raw KeyError if a future data-source integration drops a field.
    required = RAW_FEATURES + ["composite_rule_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"score_portfolio() is missing required column(s): {missing}. "
            "Ensure add_dimension_scores() has been called on the dataframe first."
        )
    df = df.copy()
    default_probs = model.predict_proba(df[RAW_FEATURES])[:, 1]
    df["ml_default_probability"] = (default_probs * 100).round(1)
    df["ml_health_score"] = ((1 - default_probs) * 100).round(1)
    df["final_health_score"] = (
        BLEND_WEIGHT_RULE * df["composite_rule_score"] + BLEND_WEIGHT_ML * df["ml_health_score"]
    ).round(1)
    df["risk_tier"] = df["final_health_score"].apply(_risk_tier)
    return df


if __name__ == "__main__":
    _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msme_synthetic_data.csv")
    raw = pd.read_csv(_csv)
    enriched = add_dimension_scores(raw)
    dims = list(DIMENSION_CONFIG.keys()) + ["composite_rule_score"]
    print(enriched[["msme_id", "archetype", "true_health"] + dims].head(8).to_string(index=False))
    enriched.to_csv(_csv, index=False)
    print("\nSaved enriched dataset with dimension scores.")
"""
MSME Financial Health Card -- Streamlit dashboard.

Run (two separate processes):
  1. uvicorn api_server:app --reload --port 8000
  2. streamlit run streamlit_app.py

ARCHITECTURE NOTE -- this version talks to the API over HTTP instead of
importing scoring_engine / the trained model directly (as an earlier
version did). This is CLIENT-SERVER SEPARATION: the dashboard (client)
knows nothing about how scoring works internally -- not the model, not
the feature engineering, not even that XGBoost is involved. It only
knows the API's HTTP contract (send a GET to /score/{id}, get JSON back).

Why this matters beyond "feels more real":
  - The dashboard and the scoring logic can now be deployed, scaled, and
    updated INDEPENDENTLY. You could swap XGBoost for a different model
    entirely and the dashboard code would need zero changes, as long as
    the JSON response shape stays the same -- this is the same principle
    as the ports-and-adapters idea from earlier, applied at the network
    boundary instead of inside one codebase.
  - Any other client (a mobile app, another bank's system, an OCEN LSP)
    could hit the same API. A dashboard with model logic baked in can
    only ever serve itself.
  - The tradeoff: you now depend on the network. If the API is down or
    slow, the dashboard must handle that gracefully instead of just
    calling a Python function that can't "fail to connect".
"""

import os

import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config via environment, not hardcoded values.
# This follows the 12-factor app principle: behaviour that differs between
# environments (local laptop vs. cloud demo vs. production) belongs in the
# environment, not in source code.  Set MSME_API_URL before launching
# Streamlit to point the dashboard at a different host or port without
# touching any Python file.
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("MSME_API_URL", "http://localhost:8000")

st.set_page_config(page_title="MSME Financial Health Card", layout="wide")


def api_get(path: str, params: dict | None = None):
    """Thin wrapper so every API call handles the 'server not running' case
    the same way, instead of repeating try/except at every call site."""
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        # Specific message for the most common dev-time failure: API not started.
        st.error(
            "Can't reach the scoring API. Start it first with:\n\n"
            "`uvicorn api_server:app --reload --port 8000`"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()
    except requests.exceptions.RequestException as e:
        # Catches everything else: Timeout, TooManyRedirects, SSLError, etc.
        # A slow API (Timeout) would otherwise crash the dashboard with an
        # unhandled exception instead of showing a friendly error.
        st.error(f"Request failed ({type(e).__name__}): {e}")
        st.stop()


def radar_chart(dimension_scores: dict, msme_id: str) -> go.Figure:
    labels = [d.replace("_score", "").replace("_", " ").title() for d in dimension_scores]
    values = list(dimension_scores.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself", name=msme_id,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(l=40, r=40, t=20, b=20), height=380,
    )
    return fig


st.title("MSME Financial Health Card")
st.caption(
    "AI-driven creditworthiness assessment built from alternate data "
    "(GST, UPI / Account Aggregator, EPFO) for New-to-Credit and "
    "New-to-Bank MSMEs. Served live from the scoring API."
)

# Fetch the portfolio once per script run -- used by BOTH tabs, so we only
# make this call once instead of once per tab.
portfolio = api_get("/portfolio/scores", params={"dataset": "test"})

tab_individual, tab_portfolio = st.tabs(["Individual health card", "Portfolio overview"])

with tab_individual:
    msme_ids = [row["msme_id"] for row in portfolio]
    selected_id = st.selectbox("Select an MSME", msme_ids)

    card = api_get(f"/score/{selected_id}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Final health score", f"{card['final_health_score']} / 100")
    col2.metric("Risk tier", card["risk_tier"])
    col3.metric("ML default probability", f"{card['ml_default_probability']}%")

    chart_col, detail_col = st.columns([1.2, 1])
    with chart_col:
        st.plotly_chart(radar_chart(card["dimension_scores"], selected_id), use_container_width=True)
    with detail_col:
        st.write(f"Rule-based composite score: **{card['rule_based_score']}**")
        st.write(f"ML-based score: **{card['ml_health_score']}**")
        st.markdown("**Strengths**")
        for s in card["top_strengths"]:
            st.write(f"- {s}")
        st.markdown("**Risk flags**")
        for r in card["top_risks"]:
            st.write(f"- {r}")

with tab_portfolio:
    risk_order = ["Low risk", "Medium risk", "High risk"]
    tier_counts = {tier: sum(1 for r in portfolio if r["risk_tier"] == tier) for tier in risk_order}

    c1, c2, c3 = st.columns(3)
    c1.metric("MSMEs scored", len(portfolio))
    c2.metric("Low risk", tier_counts["Low risk"])
    c3.metric("High risk", tier_counts["High risk"])

    invisible_viable = [r for r in portfolio if r["archetype"] == "credit_invisible_viable"]
    rescued = [r for r in invisible_viable if r["risk_tier"] != "High risk"]
    rescued_pct = len(rescued) / max(len(invisible_viable), 1)
    st.info(
        f"Of {len(invisible_viable)} thin-file / credit-invisible MSMEs in this "
        f"batch, {len(rescued)} ({rescued_pct:.0%}) scored Low or Medium risk -- "
        f"viable borrowers a traditional document-only process would likely "
        f"have rejected."
    )

    fig_bar = go.Figure(go.Bar(
        x=risk_order, y=[tier_counts[t] for t in risk_order],
        marker_color=["#1D9E75", "#EF9F27", "#E24B4A"],
    ))
    fig_bar.update_layout(
        title="Portfolio risk-tier distribution", height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Full scored portfolio")
    display_cols = [
        "msme_id", "final_health_score", "risk_tier",
        "composite_rule_score", "ml_health_score", "ml_default_probability",
    ]
    st.dataframe(
        [{k: row[k] for k in display_cols} for row in portfolio],
        use_container_width=True, hide_index=True,
    )
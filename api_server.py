"""
FastAPI layer simulating the ecosystem integration points from the problem
statement: Account Aggregator (AA) consent + data pull, and an OCEN-style
loan application / credit decision flow.

Run with:  uvicorn api_server:app --reload --port 8000
Docs at:   http://localhost:8000/docs   (FastAPI auto-generates this from
           your Pydantic models -- one of the biggest reasons to use it
           over Flask for a hackathon: you get interactive API docs for
           free, which is great for a judge or teammate to poke at live.)

WHAT'S REAL vs MOCKED:
  - The consent lifecycle (PENDING -> APPROVED -> data pull) mirrors the
    actual Sahamati Account Aggregator flow structurally, but here we
    "approve" consent ourselves instead of a real customer approving it
    in their AA app, and "fetch" from our synthetic CSV instead of a real
    FIP (Financial Information Provider) like a bank.
  - The OCEN loan flow mirrors the LSP (Lending Service Provider) ->
    lender handoff structurally: an application comes in, gets scored,
    gets a decision. A production system would exchange this over OCEN's
    actual API contracts with a real Lending Service Provider.

DESIGN NOTE -- lifespan / startup event:
Training the XGBoost model takes about a second. If we trained it INSIDE
every request handler, every API call would pay that cost -- unacceptable
for "near real-time credit assessment". Instead we train it ONCE when the
server starts (via FastAPI's `lifespan` context manager) and keep it in
memory for the life of the process. This is the API equivalent of
Streamlit's @st.cache_resource from the dashboard file -- same underlying
idea (expensive resource, load once, reuse), different framework's way of
expressing it.
"""
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Logging -- use the standard library rather than print()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from feature_engineering import add_dimension_scores
# ... rest of your original code ...

# import logging
# import traceback
# import uuid
# from contextlib import asynccontextmanager
# from datetime import datetime, timezone
# from typing import Optional

# import pandas as pd
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field, model_validator

# # ===========================================================================
# # 🔧 HACKATHON HOTFIX: SHAP / XGBoost Compatibility Patch
# # ===========================================================================
# import json
# import shap

# # Target the actual JSON loader method inside SHAP that extracts the params
# old_read_model = shap.explainers._tree.XGBTreeModelLoader.read_model

# def fixed_read_model(self, model):
#     # Let it read the model json string
#     old_read_model(self, model)
    
#     # Now, find the parsed JSON structure in SHAP's memory and fix it!
#     if hasattr(self, "base_score") and isinstance(self.base_score, str) and self.base_score.startswith('['):
#         self.base_score = float(self.base_score.strip('[]'))
        
#     # Also fix it inside the internal dictionary configuration if SHAP references it later
#     if hasattr(self, "unpacked_model") and "learner" in self.unpacked_model:
#         model_param = self.unpacked_model["learner"].get("learner_model_param", {})
#         if "base_score" in model_param and isinstance(model_param["base_score"], str) and model_param["base_score"].startswith('['):
#             model_param["base_score"] = model_param["base_score"].strip('[]')

# # Fallback extreme protection: override float handling for base_score completely
# old_init = shap.explainers._tree.XGBTreeModelLoader.__init__
# def fixed_init(self, xgb_model):
#     try:
#         old_init(self, xgb_model)
#     except ValueError as e:
#         if "could not convert string to float" in str(e):
#             # If it crashed on init, it means old_init failed at parsing.
#             # We force-inject the fix directly from the model string.
#             if hasattr(xgb_model, "save_raw"):
#                 try:
#                     raw_json = json.loads(xgb_model.save_raw(raw_format="json").decode("utf-8"))
#                     b_score = raw_json["learner"]["learner_model_param"]["base_score"]
#                     self.base_score = float(b_score.strip('[]'))
#                     # Re-trigger a minimalist completion of what init does next
#                     self.unpacked_model = raw_json
#                 except Exception:
#                     raise e
#         else:
#             raise e

# shap.explainers._tree.XGBTreeModelLoader.read_model = fixed_read_model
# shap.explainers._tree.XGBTreeModelLoader.__init__ = fixed_init
# # ===========================================================================

# # ---------------------------------------------------------------------------
# # Logging -- use the standard library rather than print()
# # ---------------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
# )
# logger = logging.getLogger(__name__)

# from msme_health_card.feature_engineering import add_dimension_scores
from msme_health_card.scoring_engine import (
    RAW_FEATURES, build_explainer, generate_health_card, score_portfolio, train_model,
)

# ... [The rest of your code continues unchanged below] ...

# In-memory stores standing in for a real database -- fine for a hackathon
# demo, but note this data disappears on restart and won't work if you ever
# run more than one server process (no shared state across processes).
CONSENTS: dict[str, dict] = {}
LOAN_APPLICATIONS: dict[str, dict] = {}

STATE: dict = {}   # holds the trained model/explainer/dataframe, set at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    raw = pd.read_csv("msme_synthetic_data.csv")
    df = add_dimension_scores(raw)
    model, test_idx = train_model(df)
    explainer, _ = build_explainer(model, df[RAW_FEATURES])

    STATE["df"] = df
    STATE["model"] = model
    STATE["explainer"] = explainer
    STATE["test_idx"] = test_idx
    logger.info("Model trained and ready. Loaded %d MSMEs.", len(df))
    yield
    STATE.clear()
    logger.info("Server shutting down; state cleared.")


app = FastAPI(title="MSME Financial Health Card API", lifespan=lifespan)


# ---------- Global exception handler ----------
# Catches any exception that is NOT already an HTTPException (those propagate
# normally so existing 404/403 responses are unaffected).  Logs the full
# traceback server-side but returns only a generic JSON body to the client --
# never expose internal stack traces to external callers.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        # Let FastAPI's built-in handler deal with HTTPExceptions.
        raise exc
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method, request.url, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please contact support."},
    )


# ---------- Pydantic schemas ----------

class ConsentRequest(BaseModel):
    msme_id: str
    purpose: str = "MSME credit assessment"


class ConsentResponse(BaseModel):
    consent_id: str
    msme_id: str
    status: str
    created_at: str


class BusinessFeatures(BaseModel):
    """Raw alternate-data inputs for scoring a business not in our dataset --
    this is what a real AA/GST/EPFO data pull would ultimately produce.

    Field constraints serve two purposes: (1) they make the 422 validation
    error message self-documenting rather than a cryptic downstream crash,
    and (2) they prevent impossible values from silently poisoning the score
    (e.g. negative turnover or a 500% filing consistency).
    """
    gst_turnover_lakhs_avg: float = Field(gt=0, description="Average monthly GST turnover in lakhs")
    gst_turnover_growth_yoy_pct: float = Field(ge=-100, le=500, description="YoY turnover growth %")
    gst_filing_delay_days_avg: float = Field(ge=0, le=365, description="Average GST filing delay in days")
    gst_filing_consistency_pct: float = Field(ge=0, le=100, description="% of returns filed on time")
    # gt=0 (strictly positive) because this is a denominator in net_margin_pct.
    upi_aa_avg_monthly_inflow_lakhs: float = Field(gt=0, description="Average monthly UPI/AA inflow in lakhs")
    upi_aa_avg_monthly_outflow_lakhs: float = Field(ge=0, description="Average monthly UPI/AA outflow in lakhs")
    cash_flow_volatility_cv: float = Field(ge=0, le=5, description="Coefficient of variation of monthly cash flow")
    bounce_rate_pct: float = Field(ge=0, le=100, description="% of payment bounces")
    avg_bank_balance_runway_days: float = Field(ge=0, le=3650, description="Days of runway at current burn rate")
    epfo_employee_count: int = Field(ge=0, description="Number of EPFO-registered employees")
    epfo_employee_growth_yoy_pct: float = Field(ge=-100, le=500, description="YoY employee headcount growth %")
    epfo_pf_compliance_score: float = Field(ge=0, le=100, description="PF compliance score (0-100)")
    epfo_avg_wage_growth_yoy_pct: float = Field(ge=-100, le=500, description="YoY average wage growth %")

    @model_validator(mode="after")
    def outflow_must_be_plausible(self) -> "BusinessFeatures":
        """Reject requests where outflow is more than 5× inflow.

        A legitimate business can run at a loss (outflow > inflow), but a ratio
        above 5x is almost certainly a data-entry error or a malformed request
        rather than real cash-flow data -- letting it through would produce a
        technically-valid but deeply misleading health score.
        """
        ratio = self.upi_aa_avg_monthly_outflow_lakhs / self.upi_aa_avg_monthly_inflow_lakhs
        if ratio > 5.0:
            raise ValueError(
                f"upi_aa_avg_monthly_outflow_lakhs ({self.upi_aa_avg_monthly_outflow_lakhs}) "
                f"is {ratio:.1f}x the inflow ({self.upi_aa_avg_monthly_inflow_lakhs}). "
                "Values above 5x inflow are not plausible and are likely a data error."
            )
        return self


class LoanApplicationRequest(BaseModel):
    msme_id: str
    loan_amount_requested_lakhs: float = Field(gt=0)


# ---------- Account Aggregator mock flow ----------

@app.post("/aa/consent", response_model=ConsentResponse)
def create_consent(req: ConsentRequest):
    if req.msme_id not in STATE["df"]["msme_id"].values:
        raise HTTPException(404, f"Unknown msme_id: {req.msme_id}")

    consent_id = str(uuid.uuid4())
    CONSENTS[consent_id] = {
        "msme_id": req.msme_id, "purpose": req.purpose,
        "status": "PENDING", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return ConsentResponse(consent_id=consent_id, **{k: v for k, v in CONSENTS[consent_id].items() if k != "purpose"})


@app.post("/aa/consent/{consent_id}/approve", response_model=ConsentResponse)
def approve_consent(consent_id: str):
    """Stands in for the business owner approving the consent in their AA app."""
    consent = CONSENTS.get(consent_id)
    if not consent:
        raise HTTPException(404, "Consent not found")
    consent["status"] = "APPROVED"
    return ConsentResponse(consent_id=consent_id, **{k: v for k, v in consent.items() if k != "purpose"})


@app.get("/aa/fetch/{consent_id}")
def fetch_data(consent_id: str):
    """Stands in for the AA pulling data from FIPs once consent is approved."""
    consent = CONSENTS.get(consent_id)
    if not consent:
        raise HTTPException(404, "Consent not found")
    if consent["status"] != "APPROVED":
        raise HTTPException(403, f"Consent status is {consent['status']}, not APPROVED")

    row = STATE["df"][STATE["df"]["msme_id"] == consent["msme_id"]].iloc[0]
    return {"msme_id": consent["msme_id"], "raw_data": row[RAW_FEATURES].to_dict()}


# ---------- Scoring ----------

@app.get("/score/{msme_id}")
def score_existing_msme(msme_id: str):
    df = STATE["df"]
    match = df[df["msme_id"] == msme_id]
    if match.empty:
        raise HTTPException(404, f"Unknown msme_id: {msme_id}")
    return generate_health_card(match.iloc[0], STATE["model"], STATE["explainer"])


@app.post("/score/raw")
def score_raw_features(features: BusinessFeatures):
    """Score a business that isn't in our dataset -- e.g. data just pulled
    live via AA/GST/EPFO for a brand-new applicant. This is the endpoint
    that demonstrates 'near real-time credit assessment' end to end."""
    row_df = pd.DataFrame([features.model_dump()])
    enriched = add_dimension_scores(row_df).iloc[0]
    enriched["msme_id"] = f"ADHOC-{uuid.uuid4().hex[:8]}"
    return generate_health_card(enriched, STATE["model"], STATE["explainer"])


@app.get("/portfolio/scores")
def portfolio_scores(dataset: str = "test"):
    """
    Batch scores for the dashboard's portfolio view. `dataset=test` (default)
    restricts to the held-out test set, so the dashboard only ever shows
    scores the model produced on data it never trained on -- same reasoning
    as the individual /score endpoint's use of test_idx.
    """
    df = STATE["df"]
    subset = df.loc[STATE["test_idx"]] if dataset == "test" else df
    scored = score_portfolio(subset, STATE["model"])

    cols = [
        "msme_id", "final_health_score", "risk_tier", "composite_rule_score",
        "ml_health_score", "ml_default_probability", "archetype",
    ]
    # archetype is our synthetic ground truth, included only so the demo can
    # show the "credit-invisible-viable businesses rescued" story -- a real
    # bank's API would never have this field, since it doesn't exist for
    # real applicants.
    return scored[cols].to_dict(orient="records")


# ---------- OCEN-style loan application flow ----------

@app.post("/ocen/loan-application")
def submit_loan_application(req: LoanApplicationRequest):
    df = STATE["df"]
    match = df[df["msme_id"] == req.msme_id]
    if match.empty:
        raise HTTPException(404, f"Unknown msme_id: {req.msme_id}")

    card = generate_health_card(match.iloc[0], STATE["model"], STATE["explainer"])

    # Simple, explainable policy layer on top of the score -- a real lender's
    # credit policy would be more elaborate, but the pattern (score -> tiered
    # decision) is the same.
    if card["risk_tier"] == "Low risk":
        decision = "APPROVED"
    elif card["risk_tier"] == "Medium risk":
        decision = "REFER_FOR_MANUAL_REVIEW"
    else:
        decision = "REJECTED"

    application_id = str(uuid.uuid4())
    LOAN_APPLICATIONS[application_id] = {
        "msme_id": req.msme_id,
        "loan_amount_requested_lakhs": req.loan_amount_requested_lakhs,
        "decision": decision,
        "health_card": card,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"application_id": application_id, **LOAN_APPLICATIONS[application_id]}


@app.get("/ocen/loan-application/{application_id}")
def get_loan_application(application_id: str):
    application = LOAN_APPLICATIONS.get(application_id)
    if not application:
        raise HTTPException(404, "Application not found")
    return application


@app.get("/health")
def health_check():
    return {"status": "ok", "msmes_loaded": len(STATE["df"]) if "df" in STATE else 0}
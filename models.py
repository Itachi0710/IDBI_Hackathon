"""
Pydantic models for the MSME Health Card API.

Centralising all request/response schemas here keeps api_server.py focused
on routing logic and makes the data contracts easy to find and evolve in one
place.
"""
from pydantic import BaseModel, Field, model_validator


class ConsentRequest(BaseModel):
    msme_id: str
    purpose: str = "MSME credit assessment"


class ConsentResponse(BaseModel):
    consent_id: str
    msme_id: str
    status: str
    created_at: str


class LoanApplicationRequest(BaseModel):
    msme_id: str
    loan_amount_requested_lakhs: float = Field(gt=0)

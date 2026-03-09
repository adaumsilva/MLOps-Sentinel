"""
Pydantic v2 request/response schemas for the prediction endpoint.
Strict validation prevents the API from crashing on malformed input.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "DEBT_CONSOLIDATION"
    CREDIT_CARD = "CREDIT_CARD"
    HOME_IMPROVEMENT = "HOME_IMPROVEMENT"
    OTHER = "OTHER"


class CreditRiskInput(BaseModel):
    """
    Input payload for the /predict endpoint.
    All ranges are validated against realistic domain constraints.
    """

    model_config = {"json_schema_extra": {
        "example": {
            "age": 35,
            "annual_income": 72000.0,
            "loan_amount": 15000.0,
            "credit_score": 680,
            "employment_years": 7,
            "debt_to_income_ratio": 0.28,
            "num_open_accounts": 5,
            "num_derogatory_marks": 0,
            "home_ownership": "MORTGAGE",
            "loan_purpose": "DEBT_CONSOLIDATION",
        }
    }}

    age: Annotated[int, Field(ge=18, le=100, description="Applicant age in years")]
    annual_income: Annotated[
        float, Field(gt=0, le=10_000_000, description="Gross annual income (USD)")
    ]
    loan_amount: Annotated[
        float, Field(gt=0, le=1_000_000, description="Requested loan amount (USD)")
    ]
    credit_score: Annotated[
        int, Field(ge=300, le=850, description="FICO credit score")
    ]
    employment_years: Annotated[
        int, Field(ge=0, le=60, description="Years at current employer")
    ]
    debt_to_income_ratio: Annotated[
        float, Field(ge=0.0, le=1.0, description="Monthly debt payments / gross monthly income")
    ]
    num_open_accounts: Annotated[
        int, Field(ge=0, le=100, description="Number of currently open credit accounts")
    ]
    num_derogatory_marks: Annotated[
        int, Field(ge=0, le=50, description="Number of derogatory marks on credit report")
    ]
    home_ownership: HomeOwnership
    loan_purpose: LoanPurpose

    @field_validator("annual_income", "loan_amount")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        import math
        if not math.isfinite(v):
            raise ValueError("Value must be finite (no inf/nan)")
        return v


class PredictionResponse(BaseModel):
    """Structured API response returned by /predict."""

    prediction: int = Field(description="Binary risk label: 1 = high-risk, 0 = low-risk")
    probability_high_risk: float = Field(
        description="Model confidence that the applicant is high-risk (0.0–1.0)"
    )
    model_version: str = Field(description="Identifier of the loaded model artefact")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

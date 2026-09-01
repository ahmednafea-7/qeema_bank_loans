"""
QeemaBank Loan Risk API
------------------------
Serves the two trained pipelines from the capstone notebook:
  - classification_pipeline.joblib  -> predicts `defaulted` (0/1) + probability
  - regression_pipeline.joblib      -> predicts a suggested `loan_amount`

Both pipelines already contain their preprocessing (ColumnTransformer with
StandardScaler / OrdinalEncoder / OneHotEncoder), so this app only has to
build a one-row DataFrame with the 9 raw fields and 2 engineered features
and call .predict() / .predict_proba().
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="QeemaBank Loan Risk API")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Load the two trained pipelines once, at startup.
# ---------------------------------------------------------------------------
clf_pipeline = joblib.load(MODELS_DIR / "classification_pipeline.joblib")
reg_pipeline = joblib.load(MODELS_DIR / "regression_pipeline.joblib")

# Same 9 raw columns feed both pipelines (see notebook: feature_cols).
EDUCATION_LEVELS = ["High School", "Bachelor", "Master", "PhD"]
GENDER_OPTIONS = ["Male", "Female"]
# Loan purposes weren't enumerated in the notebook (OneHotEncoder handle_unknown="ignore"
# makes any string safe) - edit this list to match the real categories in your CSV.
LOAN_PURPOSE_OPTIONS = [
'Education', 'Personal', 'Home', 'Car', 'Business',"Other",
]


def build_features(
    gender: str, age: float, annual_income: float, employment_years: float,
    education: str, credit_score: float, debt_to_income: float,
    num_existing_loans: int, loan_purpose: str,
) -> pd.DataFrame:
    """Recreate the exact engineered features from the notebook (cell 10)."""
    monthly_debt_est = (annual_income / 12) * debt_to_income
    credit_risk_index = (debt_to_income * (num_existing_loans + 1)) / (credit_score / 100)

    row = {
        "gender": gender,
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "education": education,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "num_existing_loans": num_existing_loans,
        "loan_purpose": loan_purpose,
        "monthly_debt_est": monthly_debt_est,
        "credit_risk_index": credit_risk_index,
    }
    return pd.DataFrame([row])


class LoanApplication(BaseModel):
    gender: str = Field(..., examples=["Male"])
    age: float = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    employment_years: float = Field(..., ge=0)
    education: str = Field(..., examples=["Bachelor"])
    credit_score: float = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0, le=2)
    num_existing_loans: int = Field(..., ge=0)
    loan_purpose: str = Field(..., examples=["Debt Consolidation"])


def run_models(payload: LoanApplication) -> dict:
    X = build_features(
        payload.gender, payload.age, payload.annual_income, payload.employment_years,
        payload.education, payload.credit_score, payload.debt_to_income,
        payload.num_existing_loans, payload.loan_purpose,
    )
    default_proba = float(clf_pipeline.predict_proba(X)[0, 1])
    default_pred = int(default_proba >= 0.5)
    suggested_amount = float(reg_pipeline.predict(X)[0])

    return {
        "default_probability": round(default_proba * 100, 1),
        "default_prediction": "Likely to default" if default_pred else "Likely to repay",
        "suggested_loan_amount": round(suggested_amount, 2),
        "recommendation": (
            "Route to human underwriter for manual review."
            if default_pred
            else "Eligible for fast-track approval after a quick human check."
        ),
    }


# ---------------------------------------------------------------------------
# JSON API (for programmatic use / curl / another app)
# ---------------------------------------------------------------------------
@app.post("/api/predict")
def api_predict(payload: LoanApplication):
    return run_models(payload)


# ---------------------------------------------------------------------------
# Simple HTML UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "genders": GENDER_OPTIONS,
            "educations": EDUCATION_LEVELS,
            "purposes": LOAN_PURPOSE_OPTIONS,
            "result": None,
        },
    )


@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    annual_income: float = Form(...),
    employment_years: float = Form(...),
    education: str = Form(...),
    credit_score: float = Form(...),
    debt_to_income: float = Form(...),
    num_existing_loans: int = Form(...),
    loan_purpose: str = Form(...),
):
    payload = LoanApplication(
        gender=gender, age=age, annual_income=annual_income,
        employment_years=employment_years, education=education,
        credit_score=credit_score, debt_to_income=debt_to_income,
        num_existing_loans=num_existing_loans, loan_purpose=loan_purpose,
    )
    result = run_models(payload)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "genders": GENDER_OPTIONS,
            "educations": EDUCATION_LEVELS,
            "purposes": LOAN_PURPOSE_OPTIONS,
            "result": result,
            "form": payload.model_dump(),
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

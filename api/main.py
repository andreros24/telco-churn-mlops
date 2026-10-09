"""
api/main.py

API FastAPI to share the churn prediction model.
Load preprocessor and trained models, show an endpoint /predict
that takes raw data of a client and return the prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
from pathlib import Path

# --- Loading models and preprocessor ---
MODELS_DIR = Path("models")

try:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model_rf = joblib.load(MODELS_DIR / "model_rf.pkl")
    model_logistic = joblib.load(MODELS_DIR / "model_logistic.pkl")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Models or preprocessor not found in {MODELS_DIR}. "
        f"Execute 'dvc pull' or 'dvc repro' before running the API."
    ) from e

app = FastAPI(
    title="Telco Churn Prediction API",
    description="API to predict the costumer churn in telco sector",
    version="1.0.0",
)


# --- Input schema with the columns of the raw dataset ---
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=500, description="Months as a client")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50,
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Transform Pydantic input into a DataFrame with 1 row,
        # with columns in the same order/names of training 
        input_df = pd.DataFrame([customer.model_dump()])

        # Apply same preprocessing (scaling + encoding) used in training
        processed = preprocessor.transform(input_df)

        prediction_rf = model_rf.predict(processed)[0]
        probability_rf = model_rf.predict_proba(processed)[0][1]

        return PredictionResponse(
            churn_prediction="Yes" if prediction_rf == 1 else "No",
            churn_probability=round(float(probability_rf), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Raise error in prediction: {str(e)}")
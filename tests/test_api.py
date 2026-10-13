from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction_rf"] in ["Yes", "No"]
    assert data["churn_prediction_logistic"] in ["Yes", "No"]
    assert data["churn_prediction_xgb"] in ["Yes", "No"]
    assert data["churn_prediction_overall"] in ["Yes", "No"]
    assert 0 <= data["churn_probability_rf"] <= 1
    assert 0 <= data["churn_probability_logistic"] <= 1
    assert 0 <= data["churn_probability_xgb"] <= 1
    assert 0 <= data["churn_probability_mean"] <= 1


def test_predict_invalid_input():
    payload = {"gender": "Unknown"}  # mancano campi obbligatori
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # errore di validazione Pydantic
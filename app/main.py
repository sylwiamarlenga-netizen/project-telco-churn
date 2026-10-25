from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# --- wczytanie modelu raz, przy starcie aplikacji ---
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]                      # cały Pipeline (preprocessing + regresja logistyczna)
THRESHOLD = artifact["threshold"]              # próg decyzyjny wybrany w Etapie 12
FEATURE_COLUMNS = artifact["feature_columns"]  # kolumny w takiej kolejności, jakiej oczekuje model

app = FastAPI(
    title="Predykcja churnu klientów telekomunikacyjnych",
    description="Zwraca prawdopodobieństwo odejścia klienta i decyzję, czy objąć go akcją retencyjną.",
)


# --- opis i walidacja danych wejściowych (Pydantic) ---
class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Staż klienta w miesiącach")
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
    MonthlyCharges: float = Field(ge=0, description="Opłata miesięczna [$]")
    TotalCharges: float = Field(ge=0, description="Suma dotychczasowych opłat [$]")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 14,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 104.85,
                "TotalCharges": 1468.9,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float


# --- endpointy ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])[FEATURE_COLUMNS]   # jeden wiersz, kolumny w kolejności modelu
    probability = float(model.predict_proba(row)[0, 1])
    return Prediction(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= THRESHOLD,
        threshold=THRESHOLD,
    )
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

# Project Paths

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Load Model, Scaler and Dataset Information

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

data = load_breast_cancer()

FEATURE_NAMES = data.feature_names.tolist()
EXPECTED_FEATURES = len(FEATURE_NAMES)

# Create FastAPI Application

app = FastAPI(
    title="AI Cancer Diagnosis API",
    description=(
        "Machine Learning API for breast tumor classification "
        "using a Random Forest Classifier."
    ),
    version="1.0.0",
)

# Input Data Model

class PatientData(BaseModel):
    features: list[float] = Field(
        ...,
        description=(
            "30 breast tumor features in the same order as the "
            "Breast Cancer Wisconsin dataset."
        ),
        min_length=30,
        max_length=30,
    )

# Home Route

@app.get("/", tags=["General"])
def home():
    return {
        "message": "AI Cancer Diagnosis API is running!",
        "status": "active",
        "version": "1.0.0",
    }

# Health Check Route

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
    }

# Prediction Route

@app.post("/predict", tags=["Prediction"])
def predict(patient: PatientData):

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame(
            [patient.features],
            columns=FEATURE_NAMES
        )

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]

        # Convert prediction to readable diagnosis
        if prediction == 0:
            diagnosis = "Malignant"
        else:
            diagnosis = "Benign"

        return {
            "prediction": int(prediction),
            "diagnosis": diagnosis,
            "malignant_probability": round(
                float(probabilities[0]) * 100, 2
            ),
            "benign_probability": round(
                float(probabilities[1]) * 100, 2
            ),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
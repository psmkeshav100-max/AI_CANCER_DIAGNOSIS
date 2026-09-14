from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# Load trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

#new data
# Sample patient data
new_patient = [[
    17.99, 10.38, 122.80, 1001.0, 0.11840,
    0.27760, 0.30010, 0.14710, 0.2419, 0.07871,
    1.0950, 0.9053, 8.589, 153.40, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.60, 2019.0, 0.16220,
    0.66560, 0.71190, 0.26540, 0.4601, 0.11890
]]

# Load feature names
data = load_breast_cancer()

# Convert new patient to DataFrame
new_patient = pd.DataFrame(
    new_patient,
    columns=data.feature_names
)

# Scale the new patient
new_patient_scaled = scaler.transform(new_patient)

# Make prediction
prediction = model.predict(new_patient_scaled)[0]

# Get prediction probabilities
probabilities = model.predict_proba(new_patient_scaled)[0]

#dispaly results
print("\n===== PREDICTION RESULTS =====")
print(f"Prediction: {prediction}")

print("\nPrediction Probability:")
print(f"Malignant: {probabilities[0] * 100:.2f}%")
print(f"Benign: {probabilities[1] * 100:.2f}%")
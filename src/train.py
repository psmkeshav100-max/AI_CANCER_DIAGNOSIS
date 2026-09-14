from pathlib import Path
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from data_preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from evaluate import evaluate_model

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# 1. Load the dataset
data = load_breast_cancer()

# 2. Create features (X) and target (y)
X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

# 3. Preprocess data
X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_test = preprocess_data(X, y)

# 4. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train_scaled, y_train)

# 6. saving scaler.pkl
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(model, MODEL_DIR / "random_forest.pkl")

# 7. Evaluate model
evaluate_model(model, X_test_scaled, y_test)

# 8. Print information
print("Training completed successfully.")
print(f"Model saved to: {MODEL_DIR / 'random_forest.pkl'}")
print(f"Scaler saved to: {MODEL_DIR / 'scaler.pkl'}")
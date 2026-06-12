import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """Load trained churn model."""
    return joblib.load(MODEL_PATH)


def predict_churn(input_data):
    """Predict customer churn."""
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    not_churn_probability = probability[0]
    churn_probability = probability[1]

    if prediction == 1:
        label = "Likely to Churn"
    else:
        label = "Not Likely to Churn"

    return {
        "prediction": label,
        "churn_probability": churn_probability,
        "not_churn_probability": not_churn_probability,
    }
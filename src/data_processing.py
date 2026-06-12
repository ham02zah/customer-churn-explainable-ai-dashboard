import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for customer churn.

    Creates:
    - AvgChargePerTenure
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

        df["AvgChargePerTenure"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
        df["AvgChargePerTenure"] = df["AvgChargePerTenure"].replace([np.inf, -np.inf], np.nan)

        return df


def load_churn_dataset(data_path):
    """Load Telco customer churn dataset."""
    df = pd.read_csv(data_path)

    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def prepare_features_and_target(df, target_column, id_column=None):
    """Prepare X and y for model training."""
    data = df.copy()

    if id_column and id_column in data.columns:
        data = data.drop(columns=[id_column])

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    data = data.dropna(subset=[target_column])

    data[target_column] = data[target_column].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y
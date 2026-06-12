DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.joblib"

REPORTS_DIR = "reports"
VISUALIZATIONS_DIR = "visualizations"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgChargePerTenure",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]
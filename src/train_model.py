import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
)
from src.data_processing import (
    ChurnFeatureEngineer,
    load_churn_dataset,
    prepare_features_and_target,
)


def create_directories():
    """Create output folders."""
    os.makedirs("models", exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def plot_class_distribution(df):
    """Save churn class distribution graph."""
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette="viridis",
        legend=False,
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/class_distribution.png")
    plt.close()


def plot_churn_by_contract(df):
    """Save churn rate by contract type."""
    plot_df = df.copy()
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    churn_rate = (
        plot_df.groupby("Contract")[TARGET_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=TARGET_COLUMN, ascending=False)
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=churn_rate,
        x="Contract",
        y=TARGET_COLUMN,
        hue="Contract",
        palette="mako",
        legend=False,
    )

    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/churn_by_contract.png")
    plt.close()


def plot_churn_by_internet_service(df):
    """Save churn rate by internet service."""
    plot_df = df.copy()
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    churn_rate = (
        plot_df.groupby("InternetService")[TARGET_COLUMN]
        .mean()
        .reset_index()
        .sort_values(by=TARGET_COLUMN, ascending=False)
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=churn_rate,
        x="InternetService",
        y=TARGET_COLUMN,
        hue="InternetService",
        palette="rocket",
        legend=False,
    )

    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/churn_by_internet_service.png")
    plt.close()


def plot_monthly_charges_distribution(df):
    """Save monthly charges distribution by churn."""
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="MonthlyCharges",
        hue=TARGET_COLUMN,
        bins=40,
        kde=True,
        palette="mako",
    )

    plt.title("Monthly Charges Distribution by Churn")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/monthly_charges_distribution.png")
    plt.close()


def plot_tenure_distribution(df):
    """Save tenure distribution by churn."""
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="tenure",
        hue=TARGET_COLUMN,
        bins=40,
        kde=True,
        palette="viridis",
    )

    plt.title("Customer Tenure Distribution by Churn")
    plt.xlabel("Tenure Months")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/tenure_distribution.png")
    plt.close()


def build_pipeline(model):
    """Build preprocessing + feature engineering + model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    return metrics, y_pred, y_prob


def plot_confusion_matrix_graph(y_test, y_pred):
    """Save confusion matrix graph."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Churn", "Churn"],
        yticklabels=["Not Churn", "Churn"],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/confusion_matrix.png")
    plt.close()


def plot_roc_curve_graph(y_test, y_prob):
    """Save ROC curve graph."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/roc_curve.png")
    plt.close()


def plot_precision_recall_curve_graph(y_test, y_prob):
    """Save precision-recall curve graph."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(recall, precision, color="purple")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/precision_recall_curve.png")
    plt.close()


def plot_model_comparison(comparison_df):
    """Save model comparison graph."""
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=plot_df,
        x="model",
        y="score",
        hue="metric",
        palette="mako",
    )

    plt.title("Customer Churn Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/model_comparison.png")
    plt.close()


def get_processed_feature_names(best_model):
    """Extract processed feature names from preprocessing pipeline."""
    preprocessor = best_model.named_steps["preprocessor"]

    numeric_names = NUMERIC_FEATURES

    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    categorical_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    return list(numeric_names) + list(categorical_names)


def plot_feature_importance(best_model, X_test, y_test):
    """Save permutation feature importance graph."""
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )

    raw_feature_names = list(X_test.columns)

    importance_df = pd.DataFrame(
        {
            "feature": raw_feature_names,
            "importance": result.importances_mean,
        }
    ).sort_values(by="importance", ascending=False).head(15)

    plt.figure(figsize=(9, 7))

    sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="rocket",
        legend=False,
    )

    plt.title("Top Features Impacting Churn Prediction")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(f"{VISUALIZATIONS_DIR}/feature_importance.png")
    plt.close()


def train_model():
    """Train, compare, evaluate, and save best churn model."""
    create_directories()

    print("Loading dataset...")
    df = load_churn_dataset(DATA_PATH)

    print("Creating exploratory visualizations...")
    plot_class_distribution(df)
    plot_churn_by_contract(df)
    plot_churn_by_internet_service(df)
    plot_monthly_charges_distribution(df)
    plot_tenure_distribution(df)

    X, y = prepare_features_and_target(df, TARGET_COLUMN, ID_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    comparison_results = []
    trained_models = {}

    for model_name, estimator in candidate_models.items():
        print(f"\nTraining {model_name}...")

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        metrics, _, _ = evaluate_model(pipeline, X_test, y_test)

        comparison_results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = pipeline

        print(f"{model_name} ROC-AUC: {metrics['roc_auc']:.4f}")

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)

    best_model_name = comparison_df.sort_values(
        by="roc_auc",
        ascending=False,
    ).iloc[0]["model"]

    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    best_metrics, y_pred, y_prob = evaluate_model(best_model, X_test, y_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"]))

    final_report = {
        "best_model": best_model_name,
        **best_metrics,
    }

    with open(f"{REPORTS_DIR}/metrics.json", "w") as file:
        json.dump(final_report, file, indent=4)

    print("Saving evaluation visualizations...")
    plot_confusion_matrix_graph(y_test, y_pred)
    plot_roc_curve_graph(y_test, y_prob)
    plot_precision_recall_curve_graph(y_test, y_prob)
    plot_model_comparison(comparison_df)
    plot_feature_importance(best_model, X_test, y_test)

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {REPORTS_DIR}/metrics.json")
    print(f"Model comparison saved to: {REPORTS_DIR}/model_comparison.csv")
    print(f"Visualizations saved inside: {VISUALIZATIONS_DIR}/")


if __name__ == "__main__":
    train_model()
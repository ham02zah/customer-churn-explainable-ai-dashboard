# Customer Churn Explainable AI Dashboard

## Project Overview

This project is an intermediate-level machine learning dashboard that predicts customer churn probability and explains customer-level churn risk drivers.

The project uses the Telco Customer Churn dataset and includes data preprocessing, feature engineering, multiple model comparison, model evaluation, feature importance analysis, customer-level risk explanation, and a modern multi-gradient Streamlit dashboard.

---

## Dataset

Dataset used:

Telco Customer Churn Dataset  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the dataset and place the CSV file here:

```text
data/telco_churn.csv
```

If the downloaded file is named:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

rename it to:

```text
telco_churn.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## Machine Learning Workflow

1. Load Telco customer churn dataset
2. Clean and convert numerical columns
3. Engineer customer behavior features
4. Handle missing values
5. Encode categorical variables
6. Scale numerical features
7. Train multiple classification models
8. Compare models using evaluation metrics
9. Select best model based on ROC-AUC
10. Generate permutation feature importance
11. Save trained model
12. Build Streamlit dashboard

---

## Models Compared

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Permutation Feature Importance

---

## Dashboard Features

- Modern multi-gradient Streamlit UI
- Collapsible navigation menu
- Customer churn probability prediction
- Retention probability score
- Customer-level risk explanation
- Model evaluation dashboard
- Visualization gallery
- Feature importance analysis
- Model comparison chart

---

## Project Structure

```text
customer-churn-explainable-ai-dashboard/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── telco_churn.csv
│
├── models/
│   └── churn_model.joblib
│
├── reports/
│   ├── metrics.json
│   └── model_comparison.csv
│
├── visualizations/
│   ├── class_distribution.png
│   ├── churn_by_contract.png
│   ├── churn_by_internet_service.png
│   ├── monthly_charges_distribution.png
│   ├── tenure_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_processing.py
    ├── train_model.py
    └── predict.py
```

---

## How to Run

### 1. Install dependencies

```bash
python3.11 -m pip install -r requirements.txt
```

### 2. Add dataset

Place the dataset here:

```text
data/telco_churn.csv
```

### 3. Train the model

```bash
python3.11 -m src.train_model
```

### 4. Run the app

```bash
python3.11 -m streamlit run app.py
```

---

## Example Use Case

A telecom company can use this dashboard to identify high-risk customers and understand which customer attributes may be contributing to churn risk.

---

## Future Improvements

- Add SHAP explainability
- Add customer retention recommendation engine
- Add batch prediction from CSV
- Deploy dashboard on Streamlit Cloud
- Add customer segmentation
- Add database storage for prediction history

---

## Author

Hamzah Jawad
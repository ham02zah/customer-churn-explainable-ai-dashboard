import os
import json

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH


METRICS_PATH = "reports/metrics.json"

PAGES = [
    "Churn Prediction",
    "Customer Explanation",
    "Model Insights",
    "Visualization Gallery",
]


st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(6, 182, 212, 0.32), transparent 28%),
        radial-gradient(circle at top right, rgba(249, 115, 22, 0.28), transparent 30%),
        radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.30), transparent 32%),
        linear-gradient(135deg, #030712 0%, #111827 48%, #020617 100%);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stRadio label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stTextInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span {
    color: #000000 !important;
}

ul[role="listbox"] {
    background-color: #ffffff !important;
}

li[role="option"],
li[role="option"] span,
li[role="option"] div {
    color: #000000 !important;
    background-color: #ffffff !important;
}

li[role="option"]:hover,
li[role="option"]:hover span,
li[role="option"]:hover div {
    color: #000000 !important;
    background-color: #cffafe !important;
}

.main-title {
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #22d3ee, #fb923c, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #d1d5db;
    max-width: 980px;
    margin-bottom: 1.5rem;
}

.glass-card {
    padding: 1.5rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 22px 45px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(16px);
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #ffffff;
}

.section-caption {
    color: #cbd5e1;
    font-size: 0.95rem;
}

.nav-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.nav-subtitle {
    color: #cbd5e1;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.nav-section {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}

.feature-box {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 18px;
    color: #cbd5e1;
    font-size: 0.95rem;
    line-height: 1.9;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 28px !important;
    background:
        linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(3, 7, 18, 0.90)),
        rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.34) !important;
}

div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 18px !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.75rem !important;
    width: 100% !important;
}

div[role="radiogroup"] label:hover {
    background: linear-gradient(90deg, #0891b2, #f97316, #7c3aed) !important;
}

div[role="radiogroup"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.result-risk {
    padding: 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.28), rgba(239, 68, 68, 0.16));
    border: 1px solid rgba(251, 146, 60, 0.50);
}

.result-safe {
    padding: 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(34, 197, 94, 0.14));
    border: 1px solid rgba(34, 211, 238, 0.50);
}

.result-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.result-text {
    font-size: 1rem;
    color: #e5e7eb;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 20px;
    background:
        linear-gradient(135deg, rgba(6, 182, 212, 0.18), rgba(249, 115, 22, 0.13)),
        rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #67e8f9;
}

.metric-label {
    font-size: 0.85rem;
    color: #cbd5e1;
}

.explain-box {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #e5e7eb;
    line-height: 1.8;
}

div.stButton > button:first-child {
    width: 100%;
    border-radius: 18px;
    height: 3.1rem;
    border: none;
    font-size: 0.95rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #0891b2, #f97316, #7c3aed);
}

div.stButton > button:first-child:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 45px rgba(249, 115, 22, 0.35);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 10px 18px;
    background: rgba(255, 255, 255, 0.08);
    color: white;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #0891b2, #f97316);
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if "menu_open" not in st.session_state:
    st.session_state.menu_open = True

if "page" not in st.session_state:
    st.session_state.page = "Churn Prediction"

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


def toggle_menu():
    st.session_state.menu_open = not st.session_state.menu_open


@st.cache_resource
def load_model():
    """Load trained churn model."""
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load model evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return None

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


def display_metric_card(label, value):
    """Render custom metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_customer_risk_level(churn_probability):
    """Return customer churn risk level."""
    if churn_probability >= 0.70:
        return "High Churn Risk"
    if churn_probability >= 0.40:
        return "Medium Churn Risk"
    return "Low Churn Risk"


def explain_customer_risk(customer_data):
    """Generate human-readable churn explanation based on customer inputs."""
    explanations = []

    if customer_data["Contract"] == "Month-to-month":
        explanations.append("Month-to-month contract increases churn risk.")

    if customer_data["tenure"] < 12:
        explanations.append("Low tenure suggests the customer is still early in the lifecycle.")

    if customer_data["InternetService"] == "Fiber optic":
        explanations.append("Fiber optic customers often show higher churn in this dataset.")

    if customer_data["TechSupport"] == "No":
        explanations.append("Lack of tech support may increase dissatisfaction risk.")

    if customer_data["OnlineSecurity"] == "No":
        explanations.append("No online security service may indicate lower product stickiness.")

    if customer_data["MonthlyCharges"] > 80:
        explanations.append("High monthly charges can increase price sensitivity.")

    if customer_data["PaperlessBilling"] == "Yes":
        explanations.append("Paperless billing is associated with higher churn in many customer segments.")

    if not explanations:
        explanations.append("No major churn risk drivers detected from the selected customer profile.")

    return explanations


model = load_model()
metrics = load_metrics()


top_col1, top_col2 = st.columns([0.16, 0.84])

with top_col1:
    button_label = "☰ Open Menu" if not st.session_state.menu_open else "✕ Close Menu"
    st.button(button_label, on_click=toggle_menu)

with top_col2:
    st.markdown(
        """
        <div class="main-title">Customer Churn Explainable AI Dashboard</div>
        <div class="main-subtitle">
            A modern churn intelligence system that predicts customer churn probability,
            explains risk drivers, compares machine learning models, and visualizes retention insights.
        </div>
        """,
        unsafe_allow_html=True,
    )


if model is None:
    st.error(
        "Model not found. Train the model first by running: "
        "`python3.11 -m src.train_model`"
    )
    st.stop()


if st.session_state.menu_open:
    menu_col, content_col = st.columns([0.25, 0.75], gap="large")
else:
    content_col = st.container()


if st.session_state.menu_open:
    with menu_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="nav-title">📉 Churn AI</div>
                <div class="nav-subtitle">Explainable ML Dashboard</div>
                <div class="nav-section">Navigation</div>
                """,
                unsafe_allow_html=True,
            )

            selected_page = st.radio(
                label="Select dashboard section",
                options=PAGES,
                index=PAGES.index(st.session_state.page),
                label_visibility="collapsed",
            )

            st.session_state.page = selected_page

            st.markdown(
                """
                <div class="nav-section">Project Features</div>
                <div class="feature-box">
                    • Churn probability scoring<br>
                    • Customer risk explanation<br>
                    • Multi-model comparison<br>
                    • Permutation importance<br>
                    • ROC-AUC evaluation<br>
                    • Modern gradient UI
                </div>
                """,
                unsafe_allow_html=True,
            )


with content_col:
    page = st.session_state.page

    if page == "Churn Prediction":
        left_col, right_col = st.columns([1.1, 0.9])

        with left_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Customer Profile Input</div>
                    <div class="section-caption">
                        Enter customer details to predict churn risk and generate a retention profile.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            customer_name = st.text_input("Customer Name", placeholder="Enter customer name")

            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            with col2:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                )
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
                total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

            predict_button = st.button("Predict Customer Churn")

        with right_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Prediction Output</div>
                    <div class="section-caption">
                        Churn score, retention score, and customer risk level.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if predict_button:
                customer_data = {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }

                input_df = pd.DataFrame([customer_data])

                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]

                not_churn_probability = probability[0]
                churn_probability = probability[1]

                customer_display_name = customer_name.strip() if customer_name.strip() else "Customer"
                risk_level = get_customer_risk_level(churn_probability)

                explanations = explain_customer_risk(customer_data)

                st.session_state.latest_prediction = {
                    "customer_name": customer_display_name,
                    "risk_level": risk_level,
                    "churn_probability": churn_probability,
                    "not_churn_probability": not_churn_probability,
                    "explanations": explanations,
                }

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-risk">
                            <div class="result-title">⚠️ {customer_display_name} May Churn</div>
                            <div class="result-text">Churn Probability: {churn_probability:.2%}</div>
                            <div class="result-text">Risk Level: {risk_level}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-safe">
                            <div class="result-title">✅ {customer_display_name} Is Likely Retained</div>
                            <div class="result-text">Retention Probability: {not_churn_probability:.2%}</div>
                            <div class="result-text">Risk Level: {risk_level}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.progress(float(churn_probability))

                score_col1, score_col2 = st.columns(2)

                with score_col1:
                    display_metric_card("Churn Score", f"{churn_probability:.1%}")

                with score_col2:
                    display_metric_card("Retention Score", f"{not_churn_probability:.1%}")

            else:
                st.info("Enter customer details and click the prediction button.")

    elif page == "Customer Explanation":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Customer-Level Explanation</div>
                <div class="section-caption">
                    This section explains the likely business reasons behind the latest churn prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        latest_prediction = st.session_state.latest_prediction

        if latest_prediction:
            display_metric_card("Customer", latest_prediction["customer_name"])
            st.write("")

            col1, col2, col3 = st.columns(3)

            with col1:
                display_metric_card("Risk Level", latest_prediction["risk_level"])

            with col2:
                display_metric_card("Churn Score", f"{latest_prediction['churn_probability']:.1%}")

            with col3:
                display_metric_card("Retention Score", f"{latest_prediction['not_churn_probability']:.1%}")

            st.write("")

            explanation_html = "<br>".join(
                [f"• {reason}" for reason in latest_prediction["explanations"]]
            )

            st.markdown(
                f"""
                <div class="explain-box">
                    <strong>Top Customer Risk Drivers</strong><br><br>
                    {explanation_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Run a churn prediction first to generate customer-level explanations.")

    elif page == "Model Insights":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Model Performance Summary</div>
                <div class="section-caption">
                    Evaluation metrics and model comparison results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if metrics:
            st.write(f"### Best Model: `{metrics['best_model']}`")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                display_metric_card("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                display_metric_card("Precision", f"{metrics['precision']:.3f}")
            with col3:
                display_metric_card("Recall", f"{metrics['recall']:.3f}")
            with col4:
                display_metric_card("F1 Score", f"{metrics['f1_score']:.3f}")
            with col5:
                display_metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        else:
            st.warning("Metrics not found. Run training first.")

        insight_graphs = {
            "Model Comparison": "visualizations/model_comparison.png",
            "Feature Importance": "visualizations/feature_importance.png",
            "Confusion Matrix": "visualizations/confusion_matrix.png",
            "ROC Curve": "visualizations/roc_curve.png",
            "Precision-Recall Curve": "visualizations/precision_recall_curve.png",
        }

        for title, path in insight_graphs.items():
            if os.path.exists(path):
                st.markdown(f"### {title}")
                st.image(path, use_container_width=True)

    elif page == "Visualization Gallery":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Visualization Gallery</div>
                <div class="section-caption">
                    Explore customer churn patterns, model performance, and feature importance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        graph_paths = {
            "Class Distribution": "visualizations/class_distribution.png",
            "Churn by Contract": "visualizations/churn_by_contract.png",
            "Churn by Internet Service": "visualizations/churn_by_internet_service.png",
            "Monthly Charges Distribution": "visualizations/monthly_charges_distribution.png",
            "Tenure Distribution": "visualizations/tenure_distribution.png",
            "Confusion Matrix": "visualizations/confusion_matrix.png",
            "ROC Curve": "visualizations/roc_curve.png",
            "Precision-Recall Curve": "visualizations/precision_recall_curve.png",
            "Model Comparison": "visualizations/model_comparison.png",
            "Feature Importance": "visualizations/feature_importance.png",
        }

        tabs = st.tabs(list(graph_paths.keys()))

        for tab, (title, path) in zip(tabs, graph_paths.items()):
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"{title} graph not found. Run training to generate it.")
import streamlit as st
import joblib

st.set_page_config(
    page_title="Credit Risk Default Prediction",
    page_icon="🏦",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>

.main-title {
    font-size:42px;
    font-weight:bold;
    color:#1f4e79;
}

.sub-title {
    font-size:20px;
    color:#555;
}

.metric-label {
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("data/models/credit_risk_xgboost.pkl")
features = joblib.load("data/models/feature_columns.pkl")

# Title
st.markdown(
    '<p class="main-title">🏦 Credit Risk Default Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI-Powered Loan Default Risk Assessment using XGBoost</p>',
    unsafe_allow_html=True
)

st.success("✅ Model Loaded Successfully")

st.divider()

# Metrics Section
c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset Size", "5000")
c2.metric("Features", len(features))
c3.metric("Accuracy", "93.6%")
c4.metric("ROC-AUC", "98.3%")

st.divider()

# Project Summary
st.subheader("📌 Project Summary")

st.write("""
This project predicts whether a customer is likely to default on a loan within the next 12 months.

### Models Used
- Logistic Regression
- Random Forest
- XGBoost (Best Model)

### Techniques Applied
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- SMOTE for Class Balancing
- Cross Validation
- Feature Importance Analysis

### Final Results
- Accuracy: 93.6%
- Precision: 83.0%
- Recall: 85.5%
- F1 Score: 84.2%
- ROC-AUC: 98.3%
""")

st.divider()

st.info(
    "Next section will allow users to enter applicant details and predict default risk."
)
import streamlit as st
import pandas as pd
import joblib

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="Credit Risk Default Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ====================================
# LOAD MODEL
# ====================================
model = joblib.load("data/models/credit_risk_xgboost.pkl")
features = joblib.load("data/models/feature_columns.pkl")

# ====================================
# CUSTOM CSS
# ====================================
st.markdown("""
<style>

.main-title{
    font-size:60px;
    font-weight:900;
    color:white;
    text-align:center;

}

.subtitle{
    font-size:18px;
    color:#666666;
    text-align:center;
    margin-bottom:25px;
}


[data-testid="stMetricValue"]{
    font-size:28px !important;
    font-weight:800 !important;
}
}

[data-testid="stMetricLabel"]{
    font-size:13px !important;
}

[data-testid="stSidebar"] label{
    font-size:12px !important;
}

</style>
""", unsafe_allow_html=True)


# ====================================
# HEADER
# ====================================

st.title("🏦 Credit Risk Default Prediction System")

st.markdown(
    "### AI Powered Loan Default Risk Assessment using XGBoost"
)

st.divider()

# ====================================
# DASHBOARD METRICS
# ====================================
m1, m2, m3, m4 = st.columns(4)

m1.metric("📊 Dataset", "5000")
m2.metric("🧠 Features", "40")
m3.metric("🎯 Accuracy", "94.7%")
m4.metric("📈 ROC-AUC", "98.5%")

st.divider()

# ====================================
# SIDEBAR
# ====================================
st.sidebar.header("Applicant Information")

age = st.sidebar.slider("Age", 18, 70, 30)

annual_income = st.sidebar.number_input(
    "Annual Income",
    min_value=10000,
    max_value=1000000,
    value=50000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=1000,
    max_value=500000,
    value=50000
)

bureau_score = st.sidebar.slider(
    "Bureau Score",
    300,
    900,
    650
)

total_debt = st.sidebar.number_input(
    "Total Debt",
    min_value=0,
    max_value=1000000,
    value=100000
)

delinquent_accounts = st.sidebar.slider(
    "Delinquent Accounts",
    0,
    10,
    0
)

num_defaults = st.sidebar.slider(
    "Previous Defaults",
    0,
    10,
    0
)

avg_dpd = st.sidebar.slider(
    "Average DPD",
    0,
    120,
    10
)

late_payments = st.sidebar.slider(
    "Late Payments",
    0,
    20,
    0
)

utilization_ratio = st.sidebar.slider(
    "Utilization Ratio",
    0.0,
    1.0,
    0.50
)

predict = st.sidebar.button("🚀 Predict Risk")

# ====================================
# TABS
# ====================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Risk Prediction",
    "📊 Model Performance",
    "📖 Project Details"
])

# ====================================
# TAB 1 - PREDICTION
# ====================================
with tab1:

    st.header("🎯 Credit Risk Assessment")

    st.info(
        "Enter applicant details in the sidebar and click Predict Risk."
    )

    if predict:

        input_dict = {}

        for col in features:
            input_dict[col] = 0

        input_dict["age"] = age
        input_dict["annual_income"] = annual_income
        input_dict["loan_amount"] = loan_amount
        input_dict["bureau_score"] = bureau_score
        input_dict["total_debt"] = total_debt
        input_dict["delinquent_accounts"] = delinquent_accounts
        input_dict["num_defaults"] = num_defaults
        input_dict["avg_dpd"] = avg_dpd
        input_dict["late_payments"] = late_payments
        input_dict["utilization_ratio"] = utilization_ratio

        # Engineered Features
        input_dict["debt_to_income"] = total_debt / max(annual_income, 1)
        input_dict["loan_to_income"] = loan_amount / max(annual_income, 1)

        input_df = pd.DataFrame([input_dict])

        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        probability = model.predict_proba(input_df)[0][1]
        risk_percent = probability * 100

        st.divider()

        st.subheader("Prediction Result")

        if risk_percent < 35:

            st.success(
                f"🟢 LOW RISK CUSTOMER ({risk_percent:.2f}%)"
            )

            st.success(
                "✅ Recommendation: APPROVE LOAN"
            )

        elif risk_percent < 65:

            st.warning(
                f"🟡 MEDIUM RISK CUSTOMER ({risk_percent:.2f}%)"
            )

            st.warning(
                "⚠️ Recommendation: MANUAL REVIEW"
            )

        else:

            st.error(
                f"🔴 HIGH RISK CUSTOMER ({risk_percent:.2f}%)"
            )

            st.error(
                "❌ Recommendation: REJECT LOAN"
            )

        st.progress(float(probability))

        st.metric(
            "Default Probability",
            f"{risk_percent:.2f}%"
        )

        # WHY THIS PREDICTION

        st.subheader("🔍 Why this Prediction?")

        reasons = []

        if num_defaults > 2:
            reasons.append("Multiple previous defaults")

        if delinquent_accounts > 2:
            reasons.append("High delinquent accounts")

        if avg_dpd > 30:
            reasons.append("High Average DPD")

        if late_payments > 5:
            reasons.append("Frequent late payments")

        if total_debt / annual_income > 0.5:
            reasons.append("High Debt-to-Income Ratio")

        if utilization_ratio > 0.7:
            reasons.append("High credit utilization")

        if len(reasons) > 0:
            for r in reasons:
                st.write("•", r)
        else:
            st.success(
                "No major risk indicators detected."
            )

# ====================================
# TAB 2 - MODEL PERFORMANCE
# ====================================
with tab2:

    st.header("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "94.7%")
    c2.metric("Precision", "83.0%")
    c3.metric("Recall", "85.5%")
    c4.metric("F1 Score", "84.2%")

    st.metric("ROC-AUC", "98.5%")

    st.success(
        "XGBoost achieved the best performance among all tested models."
    )

    importance_df = pd.DataFrame({
        "Feature": [
            "debt_to_income",
            "avg_dpd",
            "annual_income",
            "num_defaults",
            "delinquent_accounts",
            "total_debt",
            "utilization_ratio",
            "loan_to_income",
            "late_payments",
            "credit_used"
        ],
        "Importance": [
            0.148624,
            0.097943,
            0.084848,
            0.068556,
            0.062683,
            0.052307,
            0.047189,
            0.044495,
            0.037670,
            0.025368
        ]
    })

    st.subheader("Top Risk Factors")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

# ====================================
# TAB 3 - PROJECT DETAILS
# ====================================
with tab3:

    st.header("📖 Project Details")

    st.subheader("Project Objective")

    st.write("""
    Predict whether a customer is likely to default
    within the next 12 months.
    """)

    st.subheader("Machine Learning Pipeline")

    st.write("""
    • Data Collection & Integration

    • Exploratory Data Analysis

    • Feature Engineering

    • One-Hot Encoding

    • SMOTE Class Balancing

    • Logistic Regression

    • Random Forest

    • XGBoost
             
    • Hyperparameter Tuning (GridSearchCV)

    • Cross Validation

    • Feature Importance Analysis
    """)

    st.subheader("Key Risk Factors")

    st.write("""
    • Debt To Income Ratio

    • Average DPD

    • Annual Income

    • Previous Defaults

    • Delinquent Accounts
    """)
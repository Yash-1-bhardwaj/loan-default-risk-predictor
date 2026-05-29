import streamlit as st
import pandas as pd
import joblib
# Page Config
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💰",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.metric-card {
    background-color: black;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

.title {
    text-align: center;
    color: #1f4e79;
}

</style>
""", unsafe_allow_html=True)
# Load model
model = joblib.load("loan_default_model.pkl")

# Mappings
grade_mapping = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7
}

default_mapping = {
    'N': 0,
    'Y': 1
}

# Page Config
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💰",
    layout="wide"
)
# Title
st.markdown(
    """
    <h1 class='title'>
        💰 Loan Default Risk Predictor
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Predict whether a loan applicant is likely to default.</p>",
    unsafe_allow_html=True
)

st.divider()
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    income = st.number_input(
        "Annual Income",
        min_value=1000,
        value=50000
    )

    emp_length = st.number_input(
        "Employment Length",
        min_value=0,
        value=5
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=100,
        value=10000
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=12.0
    )

    credit_history_length = st.number_input(
        "Credit History Length",
        min_value=0,
        value=8
    )

st.divider()
home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

previous_default = st.selectbox(
    "Previous Default On File",
    ["N", "Y"]
)
# Predict Button
if st.button("Predict Risk"):

    loan_percent_income = loan_amount / income

    data = pd.DataFrame({
        "person_age": [age],
        "person_income": [income],
        "person_home_ownership": [home_ownership],
        "person_emp_length": [emp_length],
        "loan_intent": [loan_intent],
        "loan_grade": [grade_mapping[loan_grade]],
        "loan_amnt": [loan_amount],
        "loan_int_rate": [interest_rate],
        "loan_percent_income": [loan_percent_income],
        "cb_person_default_on_file": [
            default_mapping[previous_default]
        ],
        "cb_person_cred_hist_length": [
            credit_history_length
        ]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )
    st.metric(
    "Approval Chance",
    f"{(1-probability):.2%}"
)

    st.progress(float(probability))

    if probability < 0.20:
        risk_level = "🟢 Low Risk"

    elif probability < 0.50:
        risk_level = "🟡 Medium Risk"

    else:
        risk_level = "🔴 High Risk" 
    st.info(f"Risk Category: {risk_level}")
    st.write("---") 
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>Default Probability</h3>
            <h2>{probability:.2%}</h2>
        </div>
        """,
    unsafe_allow_html=True
    )
    st.markdown("---")

    st.caption(
        "Built using Random Forest, Scikit-Learn, Streamlit and SHAP"
    )
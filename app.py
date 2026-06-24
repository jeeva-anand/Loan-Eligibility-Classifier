
import streamlit as st
import pandas as pd
import joblib

 
# PAGE CONFIG
 

st.set_page_config(
    page_title="Loan Eligibility Classifier",
    page_icon="🏦",
    layout="wide"
)

 
# CUSTOM CSS
 

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:#0F172A;
    padding:15px;
    border-radius:15px;
    text-align:center;
    color:white;
}

.prediction-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

 
# LOAD MODEL
 

model = joblib.load("models/random_forest.pkl")

 
# HEADER
 

st.markdown("""
#  Loan Eligibility Classifier

Predict loan approval using machine learning and applicant financial information.
""")

# st.divider()

 
 
# INPUT SECTION
 

st.subheader(" Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

with col2:

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=2000
    )

with col3:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=150
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1,
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

 
# FEATURE ENGINEERING
 

total_income = applicant_income + coapplicant_income

emi = loan_amount / loan_term

balance_income = total_income - (emi * 1000)

st.divider()

 
# FINANCIAL SUMMARY
 

st.subheader("User's Financial Summary")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Total Income",
        f"${total_income:,.0f}"
    )

with m2:
    st.metric(
        "Estimated EMI",
        f"${emi:,.2f}"
    )

with m3:
    st.metric(
        "Balance Income",
        f"${balance_income:,.2f}"
    )



 
# DATAFRAME
 

input_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "TotalIncome": total_income,
    "EMI": emi,
    "BalanceIncome": balance_income
}])

input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

 
# PREDICTION
 

if st.button(
    " Submit",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader(" Prediction Result")

    if prediction == "Y":

        st.markdown(f"""
        <div class="prediction-box"
        style="background:#DCFCE7;color:#166534;">
         LOAN APPROVED
        <br><br>
        Confidence: {probability:.2%}
        </div>
        """,
                    unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="prediction-box"
        style="background:#FEE2E2;color:#991B1B;">
         LOAN REJECTED
        <br><br>
        Confidence: {probability:.2%}
        </div>
        """,
                    unsafe_allow_html=True)

    st.progress(float(probability))

    st.caption(
        f"Approval Probability: {probability:.2%}"
    )




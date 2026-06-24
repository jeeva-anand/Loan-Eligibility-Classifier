import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("models/random_forest.pkl")


# ----------------------------
# UI Header
# ----------------------------
st.title("🏦 Loan Eligibility Prediction App")
st.markdown(
    "Predict whether a customer is eligible for a loan using Machine Learning")

st.divider()

# ----------------------------
# Input Section
# ----------------------------
st.subheader("📋 Enter Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", value=2000)

with col3:
    loan_amount = st.number_input("Loan Amount", value=150)
    loan_term = st.number_input("Loan Term (in days)", value=360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

st.divider()

# ----------------------------
# Feature Engineering (same logic as training)
# ----------------------------
total_income = applicant_income + coapplicant_income
emi = loan_amount / loan_term
balance_income = total_income - (emi * 1000)

# ----------------------------
# Create Input DataFrame
# ----------------------------
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

# ----------------------------
# Encoding (must match training)
# ----------------------------
input_data = pd.get_dummies(input_data)

# Align with model features
model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Loan Eligibility"):

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == "Y":
        st.success(f" Loan Approved (Confidence: {prob:.2f})")
    else:
        st.error(f" Loan Rejected (Confidence: {prob:.2f})")

st.divider()

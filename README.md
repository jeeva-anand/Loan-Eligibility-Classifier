
# 🏦 Loan Eligibility Prediction System

### *End-to-End Machine Learning System for Automated Loan Approval*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-red?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-RandomForest%20%7C%20LogisticRegression-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-ff4b4b?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

---

## 🚀 Overview

Loan approval is a critical decision-making process in banking systems. Traditional methods rely heavily on manual review, which is:

* ⛔ Time-consuming
* ⛔ Prone to human bias
* ⛔ Not scalable for large volumes

This project builds an **end-to-end machine learning system** that predicts whether a loan application should be approved based on applicant financial and demographic information.

---

## 🎯 Objective

To develop a **data-driven loan eligibility prediction system** that:

* Automates loan approval decisions
* Reduces manual intervention
* Improves decision consistency
* Provides probabilistic confidence for predictions

---

## 📊 Dataset Description

The dataset contains historical loan application records with attributes such as:

* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* Gender, Marital Status, Education
* Property Area
* Loan Status (Target Variable)

---

## 🧠 Problem Type

> Binary Classification (Supervised Learning)

Target:

* `Y` → Loan Approved
* `N` → Loan Rejected

---

## 🏗️ Project Architecture

```text
Raw Data
   ↓
Data Cleaning & Missing Value Handling
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering (Income, EMI, Balance Income)
   ↓
Encoding & Transformation
   ↓
Model Training (Logistic Regression, Random Forest)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Model Evaluation (CV, ROC-AUC, Accuracy)
   ↓
Model Serialization (Joblib)
   ↓
Streamlit Deployment
```

---

## 📌 Key Features

### 🔹 Data Processing

* Missing value imputation (mode & median strategy)
* Outlier detection using IQR method
* Log transformation for skewed variables

### 🔹 Feature Engineering

* Total Income = Applicant + Coapplicant Income
* EMI calculation
* Balance Income feature
* Log transformations for normalization

### 🔹 Machine Learning Models

* Logistic Regression (Baseline Model)
* Random Forest Classifier (Final Model)

### 🔹 Model Optimization

* Stratified K-Fold Cross Validation
* GridSearchCV for hyperparameter tuning
* Feature importance analysis

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~78–82%  |
| Random Forest       | ~82–86%  |

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Curve

---

## 🧪 Feature Importance

The most influential features in loan approval prediction:

* Credit History ⭐ (Most important)
* Applicant Income
* Loan Amount
* EMI / Financial burden
* Property Area

---

## 🖥️ Streamlit Web App

A fully interactive ML application built using Streamlit.

### Features:

* 🧾 Manual input form for loan prediction
* 📂 Batch CSV upload for bulk predictions
* 📊 Real-time prediction with probability score
* 📉 Interactive visual feedback
* ⬇ Download prediction results

### Run Locally:

```bash
git clone https://github.com/yourusername/Loan-Eligibility-Prediction.git
cd Loan-Eligibility-Prediction
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 🧠 Business Impact

This system can be used by financial institutions to:

* ⚡ Speed up loan approval process
* 📉 Reduce operational cost
* 📊 Standardize decision-making
* 🧾 Improve risk assessment
* 🤖 Enable AI-driven lending workflows

---

## 🏗️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn / Plotly
* Streamlit
* Joblib

---

## 📁 Project Structure

```text
Loan-Eligibility-Prediction/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app/
├── reports/
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Use

### 1. Train Model

```bash
python train.py
```

### 2. Run App

```bash
streamlit run app/streamlit_app.py
```

### 3. Predict

* Enter applicant details OR upload CSV
* Get instant loan approval prediction

---

## 📊 Sample Output

* Loan Approved: ✅ Probability = 0.87
* Loan Rejected: ❌ Probability = 0.32

---

## 🔍 Key Insights from Data

* Credit history is the strongest predictor of loan approval
* Higher income does not guarantee approval
* Balanced income (after EMI) is a strong risk indicator
* Education and property area influence loan decisions

---

## 📌 Future Improvements

* 🧠 XGBoost / LightGBM model comparison
* 📊 SHAP explainability for predictions
* ☁️ Cloud deployment (AWS / Streamlit Cloud)
* 🔐 Authentication system for banking users
* 📡 API integration for real-world banking systems

---

## 👨‍💻 Author

**Jeeva Anand**
Machine Learning & Data Science Enthusiast

---

## ⭐ If You Like This Project

If you found this useful:

* ⭐ Star the repository
* 🍴 Fork and improve it
* 📢 Share with others

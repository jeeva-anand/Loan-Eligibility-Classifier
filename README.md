
#  Loan Eligibility Prediction System

### *End-to-End Machine Learning System for Automated Loan Approval*

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-red?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-RandomForest%20%7C%20LogisticRegression-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-ff4b4b?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

---

##  Overview

Loan approval is a critical decision-making process in banking systems. Traditional methods rely heavily on manual review, which is:

*  Time-consuming
*  Prone to human bias
*  Not scalable for large volumes

This project builds an **end-to-end machine learning system** that predicts whether a loan application should be approved based on applicant financial and demographic information.

---

##  Objective

To develop a **data-driven loan eligibility prediction system** that:

* Automates loan approval decisions
* Reduces manual intervention
* Improves decision consistency
* Provides probabilistic confidence for predictions

---

##  Dataset Description

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

##  Problem Type

> Binary Classification (Supervised Learning)

Target:

* `Y` → Loan Approved
* `N` → Loan Rejected

---

##  Project Architecture

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

##  Key Features

###  Data Processing

* Missing value imputation (mode & median strategy)
* Outlier detection using IQR method
* Log transformation for skewed variables

###  Feature Engineering

* Total Income = Applicant + Coapplicant Income
* EMI calculation
* Balance Income feature
* Log transformations for normalization

###  Machine Learning Models

* Logistic Regression (Baseline Model)
* Random Forest Classifier (Final Model)

###  Model Optimization

* Stratified K-Fold Cross Validation
* GridSearchCV for hyperparameter tuning
* Feature importance analysis

---

##  Model Performance

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

##  Feature Importance

The most influential features in loan approval prediction:

* Credit History 
* Applicant Income
* Loan Amount
* EMI / Financial burden
* Property Area




##  Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn / Plotly
* Streamlit
* Joblib


##  Key Insights from Data

* Credit history is the strongest predictor of loan approval
* Higher income does not guarantee approval
* Balanced income (after EMI) is a strong risk indicator
* Education and property area influence loan decisions



##  Future Improvements

*  XGBoost / LightGBM model comparison
*  SHAP explainability for predictions
*  Cloud deployment (AWS / Streamlit Cloud)
*  Authentication system for banking users
*  API integration for real-world banking systems

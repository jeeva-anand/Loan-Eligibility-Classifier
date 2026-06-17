import numpy as np


def preprocess(df):

    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    df["Loan_Amount_Term"].fillna(
        df["Loan_Amount_Term"].mode()[0],
        inplace=True
    )

    df["LoanAmount"].fillna(
        df["LoanAmount"].median(),
        inplace=True
    )

    df["LoanAmount_log"] = np.log(df["LoanAmount"])

    return df


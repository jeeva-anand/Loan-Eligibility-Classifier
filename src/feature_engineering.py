import numpy as np


def feature_engineering(df):

    df["TotalIncome"] = (
        df["ApplicantIncome"] + df["CoapplicantIncome"]
    )

    df["TotalIncome_log"] = np.log(df["TotalIncome"])

    df["EMI"] = (
        df["LoanAmount"] /
        df["Loan_Amount_Term"]
    )

    df["BalanceIncome"] = (
        df["TotalIncome"] -
        df["EMI"] * 1000
    )

    return df


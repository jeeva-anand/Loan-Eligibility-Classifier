from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def logistic_model():

    return LogisticRegression(
        random_state=42,
        max_iter=1000
    )


def random_forest():

    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            pred,
            pos_label="Y"
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            pred,
            pos_label="Y"
        )
    )

    print(
        "F1:",
        f1_score(
            y_test,
            pred,
            pos_label="Y"
        )
    )

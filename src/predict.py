import joblib


def load_model(path):

    return joblib.load(path)


def predict(model, X):

    return model.predict(X)

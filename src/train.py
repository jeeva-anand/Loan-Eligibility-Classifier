from data_loader import load_data
from preprocess import preprocess
from feature_engineering import feature_engineering
from model import random_forest
from utils import save_model
import pandas as pd

from sklearn.model_selection import train_test_split

train = pd.read_csv("../data/raw/train_ctrUa4K.csv")
test = pd.read_csv("../data/raw/test_lAUu6dG.csv")

train, test = preprocess(train,test)

trained_df = feature_engineering(train)

y = trained_df["Loan_Status"]

X = trained_df.drop(
    ["Loan_Status", "Loan_ID"],
    axis=1
)

X = pd.get_dummies(X, dtype='int64')

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = random_forest()

model.fit(xtrain, ytrain)

save_model(
    model,
    "../models/random_forest.pkl"
)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

INPUT = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT = "data/processed"

os.makedirs(
    OUTPUT,
    exist_ok=True
)

print("Loading dataset...")

df = pd.read_csv(INPUT)

df = df.drop(
    columns=["customerID"]
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.fillna(0)

for col in df.columns:

    if df[col].dtype == "object":

        encoder = LabelEncoder()

        df[col] = (
            encoder.fit_transform(
                df[col]
            )
        )

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

train = X_train.copy()
train["Churn"] = y_train

test = X_test.copy()
test["Churn"] = y_test

train.to_csv(
    "data/processed/train.csv",
    index=False
)

test.to_csv(
    "data/processed/test.csv",
    index=False
)

print("Preprocessing completed")

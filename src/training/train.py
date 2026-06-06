import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# MLflow backend
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "customer_churn"
)

print("Loading training data...")

train = pd.read_csv(
    "data/processed/train.csv"
)

X = train.drop(
    columns=["Churn"]
)

y = train["Churn"]

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    mlflow.log_param(
        "algorithm",
        "RandomForest"
    )

    mlflow.log_param(
        "trees",
        100
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    joblib.dump(
        model,
        "model.pkl"
    )

print("\nTraining Completed")
print("Accuracy:", round(accuracy, 4))

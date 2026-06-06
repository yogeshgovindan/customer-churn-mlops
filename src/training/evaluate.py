import pandas as pd
import joblib
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# MLflow
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "customer_churn"
)

print("Loading test data...")

test = pd.read_csv(
    "data/processed/test.csv"
)

X = test.drop(
    columns=["Churn"]
)

y = test["Churn"]

print("Loading model...")

model = joblib.load(
    "model.pkl"
)

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

print("\nMetrics")
print("test_accuracy :", accuracy)
print("test_precision:", precision)
print("test_recall   :", recall)
print("test_f1       :", f1)

with mlflow.start_run(
    run_name="model_evaluation"
):

    mlflow.log_metric(
        "test_accuracy",
        float(accuracy)
    )

    mlflow.log_metric(
        "test_precision",
        float(precision)
    )

    mlflow.log_metric(
        "test_recall",
        float(recall)
    )

    mlflow.log_metric(
        "test_f1",
        float(f1)
    )

print("\nLogged to MLflow")

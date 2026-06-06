from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

print("Loading model...")

model = joblib.load(
    "model.pkl"
)


@app.get("/")
def home():

    return {
        "message": "Customer Churn API Running"
    }


@app.post("/predict")
def predict(
    data: dict
):

    df = pd.DataFrame(
        [data]
    )

    prediction = (
        model.predict(df)
    )

    probability = (
        model.predict_proba(df)
    )

    return {

        "prediction":
        int(
            prediction[0]
        ),

        "probability":
        float(
            probability[0][1]
        )
    }

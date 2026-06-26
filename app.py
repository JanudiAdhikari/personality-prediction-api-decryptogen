from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "personality_model.joblib"

app = FastAPI(
    title="Personality Prediction API",
    description="API to predict whether a person is an Introvert or Extrovert.",
    version="1.0"
)

# Load the trained model once when the app starts
model = joblib.load(MODEL_PATH)


class PersonalityInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float


@app.get("/")
def home():
    return {
        "message": "Personality Prediction API is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_personality(data: PersonalityInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    probability_dict = {
        classes[i]: round(float(probabilities[i]), 4)
        for i in range(len(classes))
    }

    return {
        "prediction": prediction,
        "probabilities": probability_dict
    }
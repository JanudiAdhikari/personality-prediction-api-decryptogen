# Personality Prediction API

This project predicts whether a person is an Introvert or Extrovert based on personality-related characteristics using a machine learning classification model.

## Project Overview

The goal of this task is to build and deploy a machine learning model that accepts personality-related input data through a public API and returns the predicted personality type.

The model predicts one of the following classes:

- Introvert
- Extrovert

## Features Completed

- Loaded and explored the provided dataset
- Checked dataset structure and missing values
- Handled numerical and categorical features
- Applied preprocessing using Scikit-learn pipelines
- Trained a Gradient Boosting classification model
- Evaluated the model using accuracy and classification report
- Saved the trained model using Joblib
- Built a FastAPI application for prediction
- Created a POST `/predict` endpoint
- Deployed the API on Render
- Tested the deployed API using Swagger UI

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Render

## Project Structure

```text
personality-prediction-api-decryptogen/
├── __pycache__/
├── venv/
├── .gitignore
├── .python-version
├── app.py
├── personality_dataset.csv
├── personality_model.joblib
├── README.md
├── requirements.txt
├── train_model.ipynb
└── train_model.py
```

## Input Format

The API accepts a POST request with the following JSON format:

```json
{
  "Time_spent_Alone": 4,
  "Stage_fear": "No",
  "Social_event_attendance": 6,
  "Going_outside": 5,
  "Drained_after_socializing": "No",
  "Friends_circle_size": 10,
  "Post_frequency": 5
}
```

## API Endpoint

```http
POST /predict
```

## Live Demo

Public API URL:

```text
https://personality-prediction-api-decryptogen.onrender.com/predict
```

API documentation:

```text
https://personality-prediction-api-decryptogen.onrender.com/docs
```

## Example Request

```bash
curl -X POST https://personality-prediction-api-decryptogen.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Time_spent_Alone":4,"Stage_fear":"No","Social_event_attendance":6,"Going_outside":5,"Drained_after_socializing":"No","Friends_circle_size":10,"Post_frequency":5}'
```

## Example Response

```json
{
  "prediction": "Extrovert",
  "probabilities": {
    "Extrovert": 0.9527,
    "Introvert": 0.0473
  }
}
```

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the API locally:

```bash
uvicorn app:app --reload
```

Open the local API documentation:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The API is deployed on Render.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Notebook

The notebook `train_model.ipynb` contains the data preparation, preprocessing, model training, and evaluation steps used for this project.

## Notes

The deployed API may take a few seconds to respond if the Render free service is inactive and needs to restart.


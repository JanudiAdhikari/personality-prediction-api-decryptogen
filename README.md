# Personality Prediction API

This project predicts whether a person is an Introvert or Extrovert based on personality-related characteristics.

## Features

- Data preprocessing
- Machine learning model training
- Gradient Boosting classifier
- FastAPI prediction endpoint
- Public API deployment support

## Input Format

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

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the API:

```bash
uvicorn app:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## Deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

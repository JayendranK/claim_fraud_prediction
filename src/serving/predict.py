# src/serving/predict.py
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import mlflow
from .schemas import InputData, PredictionResponse
import mlflow.pyfunc
import os

app = FastAPI(title="Fraud Detection API", version="1.0")

# Config
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
MLFLOW_TRACKING_URI = "file:./mlruns"
MODEL_NAME = "claim-fraud-detection-best"  # if using registry
MODEL_STAGE = None

# Try loading from MLflow first
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    print(f"[INFO] Loading model from MLflow: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"⚠ Could not load model from MLflow: {e}")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from local path: {MODEL_PATH}")
    except FileNotFoundError:
        model = None
        print(f"❌ Model not found locally at {MODEL_PATH}")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_df = pd.DataFrame([data.dict()])
    pred = model.predict(input_df)[0]
    if isinstance(pred, (int, float)):
        prediction_label = "Yes" if pred == 1 else "No"
    elif isinstance(pred, str):
        prediction_label = pred
    print(f"[INFO] Prediction made: {prediction_label}")

    return PredictionResponse(prediction=prediction_label)
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "champion_model.joblib"

app = FastAPI(
    title="Real-Time ML Inference API",
    description="A small FastAPI service for serving a trained scikit-learn classifier.",
    version="1.0.0",
)

model = None

class PredictRequest(BaseModel):
    # These feature names and types must match the model used at training time.
    features: dict[str, Any] = Field(
        ...,
        examples=[{"feature_01": 0.2, "feature_02": -1.1}]
    )

class PredictResponse(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None

@app.on_event("startup")
def load_model() -> None:
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {
        "service": "Real-Time ML Inference API",
        "status": "ok",
        "docs": "/docs",
        "model_loaded": model is not None,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model found. Place a compatible model at model/champion_model.joblib."
        )
    try:
        import pandas as pd
        row = pd.DataFrame([request.features])
        prediction = model.predict(row)[0]
        probs = None
        if hasattr(model, "predict_proba"):
            raw_probs = model.predict_proba(row)[0]
            classes = getattr(model, "classes_", range(len(raw_probs)))
            probs = {str(label): float(prob) for label, prob in zip(classes, raw_probs)}
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        return PredictResponse(prediction=prediction, probabilities=probs)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not score input. Check feature names/types. Details: {exc}")

@app.get("/model-info")
def model_info():
    return {
        "model_loaded": model is not None,
        "expected_input": "JSON object with a 'features' dictionary matching the training feature names",
        "model_path": "model/champion_model.joblib",
    }

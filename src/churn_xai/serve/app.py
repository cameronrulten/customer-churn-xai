import os

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


class InputRow(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    # ... add fields as needed

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(x: InputRow):
    assert MODEL is not None, "Model not loaded"
    # transform x → vector using saved preprocessor
    # proba = MODEL.predict_proba([vec])[0,1]
    proba = 0.42
    return {"churn_probability": proba}

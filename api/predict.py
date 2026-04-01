from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

router = APIRouter()
model = joblib.load('models/xgboost/xgboost_model.pkl')

class Transaction(BaseModel):
    features: list[float]

@router.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Expected exactly 30 features")
    X = np.array(transaction.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    return {
        "fraud_probability": round(float(proba), 4),
        "prediction": "FRAUD" if proba > 0.3 else "NORMAL"
    }
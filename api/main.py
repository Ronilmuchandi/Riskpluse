from fastapi import FastAPI
from api.predict import router as predict_router
from api.drift import router as drift_router
from api.alerts import router as alerts_router

app = FastAPI(
    title="RiskPulse API",
    description="Fraud detection and risk monitoring API",
    version="1.0.0"
)

app.include_router(predict_router)
app.include_router(drift_router)
app.include_router(alerts_router)

@app.get("/")
def root():
    return {"message": "RiskPulse API is running"}
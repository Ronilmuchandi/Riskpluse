from fastapi import APIRouter
import json
import os

router = APIRouter()

@router.get("/alerts")
def alerts():
    path = 'data/processed/alerts.json'
    if not os.path.exists(path):
        return {"alerts": []}
    with open(path, 'r') as f:
        return {"alerts": json.load(f)}
from fastapi import APIRouter
import numpy as np
import pandas as pd
from monitoring.drift_detector import detect_drift

router = APIRouter()

@router.get("/drift")
def drift():
    df = pd.read_csv('data/raw/creditcard.csv')
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['Time'] = (df['Time'] - df['Time'].mean()) / df['Time'].std()
    X = df.drop(columns=['Class']).values
    split = int(len(X) * 0.8)
    report = detect_drift(X[:split], X[split:])
    return report
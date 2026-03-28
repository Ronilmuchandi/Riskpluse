# RiskPulse — Intelligent Financial Risk & Fraud Intelligence Platform

A production-style end-to-end machine learning system for fraud detection and risk monitoring, built with Python, XGBoost, TensorFlow/Keras, PyTorch, FastAPI, and deployed on AWS EC2 via Docker.

## Project Structure
```
riskpulse/
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/        # Cleaned and preprocessed data
├── models/
│   ├── xgboost/          # XGBoost fraud classifier
│   ├── autoencoder/      # TensorFlow/Keras anomaly detection
│   └── pytorch/          # PyTorch neural network
├── api/                  # FastAPI REST endpoints
├── monitoring/           # Drift detection and alerts
├── dashboard/            # Streamlit dashboard
├── notebooks/            # EDA and modeling notebooks
└── tests/                # Unit tests
```

## Parts

- **Part A** — Fraud Detection Engine: XGBoost, TensorFlow/Keras Autoencoder, PyTorch NN, A/B Testing, KS Hypothesis Test
- **Part B** — Data Drift Monitoring: Feature drift detection, accuracy tracking, automated alerts
- **Part C** — Deployment: FastAPI, Streamlit, Docker, AWS S3 + EC2

## Dataset
Credit Card Fraud Detection — Kaggle (284,807 transactions, highly imbalanced)

## Setup
```bash
git clone https://github.com/Ronilmuchandi/riskpulse.git
cd riskpulse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Author
Ronil Muchandi | MS Data Science & Analytics | University of Missouri Columbia

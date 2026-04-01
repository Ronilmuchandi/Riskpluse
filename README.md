# 🛡️ RiskPulse — Intelligent Financial Risk & Fraud Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.9-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red) ![FastAPI](https://img.shields.io/badge/FastAPI-0.103-teal) ![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-yellow) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

A production-style end-to-end fraud detection and risk monitoring platform built for financial services. Covers the full ML lifecycle — raw data ingestion, model training, statistical validation, drift monitoring, REST API, and cloud deployment.

---

## Dashboard Preview

![Dashboard](docs/Screenshot%202026-03-31%20at%207.07.24%20PM.png)

---

## Key Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| XGBoost (primary) | **0.9717** | **0.8214** |
| PyTorch Neural Network | 0.9706 | 0.7608 |
| TF/Keras Autoencoder | 0.9595 | 0.5159 |

KS Hypothesis Test confirmed all three models produce statistically significantly different fraud score distributions (p < 0.0001).

---

## Architecture

**Part A — Fraud Detection Engine**
- XGBoost classifier — primary production model
- TensorFlow/Keras Autoencoder — unsupervised anomaly detection
- PyTorch Neural Network — comparison model
- A/B Testing for model performance comparison
- KS Hypothesis Testing for statistical validation
- SMOTE oversampling to handle extreme class imbalance (0.17% fraud rate)

**Part B — Data Drift Monitoring**
- Feature drift detection using KS statistic
- Automated alert system with HIGH/MEDIUM severity
- Simulated real-world drift on live transaction data

**Part C — Deployment**
- FastAPI REST API with /predict, /drift, /alerts endpoints
- Streamlit dashboard with dark UI, live predictions, drift charts
- Docker containerization
- AWS S3 for dataset storage
- AWS EC2 for cloud deployment

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Returns fraud probability for a transaction |
| `/drift` | GET | Runs KS-based feature drift analysis |
| `/alerts` | GET | Returns active system alerts |

---

## Tech Stack

`Python 3.9` `XGBoost` `TensorFlow/Keras` `PyTorch` `Scikit-learn` `SMOTE` `SciPy` `FastAPI` `Uvicorn` `Streamlit` `Docker` `AWS S3` `AWS EC2` `Pandas` `NumPy` `Matplotlib` `Seaborn`

---

## Dataset

Kaggle Credit Card Fraud Detection — 284,807 real anonymized transactions, only 0.17% fraud. Class imbalance handled using SMOTE oversampling applied exclusively on training data.

---

## Run Locally
```bash
git clone https://github.com/Ronilmuchandi/Riskpluse.git
cd Riskpluse
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download dataset from Kaggle and place at `data/raw/creditcard.csv`, then:
```bash
python3 notebooks/02_preprocessing.py
python3 models/xgboost/train_xgboost.py
python3 models/autoencoder/train_autoencoder.py
python3 models/pytorch/train_pytorch.py

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run dashboard/app.py
```

Open `http://localhost:8501`

---

## Author

**Ronil Muchandi**
MS Data Science & Analytics | University of Missouri Columbia | GPA 3.5
[LinkedIn](https://linkedin.com/in/ronil-muchandi-892602187) · [GitHub](https://github.com/Ronilmuchandi)
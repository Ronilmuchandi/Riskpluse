import numpy as np
import joblib
import torch
import torch.nn as nn
from tensorflow import keras
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

print("Loading data and models...")
X_test = np.load('data/processed/X_test.npy').astype(np.float32)
y_test = np.load('data/processed/y_test.npy')

# Load XGBoost
xgb_model = joblib.load('models/xgboost/xgboost_model.pkl')
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Load Autoencoder
autoencoder = keras.models.load_model('models/autoencoder/autoencoder_model.h5')
reconstructions = autoencoder.predict(X_test)
ae_scores = np.mean(np.power(X_test - reconstructions, 2), axis=1)
# Normalize to 0-1 range for fair comparison
ae_proba = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())

# Load PyTorch
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

pt_model = FraudDetector(X_test.shape[1])
pt_model.load_state_dict(torch.load('models/pytorch/pytorch_model.pt'))
pt_model.eval()
with torch.no_grad():
    pt_proba = pt_model(torch.tensor(X_test)).squeeze().numpy()

# A/B Testing — compare ROC-AUC and Average Precision
print("\n===== Model Comparison Results =====")
models = {'XGBoost': xgb_proba, 'Autoencoder': ae_proba, 'PyTorch': pt_proba}
results = {}
for name, proba in models.items():
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    results[name] = {'ROC-AUC': auc, 'Avg Precision': ap}
    print(f"{name:12} — ROC-AUC: {auc:.4f} | Avg Precision: {ap:.4f}")

# KS Hypothesis Test — do fraud score distributions differ significantly?
print("\n===== KS Hypothesis Test =====")
print("Testing if fraud score distributions differ between models")
pairs = [
    ('XGBoost', 'PyTorch', xgb_proba, pt_proba),
    ('XGBoost', 'Autoencoder', xgb_proba, ae_proba),
    ('PyTorch', 'Autoencoder', pt_proba, ae_proba)
]
for name1, name2, p1, p2 in pairs:
    ks_stat, p_value = stats.ks_2samp(p1, p2)
    conclusion = "Significantly different" if p_value < 0.05 else "Not significantly different"
    print(f"{name1} vs {name2}: KS={ks_stat:.4f}, p={p_value:.4f} → {conclusion}")

# Bar chart comparing models
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results))
width = 0.35
aucs = [results[m]['ROC-AUC'] for m in results]
aps = [results[m]['Avg Precision'] for m in results]
ax.bar(x - width/2, aucs, width, label='ROC-AUC', color='steelblue')
ax.bar(x + width/2, aps, width, label='Avg Precision', color='crimson')
ax.set_xticks(x)
ax.set_xticklabels(results.keys())
ax.set_ylim(0, 1.1)
ax.set_title('Model Comparison — A/B Testing')
ax.legend()
plt.tight_layout()
plt.savefig('data/processed/model_comparison.png')
print("\nModel comparison chart saved.")
print("\nA/B testing and hypothesis testing complete.")

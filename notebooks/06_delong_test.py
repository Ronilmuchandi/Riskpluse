import numpy as np
import joblib
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score

def delong_test(y_true, y_pred1, y_pred2):
    n1 = int(y_true.sum())
    n2 = int(len(y_true) - n1)

    pos1 = y_pred1[y_true == 1]
    neg1 = y_pred1[y_true == 0]
    pos2 = y_pred2[y_true == 1]
    neg2 = y_pred2[y_true == 0]

    def auc_var(pos, neg):
        m, n = len(pos), len(neg)
        v10 = np.array([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])
        v01 = np.array([(n_i < pos).mean() + 0.5 * (n_i == pos).mean() for n_i in neg])
        s10 = np.var(v10, ddof=1) / m
        s01 = np.var(v01, ddof=1) / n
        return s10 + s01

    auc1 = np.mean([(( p > neg1).mean() + 0.5 * (p == neg1).mean()) for p in pos1])
    auc2 = np.mean([((p > neg2).mean() + 0.5 * (p == neg2).mean()) for p in pos2])

    var1 = auc_var(pos1, neg1)
    var2 = auc_var(pos2, neg2)

    v10_1 = np.array([(p > neg1).mean() + 0.5*(p == neg1).mean() for p in pos1])
    v10_2 = np.array([(p > neg2).mean() + 0.5*(p == neg2).mean() for p in pos2])
    v01_1 = np.array([(n_i < pos1).mean() + 0.5*(n_i == pos1).mean() for n_i in neg1])
    v01_2 = np.array([(n_i < pos2).mean() + 0.5*(n_i == pos2).mean() for n_i in neg2])

    cov = (np.cov(v10_1, v10_2)[0,1] / n1) + (np.cov(v01_1, v01_2)[0,1] / n2)
    var_diff = var1 + var2 - 2 * cov

    if var_diff <= 0:
        return auc1, auc2, 0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p

print("Loading test data...")
X_test = np.load('data/processed/X_test_raw.npy').astype(np.float32)
y_test = np.load('data/processed/y_test_raw.npy')
print(f"Test set: {X_test.shape}, Fraud cases: {int(y_test.sum())}")

# XGBoost
xgb = joblib.load('models/xgboost/xgboost_model.pkl')
xgb_proba = xgb.predict_proba(X_test)[:, 1]
print(f"XGBoost ROC-AUC (sklearn): {roc_auc_score(y_test, xgb_proba):.4f}")
print(f"XGBoost proba range: {xgb_proba.min():.4f} to {xgb_proba.max():.4f}")

# PyTorch
class FraudDetector(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

pt_model = FraudDetector(X_test.shape[1])
pt_model.load_state_dict(torch.load('models/pytorch/pytorch_model.pt'))
pt_model.eval()
with torch.no_grad():
    pt_proba = pt_model(torch.tensor(X_test)).squeeze().numpy()
print(f"PyTorch ROC-AUC (sklearn): {roc_auc_score(y_test, pt_proba):.4f}")
print(f"PyTorch proba range: {pt_proba.min():.4f} to {pt_proba.max():.4f}")

print("\n===== DeLong Test — XGBoost vs PyTorch =====")
auc1, auc2, z, p = delong_test(y_test, xgb_proba, pt_proba)
print(f"XGBoost AUC:  {auc1:.4f}")
print(f"PyTorch AUC:  {auc2:.4f}")
print(f"Z-statistic:  {z:.4f}")
print(f"P-value:      {p:.4f}")
if p < 0.05:
    print("Verdict: Difference is statistically significant")
else:
    print("Verdict: Difference is NOT statistically significant — models are equivalent")
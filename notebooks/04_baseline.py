import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, fbeta_score
from sklearn.metrics import fbeta_score

print("Loading test data...")
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

df_test = pd.DataFrame(X_test)

# Amount is the last column (scaled)
amount_col = df_test.iloc[:, -1]

print(f"Test set: {len(y_test)} transactions, {int(y_test.sum())} fraud cases")
print(f"Fraud rate: {y_test.mean():.4%}")

# Baseline 1 — flag top 0.17% by amount as fraud
threshold_pct = np.percentile(amount_col, 99.83)
y_baseline_amount = (amount_col >= threshold_pct).astype(int)

print("\n--- Baseline: Flag top 0.17% by transaction amount ---")
print(classification_report(y_test, y_baseline_amount, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, amount_col):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, amount_col):.4f}")
print(f"F2-Score: {fbeta_score(y_test, y_baseline_amount, beta=2):.4f}")

# Baseline 2 — flag everything as normal (majority class)
y_all_normal = np.zeros(len(y_test))
print("\n--- Baseline: Predict everything as normal ---")
print(f"Accuracy: {(y_all_normal == y_test).mean():.4%}")
print(f"Fraud recall: 0.0000 (catches no fraud)")
print("This is why accuracy is the wrong metric for imbalanced data.")

print("\nBaseline complete.")

# =============================================================
# RiskPulse — XGBoost Fraud Classifier
# Author: Ronil Muchandi
# Description: Trains an XGBoost binary classifier on the
#              balanced fraud dataset and evaluates performance
# =============================================================

import numpy as np
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score
)

# ── 1. Load preprocessed data ─────────────────────────────────
print("Loading preprocessed data...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_test  = np.load('data/processed/X_test.npy')
y_test  = np.load('data/processed/y_test.npy')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 2. Train XGBoost model ────────────────────────────────────
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=100,       # number of trees
    max_depth=6,            # tree depth
    learning_rate=0.1,      # step size shrinkage
    scale_pos_weight=1,     # balanced via SMOTE already
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
model.fit(X_train, y_train)
print("Training complete.")

# ── 3. Evaluate model ─────────────────────────────────────────
print("\nEvaluating model...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
print(f"ROC-AUC Score:          {roc_auc:.4f}")
print(f"Average Precision Score: {avg_precision:.4f}")

# ── 4. Confusion matrix plot ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.yticks([0, 1], ['Normal', 'Fraud'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig('data/processed/xgboost_confusion_matrix.png')
print("Confusion matrix saved.")

# ── 5. ROC curve plot ─────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.4f})', color='steelblue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — XGBoost')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/xgboost_roc_curve.png')
print("ROC curve saved.")

# ── 6. Save model ─────────────────────────────────────────────
joblib.dump(model, 'models/xgboost/xgboost_model.pkl')
print("\nModel saved to models/xgboost/xgboost_model.pkl")
print("\nXGBoost training complete.")

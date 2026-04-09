import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score, fbeta_score
)

print("Loading data...")
df = pd.read_csv('data/raw/creditcard.csv')
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['Time'] = (df['Time'] - df['Time'].mean()) / df['Time'].std()
df = df.sort_values('Time').reset_index(drop=True)

split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df.drop(columns=['Class']).values
y_train = train_df['Class'].values
X_test = test_df.drop(columns=['Class']).values
y_test = test_df['Class'].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4%}")
print(f"Test fraud rate: {y_test.mean():.4%}")

scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
print(f"\nscale_pos_weight: {scale_pos_weight}")

print("\nTraining XGBoost with class weights...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
f2 = fbeta_score(y_test, y_pred, beta=2)

print(f"ROC-AUC Score:              {roc_auc:.4f}")
print(f"Average Precision Score:    {avg_precision:.4f}")
print(f"F2-Score (recall weighted): {f2:.4f}")

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

joblib.dump(model, 'models/xgboost/xgboost_model.pkl')
print("\nModel saved.")
print("XGBoost training complete.")
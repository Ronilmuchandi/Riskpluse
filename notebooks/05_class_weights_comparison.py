import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score

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

fraud_count = y_train.sum()
normal_count = len(y_train) - fraud_count
scale = normal_count / fraud_count
print(f"Class weight scale_pos_weight: {scale:.2f}")

print("\nTraining XGBoost with class weights (no SMOTE)...")
model_cw = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model_cw.fit(X_train, y_train)

y_pred_cw = model_cw.predict(X_test)
y_proba_cw = model_cw.predict_proba(X_test)[:, 1]

print("\n--- Class Weights Results ---")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba_cw):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_proba_cw):.4f}")
print(f"F2-Score:      {fbeta_score(y_test, y_pred_cw, beta=2):.4f}")

X_train_smote = np.load('data/processed/X_train.npy')
y_train_smote = np.load('data/processed/y_train.npy')

print("\nTraining XGBoost with SMOTE...")
model_smote = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test)
y_proba_smote = model_smote.predict_proba(X_test)[:, 1]

print("\n--- SMOTE Results ---")
print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba_smote):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_proba_smote):.4f}")
print(f"F2-Score:      {fbeta_score(y_test, y_pred_smote, beta=2):.4f}")

print("\n--- Verdict ---")
cw_ap = average_precision_score(y_test, y_proba_cw)
smote_ap = average_precision_score(y_test, y_proba_smote)
if cw_ap > smote_ap:
    print("Class weights outperforms SMOTE on Average Precision")
else:
    print("SMOTE outperforms class weights on Average Precision")
# =============================================================
# RiskPulse — Data Preprocessing
# Author: Ronil Muchandi
# Description: Cleans, scales, and balances the fraud dataset
#              using SMOTE to handle extreme class imbalance
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── 1. Load raw data ──────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/raw/creditcard.csv')
print(f"Dataset shape: {df.shape}")

# ── 2. Scale Amount and Time columns ─────────────────────────
# V1-V28 are already PCA scaled by the dataset provider
# Amount and Time need to be scaled manually
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
df.drop(columns=['Amount', 'Time'], inplace=True)
print("Amount and Time scaled.")

# ── 3. Split features and target ─────────────────────────────
X = df.drop(columns=['Class'])
y = df['Class']
print(f"Features shape: {X.shape}")
print(f"Class distribution before SMOTE:\n{y.value_counts()}")

# ── 4. Train/test split before SMOTE ─────────────────────────
# Important: SMOTE is applied only on training data
# Test set remains original imbalanced distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 5. Apply SMOTE to training data only ─────────────────────
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Training class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")

# ── 6. Save processed data ────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)

np.save('data/processed/X_train.npy', X_train_balanced)
np.save('data/processed/y_train.npy', y_train_balanced)
np.save('data/processed/X_test.npy', X_test.values)
np.save('data/processed/y_test.npy', y_test.values)

# Save scaler for later use in API
joblib.dump(scaler, 'data/processed/scaler.pkl')

print("\nProcessed data saved to data/processed/")
print("Preprocessing complete.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

print("Loading dataset...")
df = pd.read_csv('data/raw/creditcard.csv')
print(f"Dataset shape: {df.shape}")

# Scale Amount and Time
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])
df.drop(columns=['Amount', 'Time'], inplace=True)

# Time-based split — sort by Time first to avoid temporal leakage
# In production you never train on future data
df = df.sort_values('Time_Scaled').reset_index(drop=True)
split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean():.4%}")
print(f"Test fraud rate: {y_test.mean():.4%}")

# Apply SMOTE only on training data
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Training class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")

os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', X_train_balanced)
np.save('data/processed/y_train.npy', y_train_balanced)
np.save('data/processed/X_test.npy', X_test.values)
np.save('data/processed/y_test.npy', y_test.values)
joblib.dump(scaler, 'data/processed/scaler.pkl')

print("\nProcessed data saved.")
print("Preprocessing complete.")
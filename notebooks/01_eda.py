# =============================================================
# RiskPulse — Exploratory Data Analysis
# Author: Ronil Muchandi
# Description: Initial exploration of the credit card fraud dataset
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv('data/raw/creditcard.csv')

# ── 2. Basic info ─────────────────────────────────────────────
print("Shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['Class'].value_counts())
print("\nFraud percentage: {:.4f}%".format(df['Class'].mean() * 100))

# ── 3. Statistical summary ────────────────────────────────────
print("\nStatistical summary:\n", df.describe())

# ── 4. Class imbalance plot ───────────────────────────────────
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette=['steelblue', 'crimson'])
plt.title('Class Distribution — Fraud vs Normal')
plt.xticks([0, 1], ['Normal (0)', 'Fraud (1)'])
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/processed/class_distribution.png')
print("\nClass distribution plot saved.")

# ── 5. Transaction amount by class ───────────────────────────
plt.figure(figsize=(8, 4))
df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6, label='Normal', color='steelblue')
df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6, label='Fraud', color='crimson')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/processed/amount_distribution.png')
print("Amount distribution plot saved.")

# ── 6. Correlation heatmap ────────────────────────────────────
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('data/processed/correlation_heatmap.png')
print("Correlation heatmap saved.")

print("\nEDA complete.")

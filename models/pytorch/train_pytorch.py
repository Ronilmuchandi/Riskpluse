import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, fbeta_score

print("Loading data...")
df = pd.read_csv('data/raw/creditcard.csv')
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df['Time'] = (df['Time'] - df['Time'].mean()) / df['Time'].std()
df = df.sort_values('Time').reset_index(drop=True)

split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df.drop(columns=['Class']).values.astype(np.float32)
y_train = train_df['Class'].values.astype(np.float32)
X_test = test_df.drop(columns=['Class']).values.astype(np.float32)
y_test = test_df['Class'].values.astype(np.float32)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
X_test_tensor = torch.tensor(X_test)

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

model = FraudDetector(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining PyTorch model...")
losses = []
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/20 — Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    y_proba = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC Score:           {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")
print(f"F2-Score:                {fbeta_score(y_test, y_pred, beta=2):.4f}")

plt.figure(figsize=(6, 4))
plt.plot(losses, color='steelblue', label='Train loss')
plt.title('PyTorch Training Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/pytorch_loss.png')

torch.save(model.state_dict(), 'models/pytorch/pytorch_model.pt')
print("Model saved.")
print("\nPyTorch training complete.")
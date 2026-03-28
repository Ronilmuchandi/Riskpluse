import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("Loading data...")
X_train = np.load('data/processed/X_train.npy').astype(np.float32)
y_train = np.load('data/processed/y_train.npy').astype(np.float32)
X_test = np.load('data/processed/X_test.npy').astype(np.float32)
y_test = np.load('data/processed/y_test.npy').astype(np.float32)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Simple feedforward neural network
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

# Evaluate
model.eval()
with torch.no_grad():
    y_proba = model(X_test_tensor).squeeze().numpy()
    y_pred = (y_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, y_proba):.4f}")

# Save loss plot
plt.figure(figsize=(6, 4))
plt.plot(losses, color='steelblue', label='Train loss')
plt.title('PyTorch Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/pytorch_loss.png')
print("Loss plot saved.")

torch.save(model.state_dict(), 'models/pytorch/pytorch_model.pt')
print("Model saved.")
print("\nPyTorch training complete.")

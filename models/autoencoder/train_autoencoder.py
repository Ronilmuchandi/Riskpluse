import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("Loading data...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Autoencoders learn what "normal" looks like
# so we train only on normal transactions
X_train_normal = X_train[y_train == 0]
print(f"Training on {X_train_normal.shape[0]} normal transactions")

input_dim = X_train.shape[1]

# Build autoencoder - encoder compresses, decoder reconstructs
inputs = keras.Input(shape=(input_dim,))
encoded = layers.Dense(16, activation='relu')(inputs)
encoded = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(encoded)
outputs = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

print("\nTraining autoencoder...")
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# Reconstruction error is our fraud score
# Fraud transactions will have higher error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Set threshold at 95th percentile of normal reconstruction error
normal_mse = mse[y_test == 0]
threshold = np.percentile(normal_mse, 95)
print(f"\nThreshold (95th percentile): {threshold:.4f}")

y_pred = (mse > threshold).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, mse):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, mse):.4f}")

# Save training loss plot
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('data/processed/autoencoder_loss.png')
print("Loss plot saved.")

autoencoder.save('models/autoencoder/autoencoder_model.h5')
np.save('models/autoencoder/threshold.npy', threshold)
print("Model saved.")
print("\nAutoencoder training complete.")

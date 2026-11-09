import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "demand_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

if "Demand" not in df.columns:
    raise ValueError("CSV must contain a 'Demand' column")

if "Date" in df.columns:
    df = df.sort_values("Date")

print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# -----------------------------
# FEATURE SELECTION (Demand only for scaler compatibility)
# -----------------------------
data = df[["Demand"]].values

print(f"   Using feature: Demand")
print(f"   Data range: {data.min():.1f} - {data.max():.1f}")

# -----------------------------
# SCALING
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# -----------------------------
# SEQUENCE CREATION
# -----------------------------
SEQ_LENGTH = 10

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LENGTH)

# Train/test split (80/20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n📊 Sequences created: {len(X)} total")
print(f"   Training: {len(X_train)}")
print(f"   Testing: {len(X_test)}")
print(f"   Sequence length: {SEQ_LENGTH}")

# -----------------------------
# MODEL
# -----------------------------
print("\n🔧 Building LSTM model...")
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# -----------------------------
# TRAINING
# -----------------------------
print("\n🚀 Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# EVALUATE
# -----------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Test Loss (MSE): {loss:.6f}")
print(f"📊 Test MAE: {mae:.6f}")

# Convert MAE back to original scale
mae_original = mae * (scaler.data_max_[0] - scaler.data_min_[0])
print(f"📊 MAE (original scale): {mae_original:.2f} liters")

# -----------------------------
# SAVE
# -----------------------------
model.save(MODEL_PATH)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"\n💾 Model saved: {MODEL_PATH}")
print(f"💾 Scaler saved: {SCALER_PATH}")
print("✅ Demand forecasting model training complete!")

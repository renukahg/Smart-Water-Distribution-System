import pandas as pd
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "leak_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "leak_detection_model.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print(f"   Leak rate: {df['Leak_Status'].mean()*100:.1f}%")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
feature_cols = ["Pressure", "Flow", "Temperature"]
X = df[feature_cols]
y = df["Leak_Status"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {len(X_train)} samples")
print(f"📊 Testing set:  {len(X_test)} samples")

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\n🔧 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Leak", "Leak"]))

# Feature importance
importances = model.feature_importances_
for feat, imp in zip(feature_cols, importances):
    print(f"   {feat}: {imp*100:.1f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Model saved to: {MODEL_PATH}")
print("✅ Leak detection model training complete!")

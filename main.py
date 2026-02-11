# Project Title:
# Behavioral Analytics–Based Cognitive Load Estimation
# Using Keystroke Dynamics
# -----------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. DATA GENERATION
# -----------------------------
# Synthetic dataset representing typing behavior

np.random.seed(42)
samples = 300

data = {
    "typing_speed": np.concatenate([
        np.random.normal(4.2, 0.4, samples // 3),   # Low cognitive load
        np.random.normal(3.0, 0.4, samples // 3),   # Medium cognitive load
        np.random.normal(1.8, 0.3, samples // 3)    # High cognitive load
    ]),

    "pause_duration": np.concatenate([
        np.random.normal(0.4, 0.1, samples // 3),
        np.random.normal(0.9, 0.2, samples // 3),
        np.random.normal(1.6, 0.3, samples // 3)
    ]),

    "backspace_frequency": np.concatenate([
        np.random.randint(0, 3, samples // 3),
        np.random.randint(3, 7, samples // 3),
        np.random.randint(7, 12, samples // 3)
    ]),

    "error_rate": np.concatenate([
        np.random.normal(0.05, 0.02, samples // 3),
        np.random.normal(0.18, 0.05, samples // 3),
        np.random.normal(0.35, 0.08, samples // 3)
    ]),

    "task_completion_time": np.concatenate([
        np.random.normal(120, 15, samples // 3),
        np.random.normal(260, 30, samples // 3),
        np.random.normal(420, 50, samples // 3)
    ]),

    "cognitive_load": (
        ["Low"] * (samples // 3) +
        ["Medium"] * (samples // 3) +
        ["High"] * (samples // 3)
    )
}

df = pd.DataFrame(data)

# -----------------------------
# 3. DATA PREPARATION
# -----------------------------

X = df.drop("cognitive_load", axis=1)
y = df["cognitive_load"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. MODEL TRAINING
# -----------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 5. MODEL EVALUATION
# -----------------------------

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1, 2], ["Low", "Medium", "High"])
plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# -----------------------------
# 7. FEATURE IMPORTANCE ANALYSIS
# -----------------------------

importances = model.feature_importances_
features = X.columns

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance Analysis")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=30)
plt.show()

# -----------------------------
# 8. SAMPLE REAL-TIME PREDICTION
# -----------------------------

sample_input = pd.DataFrame([{
    "typing_speed": 2.0,
    "pause_duration": 1.4,
    "backspace_frequency": 8,
    "error_rate": 0.32,
    "task_completion_time": 380
}])

prediction = model.predict(sample_input)

print("\nPredicted Cognitive Load for sample input:", prediction[0])

# ============================================================
# END OF PROJECT CODE
# ============================================================

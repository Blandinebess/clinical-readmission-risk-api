import os
# Limit OpenBLAS threads to prevent memory allocation crashes on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Generate imbalanced clinical data (15% readmission rate)
np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    'age': np.random.randint(18, 90, size=n_samples),
    'num_lab_procedures': np.random.randint(1, 100, size=n_samples),
    'num_medications': np.random.randint(1, 100, size=n_samples),
    'time_in_hospital': np.random.randint(1, 14, size=n_samples),
    'number_diagnoses': np.random.randint(1, 16, size=n_samples),
    # Imbalanced class: 15% positive class (1 = readmitted)
    'readmitted': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
})

X = data.drop('readmitted', axis=1)
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. RandomForest model with class weighting and explicit job count
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Penalizes mistakes on the minority class
    n_jobs=1,                 # Restricts multi-threading to avoid OpenBLAS memory issues
    random_state=42
)

model.fit(X_train, y_train)

# 3. Clinical evaluation metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("--- Clinical Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 4. Save trained artifact
joblib.dump(model, 'model.joblib')
print("\nTrained model saved as 'model.joblib'!")
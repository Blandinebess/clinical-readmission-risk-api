from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Clinical Readmission Risk API",
    description="Production-grade API for predicting patient readmission risk.",
    version="1.0.0"
)

# Load the trained model artifact
MODEL_PATH = "model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Input schema matching model training features
class PatientData(BaseModel):
    age: int
    num_lab_procedures: int
    num_medications: int
    time_in_hospital: int
    number_diagnoses: int

    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "num_lab_procedures": 45,
                "num_medications": 12,
                "time_in_hospital": 4,
                "number_diagnoses": 5
            }
        }

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_readmission(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Run train_clinical_model.py first.")

    # Convert incoming JSON data to model input array
    features = np.array([[
        patient.age,
        patient.num_lab_procedures,
        patient.num_medications,
        patient.time_in_hospital,
        patient.number_diagnoses
    ]])

    # Generate prediction and risk probability
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "readmission_predicted": bool(prediction),
        "readmission_risk_score": round(probability, 4),
        "risk_category": "High Risk" if probability >= 0.5 else "Low Risk"
    }
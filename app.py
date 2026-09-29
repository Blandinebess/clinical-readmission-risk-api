from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Clinical Readmission Risk API")

class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120)
    bmi: float = Field(..., ge=10.0, le=70.0)
    systolic_bp: int = Field(..., ge=70, le=250)
    num_previous_admissions: int = Field(..., ge=0)

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/predict")
def predict_readmission(patient: PatientData):
    # Dummy logic until model file is present
    risk_score = (patient.age * 0.03) + (patient.systolic_bp * 0.02) + (patient.num_previous_admissions * 0.5)
    is_high_risk = risk_score > 4.0
    
    return {
        "readmission_probability_pct": 85.0 if is_high_risk else 15.0,
        "risk_level": "HIGH RISK OF READMISSION" if is_high_risk else "LOW RISK",
        "recommended_care_plan": "Enroll patient in intensive protocol." if is_high_risk else "Standard discharge."
    }
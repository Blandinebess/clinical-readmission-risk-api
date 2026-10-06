from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Clinical Readmission Risk & Interpretability API",
    description="API for predicting hospital readmission risk and providing SHAP-based feature importance explanations.",
    version="2.0.0"
)

# Load trained model and SHAP explainer
try:
    model = joblib.load("model.joblib")
    explainer = joblib.load("explainer.joblib")
except Exception as e:
    model = None
    explainer = None

class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120, example=65)
    num_lab_procedures: int = Field(..., ge=0, example=45)
    num_medications: int = Field(..., ge=0, example=12)
    time_in_hospital: int = Field(..., ge=1, example=4)
    number_diagnoses: int = Field(..., ge=1, example=5)

@app.get("/")
def read_root():
    return {
        "service": "Clinical Readmission Risk API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_readmission(patient: PatientData):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model or explainer artifact not loaded.")
    
    # Format input data
    input_df = pd.DataFrame([patient.model_dump()])
    
    # Calculate probability and prediction
    prob_readmission = float(model.predict_proba(input_df)[0][1])
    prediction = int(model.predict(input_df)[0])
    
    # Determine risk category
    if prob_readmission >= 0.7:
        risk_category = "High Risk"
    elif prob_readmission >= 0.4:
        risk_category = "Moderate Risk"
    else:
        risk_category = "Low Risk"
        
    # Calculate SHAP values for clinical interpretability
    shap_values = explainer(input_df)
    # Get SHAP impact values for class 1 (readmission)
    if len(shap_values.values.shape) == 3:
        feature_impacts = shap_values.values[0, :, 1].tolist()
    else:
        feature_impacts = shap_values.values[0].tolist()
        
    feature_importance = dict(zip(input_df.columns, feature_impacts))
    
    return {
        "readmission_predicted": bool(prediction),
        "readmission_risk_score": round(prob_readmission, 4),
        "risk_category": risk_category,
        "feature_contributions": feature_importance
    }
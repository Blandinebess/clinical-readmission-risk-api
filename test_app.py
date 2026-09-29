from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Verify root endpoint returns HTTP 200 and online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "online"}

def test_predict_readmission_valid():
    """Verify valid patient payload returns a high-risk classification."""
    payload = {
        "age": 120,
        "bmi": 10.0,
        "systolic_bp": 70,
        "num_previous_admissions": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "readmission_probability_pct" in data
    assert "risk_level" in data
    assert data["risk_level"] == "HIGH RISK OF READMISSION"

def test_predict_readmission_invalid_age():
    """Verify Pydantic validation rejects age > 120 with HTTP 422."""
    invalid_payload = {
        "age": 150,  # Exceeds max 120 constraint
        "bmi": 25.0,
        "systolic_bp": 120,
        "num_previous_admissions": 1
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
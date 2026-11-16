
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


# Test 1: Health endpoint
def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Random-Forest-100"


# Test 2: Homepage endpoint
def test_homepage_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert "WattWise" in response.text
    assert "Electricity Consumption Dashboard" in response.text


# Test 3: Valid prediction
def test_valid_prediction():

    payload = {
        "history": [1.5] * 168,
        "last_timestamp": "2026-11-12T23:00:00"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_consumption_kw" in data
    assert "prediction_for" in data
    assert "model_used" in data
    assert "suggestions" in data

    assert isinstance(
        data["predicted_consumption_kw"], float
    )


# Test 4: Reject insufficient history
def test_invalid_short_history():

    payload = {
        "history": [1.5] * 20,
        "last_timestamp": "2026-11-12T23:00:00"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 5: Reject negative readings
def test_negative_readings():

    history = [1.5] * 168
    history[10] = -5

    payload = {
        "history": history,
        "last_timestamp": "2026-11-12T23:00:00"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

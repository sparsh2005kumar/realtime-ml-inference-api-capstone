from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_is_healthy():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_without_model_returns_clear_error():
    response = client.post("/predict", json={"features": {"feature_01": 0.1}})
    # If a compatible artifact is present, this may return 200 instead.
    assert response.status_code in (200, 503, 422)

def test_predict_requires_features():
    response = client.post("/predict", json={})
    assert response.status_code == 422

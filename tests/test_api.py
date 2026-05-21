"""
tests/test_api.py - Automated tests for the MTCars MPG Prediction API.

Run from the project root:
    python -m pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# Health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# Readiness
def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["r2"] > 0.5


# Predict - success
def test_predict_success(client):
    r = client.post("/predict", json={"wt": 3.215, "hp": 110, "cyl": 6})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_mpg" in data
    assert isinstance(data["predicted_mpg"], float)


def test_predict_realistic_range(client):
    r = client.post("/predict", json={"wt": 3.215, "hp": 110, "cyl": 6})
    mpg = r.json()["predicted_mpg"]
    assert 5 < mpg < 60, f"mpg {mpg} outside realistic range"


def test_predict_heavy_car_lower_mpg(client):
    """Heavier, more powerful car should have lower mpg."""
    light = client.post("/predict", json={"wt": 1.5,  "hp": 65,  "cyl": 4}).json()
    heavy = client.post("/predict", json={"wt": 5.0,  "hp": 300, "cyl": 8}).json()
    assert light["predicted_mpg"] > heavy["predicted_mpg"]


def test_predict_returns_predictors_used(client):
    r = client.post("/predict", json={"wt": 2.62, "hp": 110, "cyl": 4})
    assert r.json()["predictors_used"] == ["wt", "hp", "cyl"]


# Predict - validation errors
def test_predict_missing_field(client):
    r = client.post("/predict", json={"wt": 3.215, "hp": 110})  # missing cyl
    assert r.status_code == 422


def test_predict_invalid_type(client):
    r = client.post("/predict", json={"wt": "heavy", "hp": 110, "cyl": 6})
    assert r.status_code == 422


def test_predict_negative_weight(client):
    r = client.post("/predict", json={"wt": -1.0, "hp": 110, "cyl": 6})
    assert r.status_code == 422


def test_predict_zero_hp(client):
    r = client.post("/predict", json={"wt": 3.215, "hp": 0, "cyl": 6})
    assert r.status_code == 422


def test_predict_empty_body(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
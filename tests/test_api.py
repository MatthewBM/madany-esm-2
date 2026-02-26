import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """
    validate engine is initialzed before starting tests
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    # verify health is responsive and mock mode is used
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["is_mock_mode"] is True

def test_predict_valid_sequence(client):
    # Test a single valid protein sequence against /predict
    payload = {"sequence": "MAPLRKTYVLKLYVAGTPYVRRCDV"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "embedding" in response.json()
    assert isinstance(response.json()["embedding"], list)

def test_predict_invalid_characters(client):
    # Test if non-amino acid characters are rejected
    payload = {"sequence": "MAPLR-123-INVALID"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_too_long(client):
    # Test if sequences over max_length are rejected
    long_seq = "A" * 1500
    payload = {"sequence": long_seq}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "exceeds limit" in response.text

def test_batch_prediction(client):
    # Test multiple valid protein sequence against /predict/batch
    payload = {"sequences": ["MAPLR", "KTYVL", "VAGTP"]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 3

import pytest
from fastapi.testclient import TestClient

from hybrid_fraud import __version__
from hybrid_fraud.api.main import MAX_BATCH_SIZE, app

client = TestClient(app)

TX = {
    "transaction_id": "tx_001",
    "user_id": "user_123",
    "amount": 149.99,
    "merchant_id": "merch_456",
    "device_id": "dev_789",
    "timestamp": "2026-11-27T00:00:00Z",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["version"] == __version__


def test_score_placeholder():
    r = client.post("/score", json=TX)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "tx_001"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["mode"] == "placeholder"
    assert body["fusion_weights"] == {} and body["components"] == {}


@pytest.mark.parametrize("amount,decision", [(10, "approve"), (1500, "review"), (5000, "decline")])
def test_score_decisions(amount, decision):
    assert client.post("/score", json={**TX, "amount": amount}).json()["decision"] == decision


def test_score_validation():
    assert client.post("/score", json={**TX, "amount": -1}).status_code == 422
    assert client.post("/score", json={"user_id": "u"}).status_code == 422


def test_batch_score():
    r = client.post("/batch_score", json=[TX, {**TX, "transaction_id": "tx_002"}])
    assert r.status_code == 200
    assert r.json()["count"] == 2
    assert [x["transaction_id"] for x in r.json()["results"]] == ["tx_001", "tx_002"]


def test_batch_too_large():
    r = client.post("/batch_score", json=[TX] * (MAX_BATCH_SIZE + 1))
    assert r.status_code == 413

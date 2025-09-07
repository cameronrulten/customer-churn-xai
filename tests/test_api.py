from fastapi.testclient import TestClient
from churn_xai..api.main import app

def test_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict_roundtrip():
    client = TestClient(app)
    meta = client.get("/metadata").json()
    cats, nums = meta["features"]["categorical"], meta["features"]["numeric"]
    # construct one dummy row (uses None if nothing known)
    payload = {"features": {**{n: 0 for n in nums}, **{c: "" for c in cats}}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["proba"] <= 1.0
    assert isinstance(data["top_features"], list)

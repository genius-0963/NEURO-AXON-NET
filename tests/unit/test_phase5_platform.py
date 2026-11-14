"""
Unit, API, and Integration Tests for Phase 05 Inference Platform & Release Engine.
"""

import pytest
import io
import json
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from neuroaxonnet.inference.service import NeuroAxonInferenceService
from neuroaxonnet.api.server import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_png_bytes():
    arr = np.random.randint(50, 200, size=(187, 187), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_service_image_validation_success(sample_png_bytes):
    service = NeuroAxonInferenceService()
    arr, meta = service.validate_image_bytes(sample_png_bytes, {"species": "rat"})
    assert arr.shape == (187, 187)
    assert len(meta["sha256"]) == 64
    assert meta["species"] == "rat"


def test_service_rejects_empty_bytes():
    service = NeuroAxonInferenceService()
    with pytest.raises(ValueError, match="Empty or zero-byte"):
        service.validate_image_bytes(b"")


def test_service_result_schema_compliance():
    service = NeuroAxonInferenceService()
    raw = np.random.uniform(50, 200, size=(187, 187)).astype(np.float32)
    res = service.predict_patch(raw)

    # Check mandatory Section 6 schema fields
    assert "job_id" in res
    assert "status" in res
    assert "model" in res and res["model"]["name"] == "NeuroAxonNet"
    assert "input" in res and "sha256" in res["input"]
    assert "count" in res and res["count"]["unit"] == "normal_appearing_axons"
    assert "qc" in res and "flags" in res["qc"]
    assert "provenance" in res
    assert "RESEARCH PROTOTYPE ONLY" in res["disclaimer"]


def test_api_health_endpoint(test_client):
    response = test_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "RESEARCH PROTOTYPE ONLY" in data["disclaimer"]


def test_api_job_lifecycle_end_to_end(test_client, sample_png_bytes):
    # 1. POST /api/v1/jobs
    files = {"image": ("test_patch.png", sample_png_bytes, "image/png")}
    data = {"metadata": json.dumps({"species": "rat", "stain": "PPD"})}
    post_res = test_client.post("/api/v1/jobs", files=files, data=data)
    assert post_res.status_code == 200
    job_info = post_res.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]

    # 2. GET /api/v1/jobs/{job_id}
    status_res = test_client.get(f"/api/v1/jobs/{job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["progress"] == 1.0

    # 3. GET /api/v1/jobs/{job_id}/result
    result_res = test_client.get(f"/api/v1/jobs/{job_id}/result")
    assert result_res.status_code == 200
    res_data = result_res.json()
    assert res_data["count"]["unit"] == "normal_appearing_axons"

    # 4. GET /api/v1/jobs/{job_id}/artifacts/overlay
    overlay_res = test_client.get(f"/api/v1/jobs/{job_id}/artifacts/overlay")
    assert overlay_res.status_code == 200
    assert overlay_res.headers["content-type"] == "image/png"

    # 5. GET /api/v1/jobs/{job_id}/artifacts/csv
    csv_res = test_client.get(f"/api/v1/jobs/{job_id}/artifacts/csv")
    assert csv_res.status_code == 200
    assert "normal_appearing_axons" in csv_res.text

"""
AuditHub Tests - FastAPI Backend
==================================

Integration tests for the FastAPI REST endpoints using TestClient.
"""

import pytest
import io
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for FastAPI."""
    return TestClient(app)


def test_api_root(client, config_manager):
    """Test the root endpoint status and endpoints catalog."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_ingest_and_validation(client, config_manager):
    """Test file ingestion and validation REST endpoints."""
    # Build synthetic csv in-memory
    csv_content = "id,feat_1,target\n1,10.5,1\n2,20.0,0\n3,15.2,1"
    file_tuple = ("file", ("dataset.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv"))

    # Ingest test
    resp_ingest = client.post("/api/v1/ingest", files={"file": file_tuple[1]})
    assert resp_ingest.status_code == 200
    assert resp_ingest.json()["status"] == "success"
    assert "summary" in resp_ingest.json()

    # Reset file pointer for next request
    file_tuple[1][1].seek(0)

    # Validate test
    resp_val = client.post("/api/v1/validate", files={"file": file_tuple[1]}, data={"target_column": "target"})
    assert resp_val.status_code == 200
    assert resp_val.json()["status"] == "success"
    assert "report" in resp_val.json()
    assert resp_val.json()["report"]["score"] > 0

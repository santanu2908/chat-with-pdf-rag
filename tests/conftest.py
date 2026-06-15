"""Shared fixtures for the evaluation harness."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "sample_test_file.pdf"
EVAL_QUESTIONS = PROJECT_ROOT / "tests" / "eval_questions.json"


@pytest.fixture(scope="session")
def client():
    """Create a TestClient and upload the sample PDF once for all tests."""
    with TestClient(app) as c:
        with open(SAMPLE_PDF, "rb") as f:
            resp = c.post("/upload", files={"file": ("sample_test_file.pdf", f, "application/pdf")})
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        yield c


def load_eval_questions():
    """Load the 19 evaluation questions from JSON."""
    with open(EVAL_QUESTIONS) as f:
        return json.load(f)

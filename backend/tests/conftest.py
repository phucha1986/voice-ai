"""Test configuration: point the app at a throwaway data dir."""

import os
import tempfile

# Must be set before importing app modules so config picks it up.
_tmp = tempfile.mkdtemp(prefix="voice-ai-test-")
os.environ["VOICE_AI_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["VOICE_AI_DEVICE"] = "cpu"
# Tests must not download VieNeu in a background preload thread.
os.environ["VOICE_AI_VIENEU_PRELOAD"] = "0"

import pytest
from fastapi.testclient import TestClient


def _client() -> TestClient:
    from app.main import app

    # TestClient runs startup/shutdown events via the context manager.
    return TestClient(app)


@pytest.fixture()
def client() -> TestClient:
    with _client() as c:
        yield c

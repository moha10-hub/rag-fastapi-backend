import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.rag_engine import get_engine
from tests.conftest import FAKE_ANSWER

TEXT = "Il gatto dorme sul divano.".encode()


@pytest.fixture
def client(engine, monkeypatch):
    main.app.dependency_overrides[get_engine] = lambda: engine
    monkeypatch.setattr(main, "is_engine_ready", lambda: True)
    monkeypatch.setattr(main, "get_engine", lambda: engine)
    # Senza "with": il lifespan non parte, quindi non viene creato il motore reale.
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _upload(client, name="note.txt", content=TEXT):
    return client.post("/upload", files={"file": (name, content, "text/plain")})


def test_health_when_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine_ready"] is True
    assert body["indexed_chunks"] == 0


def test_health_when_engine_not_ready(client, monkeypatch):
    monkeypatch.setattr(main, "is_engine_ready", lambda: False)
    body = client.get("/health").json()
    assert body["engine_ready"] is False
    assert body["indexed_chunks"] is None


def test_upload_txt_returns_201(client):
    response = _upload(client)
    assert response.status_code == 201
    body = response.json()
    assert body["file_type"] == "txt"
    assert body["chunks_added"] >= 1


def test_upload_unsupported_format_returns_415(client):
    assert _upload(client, "a.docx", b"x").status_code == 415


def test_upload_empty_file_returns_400(client):
    assert _upload(client, "empty.txt", b"").status_code == 400


def test_upload_file_without_text_returns_422(client):
    assert _upload(client, "blank.txt", b"   \n").status_code == 422


def test_upload_file_over_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(main, "settings", Settings(_env_file=None, max_upload_mb=1))
    assert _upload(client, "big.txt", b"a" * (1024 * 1024 + 1)).status_code == 413


def test_query_without_documents_returns_409(client):
    assert client.post("/query", json={"question": "ciao"}).status_code == 409


def test_query_after_upload_returns_answer_and_sources(client):
    _upload(client)
    response = client.post("/query", json={"question": "Dove dorme il gatto?", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_ANSWER
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source"] == "note.txt"


def test_query_blank_question_returns_422(client):
    assert client.post("/query", json={"question": "   "}).status_code == 422

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import create_app


class FakeEngine:
    """Deterministic provider substitute; never makes network requests."""

    def index_pdf(self, path, filename, progress):
        assert path.read_bytes().startswith(b"%PDF-")
        progress("Testing the index")
        return SimpleNamespace(pages=2, chunks=3)

    def ask(self, document, question):
        return {"answer": "A supported answer [Source 1, page 2].", "supported": True,
                "sources": [{"id": 1, "filename": "notes.pdf", "page": 2, "excerpt": "A passage."}],
                "steps": ["Retrieved document passages", "Checked usefulness"], "revisions": 0}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    with TestClient(create_app(FakeEngine, data_dir=tmp_path)) as test_client:
        yield test_client


def upload(client, filename="notes.pdf", content=b"%PDF-1.4 test"):
    return client.post("/api/documents", files={"file": (filename, content, "application/pdf")})


def test_home_and_health(client):
    assert "Your documents." in client.get("/").text
    health = client.get("/api/health").json()
    assert health["configured"]
    assert "test-only" not in str(health)
    assert client.get("/static/app.js").status_code == 200


def test_upload_ask_and_remove(client, tmp_path):
    response = upload(client, "../../notes.pdf")
    assert response.status_code == 202
    document_id = response.json()["id"]
    document = client.get(f"/api/documents/{document_id}").json()
    assert document["filename"] == "notes.pdf"
    assert document["status"] == "ready"
    assert document["pages"] == 2
    assert list(tmp_path.iterdir()) == []  # PDF removed after indexing.
    answer = client.post("/api/ask", json={"document_id": document_id, "question": " What is this? "})
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["page"] == 2
    assert answer.json()["supported"]
    assert client.delete(f"/api/documents/{document_id}").status_code == 200
    assert client.get(f"/api/documents/{document_id}").status_code == 404


@pytest.mark.parametrize("filename,content,code", [
    ("notes.txt", b"text", 415), ("notes.pdf", b"not pdf", 415), ("notes.pdf", b"", 400),
])
def test_invalid_uploads(client, tmp_path, filename, content, code):
    assert upload(client, filename, content).status_code == code
    assert list(tmp_path.iterdir()) == []


def test_upload_limit(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    assert upload(client, content=b"%PDF-" + b"x" * 20).status_code == 413


def test_missing_keys(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert client.get("/api/health").json()["missing_keys"] == ["GEMINI_API_KEY"]
    assert upload(client).status_code == 503


@pytest.mark.parametrize("question", ["", "   ", "x" * 4001])
def test_invalid_questions(client, question):
    assert client.post("/api/ask", json={"document_id": "missing", "question": question}).status_code == 422


def test_unknown_document(client):
    assert client.post("/api/ask", json={"document_id": "missing", "question": "Why?"}).status_code == 404


def test_provider_errors_do_not_leak_keys(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("provider secret test-only")
    document_id = upload(client).json()["id"]
    monkeypatch.setattr(FakeEngine, "ask", fail)
    response = client.post("/api/ask", json={"document_id": document_id, "question": "Why?"})
    assert response.status_code == 502
    assert "test-only" not in response.text


def test_indexing_failure_has_status_and_cleans_file(client, monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("This PDF has no readable text.")
    monkeypatch.setattr(FakeEngine, "index_pdf", fail)
    document_id = upload(client).json()["id"]
    result = client.get(f"/api/documents/{document_id}").json()
    assert result["status"] == "error"
    assert "no readable text" in result["message"]
    assert list(tmp_path.iterdir()) == []
    assert client.post("/api/ask", json={"document_id": document_id, "question": "Why?"}).status_code == 409


def test_document_limit_can_be_recovered(client, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_DOCUMENTS", 1)
    document_id = upload(client).json()["id"]
    assert upload(client).status_code == 409
    assert client.delete(f"/api/documents/{document_id}").status_code == 200
    assert upload(client).status_code == 202

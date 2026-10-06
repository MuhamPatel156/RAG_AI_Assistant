"""Local FastAPI backend and static frontend for Self-RAG."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from self_rag import SelfRAG

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
logger = logging.getLogger("self_rag")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DOCUMENTS = 8


class Question(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a question first.")
        return value.strip()


@dataclass
class Document:
    id: str
    filename: str
    status: str = "processing"
    message: str = "Preparing your document…"
    pages: int = 0
    chunks: int = 0
    indexed: Any = None
    engine: Any = None
    lock: Any = field(default_factory=threading.Lock)

    def public(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in ("id", "filename", "status", "message", "pages", "chunks")}


def missing_keys() -> list[str]:
    return [key for key in ("OPENAI_API_KEY", "GEMINI_API_KEY") if not os.getenv(key, "").strip()]


def create_app(engine_factory=SelfRAG, data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Self-RAG API", version="1.0.0")
    documents: dict[str, Document] = {}
    documents_lock = threading.Lock()
    upload_dir = data_dir if data_dir is not None else BASE_DIR / ".data"
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(BASE_DIR / "static" / "index.html")

    @app.get("/api/health")
    def health():
        missing = missing_keys()
        return {"status": "ok", "configured": not missing, "missing_keys": missing,
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "max_upload_mb": 20}

    def get_document(document_id: str) -> Document:
        with documents_lock:
            document = documents.get(document_id)
        if document is None:
            raise HTTPException(404, "Document not found. Upload it again if the server restarted.")
        return document

    def process(document: Document, path: Path) -> None:
        def progress(message: str) -> None:
            with document.lock:
                document.message = message

        try:
            document.engine = engine_factory()
            indexed = document.engine.index_pdf(path, document.filename, progress)
            with document.lock:
                document.indexed = indexed
                document.pages, document.chunks = indexed.pages, indexed.chunks
                document.message, document.status = "Ready for your questions", "ready"
        except ValueError as exc:
            with document.lock:
                # Only expose our document validation errors, never provider payloads.
                safe = str(exc)
                document.message = safe if safe.startswith(("This PDF ", "This document ")) else "Indexing failed. Check the server's model and embedding configuration."
                document.status = "error"
            logger.warning("Document indexing failed (%s)", type(exc).__name__)
        except Exception as exc:
            with document.lock:
                document.message = "Indexing failed. Check your API keys, model access, quota, and connection, then try again."
                document.status = "error"
            logger.warning("Document indexing failed (%s)", type(exc).__name__)
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/documents", status_code=202)
    async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        path = None
        accepted = False
        document = None
        try:
            missing = missing_keys()
            if missing:
                raise HTTPException(503, "Add " + ", ".join(missing) + " to .env and restart the server.")
            filename = (file.filename or "document.pdf").replace("\\", "/").split("/")[-1][:200]
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(415, "Please upload a PDF file.")
            if file.size is not None and file.size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "PDF files must be 20 MB or smaller.")
            with documents_lock:
                if len(documents) >= MAX_DOCUMENTS:
                    raise HTTPException(409, "Document limit reached. Remove a document or restart the server.")
                document = Document(id=uuid4().hex, filename=filename)
                documents[document.id] = document
            upload_dir.mkdir(parents=True, exist_ok=True)
            path = upload_dir / f"{document.id}.pdf"
            size = 0
            with path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    if size == 0 and not chunk.startswith(b"%PDF-"):
                        raise HTTPException(415, "The selected file is not a valid PDF.")
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "PDF files must be 20 MB or smaller.")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")
            background_tasks.add_task(process, document, path)
            accepted = True
            return document.public()
        finally:
            await file.close()
            if not accepted:
                if path is not None:
                    path.unlink(missing_ok=True)
                if document is not None:
                    with documents_lock:
                        documents.pop(document.id, None)

    @app.get("/api/documents/{document_id}")
    def document_status(document_id: str):
        document = get_document(document_id)
        with document.lock:
            return document.public()

    @app.delete("/api/documents/{document_id}")
    def remove_document(document_id: str):
        document = get_document(document_id)
        if not document.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for the current question to finish.")
        try:
            if document.status == "processing":
                raise HTTPException(409, "Wait for indexing to finish.")
            with documents_lock:
                documents.pop(document_id, None)
            return {"deleted": True}
        finally:
            document.lock.release()

    @app.post("/api/ask")
    def ask(body: Question):
        document = get_document(body.document_id)
        if not document.lock.acquire(blocking=False):
            raise HTTPException(409, "This document is busy. Wait for the current question to finish.")
        try:
            if document.status != "ready":
                raise HTTPException(409, "Wait until the document is ready before asking a question.")
            started = time.monotonic()
            try:
                result = document.engine.ask(document.indexed, body.question)
            except Exception as exc:
                logger.warning("Question failed (%s)", type(exc).__name__)
                raise HTTPException(502, "The AI service could not complete the request. Check your keys, quota, model access, and connection, then try again.") from None
            return {**result, "elapsed_seconds": round(time.monotonic() - started, 1)}
        finally:
            document.lock.release()

    return app


app = create_app()

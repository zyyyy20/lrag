"""Document ingestion pipeline: parse -> chunk -> embed -> index."""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from ..config import get_settings
from ..database import session_scope
from ..models import Document, DocumentChunk, DocumentStatus
from .chunking import SUPPORTED_EXTENSIONS, chunk_text, extract_text
from .llm import get_embedder

logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    pass


def validate_upload(file: UploadFile, size_bytes: int) -> None:
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes <= 0:
        raise UploadValidationError("Uploaded file is empty")
    if size_bytes > max_bytes:
        raise UploadValidationError(
            f"File too large. Max allowed is {settings.max_upload_mb} MB"
        )
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    # basic filename sanitization: reject path separators
    if any(sep in name for sep in ("/", "\\", "..")):
        raise UploadValidationError("Invalid filename")


def save_upload(file: UploadFile, content: bytes) -> str:
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    target = Path(settings.upload_dir) / stored_name
    target.write_bytes(content)
    return str(target)


# DashScope's OpenAI-compatible embedding endpoint caps batch size at 10.
# Keep batch small to remain compatible across OpenAI / DashScope / SiliconFlow.
_EMBED_BATCH = 10


def _embed_in_batches(texts: list[str], batch: int = _EMBED_BATCH) -> Iterable[list[float]]:
    embedder = get_embedder()
    for i in range(0, len(texts), batch):
        for vec in embedder.embed(texts[i : i + batch]):
            yield vec


def process_document(document_id: uuid.UUID) -> None:
    """Run extraction, chunking, embedding and indexing for the given document.
    Designed to be run as a background task. Uses its own DB session."""
    settings = get_settings()
    try:
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None or doc.status == DocumentStatus.deleted:
                logger.warning("Document %s not found or deleted, skipping", document_id)
                return
            doc.status = DocumentStatus.processing
            doc.error = None
            db.flush()
            file_path = doc.file_path
            content_type = doc.content_type

        text = extract_text(file_path, content_type)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            with session_scope() as db:
                doc = db.get(Document, document_id)
                if doc is not None:
                    doc.status = DocumentStatus.failed
                    doc.error = "No extractable text found in document"
            return

        vectors = list(_embed_in_batches(chunks))
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count mismatch")

        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is None or doc.status == DocumentStatus.deleted:
                return
            # remove any stale chunks (e.g. from a previous failed run)
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete(synchronize_session=False)

            for idx, (content, vec) in enumerate(zip(chunks, vectors)):
                db.add(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=idx,
                        content=content,
                        token_count=len(content),
                        embedding=vec,
                        is_deleted=False,
                    )
                )
            doc.chunk_count = len(chunks)
            doc.status = DocumentStatus.indexed
            doc.error = None
        logger.info("Indexed document %s with %d chunks", document_id, len(chunks))
    except Exception as e:
        logger.exception("Failed to process document %s", document_id)
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc is not None and doc.status != DocumentStatus.deleted:
                doc.status = DocumentStatus.failed
                doc.error = str(e)[:1000]


def soft_delete_document(document_id: uuid.UUID) -> bool:
    """Soft-delete a document and its chunks. Returns True on success."""
    now = datetime.now(timezone.utc)
    with session_scope() as db:
        doc = db.get(Document, document_id)
        if doc is None or doc.status == DocumentStatus.deleted:
            return False
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_deleted.is_(False),
        ).update(
            {"is_deleted": True, "deleted_at": now},
            synchronize_session=False,
        )
        doc.status = DocumentStatus.deleted
        doc.deleted_at = now
        # best-effort remove the underlying file
        try:
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)
        except OSError:
            logger.warning("Failed to remove file %s", doc.file_path)
        return True


def soft_delete_documents_under_kb(knowledge_base_id: uuid.UUID) -> int:
    """Cascade soft-delete: mark all documents and chunks under a KB as deleted.
    Returns the number of documents affected."""
    now = datetime.now(timezone.utc)
    with session_scope() as db:
        docs = (
            db.query(Document)
            .filter(
                Document.knowledge_base_id == knowledge_base_id,
                Document.status != DocumentStatus.deleted,
            )
            .all()
        )
        doc_ids = [d.id for d in docs]
        if not doc_ids:
            return 0
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id.in_(doc_ids),
            DocumentChunk.is_deleted.is_(False),
        ).update(
            {"is_deleted": True, "deleted_at": now},
            synchronize_session=False,
        )
        for d in docs:
            d.status = DocumentStatus.deleted
            d.deleted_at = now
            try:
                if d.file_path and os.path.exists(d.file_path):
                    os.remove(d.file_path)
            except OSError:
                logger.warning("Failed to remove file %s", d.file_path)
        return len(docs)

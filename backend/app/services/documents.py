"""文档上传、解析、分块、向量化入库与软删除流水线。

职责划分：
- ``validate_upload``：扩展名、大小、文件名安全校验；
- ``save_upload``：将原始字节落盘到 ``UPLOAD_DIR``，文件名加 UUID 前缀防冲突；
- ``process_document``：后台任务入口——抽取全文、切块、批量 embedding、写入
  ``document_chunks`` 并更新文档状态；
- ``soft_delete_document`` / ``soft_delete_documents_under_kb``：软删除文档与
  chunk（标记 ``is_deleted``），并尽力删除磁盘文件。

状态机：``uploaded`` → ``processing`` → ``indexed`` 或 ``failed``；删除后为 ``deleted``。
"""
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
    """上传校验失败时抛出的业务异常（路由层会转为 HTTP 400）。"""

    pass


def validate_upload(file: UploadFile, size_bytes: int) -> None:
    """校验上传文件是否允许接收。

    检查项：
    - 非空文件；
    - 不超过 ``MAX_UPLOAD_MB``；
    - 扩展名在 ``SUPPORTED_EXTENSIONS`` 内；
    - 文件名不含路径穿越字符。

    Args:
        file: FastAPI 包装的 multipart 文件对象。
        size_bytes: 已读入内存的字节数。

    Raises:
        UploadValidationError: 任一检查不通过。
    """
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
    if any(sep in name for sep in ("/", "\\", "..")):
        raise UploadValidationError("Invalid filename")


def save_upload(file: UploadFile, content: bytes) -> str:
    """将上传内容写入配置目录下的唯一文件名，返回绝对路径字符串。

    Args:
        file: 用于读取原始 ``filename`` 的 UploadFile。
        content: 文件完整字节内容。

    Returns:
        磁盘上的目标路径（字符串），供 ``Document.file_path`` 持久化。
    """
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    safe_name = Path(file.filename or "upload").name
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    target = Path(settings.upload_dir) / stored_name
    target.write_bytes(content)
    return str(target)


# DashScope 等兼容接口对单次 embedding 批量条数有限制，保持较小批次以兼容多供应商
_EMBED_BATCH = 10


def _embed_in_batches(texts: list[str], batch: int = _EMBED_BATCH) -> Iterable[list[float]]:
    """按批次调用 Embedding API，逐条 yield 向量（与 chunks 顺序一致）。"""
    embedder = get_embedder()
    for i in range(0, len(texts), batch):
        for vec in embedder.embed(texts[i : i + batch]):
            yield vec


def process_document(document_id: int) -> None:
    """异步文档索引主流程：解析 → 分块 → 嵌入 → 写入数据库。

    设计为 FastAPI ``BackgroundTasks`` 调用；内部使用独立 ``session_scope``，
    避免与请求线程共享 Session。任一步失败会将文档标记为 ``failed`` 并写入
    截断后的错误信息。

    Args:
        document_id: 待处理文档主键。
    """
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
            # 重跑索引前物理删除旧 chunk 行（避免失败重试产生重复）
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


def soft_delete_document(document_id: int) -> bool:
    """软删除单个文档及其所有 chunk，并尝试删除磁盘文件。

    Args:
        document_id: 文档主键。

    Returns:
        成功标记为删除返回 ``True``；文档不存在或已删除返回 ``False``。
    """
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
        try:
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)
        except OSError:
            logger.warning("Failed to remove file %s", doc.file_path)
        return True


def soft_delete_documents_under_kb(knowledge_base_id: int) -> int:
    """软删除某知识库下所有未删文档及其 chunk（删除知识库时级联调用）。

    Args:
        knowledge_base_id: 知识库主键。

    Returns:
        受影响的文档数量。
    """
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

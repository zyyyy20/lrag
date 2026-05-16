"""文档上传与列表、软删除。

文档必须归属某个知识库（``knowledge_base_id``）。上传后写入磁盘与 DB 记录，
由 ``BackgroundTasks`` 异步调用 ``process_document`` 完成解析与向量化。列表
支持按 ``knowledge_base_id`` 过滤。删除为软删除并标记关联 chunk。"""
from __future__ import annotations

from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..config import get_settings
from ..database import get_db
from ..dependencies.auth import get_current_user
from ..models import Document, DocumentStatus, KnowledgeBase, User
from ..schemas import DocumentOut
from ..services.documents import (
    UploadValidationError,
    process_document,
    save_upload,
    soft_delete_document,
    validate_upload,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _ensure_public_document_write_allowed(doc: Document | None, user_id: int) -> None:
    mode = get_settings().public_kb_write_mode
    if mode == "disabled":
        raise HTTPException(status_code=403, detail="Public knowledge base writes are disabled")
    if mode == "creator_only" and doc is not None and doc.created_by != user_id:
        raise HTTPException(status_code=403, detail="Only the creator can modify this document")


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentOut:
    """multipart 上传：表单字段 ``knowledge_base_id`` + 文件 ``file``。

    校验 KB 存在 → 校验文件 → 落盘 → 插入 ``uploaded`` 状态记录 → 投递后台索引任务。
    """
    kb = (
        db.query(KnowledgeBase)
        .filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.is_deleted.is_(False),
        )
        .one_or_none()
    )
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    _ensure_public_document_write_allowed(None, current_user.id)

    content = await file.read()
    try:
        validate_upload(file, len(content))
    except UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stored_path = save_upload(file, content)
    doc = Document(
        user_id=current_user.id,
        created_by=current_user.id,
        knowledge_base_id=knowledge_base_id,
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        file_path=stored_path,
        file_size=len(content),
        status=DocumentStatus.uploaded,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background.add_task(process_document, doc.id)

    return DocumentOut.model_validate(doc)


@router.get("", response_model=List[DocumentOut])
def list_documents(
    knowledge_base_id: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[DocumentOut]:
    """文档列表；传 ``knowledge_base_id`` 时仅返回该库下未软删文档。"""
    q = db.query(Document).filter(
        Document.status != DocumentStatus.deleted,
    )
    if knowledge_base_id is not None:
        q = q.filter(Document.knowledge_base_id == knowledge_base_id)
    rows = q.order_by(Document.created_at.desc()).all()
    return [DocumentOut.model_validate(r) for r in rows]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """软删除文档及其 chunk；204 无响应体（满足 FastAPI 对 204 的约束）。"""
    doc = db.query(Document).filter(Document.id == document_id).one_or_none()
    if doc is None or doc.status == DocumentStatus.deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    _ensure_public_document_write_allowed(doc, current_user.id)
    ok = soft_delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

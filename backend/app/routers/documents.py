from __future__ import annotations

import uuid
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

from ..database import get_db
from ..models import Document, DocumentStatus, KnowledgeBase
from ..schemas import DocumentOut
from ..services.documents import (
    UploadValidationError,
    process_document,
    save_upload,
    soft_delete_document,
    validate_upload,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background: BackgroundTasks,
    knowledge_base_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentOut:
    kb = db.get(KnowledgeBase, knowledge_base_id)
    if kb is None or kb.is_deleted:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    content = await file.read()
    try:
        validate_upload(file, len(content))
    except UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    stored_path = save_upload(file, content)
    doc = Document(
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
    knowledge_base_id: Optional[uuid.UUID] = Query(default=None),
    db: Session = Depends(get_db),
) -> List[DocumentOut]:
    q = db.query(Document).filter(Document.status != DocumentStatus.deleted)
    if knowledge_base_id is not None:
        q = q.filter(Document.knowledge_base_id == knowledge_base_id)
    rows = q.order_by(Document.created_at.desc()).all()
    return [DocumentOut.model_validate(r) for r in rows]


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_document(document_id: uuid.UUID) -> Response:
    ok = soft_delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

"""Vector retrieval over indexed document chunks using pgvector."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Document, DocumentChunk, DocumentStatus, KnowledgeBase
from .llm import get_embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    knowledge_base_id: str
    filename: str
    chunk_index: int
    content: str
    score: float  # cosine similarity in [0, 1]


def retrieve(
    db: Session,
    query: str,
    knowledge_base_id: uuid.UUID,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """Retrieve top_k chunks belonging to the given knowledge base. Chunks are
    filtered to ensure both the chunk, its document, and the knowledge base
    are not soft-deleted."""
    settings = get_settings()
    top_k = top_k or settings.rag_top_k
    embedder = get_embedder()
    query_vec = embedder.embed_one(query)

    distance = DocumentChunk.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            Document.filename,
            Document.knowledge_base_id,
            distance,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(KnowledgeBase, KnowledgeBase.id == Document.knowledge_base_id)
        .where(
            DocumentChunk.is_deleted.is_(False),
            Document.status == DocumentStatus.indexed,
            Document.knowledge_base_id == knowledge_base_id,
            KnowledgeBase.is_deleted.is_(False),
        )
        .order_by(distance.asc())
        .limit(top_k)
    )

    rows = db.execute(stmt).all()
    results: List[RetrievedChunk] = []
    for row in rows:
        dist = float(row.distance) if row.distance is not None else 1.0
        score = max(0.0, 1.0 - dist)
        results.append(
            RetrievedChunk(
                chunk_id=str(row.id),
                document_id=str(row.document_id),
                knowledge_base_id=str(row.knowledge_base_id),
                filename=row.filename,
                chunk_index=int(row.chunk_index),
                content=row.content,
                score=score,
            )
        )
    return results

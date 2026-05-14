"""基于 pgvector 的向量检索服务。

将用户问题（或经 memory_service 组合后的检索语句）嵌入为查询向量，在
``document_chunks`` 表中按余弦距离排序，返回 top_k 条最相关片段。仅检索：
- 指定知识库 ``knowledge_base_id`` 下的文档；
- 文档状态为 ``indexed``、知识库未软删、chunk 未软删。

相似度分数：``score = max(0, 1 - cosine_distance)``，数值越大表示与查询越接近。
上层（chat 服务）再结合 ``RAG_SCORE_THRESHOLD`` 决定是否启用 RAG。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Document, DocumentChunk, DocumentStatus, KnowledgeBase
from .llm import get_embedder


@dataclass
class RetrievedChunk:
    """单条检索命中的文本块及其元数据（供 prompt 与 sources 使用）。"""

    chunk_id: int
    document_id: int
    knowledge_base_id: int
    filename: str
    chunk_index: int
    content: str
    score: float  # 余弦相似度，约落在 [0, 1]


def retrieve(
    db: Session,
    query: str,
    knowledge_base_id: int,
    top_k: int | None = None,
) -> List[RetrievedChunk]:
    """在指定知识库内执行向量检索，返回 top_k 条候选 chunk。

    流程简述：
    1. 用 Embedding API 将 ``query`` 转为向量；
    2. SQL 中 ``DocumentChunk.embedding.cosine_distance(query_vec)`` 排序；
    3. JOIN 文档与知识库表，过滤已删除/未索引状态；
    4. 将距离转为相似度分数写回 ``RetrievedChunk``。

    Args:
        db: 数据库会话。
        query: 检索用自然语言（建议为「当前问题 + 上一轮用户话」组合，见 memory_service）。
        knowledge_base_id: 严格限定检索范围的知识库 ID，禁止跨库。
        top_k: 返回条数；默认读配置 ``RAG_TOP_K``。

    Returns:
        按相似度从高到低排列的 ``RetrievedChunk`` 列表（最多 top_k 条）。
    """
    settings = get_settings()
    top_k = top_k or settings.rag_top_k
    embedder = get_embedder()
    query_vec = embedder.embed_one(query)

    # pgvector：余弦距离越小越相似；对外统一用 similarity = 1 - distance
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
                chunk_id=int(row.id),
                document_id=int(row.document_id),
                knowledge_base_id=int(row.knowledge_base_id),
                filename=row.filename,
                chunk_index=int(row.chunk_index),
                content=row.content,
                score=score,
            )
        )
    return results

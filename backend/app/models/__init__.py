"""SQLAlchemy ORM 模型包。

各表对应业务实体：知识库、会话、消息、文档、文档向量块。启动时由
``database.init_db`` 导入本包以注册 ``Base.metadata``，再 ``create_all`` 建表。
"""
from .knowledge_base import KnowledgeBase
from .session import Session, ChatMode
from .message import Message
from .document import Document, DocumentStatus
from .chunk import DocumentChunk

__all__ = [
    "KnowledgeBase",
    "Session",
    "ChatMode",
    "Message",
    "Document",
    "DocumentStatus",
    "DocumentChunk",
]

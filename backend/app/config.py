"""应用配置模块。

通过 Pydantic Settings 从环境变量与多个候选路径下的 ``.env`` 文件加载配置。
支持在仓库根目录（Docker）或 ``backend/``（PyCharm 工作目录）启动时都能定位到
同一份 ``.env``。

注意：
- ``cors_origins`` 在环境变量中以逗号分隔字符串形式存在，避免 pydantic-settings
  v2 对 ``List[str]`` 先做 JSON 解析导致解析失败；对外通过计算属性暴露为列表。
- ``get_settings`` 使用 ``lru_cache`` 单例化，进程内只解析一次环境变量。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_files() -> tuple[str, ...]:
    """查找可用的 ``.env`` 文件路径列表。

    搜索顺序：
    1. 当前工作目录及其最多 3 层父目录中的 ``.env``（适配在仓库根执行 uvicorn）；
    2. 本模块所在包的上级目录（通常为 ``backend/``）及其父目录（仓库根）。

    若均未找到，返回 ``(".env",)`` 作为占位，由 pydantic 按默认行为处理。

    Returns:
        一个或多个 ``.env`` 绝对路径组成的元组，按发现顺序排列。
    """
    here = Path.cwd().resolve()
    candidates: list[str] = []
    for d in (here, *here.parents[:3]):
        candidate = d / ".env"
        if candidate.exists():
            candidates.append(str(candidate))
    # 与当前工作目录无关：根据本文件位置探测 backend/ 与仓库根
    here_module = Path(__file__).resolve().parent.parent
    for d in (here_module, here_module.parent):
        candidate = d / ".env"
        if candidate.exists() and str(candidate) not in candidates:
            candidates.append(str(candidate))
    return tuple(candidates) or (".env",)


class Settings(BaseSettings):
    """应用级配置：数据库、LLM、Embedding、RAG、Memory、上传、调试等。

    字段名使用小写蛇形，对应环境变量同名大写（如 ``DATABASE_URL``）。
    未列出的环境变量会被 ``extra="ignore"`` 忽略，避免误配导致启动失败。
    """

    model_config = SettingsConfigDict(
        env_file=_find_env_files(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    # 原始逗号分隔字符串；避免 pydantic-settings v2 将 List 当 JSON 解析失败
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins_raw"),
    )

    database_url: str = "postgresql+psycopg://lrag:lrag@localhost:5432/lrag"

    # ----- LLM（对话生成，OpenAI SDK 兼容 DashScope / DeepSeek 等） -----
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 1

    # ----- Embedding（向量检索用） -----
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""
    embedding_base_url: str | None = None

    # ----- RAG 切片与检索 -----
    rag_top_k: int = 5
    rag_score_threshold: float = 0.25
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ----- 文件上传 -----
    upload_dir: str = "/app/uploads"
    max_upload_mb: int = 20

    agent_backend: str = "langgraph"
    langgraph_checkpoint_url: str | None = None
    public_kb_write_mode: str = "all"
    enable_agent_rag: bool = True
    static_dir: str = "/app/static"

    @computed_field  # type: ignore[misc]
    @property
    def cors_origins(self) -> List[str]:
        """解析后的浏览器跨域来源列表，供 FastAPI CORSMiddleware 使用。"""
        return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内单例 ``Settings`` 实例。

    首次调用时从环境变量与 ``.env`` 加载；之后直接返回缓存结果。
    单元测试中若需覆盖配置，需先 ``get_settings.cache_clear()`` 再改环境变量。
    """
    return Settings()

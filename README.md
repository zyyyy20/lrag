# LRAG · Lightweight Production-Grade RAG Chat

一个轻量但接近生产级别的 RAG（Retrieval-Augmented Generation）对话系统，
支持普通多轮聊天 + 文档知识库问答。

- **前端**：Next.js 14 (App Router) + React + TypeScript + TailwindCSS
- **后端**：FastAPI + SQLAlchemy 2.0 + Pydantic v2
- **数据库**：PostgreSQL 16 + [pgvector](https://github.com/pgvector/pgvector)
- **LLM**：OpenAI API（通过 `LLM_BASE_URL` 可无缝切换至 DeepSeek / Qwen DashScope 兼容模式）
- **Embedding**：`text-embedding-3-small`（1536 维）
- **部署**：Docker Compose（数据库 + 后端容器化，前端由用户本地运行）

---

## 功能一览

- 普通对话：未上传文档时也能正常 AI 多轮对话
- 会话管理：新建 / 列表 / 详情 / 软删除，首条消息后自动生成会话标题
- 文档知识库：上传 PDF / TXT / Markdown，自动解析 → chunk → embedding → 入库
- RAG：每次提问先做 top_k 向量检索，命中阈值则增强生成并返回真实 sources，否则降级为普通聊天
- 不伪造引用：sources 全部来自实际命中的 chunk
- 删除文档：软删除并立即从检索范围移除（删除其 chunks）
- 统一错误处理、类型注解、文件上传校验、CORS、健康检查

---

## 目录结构

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # Pydantic Settings
│   │   ├── database.py          # SQLAlchemy 引擎 / Session / init_db
│   │   ├── errors.py            # 统一异常处理
│   │   ├── models/              # ORM models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── routers/             # FastAPI routers
│   │   └── services/            # LLM / embedding / 检索 / 文档处理 / 聊天
│   ├── scripts/init.sql         # 启用 pgvector
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                     # Next.js App Router
│   ├── components/              # Sidebar / ChatArea / DocumentsPanel
│   ├── lib/                     # 类型 + API 客户端
│   ├── package.json
│   ├── tailwind.config.ts
│   └── tsconfig.json
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 快速开始

### 1) 准备环境变量

```bash
cp .env.example .env
# 编辑 .env，至少设置：
#   LLM_API_KEY=<your_openai_key>
#   EMBEDDING_API_KEY=<your_openai_key>
```

> 想用 DeepSeek / 通义千问？只需修改 `.env`：
>
> ```
> LLM_PROVIDER=deepseek
> LLM_MODEL=deepseek-chat
> LLM_BASE_URL=https://api.deepseek.com/v1
> LLM_API_KEY=sk-xxxx
> ```
>
> Embedding 推荐继续使用 OpenAI 的 `text-embedding-3-small`，
> 或换成同样兼容 OpenAI Embedding 协议的供应商，并相应调整 `EMBEDDING_DIM`。

### 2) 启动数据库与后端（Docker）

```bash
docker compose up --build

只启动数据库：
docker-compose -f docker-compose.db.yml up -d

```

启动后：

- Postgres + pgvector：`localhost:5432`
- 后端 API：`http://localhost:8000`
- 健康检查：`http://localhost:8000/api/health`

### 3) 启动前端（本地运行）

```bash
cd frontend
cp .env.local.example .env.local   # 可选，默认指向 http://localhost:8000
npm install
npm run dev
```

打开 [http://localhost:3000](http://localhost:3000)。

---

## API

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| GET    | `/api/health`                 | 健康检查（含数据库连通性） |
| POST   | `/api/sessions`               | 新建会话 |
| GET    | `/api/sessions`               | 会话列表（未删除） |
| GET    | `/api/sessions/{id}`          | 会话详情（含消息） |
| DELETE | `/api/sessions/{id}`          | 软删除会话 |
| POST   | `/api/chat`                   | 聊天（自动 RAG 或降级普通聊天） |
| POST   | `/api/documents/upload`       | 上传 PDF / TXT / MD |
| GET    | `/api/documents`              | 文档列表（不含已删除） |
| DELETE | `/api/documents/{id}`         | 软删除文档（同时清空 chunks） |

`/api/chat` 响应示例：

```json
{
  "session_id": "…",
  "answer": "…",
  "used_rag": true,
  "sources": [
    {
      "filename": "handbook.pdf",
      "chunk_index": 3,
      "score": 0.812,
      "content_preview": "…",
      "document_id": "…"
    }
  ]
}
```

---

## 数据模型

- `sessions(id, title, is_deleted, created_at, updated_at)`
- `messages(id, session_id, role, content, used_rag, sources, created_at)`
- `documents(id, filename, content_type, file_path, file_size, status, error, chunk_count, created_at, updated_at)`
- `document_chunks(id, document_id, chunk_index, content, token_count, embedding vector(1536), created_at)`

`document.status ∈ { uploaded, processing, indexed, failed, deleted }`。

---

## RAG 流程

1. 用户在 `/api/chat` 发送 message
2. 将 message 嵌入为 query 向量
3. 在 `document_chunks`（仅 `documents.status = indexed`）中做 cosine 距离 top_k 检索
4. `similarity = 1 - distance`；过滤 `score >= RAG_SCORE_THRESHOLD`
5. 若仍有命中：构建 `CONTEXT` 块并加上明确的"禁止伪造引用"的 system prompt → LLM
6. 若无命中：降级为普通多轮聊天
7. 返回 `answer / used_rag / sources`

可调参数（位于 `.env`）：

```
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.25
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

---

## 验收对照

- [x] 不上传文档时可以正常聊天（降级路径）
- [x] 上传文档后可以做 RAG 问答
- [x] 会话历史可查看（`GET /api/sessions/{id}`）
- [x] 删除会话有效（软删除，列表中不再显示）
- [x] 删除文档后不再参与检索（status=deleted 且 chunks 被清空）
- [x] 返回真实 sources（不伪造）
- [x] `docker compose up --build` 可启动 Postgres + 后端

---

## 备注

- 上传大小默认上限 20MB，支持 `.pdf / .docx / .txt / .md / .markdown`（暂不支持旧版 `.doc`）
- 文档处理在 FastAPI BackgroundTask 中异步执行；前端会自动轮询状态
- 后端启动时会自动 `CREATE EXTENSION vector` 并通过 SQLAlchemy 建表
- API Key 仅通过环境变量加载，不会硬编码进代码

"""Conversation invoice tool."""
from __future__ import annotations

import html
import logging
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Message
from ..models import Session as ChatSession
from ..services.llm import get_llm
from ..services.memory_service import count_tokens
from .base import ToolContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

TOOL_NAME = "generate_conversation_invoice"
DEFAULT_SUMMARY_TOKEN_BUDGET = 800
MAX_SUMMARY_TOKEN_BUDGET = 1600


def _safe_int(value: Any, default: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(100, min(parsed, upper))


def _load_session(db: Session, ctx: ToolContext) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == ctx.session_id, ChatSession.user_id == ctx.user_id)
        .one_or_none()
    )
    if session is None or session.is_deleted:
        raise ValueError("Session not found")
    return session


def _load_messages(db: Session, ctx: ToolContext) -> list[Message]:
    return (
        db.query(Message)
        .filter(Message.session_id == ctx.session_id, Message.user_id == ctx.user_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )


def _fallback_summary(messages: list[Message], budget_tokens: int) -> str:
    parts: list[str] = []
    used = 0
    for msg in messages:
        if msg.role not in {"user", "assistant"}:
            continue
        line = f"{msg.role}: {(msg.content or '').strip()}"
        tokens = count_tokens(line)
        if used + tokens > budget_tokens and parts:
            break
        parts.append(line[:1000])
        used += tokens
    if not parts:
        return "本次会话暂无可总结内容。"
    return "\n".join(parts)


def _summarize(messages: list[Message], budget_tokens: int) -> str:
    content = _fallback_summary(messages, 2600)
    prompt = (
        "请用中文总结下面这段对话。要求：\n"
        "1. 保留用户主要问题、助手主要结论和关键上下文。\n"
        "2. 不编造没有出现的信息。\n"
        f"3. 控制在约 {budget_tokens} token 以内。\n\n"
        f"{content}"
    )
    try:
        return get_llm().chat(
            [
                {"role": "system", "content": "你是严谨的对话摘要助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=budget_tokens,
        )
    except Exception:
        logger.warning("Invoice summary generation failed; using fallback", exc_info=True)
        return _fallback_summary(messages, budget_tokens)


def _static_root() -> Path:
    settings = get_settings()
    return Path(settings.static_dir).resolve()


def _invoice_dir() -> Path:
    path = _static_root() / "invoices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "session"


def _paper_html(
    *,
    invoice_no: str,
    session: ChatSession,
    generated_at: datetime,
    summary: str,
    user_count: int,
    assistant_count: int,
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    summary_tokens: int,
) -> str:
    title = html.escape(session.title or "Conversation")
    summary_html = html.escape(summary).replace("\n", "<br>")
    generated = html.escape(generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LRAG 对话用量发票 {html.escape(invoice_no)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #65758b;
      --paper: #f7f2e8;
      --line: #d8cbb8;
      --accent: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      padding: 36px 18px;
      background: #d7dde4;
      color: var(--ink);
      font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }}
    .paper {{
      position: relative;
      max-width: 860px;
      margin: 0 auto;
      padding: 44px 52px 50px;
      background:
        radial-gradient(circle at 18% 22%, rgba(255,255,255,.42) 0 1px, transparent 1.5px),
        radial-gradient(circle at 72% 64%, rgba(54,44,30,.10) 0 1px, transparent 1.4px),
        linear-gradient(90deg, rgba(255,255,255,.34), transparent 18%, rgba(93,71,48,.08) 50%, transparent 78%),
        var(--paper);
      background-size: 9px 9px, 11px 11px, 100% 100%, auto;
      border: 1px solid rgba(111, 85, 55, .25);
      box-shadow: 0 24px 60px rgba(35, 45, 64, .22), inset 0 0 42px rgba(111, 85, 55, .08);
    }}
    .paper::before {{
      content: "";
      position: absolute;
      inset: 18px;
      border: 1px dashed rgba(121, 92, 54, .34);
      pointer-events: none;
    }}
    .fold {{
      position: absolute;
      left: 0;
      right: 0;
      top: 132px;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(83, 72, 56, .22), transparent);
    }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      border-bottom: 2px solid var(--line);
      padding-bottom: 22px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: 0;
    }}
    .sub {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    .stamp {{
      transform: rotate(-8deg);
      border: 3px double var(--accent);
      color: var(--accent);
      padding: 10px 14px;
      font-weight: 800;
      letter-spacing: 2px;
      text-align: center;
      opacity: .82;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 24px;
      margin: 26px 0;
      font-size: 14px;
    }}
    .label {{ color: var(--muted); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 22px 0 28px;
      font-size: 14px;
      background: rgba(255,255,255,.2);
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 12px 14px;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: rgba(255,255,255,.26);
    }}
    td.num {{
      font-family: "SFMono-Regular", Consolas, monospace;
      text-align: right;
    }}
    .summary {{
      position: relative;
      margin-top: 24px;
      padding: 18px 18px 20px;
      border-left: 4px solid rgba(185, 28, 28, .58);
      background: rgba(255,255,255,.28);
      line-height: 1.75;
      font-size: 14px;
    }}
    .watermark {{
      position: absolute;
      right: 34px;
      bottom: 24px;
      color: rgba(60, 47, 34, .08);
      font-size: 42px;
      font-weight: 900;
      letter-spacing: 1px;
      pointer-events: none;
    }}
    footer {{
      margin-top: 30px;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }}
    @media (max-width: 700px) {{
      body {{ padding: 0; background: var(--paper); }}
      .paper {{ padding: 28px 22px 34px; box-shadow: none; border: 0; }}
      header {{ flex-direction: column; }}
      .meta {{ grid-template-columns: 1fr; }}
      .watermark {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="paper">
    <div class="fold"></div>
    <header>
      <div>
        <h1>LRAG 对话用量发票</h1>
        <div class="sub">Invoice No. {html.escape(invoice_no)}<br>Generated at {generated}</div>
      </div>
      <div class="stamp">DEBUG<br>INVOICE</div>
    </header>

    <section class="meta">
      <div><span class="label">会话标题：</span>{title}</div>
      <div><span class="label">会话编号：</span>{session.id}</div>
      <div><span class="label">用户消息：</span>{user_count}</div>
      <div><span class="label">助手消息：</span>{assistant_count}</div>
    </section>

    <table>
      <thead>
        <tr><th>项目</th><th>说明</th><th>Token</th></tr>
      </thead>
      <tbody>
        <tr><td>Prompt</td><td>用户输入与上下文估算</td><td class="num">{prompt_tokens}</td></tr>
        <tr><td>Completion</td><td>助手回答估算</td><td class="num">{completion_tokens}</td></tr>
        <tr><td>Summary</td><td>本发票摘要估算</td><td class="num">{summary_tokens}</td></tr>
        <tr><td><strong>Total</strong></td><td>当前会话累计估算</td><td class="num"><strong>{total_tokens}</strong></td></tr>
      </tbody>
    </table>

    <section class="summary">
      <strong>对话内容摘要</strong><br>
      {summary_html}
    </section>

    <footer>
      <span>Token 为服务端估算值，后续可接入模型 usage 精确计量。</span>
      <span>LRAG / Debug Guest</span>
    </footer>
    <div class="watermark">LRAG DEBUG INVOICE</div>
  </main>
</body>
</html>
"""


def generate_conversation_invoice(
    ctx: ToolContext, arguments: dict[str, Any]
) -> ToolResult:
    session = _load_session(ctx.db, ctx)
    messages = _load_messages(ctx.db, ctx)
    budget = _safe_int(
        arguments.get("summary_token_budget"),
        DEFAULT_SUMMARY_TOKEN_BUDGET,
        MAX_SUMMARY_TOKEN_BUDGET,
    )

    user_messages = [m for m in messages if m.role == "user"]
    assistant_messages = [m for m in messages if m.role == "assistant"]
    prompt_tokens = sum(count_tokens(m.content or "") for m in user_messages)
    completion_tokens = sum(count_tokens(m.content or "") for m in assistant_messages)
    summary = _summarize(messages, budget)
    summary_tokens = count_tokens(summary)
    total_tokens = prompt_tokens + completion_tokens

    now = datetime.now(timezone.utc)
    invoice_no = f"LRAG-{now.strftime('%Y%m%d%H%M%S')}-{ctx.session_id}-{secrets.token_hex(3)}"
    filename = f"{_slug(invoice_no)}.html"
    path = _invoice_dir() / filename
    path.write_text(
        _paper_html(
            invoice_no=invoice_no,
            session=session,
            generated_at=now,
            summary=summary,
            user_count=len(user_messages),
            assistant_count=len(assistant_messages),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            summary_tokens=summary_tokens,
        ),
        encoding="utf-8",
    )

    return ToolResult(
        tool=TOOL_NAME,
        ok=True,
        message="已生成本次对话用量发票。",
        data={
            "title": "对话用量发票",
            "html_url": f"/static/invoices/{filename}",
            "invoice_no": invoice_no,
            "token_total": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "summary_tokens": summary_tokens,
            "message_count": len(messages),
        },
    )


invoice_tool = ToolSpec(
    name=TOOL_NAME,
    description=(
        "Generate a paper-like HTML invoice for the current conversation, including "
        "estimated token usage and a concise conversation summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary_token_budget": {
                "type": "integer",
                "minimum": 100,
                "maximum": MAX_SUMMARY_TOKEN_BUDGET,
                "description": "Approximate token budget for the conversation summary.",
            }
        },
        "required": [],
        "additionalProperties": False,
    },
    handler=generate_conversation_invoice,
)

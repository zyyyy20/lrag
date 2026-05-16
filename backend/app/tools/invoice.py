"""Conversation invoice tool."""
from __future__ import annotations

import html
import logging
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Message
from ..models import Session as ChatSession
from ..services.llm import get_llm
from ..services.memory_service import count_tokens
from .base import ToolContext, ToolResult
from .runtime import get_tool_context

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
    issue_date = html.escape(generated_at.strftime("%Y-%m-%d"))
    check_code = html.escape(invoice_no[-12:].upper())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LRAG 对话用量电子票据 {html.escape(invoice_no)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #20242b;
      --muted: #667085;
      --paper: #f8f3e7;
      --paper-deep: #ede0c8;
      --line: #b9aa92;
      --line-soft: #d8cbb7;
      --red: #b42318;
      --blue: #245b8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      padding: 38px 18px;
      background:
        radial-gradient(circle at top left, rgba(255,255,255,.55), transparent 28rem),
        linear-gradient(135deg, #c9d0d8, #eef1f4 48%, #c7cdd5);
      color: var(--ink);
      font-family: "Microsoft YaHei", "PingFang SC", "Inter", "Segoe UI", system-ui, sans-serif;
    }}
    .sheet {{
      position: relative;
      max-width: 920px;
      margin: 0 auto;
      padding: 22px;
      background:
        radial-gradient(circle at 12% 18%, rgba(255,255,255,.55) 0 1px, transparent 1.6px),
        radial-gradient(circle at 72% 68%, rgba(68,52,34,.10) 0 1px, transparent 1.5px),
        repeating-linear-gradient(8deg, rgba(107,85,53,.045) 0 1px, transparent 1px 7px),
        linear-gradient(92deg, rgba(255,255,255,.45), transparent 20%, rgba(94,72,45,.10) 54%, transparent 82%),
        var(--paper);
      background-size: 10px 10px, 13px 13px, auto, 100% 100%, auto;
      border: 1px solid rgba(98,74,45,.34);
      box-shadow:
        0 28px 70px rgba(34, 44, 58, .28),
        inset 0 0 52px rgba(87, 63, 35, .12);
    }}
    .sheet::before,
    .sheet::after {{
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      width: 16px;
      background:
        radial-gradient(circle, #cfd6dd 0 4px, transparent 4.5px) 50% 10px / 16px 22px repeat-y;
      pointer-events: none;
    }}
    .sheet::before {{ left: -8px; }}
    .sheet::after {{ right: -8px; }}
    .inner {{
      position: relative;
      padding: 34px 42px 38px;
      border: 1px solid rgba(123, 94, 58, .42);
      background: rgba(255, 252, 245, .18);
    }}
    .inner::before {{
      content: "";
      position: absolute;
      inset: 10px;
      border: 1px dashed rgba(129, 98, 59, .36);
      pointer-events: none;
    }}
    .fold {{
      position: absolute;
      left: 0;
      right: 0;
      top: 148px;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(83, 72, 56, .22), transparent);
    }}
    .serial {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      color: var(--blue);
      font-size: 12px;
      font-family: "SFMono-Regular", Consolas, monospace;
      letter-spacing: 0;
    }}
    header {{
      position: relative;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      border-top: 3px double var(--red);
      border-bottom: 3px double var(--red);
      padding: 18px 0 16px;
    }}
    h1 {{
      margin: 0;
      color: var(--red);
      font-size: 31px;
      font-weight: 800;
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
      border: 3px double var(--red);
      border-radius: 999px;
      color: var(--red);
      padding: 16px 13px;
      min-width: 92px;
      min-height: 92px;
      font-weight: 800;
      letter-spacing: 1px;
      text-align: center;
      opacity: .78;
      display: grid;
      place-items: center;
      line-height: 1.25;
    }}
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      border: 1px solid var(--line);
      border-bottom: 0;
      margin: 22px 0 0;
      font-size: 14px;
    }}
    .info-cell {{
      min-height: 54px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }}
    .info-cell:nth-child(odd) {{ border-right: 1px solid var(--line); }}
    .label {{
      display: inline-block;
      min-width: 76px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0 0;
      font-size: 14px;
      background: rgba(255,255,255,.14);
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 11px 12px;
      text-align: left;
    }}
    th {{
      color: #574832;
      font-weight: 650;
      background: rgba(237, 224, 200, .44);
    }}
    td.num {{
      font-family: "SFMono-Regular", Consolas, monospace;
      text-align: right;
    }}
    .total-row td {{
      font-weight: 800;
      background: rgba(255, 255, 255, .30);
    }}
    .amount {{
      display: grid;
      grid-template-columns: 1fr 220px;
      border: 1px solid var(--line);
      border-top: 0;
      font-size: 14px;
    }}
    .amount div {{
      padding: 12px;
    }}
    .amount div:first-child {{
      border-right: 1px solid var(--line);
    }}
    .summary {{
      position: relative;
      margin-top: 24px;
      padding: 18px 18px 20px 20px;
      border: 1px solid var(--line);
      border-left: 5px solid rgba(180, 35, 24, .70);
      background: rgba(255,255,255,.24);
      line-height: 1.75;
      font-size: 14px;
    }}
    .signatures {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-top: 26px;
      color: var(--muted);
      font-size: 13px;
    }}
    .signature {{
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    .qr {{
      width: 78px;
      height: 78px;
      margin-left: auto;
      border: 1px solid #7b7f85;
      background:
        linear-gradient(90deg, #222 10px, transparent 10px 14px, #222 14px 22px, transparent 22px),
        linear-gradient(#222 9px, transparent 9px 15px, #222 15px 24px, transparent 24px),
        repeating-linear-gradient(45deg, #222 0 3px, #fff 3px 7px);
      background-size: 26px 26px, 28px 28px, 12px 12px;
      opacity: .76;
    }}
    .watermark {{
      position: absolute;
      left: 50%;
      top: 49%;
      transform: translate(-50%, -50%) rotate(-18deg);
      color: rgba(180, 35, 24, .055);
      font-size: 66px;
      font-weight: 900;
      letter-spacing: 1px;
      pointer-events: none;
      white-space: nowrap;
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
      .sheet {{ padding: 0; box-shadow: none; border: 0; }}
      .sheet::before, .sheet::after {{ display: none; }}
      .inner {{ padding: 24px 18px 30px; border: 0; }}
      header {{ flex-direction: column; }}
      .info-grid, .amount, .signatures {{ grid-template-columns: 1fr; }}
      .info-cell:nth-child(odd), .amount div:first-child {{ border-right: 0; }}
      .watermark {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="sheet">
    <section class="inner">
      <div class="fold"></div>
      <div class="serial">
        <span>电子票据代码 3100-DEBUG-LRAG</span>
        <span>校验码 {check_code}</span>
      </div>
      <header>
        <div>
          <h1>LRAG 对话用量电子发票</h1>
          <div class="sub">发票号码：{html.escape(invoice_no)}<br>开票日期：{issue_date}　生成时间：{generated}</div>
        </div>
        <div class="qr" aria-label="invoice verification mark"></div>
        <div class="stamp">LRAG<br>已开具</div>
      </header>

      <section class="info-grid">
        <div class="info-cell"><span class="label">购买方：</span>Debug Guest User #{session.user_id}</div>
        <div class="info-cell"><span class="label">销售方：</span>LRAG Conversation Service</div>
        <div class="info-cell"><span class="label">会话标题：</span>{title}</div>
        <div class="info-cell"><span class="label">会话编号：</span>{session.id}</div>
        <div class="info-cell"><span class="label">用户消息：</span>{user_count} 条</div>
        <div class="info-cell"><span class="label">助手消息：</span>{assistant_count} 条</div>
      </section>

      <table>
        <thead>
          <tr><th>货物或应税劳务、服务名称</th><th>规格型号</th><th>单位</th><th>数量</th><th>Token</th></tr>
        </thead>
        <tbody>
          <tr><td>*信息技术服务*用户输入与上下文</td><td>Prompt</td><td>token</td><td class="num">1</td><td class="num">{prompt_tokens}</td></tr>
          <tr><td>*信息技术服务*模型回复内容</td><td>Completion</td><td>token</td><td class="num">1</td><td class="num">{completion_tokens}</td></tr>
          <tr><td>*信息技术服务*票据摘要生成</td><td>Summary</td><td>token</td><td class="num">1</td><td class="num">{summary_tokens}</td></tr>
          <tr class="total-row"><td colspan="4">合计</td><td class="num">{total_tokens}</td></tr>
        </tbody>
      </table>
      <section class="amount">
        <div><span class="label">价税合计：</span>零元整（Debug 估算票据，不作为真实结算凭证）</div>
        <div><span class="label">Token 合计：</span><strong>{total_tokens}</strong></div>
      </section>

      <section class="summary">
        <strong>对话内容摘要</strong><br>
        {summary_html}
      </section>

      <section class="signatures">
        <div class="signature">收款人：System</div>
        <div class="signature">复核：ReAct Agent</div>
        <div class="signature">开票人：LRAG Tool</div>
      </section>

      <footer>
        <span>本票据为系统调试用途，Token 为服务端估算值；后续可接入模型 usage 精确计量。</span>
        <span>LRAG / Debug Guest</span>
      </footer>
      <div class="watermark">LRAG DEBUG INVOICE</div>
    </section>
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


class InvoiceArgs(BaseModel):
    summary_token_budget: int = Field(
        default=DEFAULT_SUMMARY_TOKEN_BUDGET,
        ge=100,
        le=MAX_SUMMARY_TOKEN_BUDGET,
        description="Approximate token budget for the generated conversation summary.",
    )


generate_conversation_invoice_impl = generate_conversation_invoice


def build_runtime_generate_conversation_invoice_tool():
    @tool(args_schema=InvoiceArgs)
    def generate_conversation_invoice(summary_token_budget: int = DEFAULT_SUMMARY_TOKEN_BUDGET) -> dict:
        """Generate a paper-like HTML invoice for the current conversation."""
        result = generate_conversation_invoice_impl(
            get_tool_context(),
            {"summary_token_budget": summary_token_budget},
        )
        return result.model_dump()

    return generate_conversation_invoice

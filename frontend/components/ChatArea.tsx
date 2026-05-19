"use client";

import { useEffect, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import { webSourcesFromToolResult } from "@/lib/webSources";
import type {
  ActionRequiredEvent,
  AgentTraceEvent,
  Message,
  SessionItem,
  Source,
  ToolResult,
  WebSource,
} from "@/lib/types";

interface Props {
  messages: Message[];
  loading: boolean;
  sending: boolean;
  error: string | null;
  notice: string | null;
  currentSession: SessionItem | null;
  onPublishCsdnDraft?: (
    messageId: Message["id"],
    payload: { cookie: string; category?: string }
  ) => void;
  onDismissAction?: (messageId: Message["id"]) => void;
  onSend: (text: string) => void;
}

export function ChatArea({
  messages,
  loading,
  sending,
  error,
  notice,
  currentSession,
  onPublishCsdnDraft,
  onDismissAction,
  onSend,
}: Props) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, sending]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    onSend(text);
    setInput("");
  };

  return (
    <section className="flex h-full flex-1 flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold text-slate-800">对话</h1>
          {currentSession?.chat_mode === "rag" && currentSession?.knowledge_base_name && (
            <span className="inline-flex items-center rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
              📚 {currentSession.knowledge_base_name}
            </span>
          )}
          {currentSession?.chat_mode === "general" && (
            <span className="inline-flex items-center rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
              普通聊天
            </span>
          )}
        </div>
        <p className="mt-0.5 text-xs text-slate-500">
          {currentSession?.chat_mode === "rag"
            ? "本会话已绑定知识库，提问时将自动基于该知识库进行 RAG 检索。"
            : "未绑定知识库。如需基于文档问答，请在右栏选择知识库并新建会话。"}
        </p>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        {loading && (
          <div className="text-center text-sm text-slate-400">加载中…</div>
        )}
        {!loading && messages.length === 0 && !sending && (
          <div className="mx-auto max-w-md rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
            在下方输入框开始提问。当检索到相关文档时，将自动使用 RAG 增强回答。
          </div>
        )}
        <ul className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.map((m, idx) => {
            const isLast = idx === messages.length - 1;
            const isStreaming = sending && isLast && m.role === "assistant";
            return (
              <MessageBubble
                key={m.id}
                message={m}
                streaming={isStreaming}
                onPublishCsdnDraft={onPublishCsdnDraft}
                onDismissAction={onDismissAction}
              />
            );
          })}
        </ul>
      </div>

      {notice && (
        <div className="mx-6 mb-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
          {notice}
        </div>
      )}
      {error && (
        <div className="mx-6 mb-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}
      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-200 bg-white px-6 py-4"
      >
        <div className="mx-auto flex max-w-3xl gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="输入消息…（Enter 发送，Shift + Enter 换行）"
            rows={2}
            className="flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            disabled={sending}
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="self-end rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            发送
          </button>
        </div>
      </form>
    </section>
  );
}

function MessageBubble({
  message,
  streaming,
  onPublishCsdnDraft,
  onDismissAction,
}: {
  message: Message;
  streaming?: boolean;
  onPublishCsdnDraft?: (
    messageId: Message["id"],
    payload: { cookie: string; category?: string }
  ) => void;
  onDismissAction?: (messageId: Message["id"]) => void;
}) {
  const isUser = message.role === "user";
  const isEmptyStreaming = streaming && !message.content;
  return (
    <li className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
          isUser ? "bg-brand-600 text-white" : "bg-white text-slate-800"
        }`}
      >
        {isEmptyStreaming ? (
          <span className="inline-block animate-pulse text-slate-500">正在思考…</span>
        ) : (
          <div className="whitespace-pre-wrap">
            {message.content}
            {streaming && (
              <span className="ml-0.5 inline-block h-3.5 w-1.5 -mb-0.5 animate-pulse bg-slate-400 align-middle" />
            )}
          </div>
        )}
        {!isUser && message.used_rag && message.sources && message.sources.length > 0 && (
          <SourcesView sources={message.sources} />
        )}
        {!isUser && message.used_web && message.web_sources && message.web_sources.length > 0 && (
          <WebSourcesView sources={message.web_sources} />
        )}
        {!isUser && (
          <AgentTraceView
            trace={message.agent_trace}
            toolResults={message.tool_results}
          />
        )}
        {!isUser && message.pending_action?.action === "csdn_cookie" && (
          <CsdnPublishCard
            message={message}
            onPublish={onPublishCsdnDraft}
            onDismiss={onDismissAction}
          />
        )}
      </div>
    </li>
  );
}

function CsdnPublishCard({
  message,
  onPublish,
  onDismiss,
}: {
  message: Message;
  onPublish?: (
    messageId: Message["id"],
    payload: { cookie: string; category?: string }
  ) => void;
  onDismiss?: (messageId: Message["id"]) => void;
}) {
  const [cookie, setCookie] = useState("");
  const [category, setCategory] = useState(
    message.pending_action?.draft?.category || "AI Engineering"
  );
  const draft = message.pending_action?.draft;
  const statusText = message.pending_action?.message;
  const isErrorStatus = statusText?.toLowerCase().includes("failed");

  if (!draft) return null;

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const cleanCookie = cookie.trim();
        if (!cleanCookie || !onPublish) return;
        onPublish(message.id, {
          cookie: cleanCookie,
          category: category.trim() || undefined,
        });
        setCookie("");
      }}
      className="mt-3 rounded-md border border-sky-200 bg-sky-50 p-3 text-xs text-slate-800"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-semibold text-sky-950">Publish CSDN draft</div>
          <div className="mt-1 truncate text-sky-800" title={draft.title}>
            {draft.title}
          </div>
          <p className="mt-1 text-sky-700">
            Enter your CSDN Cookie to publish this draft. The cookie is sent only
            with this request.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onDismiss?.(message.id)}
          className="shrink-0 rounded border border-sky-200 bg-white px-2 py-1 text-[11px] font-medium text-sky-700 hover:bg-sky-100"
        >
          Close
        </button>
      </div>
      <details className="mt-2 rounded border border-sky-100 bg-white/70 p-2">
        <summary className="cursor-pointer font-medium text-sky-800">
          Draft preview
        </summary>
        <div className="mt-2 space-y-1 text-slate-600">
          <div>Description: {draft.description}</div>
          <div>Tags: {draft.tags.join(", ") || "-"}</div>
          <pre className="mt-2 max-h-36 overflow-auto whitespace-pre-wrap rounded bg-slate-950 p-2 text-[11px] text-slate-100">
            {draft.markdown_content}
          </pre>
        </div>
      </details>
      <div className="mt-3 grid gap-2 md:grid-cols-[1fr_180px_auto]">
        <input
          type="password"
          value={cookie}
          onChange={(event) => setCookie(event.target.value)}
          placeholder="CSDN Cookie"
          className="min-w-0 rounded border border-sky-200 bg-white px-3 py-2 outline-none focus:border-sky-500"
        />
        <input
          value={category ?? ""}
          onChange={(event) => setCategory(event.target.value)}
          placeholder="Category"
          className="min-w-0 rounded border border-sky-200 bg-white px-3 py-2 outline-none focus:border-sky-500"
        />
        <button
          type="submit"
          disabled={!cookie.trim()}
          className="rounded bg-sky-600 px-3 py-2 font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          Confirm publish
        </button>
      </div>
      {statusText && (
        <div
          className={`mt-2 rounded border px-2 py-1 text-[11px] ${
            isErrorStatus
              ? "border-rose-200 bg-rose-50 text-rose-700"
              : "border-sky-100 bg-white/70 text-sky-700"
          }`}
        >
          {statusText}
        </div>
      )}
    </form>
  );
}

function traceFromToolResult(result: ToolResult, index: number): AgentTraceEvent {
  const rawSources = result.data?.sources;
  const sources = Array.isArray(rawSources) ? (rawSources as Source[]) : [];
  const webSources = webSourcesFromToolResult(result);
  const title =
    result.tool === "retrieve_knowledge_base"
      ? "查询知识库"
      : result.tool === "generate_conversation_invoice"
        ? "生成对话发票"
        : result.tool === "tavily_search"
          ? "联网搜索"
          : result.tool === "tavily_extract"
            ? "读取网页"
        : result.tool;
  const summary =
    result.tool === "retrieve_knowledge_base" && sources.length > 0
      ? `检索完成，命中 ${sources.length} 条相关内容`
      : result.tool === "generate_conversation_invoice" && result.ok
        ? "HTML 对话发票已生成"
        : webSources.length > 0
          ? `检索完成，找到 ${webSources.length} 个网页来源`
        : result.message;

  return {
    id: `saved-${result.tool}-${index}`,
    type: "tool_result",
    title,
    status: result.ok ? "success" : "error",
    tool: result.tool,
    summary,
    sources,
    web_sources: webSources,
  };
}

function AgentTraceView({
  trace,
  toolResults,
}: {
  trace?: AgentTraceEvent[] | null;
  toolResults?: ToolResult[] | null;
}) {
  const events =
    trace && trace.length > 0
      ? trace
      : (toolResults ?? []).map((result, index) =>
          traceFromToolResult(result, index)
        );
  if (events.length === 0) return null;
  const hasRunningStep = events.some((event) => event.status === "running");

  return (
    <details
      className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700"
      open={hasRunningStep || undefined}
    >
      <summary className="cursor-pointer font-medium text-slate-700">
        Agent 执行过程（{events.length}）
      </summary>
      <ol className="mt-3 space-y-2">
        {events.map((event, index) => (
          <AgentTraceItem key={`${event.id}-${index}`} event={event} />
        ))}
      </ol>
      {toolResults?.map((result, index) =>
        result.tool === "generate_conversation_invoice" ? (
          <div className="mt-2" key={`${result.tool}-${index}`}>
            <InvoiceCard result={result} />
          </div>
        ) : null
      )}
    </details>
  );
}

function AgentTraceItem({ event }: { event: AgentTraceEvent }) {
  const marker =
    event.status === "running" ? "…" : event.status === "success" ? "✓" : "!";
  const markerClass =
    event.status === "running"
      ? "border-sky-200 bg-sky-50 text-sky-700"
      : event.status === "success"
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : "border-red-200 bg-red-50 text-red-700";

  return (
    <li className="flex gap-2">
      <span
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[11px] font-semibold ${markerClass}`}
      >
        {marker}
      </span>
      <div className="min-w-0 flex-1">
        <div className="font-medium text-slate-800">{event.title}</div>
        {event.content && (
          <div className="mt-0.5 text-slate-500">{event.content}</div>
        )}
        {event.type === "tool_call" && event.args !== undefined && (
          <pre className="mt-1 max-h-28 overflow-auto rounded border border-slate-200 bg-white p-2 text-[11px] text-slate-600">
            {JSON.stringify(event.args, null, 2)}
          </pre>
        )}
        {event.summary && (
          <div className="mt-0.5 text-slate-600">{event.summary}</div>
        )}
        {event.sources && event.sources.length > 0 && (
          <div className="mt-1 text-[11px] text-slate-500">
            来源：{event.sources.map((source) => source.filename).join("、")}
          </div>
        )}
        {event.web_sources && event.web_sources.length > 0 && (
          <div className="mt-1 text-[11px] text-slate-500">
            网页来源：{event.web_sources.map((source) => source.title).join("、")}
          </div>
        )}
      </div>
    </li>
  );
}

function InvoiceCard({ result }: { result: ToolResult }) {
  const htmlUrl =
    typeof result.data.html_url === "string" ? apiUrl(result.data.html_url) : null;
  const receiptText =
    typeof result.data.receipt_text === "string" ? result.data.receipt_text : null;
  const tokenTotal =
    typeof result.data.token_total === "number" ? result.data.token_total : null;
  const invoiceNo =
    typeof result.data.invoice_no === "string" ? result.data.invoice_no : null;

  return (
    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-950 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-amber-950">
            {result.data.title || "对话用量发票"}
          </div>
          {invoiceNo && (
            <div className="mt-1 font-mono text-[11px] text-amber-700">
              {invoiceNo}
            </div>
          )}
        </div>
        {htmlUrl && (
          <a
            href={htmlUrl}
            target="_blank"
            rel="noreferrer"
            className="shrink-0 rounded border border-amber-300 bg-white px-2.5 py-1 text-xs font-medium text-amber-800 hover:bg-amber-100"
          >
            打开
          </a>
        )}
      </div>
      {receiptText && (
        <pre className="mt-3 max-h-80 overflow-auto rounded border border-stone-300 bg-stone-950 p-3 font-mono text-[11px] leading-relaxed text-stone-100 shadow-inner">
          {receiptText}
        </pre>
      )}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <div className="rounded border border-amber-200 bg-white/70 px-2 py-1.5">
          <div className="text-[11px] text-amber-700">Total tokens</div>
          <div className="font-mono text-sm font-semibold">
            {tokenTotal ?? "-"}
          </div>
        </div>
        <div className="rounded border border-amber-200 bg-white/70 px-2 py-1.5">
          <div className="text-[11px] text-amber-700">Messages</div>
          <div className="font-mono text-sm font-semibold">
            {typeof result.data.message_count === "number"
              ? result.data.message_count
              : "-"}
          </div>
        </div>
      </div>
      <div className="mt-2 text-[12px] text-amber-800">{result.message}</div>
    </div>
  );
}

function SourcesView({ sources }: { sources: Source[] }) {
  return (
    <details className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
      <summary className="cursor-pointer font-medium text-slate-600">
        引用来源（{sources.length}）
      </summary>
      <ul className="mt-2 space-y-2">
        {sources.map((s, i) => (
          <li key={i} className="rounded border border-slate-200 bg-white p-2">
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-700">{s.filename}</span>
              <span className="text-[11px] text-slate-400">
                片段 #{s.chunk_index} · 相似度 {s.score.toFixed(3)}
              </span>
            </div>
            <p className="mt-1 text-[12px] text-slate-600">{s.content_preview}</p>
          </li>
        ))}
      </ul>
    </details>
  );
}

function WebSourcesView({ sources }: { sources: WebSource[] }) {
  return (
    <details className="mt-3 rounded-md border border-sky-200 bg-sky-50 p-2 text-xs text-slate-700">
      <summary className="cursor-pointer font-medium text-sky-800">
        联网来源（{sources.length}）
      </summary>
      <ol className="mt-2 space-y-2">
        {sources.map((source, index) => (
          <li key={`${source.url}-${index}`} className="rounded border border-sky-100 bg-white p-2">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-sky-100 text-[11px] font-semibold text-sky-700">
                {index + 1}
              </span>
              <div className="min-w-0 flex-1">
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer"
                  className="block truncate font-medium text-sky-800 hover:underline"
                  title={source.title}
                >
                  {source.title}
                </a>
                <div className="mt-0.5 truncate text-[11px] text-slate-400">
                  {source.url}
                </div>
                {source.content_preview && (
                  <p className="mt-1 line-clamp-2 text-[12px] text-slate-600">
                    {source.content_preview}
                  </p>
                )}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}

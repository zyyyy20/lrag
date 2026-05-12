"use client";

import { useEffect, useRef, useState } from "react";
import type { Message, SessionItem, Source } from "@/lib/types";

interface Props {
  messages: Message[];
  loading: boolean;
  sending: boolean;
  error: string | null;
  notice: string | null;
  currentSession: SessionItem | null;
  onSend: (text: string) => void;
}

export function ChatArea({
  messages,
  loading,
  sending,
  error,
  notice,
  currentSession,
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
}: {
  message: Message;
  streaming?: boolean;
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
      </div>
    </li>
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

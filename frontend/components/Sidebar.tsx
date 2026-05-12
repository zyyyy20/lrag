"use client";

import { useState } from "react";
import type { KnowledgeBase, SessionItem } from "@/lib/types";

interface Props {
  sessions: SessionItem[];
  currentId: string | null;
  loading?: boolean;
  knowledgeBases: KnowledgeBase[];
  onNew: (opts: { knowledgeBaseId: string | null }) => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({
  sessions,
  currentId,
  loading,
  knowledgeBases,
  onNew,
  onSelect,
  onDelete,
}: Props) {
  const [hover, setHover] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <aside className="flex h-full w-72 flex-col border-r border-slate-200 bg-white">
      <div className="relative flex items-center justify-between border-b border-slate-200 p-3">
        <h2 className="text-sm font-semibold text-slate-700">会话列表</h2>
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="rounded-md bg-brand-600 px-2.5 py-1 text-xs font-medium text-white shadow-sm hover:bg-brand-700"
        >
          + 新建
        </button>
        {menuOpen && (
          <div className="absolute right-3 top-12 z-10 w-60 rounded-md border border-slate-200 bg-white py-1 shadow-lg">
            <button
              onClick={() => {
                onNew({ knowledgeBaseId: null });
                setMenuOpen(false);
              }}
              className="block w-full px-3 py-2 text-left text-sm hover:bg-slate-50"
            >
              <div className="font-medium text-slate-700">普通聊天</div>
              <div className="text-[11px] text-slate-400">不使用知识库</div>
            </button>
            {knowledgeBases.length > 0 && (
              <div className="my-1 border-t border-slate-100" />
            )}
            {knowledgeBases.map((kb) => (
              <button
                key={kb.id}
                onClick={() => {
                  onNew({ knowledgeBaseId: kb.id });
                  setMenuOpen(false);
                }}
                className="block w-full px-3 py-2 text-left text-sm hover:bg-slate-50"
              >
                <div className="truncate font-medium text-slate-700">{kb.name}</div>
                <div className="text-[11px] text-slate-400">
                  使用知识库 · {kb.document_count} 个文档
                </div>
              </button>
            ))}
            {knowledgeBases.length === 0 && (
              <div className="px-3 py-2 text-[11px] text-slate-400">
                暂无知识库。可在右侧"知识库"面板新建。
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading && <div className="p-4 text-xs text-slate-400">加载中…</div>}
        {!loading && sessions.length === 0 && (
          <div className="p-4 text-xs text-slate-400">
            暂无会话，点击右上角"新建"开始对话。
          </div>
        )}
        <ul className="space-y-1 p-2">
          {sessions.map((s) => {
            const active = s.id === currentId;
            return (
              <li
                key={s.id}
                onMouseEnter={() => setHover(s.id)}
                onMouseLeave={() => setHover(null)}
                className={`group flex items-start justify-between rounded-md px-2 py-2 text-sm cursor-pointer transition-colors ${
                  active
                    ? "bg-brand-50 text-brand-700"
                    : "text-slate-700 hover:bg-slate-100"
                }`}
                onClick={() => onSelect(s.id)}
                title={s.title}
              >
                <div className="min-w-0 flex-1 pr-2">
                  <div className="truncate">{s.title || "未命名会话"}</div>
                  {s.chat_mode === "rag" && s.knowledge_base_name && (
                    <div className="mt-0.5 inline-flex items-center rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700">
                      📚 {s.knowledge_base_name}
                    </div>
                  )}
                </div>
                {(hover === s.id || active) && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm("确定删除该会话吗？")) onDelete(s.id);
                    }}
                    className="text-xs text-slate-400 hover:text-red-500"
                    aria-label="删除会话"
                  >
                    ✕
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      </div>
      <div className="border-t border-slate-200 p-3 text-[11px] text-slate-400">
        LRAG · 轻量级 RAG 对话系统
      </div>
    </aside>
  );
}

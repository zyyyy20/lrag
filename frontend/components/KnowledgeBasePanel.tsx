"use client";

import { useRef, useState } from "react";
import type { DocumentItem, DocumentStatus, KnowledgeBase } from "@/lib/types";

interface Props {
  knowledgeBases: KnowledgeBase[];
  selectedKbId: string | null;
  documents: DocumentItem[];
  uploading: boolean;
  documentsLoading: boolean;
  error: string | null;
  onSelectKb: (id: string | null) => void;
  onCreateKb: (name: string, description?: string) => Promise<void>;
  onDeleteKb: (id: string) => void;
  onUpload: (file: File) => void;
  onDeleteDocument: (id: string) => void;
  onRefresh: () => void;
}

const STATUS_COLORS: Record<DocumentStatus, string> = {
  uploaded: "bg-slate-100 text-slate-600",
  processing: "bg-amber-100 text-amber-700",
  indexed: "bg-emerald-100 text-emerald-700",
  failed: "bg-red-100 text-red-700",
  deleted: "bg-slate-100 text-slate-500",
};

const STATUS_LABELS: Record<DocumentStatus, string> = {
  uploaded: "已上传",
  processing: "处理中",
  indexed: "已索引",
  failed: "失败",
  deleted: "已删除",
};

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function KnowledgeBasePanel({
  knowledgeBases,
  selectedKbId,
  documents,
  uploading,
  documentsLoading,
  error,
  onSelectKb,
  onCreateKb,
  onDeleteKb,
  onUpload,
  onDeleteDocument,
  onRefresh,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);

  const selectedKb =
    knowledgeBases.find((kb) => kb.id === selectedKbId) ?? null;

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await onCreateKb(name, newDesc.trim() || undefined);
      setNewName("");
      setNewDesc("");
      setShowCreate(false);
    } finally {
      setCreating(false);
    }
  };

  return (
    <aside className="flex h-full w-96 flex-col border-l border-slate-200 bg-white">
      {/* Knowledge bases section */}
      <div className="flex flex-col border-b border-slate-200">
        <div className="flex items-center justify-between p-3">
          <h2 className="text-sm font-semibold text-slate-700">知识库</h2>
          <button
            onClick={() => setShowCreate((v) => !v)}
            className="rounded-md bg-brand-600 px-2.5 py-1 text-xs font-medium text-white shadow-sm hover:bg-brand-700"
          >
            {showCreate ? "取消" : "+ 新建"}
          </button>
        </div>

        {showCreate && (
          <form onSubmit={handleCreate} className="space-y-2 px-3 pb-3">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="知识库名称（必填）"
              maxLength={255}
              className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
              autoFocus
            />
            <textarea
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="描述（可选）"
              rows={2}
              className="w-full resize-none rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            <button
              type="submit"
              disabled={creating || !newName.trim()}
              className="w-full rounded-md bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {creating ? "创建中…" : "创建知识库"}
            </button>
          </form>
        )}

        <div className="max-h-56 overflow-y-auto px-2 pb-2">
          {knowledgeBases.length === 0 && (
            <p className="px-2 py-3 text-xs text-slate-400">
              暂无知识库。点击"+ 新建"创建第一个。
            </p>
          )}
          <ul className="space-y-1">
            {knowledgeBases.map((kb) => {
              const active = kb.id === selectedKbId;
              return (
                <li
                  key={kb.id}
                  onClick={() => onSelectKb(active ? null : kb.id)}
                  className={`group flex items-center justify-between rounded-md px-2 py-2 text-sm cursor-pointer ${
                    active
                      ? "bg-brand-50 text-brand-700"
                      : "text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  <div className="min-w-0 flex-1 pr-2">
                    <div className="truncate font-medium">{kb.name}</div>
                    <div className="text-[11px] text-slate-400">
                      {kb.document_count} 个文档
                      {kb.description ? ` · ${kb.description}` : ""}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (
                        confirm(
                          `确定删除知识库"${kb.name}"吗？该知识库下所有文档与索引将被一并删除。`
                        )
                      )
                        onDeleteKb(kb.id);
                    }}
                    className="opacity-0 transition-opacity group-hover:opacity-100 text-slate-400 hover:text-red-500"
                    aria-label="删除知识库"
                  >
                    ✕
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      {/* Documents section */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-200 p-3">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-slate-700">
              {selectedKb ? `文档 · ${selectedKb.name}` : "文档"}
            </h2>
            {!selectedKb && (
              <p className="text-[11px] text-slate-400">请先选择左上方的知识库</p>
            )}
          </div>
          <button
            onClick={onRefresh}
            className="text-xs text-slate-500 hover:text-slate-800"
            disabled={!selectedKb}
          >
            刷新
          </button>
        </div>

        <div className="border-b border-slate-200 p-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.markdown"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = "";
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || !selectedKb}
            className="w-full rounded-md border border-dashed border-slate-300 px-3 py-4 text-sm text-slate-600 hover:border-brand-500 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {!selectedKb
              ? "请先选择知识库"
              : uploading
              ? "上传中…"
              : "+ 上传 PDF / TXT / Markdown"}
          </button>
          {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {documentsLoading && (
            <p className="p-4 text-xs text-slate-400">加载中…</p>
          )}
          {!documentsLoading && selectedKb && documents.length === 0 && (
            <p className="p-4 text-xs text-slate-400">
              当前知识库暂无文档，上传后将显示在此处。
            </p>
          )}
          {!selectedKb && !documentsLoading && (
            <p className="p-4 text-xs text-slate-400">
              选择左上方的某个知识库以查看其文档。
            </p>
          )}
          <ul className="space-y-2">
            {documents.map((d) => (
              <li
                key={d.id}
                className="group rounded-md border border-slate-200 p-2 text-xs"
              >
                <div className="flex items-start justify-between gap-2">
                  <span
                    className="break-all font-medium text-slate-700"
                    title={d.filename}
                  >
                    {d.filename}
                  </span>
                  <button
                    onClick={() => {
                      if (confirm(`确定删除"${d.filename}"吗？`))
                        onDeleteDocument(d.id);
                    }}
                    className="text-slate-400 hover:text-red-500"
                    aria-label="删除文档"
                  >
                    ✕
                  </button>
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                  <span
                    className={`rounded px-1.5 py-0.5 font-medium ${
                      STATUS_COLORS[d.status]
                    }`}
                  >
                    {STATUS_LABELS[d.status]}
                  </span>
                  <span>{formatSize(d.file_size)}</span>
                  {d.status === "indexed" && (
                    <span>{d.chunk_count} 个片段</span>
                  )}
                </div>
                {d.status === "failed" && d.error && (
                  <p className="mt-1 text-[11px] text-red-600">{d.error}</p>
                )}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </aside>
  );
}

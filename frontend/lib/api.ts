import type {
  ChatMode,
  ChatResponse,
  DocumentItem,
  KnowledgeBase,
  SessionDetail,
  SessionItem,
} from "./types";

const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE) ||
  "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let message = `Request failed: ${res.status}`;
    try {
      const body = await res.json();
      message = body?.error?.message || body?.detail || message;
    } catch {
      // ignore
    }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  // ---- Knowledge bases ----
  listKnowledgeBases: () =>
    request<KnowledgeBase[]>("/api/knowledge-bases"),
  createKnowledgeBase: (name: string, description?: string) =>
    request<KnowledgeBase>("/api/knowledge-bases", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),
  getKnowledgeBase: (id: string) =>
    request<KnowledgeBase>(`/api/knowledge-bases/${id}`),
  updateKnowledgeBase: (id: string, payload: { name?: string; description?: string }) =>
    request<KnowledgeBase>(`/api/knowledge-bases/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteKnowledgeBase: (id: string) =>
    request<void>(`/api/knowledge-bases/${id}`, { method: "DELETE" }),

  // ---- Sessions ----
  listSessions: () => request<SessionItem[]>("/api/sessions"),
  createSession: (payload: {
    title?: string;
    chat_mode?: ChatMode;
    knowledge_base_id?: string | null;
  }) =>
    request<SessionItem>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),
  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  // ---- Chat ----
  chat: (message: string, sessionId?: string | null) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId ?? null }),
    }),

  // ---- Documents ----
  listDocuments: (knowledgeBaseId?: string | null) => {
    const qs = knowledgeBaseId ? `?knowledge_base_id=${knowledgeBaseId}` : "";
    return request<DocumentItem[]>(`/api/documents${qs}`);
  },
  uploadDocument: (file: File, knowledgeBaseId: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("knowledge_base_id", knowledgeBaseId);
    return request<DocumentItem>("/api/documents/upload", {
      method: "POST",
      body: fd,
    });
  },
  deleteDocument: (id: string) =>
    request<void>(`/api/documents/${id}`, { method: "DELETE" }),
};

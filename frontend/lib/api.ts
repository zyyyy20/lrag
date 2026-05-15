import type {
  ChatMode,
  ChatResponse,
  DocumentItem,
  KnowledgeBase,
  SessionDetail,
  SessionItem,
} from "./types";
import { getDebugGuestHeaders } from "./guest";

const API_BASE =
  (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE) ||
  "http://localhost:8000";

export function apiUrl(pathOrUrl: string): string {
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  return `${API_BASE}${pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const guestHeaders = await getDebugGuestHeaders();
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...guestHeaders,
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
  getKnowledgeBase: (id: number) =>
    request<KnowledgeBase>(`/api/knowledge-bases/${id}`),
  updateKnowledgeBase: (id: number, payload: { name?: string; description?: string }) =>
    request<KnowledgeBase>(`/api/knowledge-bases/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteKnowledgeBase: (id: number) =>
    request<void>(`/api/knowledge-bases/${id}`, { method: "DELETE" }),

  // ---- Sessions ----
  listSessions: () => request<SessionItem[]>("/api/sessions"),
  createSession: (payload: {
    title?: string;
    chat_mode?: ChatMode;
    knowledge_base_id?: number | null;
  }) =>
    request<SessionItem>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  getSession: (id: number) => request<SessionDetail>(`/api/sessions/${id}`),
  deleteSession: (id: number) =>
    request<void>(`/api/sessions/${id}`, { method: "DELETE" }),

  // ---- Chat ----
  chat: (message: string, sessionId?: number | null) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId ?? null }),
    }),

  chatStream: (
    message: string,
    sessionId: number | null,
    handlers: ChatStreamHandlers,
    signal?: AbortSignal
  ) => chatStream(message, sessionId, handlers, signal),

  // ---- Documents ----
  listDocuments: (knowledgeBaseId?: number | null) => {
    const qs = knowledgeBaseId ? `?knowledge_base_id=${knowledgeBaseId}` : "";
    return request<DocumentItem[]>(`/api/documents${qs}`);
  },
  uploadDocument: (file: File, knowledgeBaseId: number) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("knowledge_base_id", String(knowledgeBaseId));
    return request<DocumentItem>("/api/documents/upload", {
      method: "POST",
      body: fd,
    });
  },
  deleteDocument: (id: number) =>
    request<void>(`/api/documents/${id}`, { method: "DELETE" }),
};


// ===== SSE streaming chat =====

export interface ChatStreamMeta {
  session_id: number;
  used_rag: boolean;
  sources: import("./types").Source[];
  notice?: string | null;
}

export interface ChatStreamDone {
  session_id: number;
  title: string;
}

export interface ChatStreamHandlers {
  onMeta?: (meta: ChatStreamMeta) => void;
  onDelta?: (delta: string) => void;
  onToolResult?: (result: import("./types").ToolResult) => void;
  onDone?: (done: ChatStreamDone) => void;
  onError?: (message: string) => void;
}

async function chatStream(
  message: string,
  sessionId: number | null,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const guestHeaders = await getDebugGuestHeaders();
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      ...guestHeaders,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = `Stream failed: ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.error?.message || body?.detail || detail;
    } catch {
      /* ignore */
    }
    handlers.onError?.(detail);
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const dispatch = (rawEvent: string) => {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      // ignore comments (":") and other fields
    }
    if (dataLines.length === 0) return;
    const dataStr = dataLines.join("\n");
    let data: unknown;
    try {
      data = JSON.parse(dataStr);
    } catch {
      return;
    }
    if (eventName === "meta") {
      handlers.onMeta?.(data as ChatStreamMeta);
    } else if (eventName === "delta") {
      const content = (data as { content?: string }).content;
      if (typeof content === "string" && content.length > 0) {
        handlers.onDelta?.(content);
      }
    } else if (eventName === "done") {
      handlers.onDone?.(data as ChatStreamDone);
    } else if (eventName === "tool_result") {
      handlers.onToolResult?.(data as import("./types").ToolResult);
    } else if (eventName === "error") {
      const msg = (data as { message?: string }).message ?? "stream error";
      handlers.onError?.(msg);
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line ("\n\n")
      let sepIdx: number;
      while ((sepIdx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sepIdx);
        buffer = buffer.slice(sepIdx + 2);
        if (frame.trim()) dispatch(frame);
      }
    }
    // flush any trailing frame
    const tail = buffer.trim();
    if (tail) dispatch(tail);
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    handlers.onError?.((e as Error).message);
    throw e;
  }
}

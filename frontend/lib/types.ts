export type DocumentStatus =
  | "uploaded"
  | "processing"
  | "indexed"
  | "failed"
  | "deleted";

export type ChatMode = "general" | "rag";

export interface KnowledgeBase {
  id: number;
  name: string;
  description?: string | null;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface Source {
  knowledge_base_id?: number | null;
  document_id?: number | null;
  filename: string;
  chunk_index: number;
  score: number;
  content_preview: string;
}

export interface Message {
  id: number | string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: Source[] | null;
  used_rag?: boolean | null;
  created_at: string;
}

export interface SessionItem {
  id: number;
  title: string;
  chat_mode: ChatMode;
  knowledge_base_id?: number | null;
  knowledge_base_name?: string | null;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail extends SessionItem {
  messages: Message[];
}

export interface ChatResponse {
  session_id: number;
  answer: string;
  used_rag: boolean;
  sources: Source[];
  notice?: string | null;
}

export interface DocumentItem {
  id: number;
  knowledge_base_id: number;
  filename: string;
  content_type: string;
  file_size: number;
  status: DocumentStatus;
  chunk_count: number;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

export type DocumentStatus =
  | "uploaded"
  | "processing"
  | "indexed"
  | "failed"
  | "deleted";

export type ChatMode = "general" | "rag";

export interface KnowledgeBase {
  id: string;
  name: string;
  description?: string | null;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface Source {
  knowledge_base_id?: string | null;
  document_id?: string | null;
  filename: string;
  chunk_index: number;
  score: number;
  content_preview: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: Source[] | null;
  used_rag?: boolean | null;
  created_at: string;
}

export interface SessionItem {
  id: string;
  title: string;
  chat_mode: ChatMode;
  knowledge_base_id?: string | null;
  knowledge_base_name?: string | null;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail extends SessionItem {
  messages: Message[];
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  used_rag: boolean;
  sources: Source[];
  notice?: string | null;
}

export interface DocumentItem {
  id: string;
  knowledge_base_id: string;
  filename: string;
  content_type: string;
  file_size: number;
  status: DocumentStatus;
  chunk_count: number;
  error?: string | null;
  created_at: string;
  updated_at: string;
}

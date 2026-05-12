"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatArea } from "@/components/ChatArea";
import { KnowledgeBasePanel } from "@/components/KnowledgeBasePanel";
import { Sidebar } from "@/components/Sidebar";
import { api } from "@/lib/api";
import type {
  DocumentItem,
  KnowledgeBase,
  Message,
  SessionItem,
} from "@/lib/types";

export default function HomePage() {
  // ---- Sessions ----
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatNotice, setChatNotice] = useState<string | null>(null);

  // ---- Knowledge bases ----
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedKbId, setSelectedKbId] = useState<string | null>(null);

  // ---- Documents (for selected KB) ----
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);

  const docsPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const currentSession = useMemo(
    () => sessions.find((s) => s.id === currentSessionId) ?? null,
    [sessions, currentSessionId]
  );

  const refreshSessions = useCallback(async () => {
    setLoadingSessions(true);
    try {
      const list = await api.listSessions();
      setSessions(list);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingSessions(false);
    }
  }, []);

  const refreshKBs = useCallback(async () => {
    try {
      const list = await api.listKnowledgeBases();
      setKnowledgeBases(list);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const refreshDocuments = useCallback(
    async (kbId: string | null) => {
      if (!kbId) {
        setDocuments([]);
        return;
      }
      setDocumentsLoading(true);
      try {
        const list = await api.listDocuments(kbId);
        setDocuments(list);
      } catch (e) {
        console.error(e);
      } finally {
        setDocumentsLoading(false);
      }
    },
    []
  );

  const loadSession = useCallback(async (id: string) => {
    setLoadingMessages(true);
    setChatError(null);
    setChatNotice(null);
    try {
      const detail = await api.getSession(id);
      setMessages(detail.messages);
    } catch (e) {
      setChatError(e instanceof Error ? e.message : String(e));
      setMessages([]);
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
    refreshKBs();
  }, [refreshSessions, refreshKBs]);

  useEffect(() => {
    refreshDocuments(selectedKbId);
  }, [selectedKbId, refreshDocuments]);

  // Poll while any doc in current KB is pending
  useEffect(() => {
    const pending = documents.some(
      (d) => d.status === "processing" || d.status === "uploaded"
    );
    if (pending && selectedKbId) {
      if (!docsPollRef.current) {
        docsPollRef.current = setInterval(
          () => refreshDocuments(selectedKbId),
          2000
        );
      }
    } else if (docsPollRef.current) {
      clearInterval(docsPollRef.current);
      docsPollRef.current = null;
    }
    return () => {
      if (docsPollRef.current && !pending) {
        clearInterval(docsPollRef.current);
        docsPollRef.current = null;
      }
    };
  }, [documents, selectedKbId, refreshDocuments]);

  useEffect(() => {
    if (currentSessionId) loadSession(currentSessionId);
    else setMessages([]);
  }, [currentSessionId, loadSession]);

  // ---- Handlers ----
  const handleNewSession = async ({
    knowledgeBaseId,
  }: {
    knowledgeBaseId: string | null;
  }) => {
    try {
      const s = await api.createSession({
        chat_mode: knowledgeBaseId ? "rag" : "general",
        knowledge_base_id: knowledgeBaseId,
      });
      await refreshSessions();
      setCurrentSessionId(s.id);
      setMessages([]);
      setChatError(null);
      setChatNotice(null);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  const handleDeleteSession = async (id: string) => {
    try {
      await api.deleteSession(id);
      if (currentSessionId === id) {
        setCurrentSessionId(null);
        setMessages([]);
      }
      await refreshSessions();
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSend = async (text: string) => {
    setSending(true);
    setChatError(null);
    setChatNotice(null);
    const optimistic: Message = {
      id: `tmp-${Date.now()}`,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimistic]);
    try {
      const res = await api.chat(text, currentSessionId);
      const assistant: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: res.answer,
        used_rag: res.used_rag,
        sources: res.sources,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistant]);
      if (res.notice) setChatNotice(res.notice);
      if (!currentSessionId) {
        setCurrentSessionId(res.session_id);
      }
      refreshSessions();
    } catch (e) {
      setChatError(e instanceof Error ? e.message : String(e));
      setMessages((prev) => prev.filter((m) => m.id !== optimistic.id));
    } finally {
      setSending(false);
    }
  };

  const handleCreateKb = async (name: string, description?: string) => {
    try {
      const kb = await api.createKnowledgeBase(name, description);
      await refreshKBs();
      setSelectedKbId(kb.id);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  const handleDeleteKb = async (id: string) => {
    try {
      await api.deleteKnowledgeBase(id);
      if (selectedKbId === id) setSelectedKbId(null);
      await Promise.all([refreshKBs(), refreshSessions()]);
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  const handleUpload = async (file: File) => {
    if (!selectedKbId) return;
    setUploading(true);
    setDocError(null);
    try {
      await api.uploadDocument(file, selectedKbId);
      await refreshDocuments(selectedKbId);
      await refreshKBs();
    } catch (e) {
      setDocError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      await api.deleteDocument(id);
      await refreshDocuments(selectedKbId);
      await refreshKBs();
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <main className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        sessions={sessions}
        currentId={currentSessionId}
        loading={loadingSessions}
        knowledgeBases={knowledgeBases}
        onNew={handleNewSession}
        onSelect={setCurrentSessionId}
        onDelete={handleDeleteSession}
      />
      <ChatArea
        messages={messages}
        loading={loadingMessages}
        sending={sending}
        error={chatError}
        notice={chatNotice}
        currentSession={currentSession}
        onSend={handleSend}
      />
      <KnowledgeBasePanel
        knowledgeBases={knowledgeBases}
        selectedKbId={selectedKbId}
        documents={documents}
        uploading={uploading}
        documentsLoading={documentsLoading}
        error={docError}
        onSelectKb={setSelectedKbId}
        onCreateKb={handleCreateKb}
        onDeleteKb={handleDeleteKb}
        onUpload={handleUpload}
        onDeleteDocument={handleDeleteDocument}
        onRefresh={() => refreshDocuments(selectedKbId)}
      />
    </main>
  );
}

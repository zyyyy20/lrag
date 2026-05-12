import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LRAG · 轻量级 RAG 对话系统",
  description: "一个轻量级、接近生产级的 RAG 知识库对话系统。",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

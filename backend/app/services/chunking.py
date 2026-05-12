"""Text extraction and chunking utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

# Common MIME types we accept. Word's content-type tends to be long, so we just
# look for the well-known token in a case-insensitive way.
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def extract_text(file_path: str, content_type: str | None = None) -> str:
    p = Path(file_path)
    ext = p.suffix.lower()
    ct = (content_type or "").lower()

    if ext == ".pdf" or "pdf" in ct:
        return _extract_pdf(p)
    if ext == ".docx" or "wordprocessingml" in ct or ct == DOCX_MIME:
        return _extract_docx(p)
    if ext in {".txt", ".md", ".markdown"} or "text" in ct:
        return p.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file extension: {ext}")


def _extract_pdf(p: Path) -> str:
    parts: List[str] = []
    with p.open("rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n".join(parts)


def _extract_docx(p: Path) -> str:
    """Extract text from a .docx file. Includes paragraphs and table cells."""
    doc = DocxDocument(str(p))
    parts: List[str] = []
    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if text:
            parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [
                (cell.text or "").strip()
                for cell in row.cells
                if (cell.text or "").strip()
            ]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Character-based chunking that prefers splitting on paragraph / sentence
    boundaries. Robust for mixed Chinese/English content."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))

    text = normalize_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to break on a natural boundary within the last 30% of the window
        if end < n:
            window_start = start + int(chunk_size * 0.7)
            slice_ = text[window_start:end]
            for sep in ["\n\n", "\n", "。", ". ", "！", "？", "!", "?", "; ", "；"]:
                idx = slice_.rfind(sep)
                if idx != -1:
                    end = window_start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

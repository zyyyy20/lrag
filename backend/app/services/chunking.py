"""Text extraction and chunking utilities."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def extract_text(file_path: str, content_type: str | None = None) -> str:
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext == ".pdf" or (content_type and "pdf" in content_type):
        return _extract_pdf(p)
    if ext in {".txt", ".md", ".markdown"} or (content_type and "text" in (content_type or "")):
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
